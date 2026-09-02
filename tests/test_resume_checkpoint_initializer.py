"""Resume Poll 初始 Checkpoint 的显式位置和不覆盖约束测试。"""

from datetime import datetime

from expert_index.models import SourceCheckpoint, SourceScanCursor
from resume_index.checkpoint_initializer import initialize_resume_checkpoints
from resume_index.config import ResumeSettings
from resume_index.dataset import RESUME_SPEC


def _settings() -> ResumeSettings:
    return ResumeSettings.from_env(
        {
            "RESUME_SOURCE_DB_HOST": "source.mysql.local",
            "RESUME_SOURCE_DB_PORT": "3306",
            "RESUME_SOURCE_DB_NAME": "shared_source",
            "RESUME_SOURCE_DB_USER": "resume_reader",
            "RESUME_SOURCE_DB_PASSWORD": "source-secret",
            "RESUME_INDEX_DB_HOST": "127.0.0.1",
            "RESUME_INDEX_DB_PORT": "3307",
            "RESUME_INDEX_DB_NAME": "resume_index",
            "RESUME_INDEX_DB_USER": "resume_writer",
            "RESUME_INDEX_DB_PASSWORD": "index-secret",
            "RESUME_MILVUS_COLLECTION_NAME": "resume_candidates_v15_contract",
            "SOURCE_DB_HOST": "source.mysql.local",
            "SOURCE_DB_PORT": "3306",
            "SOURCE_DB_NAME": "shared_source",
            "INDEX_DB_HOST": "127.0.0.1",
            "INDEX_DB_PORT": "3306",
            "INDEX_DB_NAME": "expert_index",
            "MILVUS_COLLECTION_NAME": "expert_candidates_v15_contract",
        }
    )


class StateRepository:
    def __init__(self, existing: dict[str, SourceCheckpoint] | None = None) -> None:
        self.checkpoints = dict(existing or {})
        self.inserted: list[SourceCheckpoint] = []

    def get_source_checkpoint(self, source_table: str) -> SourceCheckpoint | None:
        return self.checkpoints.get(source_table)

    def initialize_source_checkpoint(self, checkpoint: SourceCheckpoint) -> None:
        assert checkpoint.source_table not in self.checkpoints
        self.checkpoints[checkpoint.source_table] = checkpoint
        self.inserted.append(checkpoint)


class Source:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_source_scan_upper_bound(self, source_table: str) -> SourceScanCursor | None:
        self.calls.append(source_table)
        if source_table == RESUME_SPEC.education_table:
            return None
        return SourceScanCursor(datetime(2026, 9, 2, 12, 0, 0), 99)  # noqa: DTZ001


def test_beginning_initializes_all_tables_without_reading_source() -> None:
    repository = StateRepository()

    result = initialize_resume_checkpoints(
        _settings(),
        position="beginning",
        source=Source(),
        state_repository=repository,
    )

    assert result.initialized_tables == RESUME_SPEC.source_tables
    assert result.existing_tables == ()
    assert all(item.last_id == 0 for item in repository.inserted)
    assert all(item.last_updated_at == datetime(1970, 1, 1) for item in repository.inserted)  # noqa: DTZ001


def test_current_uses_each_upper_bound_and_never_overwrites_existing_checkpoint() -> None:
    existing = SourceCheckpoint(
        source_table=RESUME_SPEC.profile_table,
        last_updated_at=datetime(2026, 9, 1),  # noqa: DTZ001
        last_id=17,
    )
    repository = StateRepository({existing.source_table: existing})
    source = Source()

    result = initialize_resume_checkpoints(
        _settings(),
        position="current",
        source=source,
        state_repository=repository,
    )

    assert result.existing_tables == (RESUME_SPEC.profile_table,)
    assert RESUME_SPEC.profile_table not in source.calls
    assert repository.checkpoints[RESUME_SPEC.profile_table] is existing
    assert repository.checkpoints[RESUME_SPEC.employment_table].last_id == 99
    assert repository.checkpoints[RESUME_SPEC.education_table].last_id == 0
