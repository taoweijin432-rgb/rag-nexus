"""Resume 全量 Pipeline 的身份、游标、状态和 Contract 复用测试。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from threading import Barrier, Lock, get_ident
from typing import Any

from expert_contracts import ExpertChunkEmbeddingInput
from expert_index.models import ExpertAggregate, ExpertSource
from expert_index.ports import ExpertBatchSyncInput
from expert_index.vector_sync import ExpertVectorSyncResult
from resume_index.config import ResumeSettings
from resume_index.pipeline import ResumeIndexPipeline
from resume_index.source import ResumeCandidate


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


def _aggregate(resume_id: int) -> ExpertAggregate:
    when = datetime(2026, 9, 2, 12, 0, 0)  # noqa: DTZ001
    return ExpertAggregate(
        profile=ExpertSource(
            id=resume_id,
            serial_number=f"RESUME-{resume_id}",
            company="示例科技有限公司",
            job_status=1,
            job_name="项目经理",
            contact_status=3,
            doubtful_flag=1,
            abroad_flag=2,
            background="负责企业知识库建设。",
            work_years=6.0,
            updated_at=when,
            created_at=when,
            deleted_at=None,
        ),
        employments=(),
        projects=(),
        educations=(),
    )


class FakeResumeSource:
    def __init__(self, candidates: list[ResumeCandidate]) -> None:
        self.candidates = candidates
        self.loaded_resume_ids: list[int] = []
        self.upper_bound_limits: list[int | None] = []
        self.page_calls: list[tuple[int, int, int]] = []

    def get_full_scan_upper_bound(self, *, max_resumes: int | None = None) -> int:
        self.upper_bound_limits.append(max_resumes)
        scoped = self.candidates if max_resumes is None else self.candidates[:max_resumes]
        return max((item.source_record_id for item in scoped), default=0)

    def load_full_scan_candidates(
        self,
        *,
        after_source_record_id: int,
        upper_bound_source_record_id: int,
        limit: int,
    ) -> list[ResumeCandidate]:
        self.page_calls.append((after_source_record_id, upper_bound_source_record_id, limit))
        return [
            item
            for item in self.candidates
            if after_source_record_id < item.source_record_id <= upper_bound_source_record_id
        ][:limit]

    def load_resume(self, resume_id: int) -> ExpertAggregate:
        self.loaded_resume_ids.append(resume_id)
        return _aggregate(resume_id)

    def load_resume_candidate(self, resume_id: int) -> ResumeCandidate:
        return next(item for item in self.candidates if item.resume_id == resume_id)


class ConcurrentFakeResumeSource(FakeResumeSource):
    """用屏障证明同一页的两个 Resume 来源快照确实由不同线程并发读取。"""

    def __init__(self, candidates: list[ResumeCandidate]) -> None:
        super().__init__(candidates)
        self.read_barrier = Barrier(2)
        self.read_thread_ids: set[int] = set()
        self.read_thread_ids_lock = Lock()

    def load_resume(self, resume_id: int) -> ExpertAggregate:
        with self.read_thread_ids_lock:
            self.read_thread_ids.add(get_ident())
        # 单线程实现会在这里超时并使测试失败；两个 Worker 会同时到达后继续。
        self.read_barrier.wait(timeout=2)
        return super().load_resume(resume_id)


class FakeRepository:
    def __init__(self) -> None:
        self.lock_modes: list[str] = []
        self.start_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.finish_calls: list[tuple[int, dict[str, Any]]] = []
        self.prepared_ids: list[int] = []
        self.running_ids: list[int] = []
        self.finished_ids: list[int] = []
        self.progress_cursor_ids: list[int] = []
        self.write_thread_ids: set[int] = set()

    def _record_write_thread(self) -> None:
        self.write_thread_ids.add(get_ident())

    @contextmanager
    def build_lock(self, run_mode: str) -> Iterator[None]:
        self.lock_modes.append(run_mode)
        yield

    def start_batch(self, batch_no: str, run_mode: str, **kwargs: Any) -> int:
        self._record_write_thread()
        self.start_calls.append((batch_no, run_mode, kwargs))
        return 501

    def finish_batch(self, batch_id: int, **kwargs: Any) -> None:
        self._record_write_thread()
        self.finish_calls.append((batch_id, kwargs))

    def prepare_batch_items(self, batch_id: int, expert_ids: list[int]) -> None:
        self._record_write_thread()
        assert batch_id == 501
        self.prepared_ids.extend(expert_ids)

    def mark_batch_item_running(self, batch_id: int, expert_id: int) -> None:
        self._record_write_thread()
        assert batch_id == 501
        self.running_ids.append(expert_id)

    def finish_batch_item(self, batch_id: int, expert_id: int, **_kwargs: Any) -> None:
        self._record_write_thread()
        assert batch_id == 501
        self.finished_ids.append(expert_id)

    def update_batch_progress(self, batch_id: int, **kwargs: Any) -> None:
        self._record_write_thread()
        assert batch_id == 501
        self.progress_cursor_ids.append(kwargs["cursor_id"])


class FakeVectorSync:
    def __init__(self) -> None:
        self.calls: list[tuple[int, list[ExpertChunkEmbeddingInput]]] = []
        self.batch_calls: list[list[ExpertBatchSyncInput]] = []
        self.batch_thread_ids: list[int] = []

    def sync_expert(
        self,
        expert_id: int,
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> ExpertVectorSyncResult:
        snapshot = list(records)
        self.calls.append((expert_id, snapshot))
        return ExpertVectorSyncResult(
            expert_id=expert_id,
            current_count=len(snapshot),
            eligible_count=len(snapshot),
            embedded_count=len(snapshot),
            metadata_updated_count=0,
            unchanged_count=0,
            deleted_count=0,
            delete_not_found_count=0,
        )

    def sync_expert_batch(
        self,
        experts: Sequence[ExpertBatchSyncInput],
    ) -> list[ExpertVectorSyncResult]:
        self.batch_thread_ids.append(get_ident())
        batch = list(experts)
        self.batch_calls.append(batch)
        return [self.sync_expert(item.expert_id, item.records) for item in batch]


def _pipeline(
    source: FakeResumeSource,
    repository: FakeRepository,
    vector_sync: FakeVectorSync,
) -> ResumeIndexPipeline:
    return ResumeIndexPipeline(
        _settings(),
        source=source,  # type: ignore[arg-type]
        build_lock_repository=repository,  # type: ignore[arg-type]
        batch_repository=repository,  # type: ignore[arg-type]
        batch_item_repository=repository,  # type: ignore[arg-type]
        vector_sync=vector_sync,
    )


def test_single_resume_uses_resume_labels_and_internal_contract_identity() -> None:
    source = FakeResumeSource(
        [ResumeCandidate(source_record_id=41, resume_id=70001, deleted_at=None)]
    )
    repository = FakeRepository()
    vector_sync = FakeVectorSync()

    result = _pipeline(source, repository, vector_sync).build_single_resume(
        70001,
        data_version="resume-full-v1",
        batch_no="resume_single_test",
    )

    assert result.resume_id == 70001
    assert result.chunk_count == 1
    assert result.vector_sync.resume_count == 1
    assert source.loaded_resume_ids == [70001]
    assert repository.lock_modes == ["single"]
    assert repository.start_calls == [("resume_single_test", "single", {})]

    synced_id, records = vector_sync.calls[0]
    assert synced_id == 70001
    assert records[0].expert_id == 70001
    assert records[0].source.source_table == ("resume+resume_education_experience")
    assert records[0].source.source_field_id == "70001"
    assert records[0].vector_doc_id.startswith("resume:70001:profile_summary:")


def test_single_deleted_resume_disables_vectors_without_reading_body() -> None:
    deleted_at = datetime(2026, 9, 2, 13, 0, 0)  # noqa: DTZ001
    source = FakeResumeSource(
        [ResumeCandidate(source_record_id=41, resume_id=70001, deleted_at=deleted_at)]
    )
    repository = FakeRepository()
    vector_sync = FakeVectorSync()

    result = _pipeline(source, repository, vector_sync).build_single_resume(
        70001,
        data_version="resume-full-v1",
    )

    assert result.resume_id == 70001
    assert result.chunk_count == 0
    assert source.loaded_resume_ids == []
    assert vector_sync.calls == [(70001, [])]
    assert repository.finish_calls[-1][1]["status"] == "success"


def test_batch_uses_physical_cursor_but_persists_and_syncs_resume_id() -> None:
    deleted_at = datetime(2026, 9, 2, 13, 0, 0)  # noqa: DTZ001
    source = FakeResumeSource(
        [
            ResumeCandidate(source_record_id=41, resume_id=70001, deleted_at=None),
            ResumeCandidate(
                source_record_id=42,
                resume_id=90000007,
                deleted_at=deleted_at,
            ),
        ]
    )
    repository = FakeRepository()
    vector_sync = FakeVectorSync()

    result = _pipeline(source, repository, vector_sync).build_resume_batch(
        data_version="resume-full-v1",
        page_size=20,
        max_resumes=2,
        batch_no="resume_full_test",
    )

    assert result.status == "success"
    assert result.upper_bound_source_record_id == 42
    assert result.cursor_source_record_id == 42
    assert result.success_count == 1
    assert result.deactivated_count == 1
    assert source.upper_bound_limits == [2]
    assert source.page_calls == [(0, 42, 20)]
    assert source.loaded_resume_ids == [70001]

    # Resume Index 内部 expert_id 字段承载 resume_id；物理 id 只进入批次游标。
    assert repository.prepared_ids == [70001, 90000007]
    assert repository.running_ids == [70001, 90000007]
    assert repository.finished_ids == [70001, 90000007]
    assert repository.progress_cursor_ids == [42]
    assert [item.expert_id for item in vector_sync.batch_calls[0]] == [
        70001,
        90000007,
    ]
    assert len(vector_sync.batch_calls[0][0].records) == 1
    assert list(vector_sync.batch_calls[0][1].records) == []


def test_batch_worker_count_prepares_resume_page_concurrently_and_commits_in_order() -> None:
    """页内并发只加速领域准备，向量同步和可靠游标仍保持来源顺序。"""

    source = ConcurrentFakeResumeSource(
        [
            ResumeCandidate(source_record_id=41, resume_id=70001, deleted_at=None),
            ResumeCandidate(source_record_id=42, resume_id=70002, deleted_at=None),
        ]
    )
    repository = FakeRepository()
    vector_sync = FakeVectorSync()
    coordinator_thread_id = get_ident()

    result = _pipeline(source, repository, vector_sync).build_resume_batch(
        data_version="resume-parallel-v1",
        page_size=2,
        batch_no="resume_parallel_test",
        worker_count=2,
    )

    assert result.status == "success"
    assert result.cursor_source_record_id == 42
    assert len(source.read_thread_ids) == 2
    assert [item.expert_id for item in vector_sync.batch_calls[0]] == [70001, 70002]
    assert vector_sync.batch_thread_ids == [coordinator_thread_id]
    assert repository.write_thread_ids == {coordinator_thread_id}
    assert repository.finished_ids == [70001, 70002]
    assert repository.progress_cursor_ids == [42]
