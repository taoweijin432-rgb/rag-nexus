"""Resume Poll 的任务映射、运行时隔离和最小范围构建测试。"""

from __future__ import annotations

from datetime import datetime

import pytest

from expert_index.chunk_builder import ChunkFactory
from expert_index.models import (
    EmploymentSource,
    ExpertSource,
    ScopedCurrentFacts,
    SourceChangeCandidate,
    SourceSyncTask,
    impact_scope_from_target,
    source_sync_task_to_read_request,
)
from expert_index.preparation_dispatcher import build_prepared_chunk_set
from expert_index.scanner import (
    IncrementalScanCoordinator,
    IncrementalScanResult,
    map_poll_candidate_to_task,
)
from rag_vector_store import MilvusSettings
from resume_index.config import ResumeSettings
from resume_index.dataset import RESUME_SPEC
from resume_index.incremental_runtime import build_resume_incremental_runtime
from resume_index.preflight import ResumeSourcePreflightResult
from resume_index.source import MySQLResumeSourceAdapter

NOW = datetime(2026, 9, 2, 10, 0, 0)  # noqa: DTZ001


def _settings(*, enabled: bool = True) -> ResumeSettings:
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
            "RESUME_INCREMENTAL_ENABLED": "1" if enabled else "0",
            "RESUME_INCREMENTAL_BATCH_SIZE": "200",
            "RESUME_INCREMENTAL_LEASE_SECONDS": "120",
            "SOURCE_DB_HOST": "source.mysql.local",
            "SOURCE_DB_PORT": "3306",
            "SOURCE_DB_NAME": "shared_source",
            "INDEX_DB_HOST": "127.0.0.1",
            "INDEX_DB_PORT": "3306",
            "INDEX_DB_NAME": "expert_index",
            "MILVUS_COLLECTION_NAME": "expert_candidates_v15_contract",
        }
    )


def _preflight(settings: ResumeSettings) -> ResumeSourcePreflightResult:
    return ResumeSourcePreflightResult(
        source_schema=settings.source_db.database,
        source_tables=RESUME_SPEC.source_tables,
        index_schema=settings.index_db.database,
        milvus_collection_name=settings.milvus_collection_name,
    )


class _VectorRepository:
    settings = MilvusSettings(collection_name="resume_candidates_v15_contract")
    _owner_field = "resume_id"
    _vector_doc_id_prefix = "resume:"


class _VectorService:
    repository = _VectorRepository()


def _task(new_task) -> SourceSyncTask:
    return SourceSyncTask(
        task_id=91,
        dedupe_key=new_task.dedupe_key,
        batch_id=new_task.batch_id,
        event_source=new_task.event_source,
        source_table=new_task.source_table,
        source_record_id=new_task.source_record_id,
        source_updated_at=new_task.source_updated_at,
        expert_id=new_task.expert_id,
        operation_type=new_task.operation_type,
        target_type=new_task.target_type,
        processing_stage="candidate",
        sync_status="pending",
        retry_count=0,
        policy_version=new_task.policy_version,
        created_at=NOW,
        updated_at=NOW,
    )


def test_resume_candidate_maps_physical_id_to_resume_person_scope() -> None:
    candidate = SourceChangeCandidate(
        source_table="resume",
        record_id=17,
        expert_id=70001,
        updated_at=NOW,
        deleted_at=None,
    )

    new_task = map_poll_candidate_to_task(
        candidate,
        batch_id=8,
        policy_version="v1.3",
        dataset_spec=RESUME_SPEC,
    )
    request = source_sync_task_to_read_request(
        _task(new_task),
        dataset_spec=RESUME_SPEC,
    )

    assert new_task.source_record_id == 17
    assert new_task.expert_id == 70001
    assert new_task.target_type == "expert"
    assert request.impact_scope.scope_type == "expert"
    assert request.trigger_source_field_id == "17"


def test_resume_scan_coordinator_uses_frozen_four_table_order() -> None:
    calls: list[tuple[str, str, str]] = []

    class ScanService:
        def scan_source(
            self,
            source_table: str,
            lease_owner: str,
            *,
            batch_no: str,
        ) -> IncrementalScanResult:
            calls.append((source_table, lease_owner, batch_no))
            return IncrementalScanResult(
                source_table=source_table,
                batch_source=None,
                recovered=False,
                page_count=0,
                scheduled_candidate_count=0,
                lease_released=True,
                lease_owner=lease_owner,
            )

    result = IncrementalScanCoordinator(
        ScanService(),  # type: ignore[arg-type]
        dataset_spec=RESUME_SPEC,
    ).scan_all_sources("resume-worker", cycle_id="cycle-1")

    assert tuple(source_table for source_table, _owner, _batch in calls) == (
        RESUME_SPEC.source_tables
    )
    assert tuple(outcome.source_table for outcome in result.outcomes) == (RESUME_SPEC.source_tables)
    assert all("resume" not in batch_no for _source, _owner, batch_no in calls)


def test_resume_employment_scope_builds_resume_source_labels() -> None:
    new_task = map_poll_candidate_to_task(
        SourceChangeCandidate(
            source_table="resume_work_experience",
            record_id=81001,
            expert_id=70001,
            updated_at=NOW,
            deleted_at=None,
        ),
        batch_id=8,
        policy_version="v1.3",
        dataset_spec=RESUME_SPEC,
    )
    task = _task(new_task)
    scope = impact_scope_from_target(
        target_type="employment",
        expert_id=70001,
        source_field_id="81001",
        dataset_spec=RESUME_SPEC,
    )
    facts = ScopedCurrentFacts(
        scope=scope,
        expert=ExpertSource(
            id=70001,
            serial_number="RESUME-001",
            company="示例科技",
            job_status=1,
            job_name="项目经理",
            contact_status=3,
            doubtful_flag=1,
            abroad_flag=2,
            background="负责知识库建设。",
            work_years=5,
            updated_at=NOW,
            created_at=NOW,
            deleted_at=None,
        ),
        employments=(
            EmploymentSource(
                id=81001,
                expert_id=70001,
                company_name="示例科技",
                position_title="项目经理",
                start_date=NOW,
                end_date=None,
                work_description="工作职责：负责知识库建设。",
                is_employed=1,
                updated_at=NOW,
                created_at=NOW,
                deleted_at=None,
            ),
        ),
    )
    factory = ChunkFactory(
        _settings().to_processing_settings(),
        dataset_spec=RESUME_SPEC,
    )

    prepared = build_prepared_chunk_set(task, facts, factory)

    assert prepared.impact_scope.source_table == "resume_work_experience"
    assert prepared.desired_chunks
    assert {chunk.source_table for chunk in prepared.desired_chunks} == {"resume_work_experience"}


def test_resume_employment_allows_nullable_company_when_role_exists() -> None:
    """Resume 与 Expert 共用同一可空公司契约，岗位存在时仍应生成 Chunk。"""

    new_task = map_poll_candidate_to_task(
        SourceChangeCandidate(
            source_table="resume_work_experience",
            record_id=81001,
            expert_id=70001,
            updated_at=NOW,
            deleted_at=None,
        ),
        batch_id=8,
        policy_version="v1.3",
        dataset_spec=RESUME_SPEC,
    )
    task = _task(new_task)
    scope = impact_scope_from_target(
        target_type="employment",
        expert_id=70001,
        source_field_id="81001",
        dataset_spec=RESUME_SPEC,
    )
    facts = ScopedCurrentFacts(
        scope=scope,
        expert=ExpertSource(
            id=70001,
            serial_number="RESUME-001",
            company="示例科技",
            job_status=1,
            job_name="项目经理",
            contact_status=3,
            doubtful_flag=1,
            abroad_flag=2,
            background="负责知识库建设。",
            work_years=5,
            updated_at=NOW,
            created_at=NOW,
            deleted_at=None,
        ),
        employments=(
            EmploymentSource(
                id=81001,
                expert_id=70001,
                company_name=None,
                position_title="项目经理",
                start_date=NOW,
                end_date=None,
                work_description="工作职责：负责知识库建设。",
                is_employed=1,
                updated_at=NOW,
                created_at=NOW,
                deleted_at=None,
            ),
        ),
    )
    factory = ChunkFactory(
        _settings().to_processing_settings(),
        dataset_spec=RESUME_SPEC,
    )

    prepared = build_prepared_chunk_set(task, facts, factory)

    assert len(prepared.desired_chunks) == 1
    chunk = prepared.desired_chunks[0]
    assert chunk.source_table == "resume_work_experience"
    assert chunk.company_name_raw is None
    assert chunk.role_name_raw == "项目经理"


def test_resume_incremental_runtime_binds_four_resume_sources_without_cdc() -> None:
    settings = _settings()
    source = MySQLResumeSourceAdapter(settings.source_db)

    runtime = build_resume_incremental_runtime(
        settings,
        max_attempts=3,
        retry_delay_seconds=30,
        task_lease_seconds=120,
        source=source,
        vector_service=_VectorService(),  # type: ignore[arg-type]
        preflight_runner=_preflight,
    )

    scanner = runtime.runtime.scanner
    assert scanner._dataset_spec is RESUME_SPEC
    assert scanner.scan_service.source_adapter is source
    assert scanner._dataset_spec.source_tables == RESUME_SPEC.source_tables
    assert runtime.preflight.index_schema == "resume_index"


def test_disabled_resume_incremental_fails_before_preflight() -> None:
    called = False

    def preflight(_settings: ResumeSettings) -> ResumeSourcePreflightResult:
        nonlocal called
        called = True
        raise AssertionError("禁用时不得执行来源预检")

    with pytest.raises(RuntimeError, match="RESUME_INCREMENTAL_ENABLED"):
        build_resume_incremental_runtime(
            _settings(enabled=False),
            max_attempts=3,
            retry_delay_seconds=30,
            task_lease_seconds=120,
            vector_service=_VectorService(),  # type: ignore[arg-type]
            preflight_runner=preflight,
        )

    assert called is False
