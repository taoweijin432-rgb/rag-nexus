"""Resume CDC Runtime 的协议、数据集和资源隔离测试。"""

from __future__ import annotations

import pytest

from expert_index.cdc import (
    CanalBatch,
    CanalChangeEvent,
    CdcRuntimeDisabledError,
)
from rag_vector_store import MilvusSettings
from rag_vector_store.milvus.repository_v15 import RESUME_V15_PROFILE
from resume_index.cdc_runtime import (
    build_resume_cdc_event_id,
    build_resume_cdc_protocol_baseline,
    build_resume_cdc_runtime,
)
from resume_index.config import (
    RESUME_CANAL_DESTINATION,
    RESUME_CANAL_SUBSCRIPTION,
    ResumeSettings,
)
from resume_index.dataset import RESUME_SPEC
from resume_index.preflight import ResumeSourcePreflightResult
from resume_index.source import MySQLResumeSourceAdapter


def _settings(*, cdc_enabled: bool = True) -> ResumeSettings:
    return ResumeSettings.from_env(
        {
            "RESUME_SOURCE_DB_HOST": "source.mysql.local",
            "RESUME_SOURCE_DB_PORT": "3306",
            "RESUME_SOURCE_DB_NAME": "expert",
            "RESUME_SOURCE_DB_USER": "resume_reader",
            "RESUME_SOURCE_DB_PASSWORD": "source-secret",
            "RESUME_INDEX_DB_HOST": "127.0.0.1",
            "RESUME_INDEX_DB_PORT": "3307",
            "RESUME_INDEX_DB_NAME": "resume_index",
            "RESUME_INDEX_DB_USER": "resume_writer",
            "RESUME_INDEX_DB_PASSWORD": "index-secret",
            "RESUME_MILVUS_COLLECTION_NAME": "resume_candidates_v15_contract",
            "RESUME_INCREMENTAL_ENABLED": "1",
            "RESUME_INCREMENTAL_BATCH_SIZE": "200",
            "RESUME_INCREMENTAL_LEASE_SECONDS": "120",
            "RESUME_CDC_ENABLED": "1" if cdc_enabled else "0",
            "RESUME_CANAL_DESTINATION": RESUME_CANAL_DESTINATION,
            "RESUME_CANAL_CLIENT_ID": "2002" if cdc_enabled else "",
            "RESUME_CANAL_SUBSCRIPTION": RESUME_CANAL_SUBSCRIPTION,
            "RESUME_CDC_SOURCE_CONNECTED": "1",
            "RESUME_CDC_PROBE_BATCH_ID": "17",
            "RESUME_CDC_PROBE_ENTRY_COUNT": "8",
            "RESUME_CDC_ACK_CONFIRMED": "1",
            "RESUME_CDC_PROTOCOL_READY": "1",
            "SOURCE_DB_HOST": "source.mysql.local",
            "SOURCE_DB_PORT": "3306",
            "SOURCE_DB_NAME": "expert",
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
    profile = RESUME_V15_PROFILE


class _VectorService:
    repository = _VectorRepository()
    owner_field = "resume_id"


class _CanalClient:
    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def fetch_without_ack(self, _batch_size: int) -> CanalBatch:
        return CanalBatch(-1, (), RESUME_CANAL_DESTINATION)

    def rollback(self, _batch_id: int) -> None:
        return None

    def ack(self, _batch_id: int) -> None:
        return None


def test_resume_protocol_baseline_uses_frozen_contract() -> None:
    baseline = build_resume_cdc_protocol_baseline(_settings())

    assert baseline.destination == RESUME_CANAL_DESTINATION
    assert baseline.subscription == RESUME_CANAL_SUBSCRIPTION
    assert baseline.production_ready is True


def test_resume_event_identity_uses_resume_destination_namespace() -> None:
    event = CanalChangeEvent(
        "expert",
        "resume_work_experience",
        81001,
        70001,
        "binlog.000021",
        100,
        1,
        1787747052000,
        0,
        0,
        "delete",
    )

    assert build_resume_cdc_event_id(event) == build_resume_cdc_event_id(event)
    from expert_index.cdc import build_cdc_event_id

    assert build_resume_cdc_event_id(event) != build_cdc_event_id(event)


def test_resume_runtime_injects_resume_dataset_source_and_collection() -> None:
    settings = _settings()
    source = MySQLResumeSourceAdapter(settings.source_db)
    baseline = build_resume_cdc_protocol_baseline(settings)

    runtime = build_resume_cdc_runtime(
        settings,
        client=_CanalClient(),
        protocol_baseline=baseline,
        max_attempts=4,
        retry_delay_seconds=30,
        task_lease_seconds=300,
        source=source,
        vector_service=_VectorService(),  # type: ignore[arg-type]
        preflight_runner=_preflight,
    )

    coordinator = runtime.runtime.coordinator
    assert coordinator._receiver.dataset_spec is RESUME_SPEC
    assert coordinator._receiver.destination == RESUME_CANAL_DESTINATION
    assert coordinator._receiver.source_schema == "expert"
    assert coordinator._task_processor._source is source
    assert coordinator._task_processor._dataset_spec is RESUME_SPEC
    assert runtime.preflight.index_schema == "resume_index"

    result = runtime.consume_once(batch_size=10, local_batch_id=1)
    runtime.stop()

    assert result.status == "empty"
    assert result.local_batch_id == 1


def test_disabled_resume_cdc_fails_before_preflight() -> None:
    called = False
    settings = _settings(cdc_enabled=False)

    def preflight(_settings: ResumeSettings) -> ResumeSourcePreflightResult:
        nonlocal called
        called = True
        raise AssertionError("禁用时不得执行来源预检")

    with pytest.raises(CdcRuntimeDisabledError, match="RESUME_CDC_ENABLED"):
        build_resume_cdc_runtime(
            settings,
            client=_CanalClient(),
            protocol_baseline=build_resume_cdc_protocol_baseline(settings),
            max_attempts=4,
            retry_delay_seconds=30,
            task_lease_seconds=300,
            vector_service=_VectorService(),  # type: ignore[arg-type]
            preflight_runner=preflight,
        )

    assert called is False
