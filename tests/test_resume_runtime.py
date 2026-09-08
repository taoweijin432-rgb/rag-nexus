"""Resume Runtime 的预检顺序和 Milvus Collection 绑定测试。"""

from __future__ import annotations

from typing import Any

import pytest

from expert_index.config import DatabaseBoundaryError
from rag_vector_store import MilvusSettings
from rag_vector_store.milvus.repository_v15 import RESUME_V15_PROFILE
from resume_index.config import ResumeSettings
from resume_index.preflight import ResumeSourcePreflightResult
from resume_index.runtime import build_resume_runtime


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


def _preflight(settings: ResumeSettings) -> ResumeSourcePreflightResult:
    return ResumeSourcePreflightResult(
        source_schema=settings.source_db.database,
        source_tables=(
            "resume",
            "resume_work_experience",
            "resume_project_experience",
            "resume_education_experience",
        ),
        index_schema=settings.index_db.database,
        milvus_collection_name=settings.milvus_collection_name,
    )


class FakeVectorSync:
    def __init__(self) -> None:
        self.store = FakeResumeVectorService()

    def sync_expert(self, *_args: object, **_kwargs: object):
        raise AssertionError("本测试不执行向量同步")

    def sync_expert_batch(self, *_args: object, **_kwargs: object):
        raise AssertionError("本测试不执行向量同步")


class FakeResumeRepository:
    settings = MilvusSettings(collection_name="resume_candidates_v15_contract")
    profile = RESUME_V15_PROFILE


class FakeResumeVectorService:
    repository = FakeResumeRepository()


def test_runtime_with_injected_vector_sync_keeps_safe_preflight_summary() -> None:
    settings = _settings()

    runtime = build_resume_runtime(
        settings,
        vector_sync=FakeVectorSync(),  # type: ignore[arg-type]
        preflight_runner=_preflight,
    )

    assert runtime.preflight.index_schema == "resume_index"
    assert runtime.preflight.milvus_collection_name == ("resume_candidates_v15_contract")
    assert runtime.pipeline.settings is settings


def test_runtime_preflights_before_constructing_explicit_resume_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    events: list[str] = []
    captured: dict[str, Any] = {}

    def preflight(received: ResumeSettings) -> ResumeSourcePreflightResult:
        assert received is settings
        events.append("preflight")
        return _preflight(received)

    def fake_create_service(**kwargs: Any) -> object:
        events.append("vector_service")
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "resume_index.runtime.create_resume_vector_ingestion_service",
        fake_create_service,
    )
    base = MilvusSettings(
        uri="http://milvus.test:19530",
        collection_name="expert_candidates_v15_contract",
    )

    build_resume_runtime(
        settings,
        base_milvus_settings=base,
        preflight_runner=preflight,
    )

    assert events == ["preflight", "vector_service"]
    resolved = captured["milvus_settings"]
    assert isinstance(resolved, MilvusSettings)
    assert resolved.collection_name == "resume_candidates_v15_contract"
    assert resolved.uri == "http://milvus.test:19530"


def test_runtime_rejects_two_vector_assembly_paths() -> None:
    with pytest.raises(ValueError, match="只能显式传入一个"):
        build_resume_runtime(
            _settings(),
            vector_sync=FakeVectorSync(),  # type: ignore[arg-type]
            vector_service=object(),  # type: ignore[arg-type]
            preflight_runner=_preflight,
        )


def test_runtime_rejects_vector_service_bound_to_expert_collection() -> None:
    class FakeRepository:
        settings = MilvusSettings(collection_name="expert_candidates_v15_contract")

    class FakeVectorService:
        repository = FakeRepository()

    with pytest.raises(DatabaseBoundaryError, match="未绑定"):
        build_resume_runtime(
            _settings(),
            vector_service=FakeVectorService(),  # type: ignore[arg-type]
            preflight_runner=_preflight,
        )
