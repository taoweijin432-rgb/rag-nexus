"""Resume Adapter、V15 映射和领域隔离的 Fake 回归测试。

测试只使用合成 Contract 和内存 Repository，不连接 Resume 来源库、不访问 Milvus，
也不调用真实 Embedding API。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from expert_contracts import adapt_expert_chunk_mapping
from expert_contracts.models import ExpertChunkEmbeddingInput
from expert_index.config import DatabaseBoundaryError
from rag_vector_store import (
    FORMAL_EMBEDDING_MODEL_SPACE_ID,
    MilvusSettings,
    MutationResult,
    ResumeVectorIngestionService,
    VectorContractError,
    create_resume_vector_ingestion_service,
    create_vector_ingestion_service,
)
from rag_vector_store.core.embedding import DomainEmbeddingWorker
from rag_vector_store.core.ingestion import DomainIngestionService
from rag_vector_store.domains import ExpertContractAdapter, ResumeContractAdapter
from rag_vector_store.milvus.contract_v15 import (
    MilvusV15RowMapper,
    ResumeMilvusV15RowMapper,
)
from rag_vector_store.milvus.repository_v15 import (
    EXPERT_V15_PROFILE,
    RESUME_V15_PROFILE,
    ContractMilvusV15VectorRepository,
)
from resume_index.config import ResumeSettings
from resume_index.preflight import ResumeSourcePreflightResult
from resume_index.runtime import build_resume_runtime

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)


def _resume_input() -> ExpertChunkEmbeddingInput:
    """把合成 Expert Fixture 转为合法 Resume namespace Contract。"""

    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        mapping = json.loads(next(line for line in stream if line.strip()))
    record = adapt_expert_chunk_mapping(mapping)
    return record.model_copy(
        update={
            "expert_id": 123,
            "vector_doc_id": "resume:123:profile_summary:900001:0",
            "source": record.source.model_copy(
                update={
                    "source_table": "resume",
                    "source_field_id": "900001",
                }
            ),
        }
    )


class FakeEmbeddingModel:
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.25] * 1024 for _ in texts]


class FakeResumeRepository:
    """保存 V15 行的内存 Repository，模拟四字段 Milvus 读写。"""

    def __init__(self, mapper: ResumeMilvusV15RowMapper) -> None:
        self.mapper = mapper
        self.rows: dict[str, dict[str, object]] = {}
        self.upsert_calls: list[list[Any]] = []

    def upsert_batch(self, records: list[Any]) -> MutationResult:
        self.upsert_calls.append(list(records))
        for record in records:
            row = self.mapper.to_row(record)
            self.rows[str(row["vector_doc_id"])] = row
        return MutationResult(
            affected_count=len(records),
            primary_keys=[self.mapper.adapter.vector_doc_id(record) for record in records],
        )

    def get_by_vector_doc_id(
        self,
        vector_doc_id: str,
        *,
        include_embedding: bool = True,
    ) -> dict[str, object] | None:
        row = self.rows.get(vector_doc_id)
        if row is None:
            return None
        result = dict(row)
        if not include_embedding:
            result.pop("embedding", None)
        return result

    def get_many_by_vector_doc_ids(
        self,
        vector_doc_ids: list[str],
        *,
        include_embedding: bool = True,
    ) -> dict[str, dict[str, object]]:
        return {
            vector_doc_id: row
            for vector_doc_id in vector_doc_ids
            if (
                row := self.get_by_vector_doc_id(
                    vector_doc_id,
                    include_embedding=include_embedding,
                )
            )
            is not None
        }

    def query_by_owner_id(
        self,
        owner_id: int,
        *,
        output_fields: tuple[str, ...] | None = None,
    ) -> list[dict[str, object]]:
        del output_fields
        return [
            {"vector_doc_id": row["vector_doc_id"]}
            for row in self.rows.values()
            if row["metadata"].get("resume_id") == owner_id  # type: ignore[union-attr]
        ]


def _resume_service(
    *,
    model: FakeEmbeddingModel | None = None,
) -> tuple[ResumeVectorIngestionService, FakeEmbeddingModel, FakeResumeRepository]:
    adapter = ResumeContractAdapter()
    mapper = ResumeMilvusV15RowMapper(adapter)
    repository = FakeResumeRepository(mapper)
    model = model or FakeEmbeddingModel()
    worker = DomainEmbeddingWorker(
        model,
        adapter,
        expected_dimension=1024,
    )
    ingestion = DomainIngestionService(worker, repository, adapter)
    service = ResumeVectorIngestionService(
        ingestion_service=ingestion,
        repository=repository,  # type: ignore[arg-type]
        validator=adapter.validator,
        row_mapper=mapper,
        expected_dimension=1024,
        formal_model_space_id=FORMAL_EMBEDDING_MODEL_SPACE_ID,
        adapter=adapter,
    )
    return service, model, repository


def test_resume_adapter_and_row_mapper_use_only_resume_identity() -> None:
    record = _resume_input()
    adapter = ResumeContractAdapter()

    assert adapter.is_input(record)
    prepared = adapter.validate(record)
    embedded = adapter.attach_embedding(
        prepared,
        [0.1] * 1024,
        embedding_model_alias="alibaba-bailian:text-embedding-v4",
        embedding_model_version="v1",
        embedded_at="2026-09-03T00:00:00+00:00",
    )
    projection = adapter.storage_projection(embedded)
    assert projection["metadata"]["resume_id"] == 123
    assert "expert_id" not in projection["metadata"]

    mapper = ResumeMilvusV15RowMapper(adapter)
    row = mapper.to_row(embedded)
    assert row["vector_doc_id"].startswith("resume:")
    assert row["metadata"]["resume_id"] == 123
    assert "expert_id" not in row["metadata"]

    restored = mapper.from_row(row)
    assert restored.vector_doc_id == embedded.vector_doc_id
    assert restored.expert_id == 123
    assert restored.embedding == embedded.embedding
    assert restored.source.source_table == "resume"


def test_resume_adapter_rejects_expert_namespace() -> None:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        expert_record = adapt_expert_chunk_mapping(
            json.loads(next(line for line in stream if line.strip()))
        )
    adapter = ResumeContractAdapter()

    assert not adapter.is_input(expert_record)
    with pytest.raises(ValueError, match="resume:"):
        adapter.vector_doc_id(expert_record)


def test_resume_row_mapper_rejects_legacy_expert_owner_metadata() -> None:
    record = _resume_input()
    adapter = ResumeContractAdapter()
    prepared = adapter.validate(record)
    embedded = adapter.attach_embedding(
        prepared,
        [0.1] * 1024,
        embedding_model_alias="alibaba-bailian:text-embedding-v4",
        embedding_model_version="v1",
        embedded_at="2026-09-03T00:00:00+00:00",
    )
    row = ResumeMilvusV15RowMapper(adapter).to_row(embedded)
    metadata = dict(row["metadata"])
    metadata.pop("resume_id")
    metadata["expert_id"] = 123
    row["metadata"] = metadata

    with pytest.raises(ValueError, match="不得包含内部 owner 字段"):
        ResumeMilvusV15RowMapper(adapter).from_row(row)


def test_resume_metadata_only_and_disable_are_embedding_free_and_idempotent() -> None:
    service, model, repository = _resume_service()
    record = _resume_input()

    initial = service.upsert_chunks([record])
    assert initial.embedded_count == 1
    assert model.calls == [[record.chunk_text]]
    original_embedding = repository.rows[record.vector_doc_id]["embedding"]
    original_embedded_at = repository.rows[record.vector_doc_id]["metadata"][
        "embedded_at"
    ]

    changed = record.model_copy(update={"metadata_hash": "1" * 64})
    updated = service.update_metadata([changed])
    assert updated.metadata_updated_count == 1
    assert model.calls == [[record.chunk_text]]
    assert repository.rows[record.vector_doc_id]["embedding"] == original_embedding
    assert (
        repository.rows[record.vector_doc_id]["metadata"]["embedded_at"]
        == original_embedded_at
    )
    assert repository.rows[record.vector_doc_id]["metadata"]["metadata_hash"] == "1" * 64

    first_disable = service.disable_vectors([record.vector_doc_id, record.vector_doc_id])
    assert first_disable.requested_count == 1
    assert first_disable.deleted_count == 1
    assert first_disable.not_found_count == 0
    calls_after_first_disable = len(repository.upsert_calls)

    second_disable = service.disable_vectors([record.vector_doc_id])
    assert second_disable.deleted_count == 1
    assert len(repository.upsert_calls) == calls_after_first_disable
    assert repository.rows[record.vector_doc_id]["metadata"]["is_active"] is False

    listed = service.list_vectors_by_owner(123)
    assert listed.owner_id == 123
    assert listed.vector_doc_ids == [record.vector_doc_id]


def test_resume_legacy_owner_list_warns_and_delegates() -> None:
    service, _model, _repository = _resume_service()

    with pytest.warns(DeprecationWarning, match="list_vectors_by_resume 已弃用"):
        listed = service.list_vectors_by_resume(123)

    assert listed.resume_id == 123
    assert listed.vector_doc_ids == []


def test_resume_service_rejects_expert_namespace_for_reads_and_disable() -> None:
    service, _model, _repository = _resume_service()
    expert_id = "expert:123:profile_summary:900001:0"

    with pytest.raises(VectorContractError, match="前缀"):
        service.get_vectors([expert_id])
    with pytest.raises(VectorContractError, match="前缀"):
        service.disable_vectors([expert_id])


def test_repository_profiles_have_distinct_owner_filters_and_prefixes() -> None:
    assert EXPERT_V15_PROFILE.owner_field == "expert_id"
    assert EXPERT_V15_PROFILE.vector_doc_id_prefix == "expert:"
    assert RESUME_V15_PROFILE.owner_field == "resume_id"
    assert RESUME_V15_PROFILE.vector_doc_id_prefix == "resume:"
    assert EXPERT_V15_PROFILE.owner_filter_fn(7) == 'metadata["expert_id"] == 7'
    assert RESUME_V15_PROFILE.owner_filter_fn(7) == 'metadata["resume_id"] == 7'


def test_resume_repository_rejects_legacy_expert_owner_during_model_space_gate() -> None:
    repository = ContractMilvusV15VectorRepository(
        MilvusSettings(collection_name="resume_candidates_v2_test"),
        1024,
        profile=RESUME_V15_PROFILE,
    )

    with pytest.raises(ValueError, match="resume_id"):
        repository._model_space_from_row(
            {
                "metadata": {
                    "expert_id": 123,
                    "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
                    "embedding_model_version": "v1",
                    "embedding_dim": 1024,
                }
            }
        )


def test_resume_factory_binds_resume_profile_and_collection() -> None:
    resume_service = create_resume_vector_ingestion_service(
        embedding_model=FakeEmbeddingModel(),
        milvus_settings=MilvusSettings(
            collection_name="resume_candidates_v2_test",
        ),
    )
    assert isinstance(resume_service, ResumeVectorIngestionService)
    assert resume_service.repository.profile.owner_field == "resume_id"
    assert resume_service.repository.profile.vector_doc_id_prefix == "resume:"
    assert resume_service.repository._allow_collection_creation is False
    assert isinstance(resume_service.row_mapper, ResumeMilvusV15RowMapper)
    assert (
        resume_service.repository.settings.collection_name
        == "resume_candidates_v2_test"
    )


def test_resume_factory_injected_worker_does_not_read_embedding_environment() -> None:
    adapter = ResumeContractAdapter()
    worker = DomainEmbeddingWorker(
        FakeEmbeddingModel(),
        adapter,
        expected_dimension=1024,
    )
    with patch(
        "rag_vector_store.ingestion.factory.CompanyCloudEmbeddingModel.from_env",
        side_effect=AssertionError("不应读取 Embedding 环境配置"),
    ) as from_env:
        service = create_resume_vector_ingestion_service(
            milvus_settings=MilvusSettings(
                collection_name="resume_candidates_v2_test",
            ),
            embedding_worker=worker,
        )

    assert isinstance(service, ResumeVectorIngestionService)
    from_env.assert_not_called()


def test_resume_factory_injected_repository_does_not_read_milvus_environment() -> None:
    repository = ContractMilvusV15VectorRepository(
        MilvusSettings(collection_name="resume_candidates_v2_test"),
        1024,
        profile=RESUME_V15_PROFILE,
    )
    with patch(
        "rag_vector_store.ingestion.factory.MilvusSettings.from_env",
        side_effect=AssertionError("不应读取 Milvus 环境配置"),
    ) as from_env:
        service = create_resume_vector_ingestion_service(
            repository=repository,
            embedding_worker=object(),
        )

    assert service.repository is repository
    from_env.assert_not_called()


def test_resume_factory_rejects_expert_repository_and_mapper() -> None:
    settings = MilvusSettings(collection_name="resume_candidates_v2_test")
    expert_repository = ContractMilvusV15VectorRepository(
        settings.model_copy(update={"collection_name": "expert_candidates_v15_contract"}),
        1024,
    )

    with pytest.raises(ValueError, match="Resume Profile"):
        create_resume_vector_ingestion_service(
            repository=expert_repository,
            embedding_worker=object(),
        )

    with pytest.raises(ValueError, match="ResumeMilvusV15RowMapper"):
        create_resume_vector_ingestion_service(
            milvus_settings=settings,
            embedding_worker=object(),
            row_mapper=MilvusV15RowMapper(),
        )


def test_factories_reject_cross_domain_repository_mapper_and_worker() -> None:
    expert_settings = MilvusSettings(
        collection_name="expert_candidates_v15_contract",
    )
    resume_settings = MilvusSettings(
        collection_name="resume_candidates_v2_test",
    )
    resume_repository = ContractMilvusV15VectorRepository(
        resume_settings,
        1024,
        profile=RESUME_V15_PROFILE,
    )
    resume_worker = DomainEmbeddingWorker(
        FakeEmbeddingModel(),
        ResumeContractAdapter(),
        expected_dimension=1024,
    )
    expert_worker = DomainEmbeddingWorker(
        FakeEmbeddingModel(),
        ExpertContractAdapter(),
        expected_dimension=1024,
    )

    with pytest.raises(ValueError, match="Expert Profile"):
        create_vector_ingestion_service(
            repository=resume_repository,
            embedding_worker=expert_worker,
        )
    with pytest.raises(TypeError, match="MilvusV15RowMapper"):
        create_vector_ingestion_service(
            milvus_settings=expert_settings,
            embedding_worker=expert_worker,
            row_mapper=ResumeMilvusV15RowMapper(),
        )
    with pytest.raises(ValueError, match="Expert 身份空间"):
        create_vector_ingestion_service(
            milvus_settings=expert_settings,
            embedding_worker=resume_worker,
        )
    with pytest.raises(ValueError, match="Resume 身份空间"):
        create_resume_vector_ingestion_service(
            milvus_settings=resume_settings,
            embedding_worker=expert_worker,
        )


def _resume_settings() -> ResumeSettings:
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


def test_resume_runtime_requires_explicit_resume_profile() -> None:
    class FakeRepository:
        settings = MilvusSettings(collection_name="resume_candidates_v15_contract")

    class FakeVectorService:
        repository = FakeRepository()

    def preflight(settings: ResumeSettings) -> ResumeSourcePreflightResult:
        return ResumeSourcePreflightResult(
            source_schema=settings.source_db.database,
            source_tables=("resume",),
            index_schema=settings.index_db.database,
            milvus_collection_name=settings.milvus_collection_name,
        )

    with pytest.raises(DatabaseBoundaryError, match="Resume Profile"):
        build_resume_runtime(
            _resume_settings(),
            vector_service=FakeVectorService(),  # type: ignore[arg-type]
            preflight_runner=preflight,
        )
