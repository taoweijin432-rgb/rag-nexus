"""双领域 Factory Profile 绑定测试（Fake，不联网）。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_vector_store import (
    ContractEmbeddingWorker,
    MilvusSettings,
    ResumeVectorIngestionService,
    VectorIngestionService,
    create_resume_vector_ingestion_service,
    create_vector_ingestion_service,
)
from rag_vector_store.chunking.contract_pipeline import ContractValidator
from rag_vector_store.core.embedding import DomainEmbeddingWorker
from rag_vector_store.domains import ResumeContractAdapter
from rag_vector_store.milvus.contract_v15 import (
    MilvusV15RowMapper,
    ResumeMilvusV15RowMapper,
)
from rag_vector_store.milvus.repository_v15 import (
    RESUME_V15_PROFILE,
    ContractMilvusV15VectorRepository,
)


class FakeEmbeddingModel:
    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


def settings(name: str = "expert_candidates_v15_contract_dev") -> MilvusSettings:
    return MilvusSettings(
        collection_name=name,
        index_type="HNSW",
        metric_type="COSINE",
        index_params={"M": 16, "efConstruction": 200},
    )


def expert_worker() -> ContractEmbeddingWorker:
    return ContractEmbeddingWorker(FakeEmbeddingModel(), expected_dimension=1024)


def resume_worker() -> DomainEmbeddingWorker:
    adapter = ResumeContractAdapter(ContractValidator())
    return DomainEmbeddingWorker(
        FakeEmbeddingModel(),
        adapter,
        expected_dimension=1024,
    )


def test_expert_factory_uses_v15_domain_core() -> None:
    service = create_vector_ingestion_service(
        milvus_settings=settings(),
        embedding_worker=expert_worker(),
    )
    assert isinstance(service, VectorIngestionService)
    assert isinstance(service.ingestion, object)
    assert isinstance(service.row_mapper, MilvusV15RowMapper)
    assert service.repository.profile.owner_field == "expert_id"
    assert service.repository.profile.vector_doc_id_prefix == "expert:"


def test_resume_factory_uses_resume_profile() -> None:
    service = create_resume_vector_ingestion_service(
        milvus_settings=settings("resume_candidates_v15_contract"),
        embedding_worker=resume_worker(),
    )
    assert isinstance(service, ResumeVectorIngestionService)
    assert isinstance(service.row_mapper, ResumeMilvusV15RowMapper)
    assert service.repository.profile is RESUME_V15_PROFILE
    assert service.repository.profile.owner_field == "resume_id"
    assert service.repository.profile.vector_doc_id_prefix == "resume:"


def test_resume_factory_rejects_expert_v16_settings() -> None:
    with pytest.raises(ValueError, match="Expert V16 Collection"):
        create_resume_vector_ingestion_service(
            milvus_settings=MilvusSettings(
                collection_name="expert_candidates_v16_contract",
                expert_storage_profile="v16",
            ),
            embedding_worker=resume_worker(),
        )


def test_resume_factory_forces_v15_profile_for_resume_collection() -> None:
    expert_v16 = MilvusSettings(
        collection_name="expert_candidates_v16_contract",
        expert_storage_profile="v16",
    )
    resume_settings = expert_v16.model_copy(
        update={"collection_name": "resume_candidates_v15_contract"}
    )

    service = create_resume_vector_ingestion_service(
        milvus_settings=resume_settings,
        embedding_worker=resume_worker(),
    )

    assert service.repository.settings.expert_storage_profile == "v15"
    assert service.repository.profile is RESUME_V15_PROFILE


def test_resume_factory_rejects_expert_worker() -> None:
    with pytest.raises(ValueError, match="跨领域"):
        create_resume_vector_ingestion_service(
            milvus_settings=settings("resume_candidates_v15_contract"),
            embedding_worker=expert_worker(),
        )


def test_expert_factory_rejects_resume_worker() -> None:
    with pytest.raises(ValueError, match="跨领域"):
        create_vector_ingestion_service(
            milvus_settings=settings(),
            embedding_worker=resume_worker(),
        )


def test_resume_factory_rejects_expert_mapper() -> None:
    with pytest.raises(TypeError, match="Resume"):
        create_resume_vector_ingestion_service(
            milvus_settings=settings("resume_candidates_v15_contract"),
            embedding_worker=resume_worker(),
            row_mapper=MilvusV15RowMapper(),
        )


def test_expert_factory_rejects_resume_mapper() -> None:
    with pytest.raises(TypeError, match="Expert"):
        create_vector_ingestion_service(
            milvus_settings=settings(),
            embedding_worker=expert_worker(),
            row_mapper=ResumeMilvusV15RowMapper(),
        )


def test_explicit_repository_profile_must_match_domain() -> None:
    resume_repository = ContractMilvusV15VectorRepository(
        settings("resume_candidates_v15_contract"),
        profile=RESUME_V15_PROFILE,
    )
    with pytest.raises(ValueError, match="Profile"):
        create_vector_ingestion_service(
            repository=resume_repository,
            embedding_worker=expert_worker(),
        )


def test_factory_never_initializes_collection() -> None:
    service = create_vector_ingestion_service(
        milvus_settings=settings(),
        embedding_worker=expert_worker(),
    )
    assert service.repository._client is None
