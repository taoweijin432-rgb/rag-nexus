"""版本间继承解除后，共享采样/缓存与领域门禁的回归。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_vector_store.config import EXPERT_V16_COLLECTION_NAME, MilvusSettings
from rag_vector_store.milvus.repository_metadata import MetadataModelSpaceRepository
from rag_vector_store.milvus.repository_v15 import (
    RESUME_V15_PROFILE,
    ContractMilvusV15VectorRepository,
)
from rag_vector_store.milvus.repository_v16 import ContractMilvusV16VectorRepository


class Client:
    def __init__(self):
        self.count = 1000
        self.query_calls = []
        self.metadata = {
            "expert_id": 1,
            "embedding_model_alias": "fixture-model",
            "embedding_model_version": "v1",
            "embedding_dim": 1024,
        }

    def get_collection_stats(self, **kwargs):
        return {"row_count": str(self.count)}

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return [{"metadata": dict(self.metadata)}]


def _record():
    return SimpleNamespace(
        embedding_model_alias="fixture-model", embedding_model_version="v1", embedding_dim=1024
    )


def _repository(version, client):
    if version == "v16":
        settings = MilvusSettings(
            collection_name=EXPERT_V16_COLLECTION_NAME, expert_storage_profile="v16"
        )
        return ContractMilvusV16VectorRepository(settings, 1024, client=client)
    return ContractMilvusV15VectorRepository(MilvusSettings(), 1024, client=client)


def test_versions_are_siblings_and_share_model_space_implementation():
    assert issubclass(ContractMilvusV15VectorRepository, MetadataModelSpaceRepository)
    assert issubclass(ContractMilvusV16VectorRepository, MetadataModelSpaceRepository)
    assert not issubclass(ContractMilvusV16VectorRepository, ContractMilvusV15VectorRepository)
    assert (
        ContractMilvusV15VectorRepository._validate_existing_model_space
        is ContractMilvusV16VectorRepository._validate_existing_model_space
    )
    assert (
        ContractMilvusV15VectorRepository.query_by_owner_id
        is not ContractMilvusV16VectorRepository.query_by_owner_id
    )


@pytest.mark.parametrize("version", ["v15", "v16"])
def test_sampling_limit_random_filter_and_instance_cache_are_preserved(version):
    client = Client()
    repository = _repository(version, client)
    repository._validate_existing_model_space(_record())
    repository._validate_existing_model_space(_record())
    assert len(client.query_calls) == 1
    assert client.query_calls[0]["limit"] == 500
    assert client.query_calls[0]["filter"].startswith("RANDOM_SAMPLE(")
    assert client.query_calls[0]["output_fields"] == ["vector_doc_id", "metadata"]
    other = _repository(version, client)
    other._validate_existing_model_space(_record())
    assert len(client.query_calls) == 2


@pytest.mark.parametrize("version", ["v15", "v16"])
def test_failed_model_space_does_not_poison_cache(version):
    client = Client()
    client.metadata["embedding_model_alias"] = "wrong-model"
    repository = _repository(version, client)
    with pytest.raises(ValueError, match="model-space"):
        repository._validate_existing_model_space(_record())
    assert repository._validated_model_space is None
    client.metadata["embedding_model_alias"] = "fixture-model"
    repository._validate_existing_model_space(_record())
    assert len(client.query_calls) == 2


def test_resume_metadata_identity_checks_remain_bound_to_profile():
    client = Client()
    repository = ContractMilvusV15VectorRepository(
        MilvusSettings(collection_name="resume_fixture"),
        1024,
        client=client,
        profile=RESUME_V15_PROFILE,
    )
    with pytest.raises(ValueError, match="resume_id"):
        repository._validate_existing_model_space(_record())
    client.metadata["resume_id"] = 1
    with pytest.raises(ValueError, match="expert_id"):
        repository._validate_existing_model_space(_record())
    del client.metadata["expert_id"]
    repository._validate_existing_model_space(_record())
