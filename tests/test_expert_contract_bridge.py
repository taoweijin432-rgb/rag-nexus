from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from expert_contracts import (
    ExpertChunkEmbeddingInput,
    ExpertChunkMappingAdapter,
    ExpertChunkMappingError,
)
from rag_vector_store import (
    ContractEmbeddingWorker,
    ContractIngestionService,
    MutationResult,
)


class FakeEmbeddingModel:
    """纯内存 Embedding double，只记录收到的 chunk_text。"""

    dimension = 3
    model_alias = "bridge-fake-model"
    model_version = "bridge-fake-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class InMemoryContractRepository:
    """纯内存 Repository double，不连接 Milvus。"""

    def __init__(self) -> None:
        self.records = []

    def upsert_batch(self, records) -> MutationResult:
        items = list(records)
        self.records.extend(items)
        return MutationResult(
            affected_count=len(items),
            primary_keys=[item.vector_doc_id for item in items],
        )


def fixed_clock() -> datetime:
    return datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def expert_chunk_mapping() -> dict[str, object]:
    text = "[专家能力概述]\n上游生成的共享 Contract 测试记录。"
    vector_doc_id = "upstream-owned-vector-doc-900001"
    return {
        "contract_version": "2.0.0",
        "chunk_id": "expert:900001:profile_summary:bridge-profile:0",
        "expert_id": 900001,
        "serial_number": "DEV900001",
        "chunk_type": "profile_summary",
        "chunk_subtype": None,
        "chunk_index": 0,
        "chunk_total": 1,
        "chunk_text": text,
        "char_count": len(text),
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "metadata": {
            "vector_doc_id": vector_doc_id,
            "company_name_raw": None,
            "company_name_norm": None,
            "company_normalization_status": None,
            "role_name_raw": None,
            "role_code": None,
            "role_level": None,
            "start_month": None,
            "end_month": None,
            "is_current": None,
            "industry_codes": ["artificial_intelligence"],
            "topic_codes": ["knowledge_retrieval"],
            "origin_type": "source_template",
            "verification_status": "source_evidence",
            "weight_tier": "B",
        },
        "source": {
            "source_table": "experts",
            "source_item_key": "bridge-profile",
            "source_field": "background",
            "source_field_id": "900001",
            "source_version": "2026-08-12T10:00:00+08:00",
            "source_updated_at": "2026-08-12T10:00:00+08:00",
            "source_hash": "a" * 64,
        },
        "versions": {
            "data_version": "dev_20260812_01",
            "chunk_policy_version": "1.2",
            "metadata_schema_version": "2.0",
            "dictionary_version": "dev_2026.08",
        },
        "is_active": True,
        "updated_at": "2026-08-12T10:05:00+08:00",
    }


def test_mapping_contract_and_ingestion_are_connected_in_memory() -> None:
    mapping = expert_chunk_mapping()
    metadata = dict(mapping["metadata"])
    source = dict(mapping["source"])
    versions = dict(mapping["versions"])

    contract = ExpertChunkMappingAdapter().from_mapping(mapping)

    assert isinstance(contract, ExpertChunkEmbeddingInput)
    assert contract.vector_doc_id == metadata["vector_doc_id"]
    assert contract.metadata.model_dump(exclude_unset=True) == metadata
    assert contract.source.model_dump() == source
    assert contract.versions.model_dump() == versions
    assert "vector_doc_id" not in mapping

    model = FakeEmbeddingModel()
    repository = InMemoryContractRepository()
    service = ContractIngestionService(
        ContractEmbeddingWorker(model, expected_dimension=3, clock=fixed_clock),
        repository,
    )

    result = service.ingest_one(contract)

    assert result.status == "embedded"
    assert result.vector_doc_id == metadata["vector_doc_id"]
    assert model.calls == [[mapping["chunk_text"]]]
    assert len(repository.records) == 1
    assert repository.records[0].vector_doc_id == metadata["vector_doc_id"]


def test_mapping_adapter_rejects_top_level_vector_doc_id_mismatch() -> None:
    mapping = expert_chunk_mapping()
    mapping["vector_doc_id"] = "different-upstream-id"

    with pytest.raises(ExpertChunkMappingError, match="vector_doc_id"):
        ExpertChunkMappingAdapter().from_mapping(mapping)


def test_mapping_adapter_requires_upstream_metadata_vector_doc_id() -> None:
    mapping = expert_chunk_mapping()
    del mapping["metadata"]["vector_doc_id"]

    with pytest.raises(ExpertChunkMappingError, match="metadata.vector_doc_id"):
        ExpertChunkMappingAdapter().from_mapping(mapping)


def test_mapping_adapter_rejects_deprecated_source_record_id() -> None:
    mapping = expert_chunk_mapping()
    mapping["source"]["source_record_id"] = "deprecated"

    with pytest.raises(ExpertChunkMappingError, match="source_record_id"):
        ExpertChunkMappingAdapter().from_mapping(mapping)
