"""Expert V16 正式存储、字段双落与 Factory 测试（Fake，不联网）。"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from expert_contracts import EmbeddedContractPreparedChunk, adapt_expert_chunk_mapping
from rag_vector_store import ContractEmbeddingWorker, MilvusSettings, MutationResult
from rag_vector_store.domains import ExpertContractAdapter
from rag_vector_store.ingestion.factory import create_vector_ingestion_service
from rag_vector_store.milvus.compatibility import compare_contract_indexes
from rag_vector_store.milvus.contract_v15 import MilvusV15RowMapper
from rag_vector_store.milvus.contract_v16 import (
    MilvusV16RowMapper,
    map_flat_v15_row_to_v16,
    validate_v16_mirrors,
)
from rag_vector_store.milvus.indexes import index_specs_v16
from rag_vector_store.milvus.repository_v16 import (
    EXPERT_V16_PROFILE,
    ContractMilvusV16VectorRepository,
)
from rag_vector_store.milvus.schemas.contract_v16 import build_v16_schema

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)


class FakeEmbeddingModel:
    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1] * 1024 for _ in texts]


class MemoryV16Repository:
    profile = EXPERT_V16_PROFILE

    def __init__(self) -> None:
        self.settings = MilvusSettings(
            collection_name="expert_candidates_v16_contract",
            expert_storage_profile="v16",
        )
        self.mapper = MilvusV16RowMapper()
        self.rows: dict[str, dict[str, object]] = {}

    def upsert_batch(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> MutationResult:
        items = list(records)
        for item in items:
            self.rows[item.vector_doc_id] = self.mapper.to_row(item)
        return MutationResult(
            affected_count=len(items),
            primary_keys=[item.vector_doc_id for item in items],
        )

    def get_many_by_vector_doc_ids(
        self,
        vector_doc_ids: Sequence[str],
        *,
        include_embedding: bool = True,
    ) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for vector_doc_id in vector_doc_ids:
            row = self.rows.get(vector_doc_id)
            if row is None:
                continue
            result[vector_doc_id] = {
                key: value
                for key, value in row.items()
                if include_embedding or key != "dense_embedding"
            }
        return result

    def query_by_owner_id(self, owner_id: int) -> list[dict[str, object]]:
        return [
            {"vector_doc_id": key}
            for key, row in self.rows.items()
            if row["expert_id"] == owner_id
        ]


class QueryClient:
    def __init__(self, row: dict[str, object]) -> None:
        self.row = row
        self.query_calls: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> list[dict[str, object]]:
        self.query_calls.append(kwargs)
        fields = kwargs["output_fields"]
        return [{key: value for key, value in self.row.items() if key in fields}]


class RowIterator:
    def __init__(self, row: dict[str, object], fields: Sequence[str]) -> None:
        self.row = row
        self.fields = fields
        self.returned = False

    def next(self) -> list[dict[str, object]]:
        if self.returned:
            return []
        self.returned = True
        return [{key: value for key, value in self.row.items() if key in self.fields}]

    def close(self) -> None:
        return None


class IteratorClient:
    def __init__(self, row: dict[str, object]) -> None:
        self.row = row
        self.iterator_calls: list[dict[str, Any]] = []

    def query_iterator(self, **kwargs: Any) -> RowIterator:
        self.iterator_calls.append(kwargs)
        return RowIterator(self.row, kwargs["output_fields"])


def _records() -> list[Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        return [
            adapt_expert_chunk_mapping(json.loads(line))
            for line in stream
            if line.strip()
        ]


def _embedded() -> list[EmbeddedContractPreparedChunk]:
    return ContractEmbeddingWorker(
        FakeEmbeddingModel(),
        clock=lambda: datetime(2026, 9, 17, tzinfo=timezone.utc),
    ).embed(_records()).embedded_chunks


def _service() -> tuple[Any, FakeEmbeddingModel, MemoryV16Repository]:
    model = FakeEmbeddingModel()
    adapter = ExpertContractAdapter()
    worker = ContractEmbeddingWorker(
        model,
        clock=lambda: datetime(2026, 9, 17, tzinfo=timezone.utc),
    )
    repository = MemoryV16Repository()
    service = create_vector_ingestion_service(
        repository=repository,  # type: ignore[arg-type]
        embedding_worker=worker,
        row_mapper=MilvusV16RowMapper(adapter),
    )
    return service, model, repository


def test_v16_schema_and_indexes_match_frozen_contract() -> None:
    schema = build_v16_schema(
        MilvusSettings(),
        1024,
    ).to_dict()
    fields = {field["name"]: field for field in schema["fields"]}

    assert list(fields) == [
        "vector_doc_id",
        "expert_id",
        "chunk_text",
        "chunk_type",
        "is_active",
        "eligible_for_recommendation",
        "dense_embedding",
        "sparse_embedding",
        "job_status_norm",
        "abroad_flag_norm",
        "work_years",
        "metadata",
    ]
    assert fields["chunk_text"]["params"]["analyzer_params"] == '{"type":"chinese"}'
    assert fields["dense_embedding"]["params"]["dim"] == 1024
    assert all(fields[name]["nullable"] for name in (
        "job_status_norm", "abroad_flag_norm", "work_years"
    ))
    assert schema["functions"][0]["input_field_names"] == ["chunk_text"]
    assert schema["functions"][0]["output_field_names"] == ["sparse_embedding"]
    assert [item.index_name for item in index_specs_v16(MilvusSettings())] == [
        "dense_embedding_hnsw",
        "sparse_embedding_bm25",
    ]


def test_v16_index_gate_compares_dense_and_sparse_numeric_params() -> None:
    specs = index_specs_v16(MilvusSettings())
    descriptions = {
        "dense_embedding_hnsw": {
            "field_name": "dense_embedding",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": "16", "efConstruction": "200"},
        },
        "sparse_embedding_bm25": {
            "field_name": "sparse_embedding",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": "1.2",
                "bm25_b": "0.75",
            },
        },
    }

    result = compare_contract_indexes(list(descriptions), descriptions, specs)

    assert result.compatible is True


def test_shared_mapper_writes_metadata_and_normalized_top_level_once() -> None:
    chunk = _embedded()[0]
    row = MilvusV16RowMapper().to_row(chunk)

    assert tuple(row) == (
        "vector_doc_id",
        "expert_id",
        "chunk_text",
        "chunk_type",
        "is_active",
        "eligible_for_recommendation",
        "dense_embedding",
        "job_status_norm",
        "abroad_flag_norm",
        "work_years",
        "metadata",
    )
    assert "sparse_embedding" not in row
    assert row["expert_id"] == row["metadata"]["expert_id"]  # type: ignore[index]
    assert row["chunk_type"] == row["metadata"]["chunk_type"]  # type: ignore[index]
    validate_v16_mirrors(row)


def test_formal_migration_mapper_preserves_flat_v15_metadata_exactly() -> None:
    chunk = _embedded()[0]
    v15_mapper = MilvusV15RowMapper()
    v16_mapper = MilvusV16RowMapper()
    source = v15_mapper.to_row(chunk)

    target = v16_mapper.to_row(v15_mapper.from_row(source))
    rolled_back = v15_mapper.to_row(v16_mapper.from_row(target))

    assert target["metadata"] == source["metadata"]
    assert rolled_back == source


@pytest.mark.parametrize("value", [None, pytest.param("missing", id="missing")])
def test_eligible_null_or_missing_normalizes_false(value: object) -> None:
    chunk = _embedded()[0]
    flat = MilvusV16RowMapper()._flat_mapper.to_row(chunk)
    metadata = dict(flat["metadata"])
    if value == "missing":
        metadata.pop("eligible_for_recommendation", None)
    else:
        metadata["eligible_for_recommendation"] = None
    flat["metadata"] = metadata

    row = map_flat_v15_row_to_v16(flat)

    assert row["eligible_for_recommendation"] is False


def test_p1_values_and_nonfinite_work_years() -> None:
    chunk = _embedded()[0]
    flat = MilvusV16RowMapper()._flat_mapper.to_row(chunk)
    metadata = dict(flat["metadata"])
    metadata.update(
        job_status_norm="active",
        abroad_flag_norm=False,
        work_years="12.50",
    )
    flat["metadata"] = metadata

    row = map_flat_v15_row_to_v16(flat)

    assert row["job_status_norm"] == "active"
    assert row["abroad_flag_norm"] is False
    assert row["work_years"] == 12.5
    metadata["work_years"] = "NaN"
    with pytest.raises(ValueError, match="有限数值"):
        map_flat_v15_row_to_v16(flat)


def test_work_years_is_normalized_to_milvus_float32() -> None:
    flat = MilvusV16RowMapper()._flat_mapper.to_row(_embedded()[0])
    metadata = dict(flat["metadata"])
    metadata["work_years"] = "12.34"
    flat["metadata"] = metadata

    row = map_flat_v15_row_to_v16(flat)

    assert row["work_years"] == struct.unpack("!f", struct.pack("!f", 12.34))[0]
    validate_v16_mirrors(row)
    metadata["work_years"] = "1e39"
    with pytest.raises(ValueError, match="Milvus FLOAT"):
        map_flat_v15_row_to_v16(flat)


def test_metadata_only_and_disable_keep_top_level_in_sync_without_embedding() -> None:
    service, model, repository = _service()
    item = _records()[0]
    service.upsert_chunks([item])
    original_dense = list(repository.rows[item.vector_doc_id]["dense_embedding"])
    changed_metadata = item.metadata.model_copy(
        update={"eligible_for_recommendation": False}
    )
    changed = item.model_copy(
        update={"metadata": changed_metadata, "metadata_hash": "f" * 64}
    )

    result = service.update_metadata([changed])
    disabled = service.disable_vectors([item.vector_doc_id])

    assert result.metadata_updated_count == 1
    assert disabled.deleted_count == 1
    assert model.calls == [[item.chunk_text]]
    row = repository.rows[item.vector_doc_id]
    assert row["dense_embedding"] == original_dense
    assert row["eligible_for_recommendation"] is False
    assert row["is_active"] is False
    assert row["metadata"]["is_active"] is False  # type: ignore[index]
    validate_v16_mirrors(row)


def test_repository_query_excludes_function_output_and_detects_mirror_drift() -> None:
    row = MilvusV16RowMapper().to_row(_embedded()[0])
    client = QueryClient(row)
    settings = MilvusSettings(
        collection_name="expert_candidates_v16_contract",
        expert_storage_profile="v16",
    )
    repository = ContractMilvusV16VectorRepository(settings, 1024, client=client)

    result = repository.get_many_by_vector_doc_ids(
        [str(row["vector_doc_id"])], include_embedding=False
    )

    assert result
    fields = client.query_calls[0]["output_fields"]
    assert "dense_embedding" not in fields
    assert "sparse_embedding" not in fields
    client.row["is_active"] = not bool(client.row["is_active"])
    with pytest.raises(ValueError, match="is_active"):
        repository.get_many_by_vector_doc_ids([str(row["vector_doc_id"])])


def test_owner_and_activity_reads_validate_all_mirrored_fields() -> None:
    row = MilvusV16RowMapper().to_row(_embedded()[0])
    client = IteratorClient(row)
    repository = ContractMilvusV16VectorRepository(
        MilvusSettings(
            collection_name="expert_candidates_v16_contract",
            expert_storage_profile="v16",
        ),
        1024,
        client=client,
    )

    owner_rows = repository.query_by_owner_id(int(row["expert_id"]))
    activity_rows = repository.list_all_vector_activity_states()

    assert owner_rows == [{"vector_doc_id": row["vector_doc_id"]}]
    assert activity_rows[0]["is_active"] is row["is_active"]
    assert "content_hash" in activity_rows[0]
    assert "metadata_hash" in activity_rows[0]
    assert "metadata" not in activity_rows[0]
    row["eligible_for_recommendation"] = not bool(
        row["eligible_for_recommendation"]
    )
    with pytest.raises(ValueError, match="eligible_for_recommendation"):
        repository.list_all_vector_activity_states()


def test_mirror_validation_rejects_bool_integer_aliasing() -> None:
    row = MilvusV16RowMapper().to_row(_embedded()[0])
    row["expert_id"] = True

    with pytest.raises(ValueError, match="expert_id"):
        validate_v16_mirrors(row)


def test_factory_selects_v16_only_from_constructed_settings() -> None:
    service = create_vector_ingestion_service(
        milvus_settings=MilvusSettings(
            collection_name="expert_candidates_v16_contract",
            expert_storage_profile="v16",
        ),
        embedding_worker=ContractEmbeddingWorker(FakeEmbeddingModel()),
    )

    assert isinstance(service.repository, ContractMilvusV16VectorRepository)
    assert isinstance(service.row_mapper, MilvusV16RowMapper)
    with pytest.raises(ValueError, match="expert_candidates_v16_contract"):
        MilvusSettings(
            collection_name="wrong",
            expert_storage_profile="v16",
        )
    with pytest.raises(ValueError, match="EXPERT_MILVUS_STORAGE_PROFILE"):
        ContractMilvusV16VectorRepository(
            MilvusSettings(collection_name="expert_candidates_v16_contract"),
            1024,
        )
