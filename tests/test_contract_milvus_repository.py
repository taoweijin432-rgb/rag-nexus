from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pymilvus import DataType

from expert_contracts import adapt_expert_chunk_mapping
from rag_vector_store import (
    CONTRACT_SCALAR_INDEX_FIELDS,
    ContractEmbeddingWorker,
    ContractMilvusVectorRepository,
    EmbeddedContractPreparedChunk,
    MilvusRowMapper,
    MilvusSettings,
    SearchHit,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v2.jsonl"
)
_DEFAULT_UPSERT_RESPONSE = object()


class FakeEmbeddingModel:
    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


class FakeIndexParams:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FakeQueryIterator:
    def __init__(
        self,
        batches: list[object],
        *,
        error_on_call: int | None = None,
    ) -> None:
        self.batches = list(batches)
        self.error_on_call = error_on_call
        self.next_calls = 0
        self.closed = False

    def next(self) -> object:
        self.next_calls += 1
        if self.error_on_call == self.next_calls:
            raise RuntimeError("iterator query failed")
        return self.batches.pop(0) if self.batches else []

    def close(self) -> None:
        self.closed = True


class FakeContractMilvusClient:
    def __init__(self, *, exists: bool) -> None:
        self.exists = exists
        self.description: dict[str, Any] | None = None
        self.index_params = FakeIndexParams()
        self.create_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.query_iterator_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.drop_calls: list[str] = []
        self.upsert_response: object = _DEFAULT_UPSERT_RESPONSE
        self.model_space_batches: list[object] | None = None
        self.iterator_error_on_call: int | None = None
        self.last_query_iterator: FakeQueryIterator | None = None

    def has_collection(self, collection_name: str) -> bool:
        return self.exists

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        if self.description is None:
            raise AssertionError("测试未配置 Collection description")
        return self.description

    def prepare_index_params(self) -> FakeIndexParams:
        return self.index_params

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)
        self.exists = True

    def list_indexes(self, collection_name: str) -> list[str]:
        return [
            "embedding_vector",
            *(f"{field_name}_inverted" for field_name in CONTRACT_SCALAR_INDEX_FIELDS),
        ]

    def describe_index(
        self,
        collection_name: str,
        index_name: str,
    ) -> dict[str, Any]:
        if index_name == "embedding_vector":
            return {
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "M": "16",
                "efConstruction": "200",
            }
        return {
            "field_name": index_name.removesuffix("_inverted"),
            "index_type": "INVERTED",
        }

    def upsert(self, **kwargs: Any) -> Any:
        self.upsert_calls.append(kwargs)
        for row in kwargs["data"]:
            self.rows[str(row["vector_doc_id"])] = dict(row)
        response = {
            "upsert_count": len(kwargs["data"]),
            "ids": [row["vector_doc_id"] for row in kwargs["data"]],
        }
        return (
            response
            if self.upsert_response is _DEFAULT_UPSERT_RESPONSE
            else self.upsert_response
        )

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append(kwargs)
        ids = kwargs.get("ids")
        if ids is None:
            return list(self.rows.values())
        return [self.rows[vector_doc_id] for vector_doc_id in ids if vector_doc_id in self.rows]

    def query_iterator(self, **kwargs: Any) -> FakeQueryIterator:
        self.query_iterator_calls.append(kwargs)
        batches = (
            [list(self.rows.values())]
            if self.model_space_batches is None
            else self.model_space_batches
        )
        iterator = FakeQueryIterator(
            batches,
            error_on_call=self.iterator_error_on_call,
        )
        self.last_query_iterator = iterator
        return iterator

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.search_calls.append(kwargs)
        rows = list(self.rows.values())
        if kwargs["filter"] == "chunk_type == 'employment'":
            rows = [row for row in rows if row["chunk_type"] == "employment"]
        output_fields = kwargs["output_fields"]
        hits = [
            {
                "id": row["vector_doc_id"],
                "distance": 0.9,
                "entity": {field: row[field] for field in output_fields},
            }
            for row in rows[: kwargs["limit"]]
        ]
        return [hits]

    def drop_collection(self, collection_name: str) -> None:
        self.drop_calls.append(collection_name)
        raise AssertionError("契约安全创建不得调用 drop_collection")


def embedded_contract_chunks() -> list[EmbeddedContractPreparedChunk]:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        records = [
            adapt_expert_chunk_mapping(json.loads(line))
            for line in stream
            if line.strip()
        ]
    result = ContractEmbeddingWorker(
        FakeEmbeddingModel(),
        clock=lambda: datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
    ).embed(records)
    return result.embedded_chunks


def contract_settings() -> MilvusSettings:
    return MilvusSettings(
        collection_name="expert_candidates_v13_contract_dev",
        index_type="HNSW",
        metric_type="COSINE",
        index_params={"M": 16, "efConstruction": 200},
    )


def test_milvus_row_mapper_preserves_scalars_and_nested_contracts() -> None:
    chunk = embedded_contract_chunks()[1]

    entity = MilvusRowMapper().to_entity(chunk)

    assert entity.vector_doc_id == chunk.vector_doc_id
    assert entity.chunk_text == chunk.chunk_text
    assert entity.content_hash == chunk.content_hash
    assert entity.company_name_norm == chunk.metadata.company_name_norm
    assert entity.role_code == chunk.metadata.role_code
    assert entity.source_table == chunk.source.source_table
    assert entity.source_field_id == chunk.source.source_field_id
    assert "source_record_id" not in entity.model_dump()
    assert entity.chunk_policy_version == chunk.versions.chunk_policy_version
    assert entity.metadata_ext == chunk.metadata.model_dump(mode="json")
    assert entity.source_ext == chunk.source.model_dump(mode="json")
    assert entity.versions_ext == chunk.versions.model_dump(mode="json")
    assert entity.embedding_dim == 1024


def test_milvus_row_mapper_round_trip_preserves_upstream_vector_doc_id() -> None:
    chunk = embedded_contract_chunks()[1]
    entity = MilvusRowMapper().to_entity(chunk)

    restored = MilvusRowMapper().from_row(entity.model_dump(mode="python"))

    assert restored.vector_doc_id == chunk.vector_doc_id
    assert restored.metadata.vector_doc_id == chunk.vector_doc_id


def test_contract_schema_uses_json_nullable_and_1024_dimension() -> None:
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=object(),
    )

    schema = repository._build_schema(1024)
    fields = {field.name: field for field in schema.fields}

    assert fields["vector_doc_id"].is_primary is True
    assert fields["vector_doc_id"].params["max_length"] == 2048
    assert fields["embedding"].dtype == DataType.FLOAT_VECTOR
    assert fields["embedding"].params["dim"] == 1024
    assert fields["metadata_ext"].dtype == DataType.JSON
    assert fields["source_ext"].dtype == DataType.JSON
    assert fields["versions_ext"].dtype == DataType.JSON
    assert fields["serial_number"].nullable is True
    assert fields["company_name_norm"].nullable is True
    assert fields["is_current"].nullable is True
    assert fields["start_month"].dtype == DataType.VARCHAR


def test_contract_repository_requires_contract_schema_mode() -> None:
    settings = contract_settings().model_copy(update={"schema_mode": "legacy"})

    with pytest.raises(ValueError, match="MILVUS_SCHEMA_MODE=contract"):
        ContractMilvusVectorRepository(settings, vector_dimension=1024, client=object())


def test_missing_contract_collection_creates_schema_and_all_indexes() -> None:
    client = FakeContractMilvusClient(exists=False)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )

    repository.initialize(1024)

    assert len(client.create_calls) == 1
    assert client.drop_calls == []
    index_calls = client.index_params.calls
    assert len(index_calls) == 1 + len(CONTRACT_SCALAR_INDEX_FIELDS)
    assert index_calls[0] == {
        "field_name": "embedding",
        "index_name": "embedding_vector",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    assert all(call["index_type"] == "INVERTED" for call in index_calls[1:])


def test_existing_compatible_contract_collection_is_not_recreated() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()

    repository.initialize(1024)

    assert client.create_calls == []
    assert client.drop_calls == []


def test_existing_incompatible_contract_collection_stops_without_drop() -> None:
    client = FakeContractMilvusClient(exists=True)
    client.description = {
        "auto_id": False,
        "enable_dynamic_field": False,
        "fields": [{"name": "vector_doc_id", "type": DataType.VARCHAR}],
    }
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )

    with pytest.raises(ValueError, match="不兼容"):
        repository.initialize(1024)

    assert client.create_calls == []
    assert client.drop_calls == []


def test_contract_repository_upsert_uses_row_mapper() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    chunks = embedded_contract_chunks()

    result = repository.upsert_batch(chunks)

    assert result.affected_count == 3
    assert len(client.upsert_calls) == 1
    rows = client.upsert_calls[0]["data"]
    assert rows[0]["metadata_ext"] == chunks[0].metadata.model_dump(mode="json")
    assert rows[1]["source_ext"] == chunks[1].source.model_dump(mode="json")
    assert rows[2]["versions_ext"] == chunks[2].versions.model_dump(mode="json")
    assert client.drop_calls == []
    assert client.query_iterator_calls[0]["filter"] == "embedding_dim >= 0"
    assert client.query_iterator_calls[0]["batch_size"] == 1000
    assert client.query_iterator_calls[0]["limit"] == -1
    assert client.last_query_iterator is not None
    assert client.last_query_iterator.closed is True


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (None, "必须是 Mapping"),
        ({"ids": []}, "upsert_count"),
        ({"upsert_count": 3}, "缺少 ids 或 primary_keys"),
        ({"upsert_count": 2, "ids": []}, "请求数量不一致"),
        ({"upsert_count": 3, "ids": ["wrong-id"]}, "主键与请求主键不一致"),
    ],
)
def test_contract_repository_rejects_unconfirmed_mutation_response(
    response: object,
    message: str,
) -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    client.upsert_response = response

    with pytest.raises((TypeError, ValueError), match=message):
        repository.upsert_batch(embedded_contract_chunks())


def test_contract_repository_rejects_mixed_batch_model_space_before_upsert() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    chunks = embedded_contract_chunks()
    chunks[1] = chunks[1].model_copy(update={"embedding_model_version": "v2"})

    with pytest.raises(ValueError, match="不能混用 Embedding model-space"):
        repository.upsert_batch(chunks)

    assert client.upsert_calls == []


def test_contract_repository_rejects_existing_mismatched_model_space() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    client.rows["old-model-row"] = {
        "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
        "embedding_model_version": "v0",
        "embedding_dim": 1024,
    }

    with pytest.raises(ValueError, match="已有 Entity 与当前 Embedding model-space"):
        repository.upsert_batch(embedded_contract_chunks())

    assert client.upsert_calls == []


def test_contract_repository_checks_all_model_space_iterator_batches() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    matching = {
        "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
        "embedding_model_version": "v1",
        "embedding_dim": 1024,
    }
    client.model_space_batches = [
        [matching],
        [{**matching, "embedding_model_version": "v0"}],
    ]

    with pytest.raises(ValueError, match="已有 Entity 与当前 Embedding model-space"):
        repository.upsert_batch(embedded_contract_chunks())

    assert client.last_query_iterator is not None
    assert client.last_query_iterator.next_calls == 2
    assert client.last_query_iterator.closed is True
    assert client.upsert_calls == []


def test_contract_repository_rejects_missing_model_space_field_and_closes_iterator() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    client.model_space_batches = [[{"embedding_model_dim": 1024}]]

    with pytest.raises(ValueError, match="已有 Entity 与当前 Embedding model-space"):
        repository.upsert_batch(embedded_contract_chunks())

    assert client.last_query_iterator is not None
    assert client.last_query_iterator.closed is True
    assert client.upsert_calls == []


def test_contract_repository_closes_model_space_iterator_when_query_fails() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    client.iterator_error_on_call = 2
    client.model_space_batches = [[{
        "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
        "embedding_model_version": "v1",
        "embedding_dim": 1024,
    }]]

    with pytest.raises(RuntimeError, match="iterator query failed"):
        repository.upsert_batch(embedded_contract_chunks())

    assert client.last_query_iterator is not None
    assert client.last_query_iterator.next_calls == 2
    assert client.last_query_iterator.closed is True
    assert client.upsert_calls == []


def test_contract_repository_rejects_placeholder_model_version() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    chunks = embedded_contract_chunks()
    chunks[0] = chunks[0].model_copy(update={"embedding_model_version": "dev-unknown"})

    with pytest.raises(ValueError, match="禁止使用 embedding_model_version=dev-unknown"):
        repository.upsert_batch(chunks)

    assert client.upsert_calls == []


def test_contract_repository_search_preserves_search_hit_and_filter() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    chunks = embedded_contract_chunks()
    repository.upsert_batch(chunks)

    hits = repository.search([0.1] * 1024, limit=3, chunk_type="employment")

    assert len(hits) == 1
    assert isinstance(hits[0], SearchHit)
    assert hits[0].chunk_type == "employment"
    assert hits[0].vector_doc_id == chunks[1].vector_doc_id
    assert client.search_calls[0]["anns_field"] == "embedding"
    assert client.search_calls[0]["filter"] == "chunk_type == 'employment'"
    assert client.search_calls[0]["search_params"] == {
        "metric_type": "COSINE",
        "params": {"ef": 64},
    }
