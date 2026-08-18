from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal
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
    / "expert_chunk_embedding_input_v4.jsonl"
)
_DEFAULT_UPSERT_RESPONSE = object()
_DEFAULT_DELETE_RESPONSE = object()


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
        self.delete_calls: list[dict[str, Any]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.drop_calls: list[str] = []
        self.upsert_response: object = _DEFAULT_UPSERT_RESPONSE
        self.delete_response: object = _DEFAULT_DELETE_RESPONSE
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

    def delete(self, **kwargs: Any) -> Any:
        self.delete_calls.append(kwargs)
        ids = [str(item) for item in kwargs["ids"]]
        deleted = [item for item in ids if item in self.rows]
        for item in deleted:
            del self.rows[item]
        response = {"delete_count": len(deleted)}
        return (
            response
            if self.delete_response is _DEFAULT_DELETE_RESPONSE
            else self.delete_response
        )

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append(kwargs)
        ids = kwargs.get("ids")
        if ids is None:
            return list(self.rows.values())
        return [self.rows[vector_doc_id] for vector_doc_id in ids if vector_doc_id in self.rows]

    def query_iterator(self, **kwargs: Any) -> FakeQueryIterator:
        self.query_iterator_calls.append(kwargs)
        rows = list(self.rows.values())
        if kwargs.get("filter", "").startswith("expert_id == "):
            expert_id = int(kwargs["filter"].split("== ")[1])
            rows = [
                row
                for row in rows
                if row.get("expert_id") == expert_id
                or str(row.get("expert_id")) == str(expert_id)
            ]
        output_fields = kwargs.get("output_fields")
        if output_fields:
            rows = [
                {field: row.get(field) for field in output_fields if field in row}
                for row in rows
            ]
        batches = (
            [rows]
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
        collection_name="expert_candidates_v14_contract_dev",
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
    assert entity.metadata_hash == chunk.metadata_hash
    assert entity.verification_status == chunk.metadata.verification_status
    assert entity.weight_tier == chunk.metadata.weight_tier
    assert entity.eligible_for_recommendation == (
        chunk.metadata.eligible_for_recommendation
    )
    # source_field 只进入 source_ext，不保留独立 Scalar
    assert entity.source_ext["source_field"] == chunk.source.source_field
    # 不存在旧 v13 标量投影，也不存在已废弃字段
    dumped = entity.model_dump()
    for removed in (
        "chunk_id",
        "source_item_key",
        "data_version",
        "dictionary_version",
        "project_role",
        "company_normalization_status",
        "company_name_norm",
        "role_code",
        "role_level",
        "start_month",
        "end_month",
        "is_current",
        "origin_type",
        "source_table",
        "source_field",
        "source_field_id",
        "chunk_policy_version",
        "metadata_schema_version",
    ):
        assert removed not in dumped
    assert entity.metadata_ext == chunk.metadata.model_dump(mode="json")
    assert entity.source_ext == chunk.source.model_dump(mode="json")
    assert entity.versions_ext == chunk.versions.model_dump(mode="json")
    assert entity.embedding_dim == 1024


def test_milvus_row_mapper_round_trip_preserves_upstream_vector_doc_id() -> None:
    chunk = embedded_contract_chunks()[1]
    entity = MilvusRowMapper().to_entity(chunk)

    restored = MilvusRowMapper().from_row(entity.model_dump(mode="python"))

    assert restored.vector_doc_id == chunk.vector_doc_id
    # metadata 不保留 vector_doc_id 副本，因此恢复后顶层主键仍然成立即可；
    # 不再要求 metadata.vector_doc_id 存在。
    assert "vector_doc_id" not in restored.metadata.model_dump()
    assert restored.metadata_hash == chunk.metadata_hash
    assert restored.metadata.verification_status == chunk.metadata.verification_status
    # source_field 从 source_ext 还原
    assert restored.source.source_field == chunk.source.source_field
    assert restored.source.source_table == chunk.source.source_table


def test_milvus_row_mapper_source_ext_is_complete_source_snapshot() -> None:
    chunk = embedded_contract_chunks()[1]
    entity = MilvusRowMapper().to_entity(chunk)

    source = entity.source_ext
    assert set(source) == {
        "source_table",
        "source_field",
        "source_field_id",
        "source_version",
        "source_updated_at",
        "source_hash",
    }
    assert source == chunk.source.model_dump(mode="json")
    restored = MilvusRowMapper().from_row(entity.model_dump(mode="python"))
    assert restored.source == chunk.source
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
    assert fields["eligible_for_recommendation"].nullable is True
    assert fields["metadata_hash"].dtype == DataType.VARCHAR
    assert fields["verification_status"].dtype == DataType.VARCHAR
    assert fields["is_active"].dtype == DataType.BOOL
    # source_field 只进入 source_ext JSON，不保留独立 Scalar 列
    assert "source_field" not in fields
    assert fields["source_ext"].dtype == DataType.JSON
    # 已废弃/非检索字段不进入正式 Schema
    for removed in (
        "chunk_id",
        "source_item_key",
        "data_version",
        "company_normalization_status",
        "project_role",
        "company_name_norm",
        "start_month",
    ):
        assert removed not in fields


def test_milvus_work_years_decimal_is_serialized_as_string_not_float() -> None:
    chunk = embedded_contract_chunks()[0]  # profile：work_years=Decimal('12.00')
    entity = MilvusRowMapper().to_entity(chunk)

    work_years = entity.metadata_ext["work_years"]
    assert isinstance(work_years, str)
    assert work_years == "12.00"
    assert isinstance(chunk.metadata.work_years, Decimal)
    # JSON 快照可稳定序列化（不含 Decimal / float）
    json.dumps(entity.metadata_ext)
    # 从 backend 字典还原后 work_years 仍是 Decimal
    restored = MilvusRowMapper().from_row(entity.model_dump(mode="python"))
    assert restored.metadata.work_years == Decimal("12.00")


def test_milvus_json_snapshots_match_contract_and_enum_values() -> None:
    chunks = embedded_contract_chunks()
    for chunk in chunks:
        entity = MilvusRowMapper().to_entity(chunk)
        assert entity.metadata_ext == chunk.metadata.model_dump(mode="json")
        assert entity.source_ext == chunk.source.model_dump(mode="json")
        assert entity.versions_ext == chunk.versions.model_dump(mode="json")
        # 顶层 vector_doc_id 不进入 metadata JSON
        assert "vector_doc_id" not in entity.metadata_ext
    # verification_status 正式三值
    statuses = {c.metadata.verification_status for c in chunks}
    assert "verified" in statuses
    assert "source_summary" in statuses
    # origin_type 正式两值
    origins = {c.metadata.origin_type for c in chunks}
    assert {"source_parsed", "compressed_from_source"} == origins
    # YYYYMM start_month / end_month 出现在 metadata JSON 快照中，
    # 且不作为独立标量字段（不属于有明确检索需求的投影）
    employment = chunks[1]
    employment_entity = MilvusRowMapper().to_entity(employment)
    assert employment_entity.metadata_ext["start_month"] == "202301"
    assert employment_entity.metadata_ext["end_month"] is None
    assert "start_month" not in employment_entity.model_dump()

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


def test_contract_repository_upsert_rows_have_no_deprecated_fields() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()

    repository.upsert_batch(embedded_contract_chunks())

    row = client.upsert_calls[0]["data"][0]
    for removed in (
        "chunk_id",
        "source_item_key",
        "data_version",
        "dictionary_version",
        "project_role",
        "company_normalization_status",
        "company_name_norm",
        "role_code",
        "start_month",
        "end_month",
        "is_current",
        "origin_type",
        "metadata.vector_doc_id",
    ):
        assert removed not in row
    assert row["metadata_hash"] not in (None, "")
    assert row["vector_doc_id"] == client.upsert_calls[0]["data"][0]["vector_doc_id"]
    assert row["embedding_dim"] == 1024


def test_contract_repository_rejects_duplicate_vector_doc_id_batch() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    chunks = embedded_contract_chunks()
    chunks[1] = chunks[1].model_copy(
        update={"vector_doc_id": chunks[0].vector_doc_id}
    )

    with pytest.raises(ValueError, match="重复"):
        repository.upsert_batch(chunks)

    assert client.upsert_calls == []


def test_contract_repository_dimension_error_uses_vector_doc_id() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        # 不预先锁定维度，让批次维度检查走 vector_doc_id 定位的错误分支
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    chunks = embedded_contract_chunks()
    chunks[0] = chunks[0].model_copy(update={"embedding": [0.1] * 512})

    with pytest.raises(ValueError, match=r"vector_doc_id.*employment:21307:0|employment:21307:0"):
        repository.upsert_batch(chunks)

    assert client.upsert_calls == []


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


def test_contract_repository_delete_by_vector_doc_ids() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    chunks = embedded_contract_chunks()
    repository.upsert_batch(chunks)

    target = chunks[0].vector_doc_id
    result = repository.delete_by_vector_doc_ids([target])

    assert client.delete_calls == [
        {
            "collection_name": "expert_candidates_v14_contract_dev",
            "ids": [target],
        }
    ]
    assert result.affected_count == 1
    assert result.primary_keys == []
    # 其余记录仍保留
    assert client.rows.get(chunks[1].vector_doc_id)


def test_contract_repository_delete_empty_list_is_noop() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )

    result = repository.delete_by_vector_doc_ids([])

    assert result.affected_count == 0
    assert result.primary_keys == []
    assert client.delete_calls == []
    assert client.query_iterator_calls == []


def test_contract_repository_delete_multiple_ids() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    chunks = embedded_contract_chunks()
    repository.upsert_batch(chunks)

    ids = [chunk.vector_doc_id for chunk in chunks]
    result = repository.delete_by_vector_doc_ids(ids)

    assert result.affected_count == 3
    assert client.delete_calls[0]["ids"] == ids
    assert client.rows == {}


def test_contract_repository_delete_dedupes_ids() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    chunks = embedded_contract_chunks()
    target = chunks[0].vector_doc_id

    repository.delete_by_vector_doc_ids([target, target])

    assert client.delete_calls[0]["ids"] == [target]


def test_contract_repository_delete_non_existent_is_not_error() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )

    result = repository.delete_by_vector_doc_ids(["expert:1:profile_summary:1:0"])

    assert result.affected_count == 0
    assert result.primary_keys == []


def test_contract_repository_delete_rejects_invalid_ids() -> None:
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=object(),
    )

    with pytest.raises(ValueError, match="必须是"):
        repository.delete_by_vector_doc_ids([""])

    with pytest.raises(ValueError, match="必须是"):
        repository.delete_by_vector_doc_ids(["valid", 123])


def test_contract_repository_delete_wraps_driver_error() -> None:
    client = FakeContractMilvusClient(exists=True)
    client.delete_response = {"delete_count": "not-an-int"}
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )

    with pytest.raises(TypeError, match="delete_count"):
        repository.delete_by_vector_doc_ids(["expert:1:profile_summary:1:0"])


def test_contract_repository_query_by_expert_id_returns_minimal_fields() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    chunks = embedded_contract_chunks()
    repository.upsert_batch(chunks)

    rows = repository.query_by_expert_id(900001)

    assert len(rows) == 3
    assert all(set(row) == {"vector_doc_id", "expert_id", "content_hash", "metadata_hash"}
               for row in rows)
    # embedding 不在输出字段
    assert "embedding" not in client.query_iterator_calls[-1]["output_fields"]
    # 过滤表达式正确
    assert client.query_iterator_calls[-1]["filter"] == "expert_id == 900001"


def test_contract_repository_query_by_expert_id_empty() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    repository.upsert_batch(embedded_contract_chunks())

    rows = repository.query_by_expert_id(999999)

    assert rows == []
    assert client.last_query_iterator is not None
    assert client.last_query_iterator.closed is True


def test_contract_repository_query_by_expert_id_custom_output_fields() -> None:
    client = FakeContractMilvusClient(exists=True)
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=client,
    )
    client.description = repository._build_schema(1024).to_dict()
    repository.upsert_batch(embedded_contract_chunks())

    rows = repository.query_by_expert_id(
        900001,
        output_fields=["vector_doc_id", "expert_id"],
    )

    assert all(set(row) == {"vector_doc_id", "expert_id"} for row in rows)


def test_contract_repository_query_by_expert_id_rejects_bad_expert_id() -> None:
    repository = ContractMilvusVectorRepository(
        contract_settings(),
        vector_dimension=1024,
        client=object(),
    )

    with pytest.raises(ValueError, match="expert_id"):
        repository.query_by_expert_id(0)
    with pytest.raises(ValueError, match="expert_id"):
        repository.query_by_expert_id(True)
