"""V15 精简 Contract Collection Repository 的接通测试（Fake/Mock，不连真实 Milvus）。

覆盖 Task5-2 要求：
    1. V14 repository 行为完全不变（profile 默认值断言 + 现有 V14 测试套件）；
    2. V15 initialize 使用 7 个字段；
    3. V15 initialize 使用 2 个索引（embedding HNSW/COSINE + expert_id INVERTED）；
    4. V15 upsert 使用 MilvusV15RowMapper；
    5. V15 get 不请求 V14-only 字段；
    6. V15 query_by_expert_id 使用一级 expert_id 过滤；
    7. V15 list_all_vector_states 只取 3 个轻量字段；
    8. V15 destructive compatibility 使用 V15 schema；
    9. V15 不读取整个 metadata 做 reconciliation；
    10. V14/V15 mapper/spec/index 不串线。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from expert_contracts import adapt_expert_chunk_mapping
from rag_vector_store import (
    ContractEmbeddingWorker,
    ContractMilvusVectorRepository,
    EmbeddedContractPreparedChunk,
    MilvusSettings,
)
from rag_vector_store.milvus import ContractMilvusV15VectorRepository
from rag_vector_store.milvus.contract import MilvusRowMapper
from rag_vector_store.milvus.contract_v15 import MilvusV15RowMapper
from rag_vector_store.milvus.indexes import index_specs, index_specs_v15
from rag_vector_store.milvus.schemas.contract import (
    build_contract_schema,
    contract_milvus_field_specs,
)
from rag_vector_store.milvus.schemas.contract_v15 import v15_milvus_field_specs

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)
_DEFAULT_UPSERT_RESPONSE = object()
_DEFAULT_DELETE_RESPONSE = object()

V15_FIELD_NAMES = [
    "vector_doc_id",
    "expert_id",
    "content_hash",
    "metadata_hash",
    "chunk_text",
    "embedding",
    "metadata",
]
V15_LIGHT_STATE_FIELDS = ["vector_doc_id", "content_hash", "metadata_hash"]
# query_by_expert_id 默认轻量输出：含 expert_id（保持一级过滤键）。
V15_EXPERT_STATE_FIELDS = [
    "vector_doc_id",
    "expert_id",
    "content_hash",
    "metadata_hash",
]
# V15 一级物理字段：禁止出现在 flat metadata JSON 中（与 contract_v15 约束一致）。
V15_PHYSICAL_FIELDS = frozenset(V15_FIELD_NAMES) - {"metadata"}
# V14-only 一级字段：V15 中这些字段必须全部进入 flat metadata JSON。
V14_ONLY_TOP_LEVEL_FIELDS = (
    "serial_number",
    "chunk_type",
    "chunk_subtype",
    "chunk_index",
    "chunk_total",
    "char_count",
    "contract_version",
    "verification_status",
    "eligible_for_recommendation",
    "weight_tier",
    "is_active",
    "updated_at",
    "embedded_at",
    "embedding_dim",
    "embedding_model_alias",
    "embedding_model_version",
    "metadata_ext",
    "source_ext",
    "versions_ext",
)


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


class FakeV15MilvusClient:
    """记录式 V15 Fake：支持创建/描述/upsert/query/delete/iterator。"""

    def __init__(self, *, exists: bool) -> None:
        self.exists = exists
        self.description: dict[str, Any] | None = None
        self.index_params = FakeIndexParams()
        self.create_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.query_iterator_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.upsert_response: object = _DEFAULT_UPSERT_RESPONSE
        self.delete_response: object = _DEFAULT_DELETE_RESPONSE
        self.model_space_batches: list[object] | None = None
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
        return ["embedding_vector", "expert_id_inverted"]

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
            rows = list(self.rows.values())
        else:
            rows = [
                self.rows[vector_doc_id]
                for vector_doc_id in ids
                if vector_doc_id in self.rows
            ]
        output_fields = kwargs.get("output_fields")
        if output_fields:
            rows = [
                {field: row.get(field) for field in output_fields if field in row}
                for row in rows
            ]
        return rows

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
        iterator = FakeQueryIterator(batches)
        self.last_query_iterator = iterator
        return iterator

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        raise AssertionError("V15 测试不应触发 search")


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


def v15_settings(
    collection_name: str = "expert_candidates_v15_contract_dev",
    *,
    allowed: tuple[str, ...] = (),
) -> MilvusSettings:
    return MilvusSettings(
        collection_name=collection_name,
        index_type="HNSW",
        metric_type="COSINE",
        index_params={"M": 16, "efConstruction": 200},
        destructive_maintenance_collections=list(allowed),
    )


def make_v15_repository(
    client: FakeV15MilvusClient | None = None,
    **settings_kwargs: Any,
) -> ContractMilvusV15VectorRepository:
    return ContractMilvusV15VectorRepository(
        v15_settings(**settings_kwargs),
        1024,
        client=client,
        allow_collection_creation=True,
    )


def v15_description(repository: Any) -> dict[str, Any]:
    return repository._build_schema(1024).to_dict()


def v14_description() -> dict[str, Any]:
    return build_contract_schema(
        MilvusSettings(
            collection_name="expert_candidates_v14_contract_dev",
            index_type="HNSW",
            metric_type="COSINE",
            index_params={"M": 16, "efConstruction": 200},
        ),
        1024,
    ).to_dict()


# ---------------------------------------------------------------------------
# 1. V14 默认 profile 与行为不变
# ---------------------------------------------------------------------------


def test_v14_repository_profile_defaults_are_still_v14() -> None:
    """V14 Repository 的持久化 profile 必须仍指向 V14 符号（不串线）。"""

    repository = ContractMilvusVectorRepository(
        v15_settings("expert_candidates_v14_contract_dev"),
        1024,
    )
    assert repository._schema_field_specs is contract_milvus_field_specs
    assert repository._row_mapper_cls is MilvusRowMapper
    assert repository._index_specs_fn is index_specs
    assert repository._full_scan_filter == "embedding_dim >= 0"
    assert isinstance(repository._contract_row_mapper, MilvusRowMapper)
    assert not isinstance(repository._contract_row_mapper, MilvusV15RowMapper)


# ---------------------------------------------------------------------------
# 2/3. V15 initialize：7 字段、dynamic=False、2 索引
# ---------------------------------------------------------------------------


def test_v15_initialize_creates_7_field_schema() -> None:
    client = FakeV15MilvusClient(exists=False)
    repository = make_v15_repository(client)
    repository.initialize(1024)

    assert len(client.create_calls) == 1
    schema = client.create_calls[0]["schema"]
    description = schema.to_dict()
    fields = [
        field.get("name")
        for field in description.get("fields", [])
        if isinstance(field, Mapping)
    ]
    assert fields == V15_FIELD_NAMES
    assert description.get("auto_id") is False
    assert description.get("enable_dynamic_field") is False
    # 主键唯一且 auto_id=False
    primary = [
        field.get("name")
        for field in description.get("fields", [])
        if isinstance(field, Mapping) and field.get("is_primary")
    ]
    assert primary == ["vector_doc_id"]


def test_v15_initialize_declares_exactly_two_indexes() -> None:
    client = FakeV15MilvusClient(exists=False)
    repository = make_v15_repository(client)
    repository.initialize(1024)

    indexes = client.index_params.calls
    assert [(item["field_name"], item["index_name"]) for item in indexes] == [
        ("embedding", "embedding_vector"),
        ("expert_id", "expert_id_inverted"),
    ]
    vector_index, scalar_index = indexes[0], indexes[1]
    assert vector_index["index_type"] == "HNSW"
    assert vector_index["metric_type"] == "COSINE"
    assert scalar_index["index_type"] == "INVERTED"
    # 不允许出现 V14 才有的 chunk_type / eligible_for_recommendation 索引。
    for item in indexes:
        assert item["field_name"] not in (
            "chunk_type",
            "eligible_for_recommendation",
        )


def test_v15_initialize_existing_collection_passes_v15_gate() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    repository.initialize(1024)  # 不应抛异常
    assert client.create_calls == []


def test_v15_initialize_rejects_v14_collection() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v14_description()
    with pytest.raises(ValueError, match="不兼容"):
        repository.initialize(1024)
    assert client.create_calls == []


# ---------------------------------------------------------------------------
# 4. V15 upsert 走 MilvusV15RowMapper
# ---------------------------------------------------------------------------


def test_v15_upsert_uses_v15_row_mapper_and_7_field_rows() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)

    result = repository.upsert_batch(chunks)

    assert result.affected_count == len(chunks)
    assert set(result.primary_keys) == {chunk.vector_doc_id for chunk in chunks}
    assert isinstance(repository._contract_row_mapper, MilvusV15RowMapper)
    rows = client.upsert_calls[0]["data"]
    assert len(rows) == len(chunks)
    for row in rows:
        assert set(row) == set(V15_FIELD_NAMES)
        # V14-only 一级字段不允许出现在 V15 行。
        for forbidden in V14_ONLY_TOP_LEVEL_FIELDS:
            assert forbidden not in row
        # flat metadata 承载其余信息。
        assert row["metadata"]["contract_version"] == "4.0.0"
        assert "chunk_type" in row["metadata"]
        assert "source_table" in row["metadata"]
        assert "chunk_policy_version" in row["metadata"]
        # 一级字段不在 metadata 中重复。
        for physical in V15_PHYSICAL_FIELDS:
            assert physical not in row["metadata"]


def test_v15_upsert_preserves_embedding_and_hashes() -> None:
    chunk = embedded_contract_chunks()[0]
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)

    repository.upsert_batch([chunk])

    row = client.upsert_calls[0]["data"][0]
    assert row["vector_doc_id"] == chunk.vector_doc_id
    assert row["embedding"] == list(chunk.embedding)
    assert row["content_hash"] == chunk.content_hash
    assert row["metadata_hash"] == chunk.metadata_hash
    assert row["chunk_text"] == chunk.chunk_text
    assert row["expert_id"] == chunk.expert_id


def test_v15_upsert_model_space_check_reads_flat_metadata() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.model_space_batches = [
        [
            {
                "metadata": {
                    "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
                    "embedding_model_version": "v1",
                    "embedding_dim": 1024,
                }
            }
        ]
    ]

    repository.upsert_batch(chunks)  # 模型空间一致，不应抛异常

    scan_call = client.query_iterator_calls[0]
    # 全量扫描 filter 基于一级 expert_id（不依赖 JSON）。
    assert scan_call["filter"] == "expert_id >= 0"
    # model-space validation 仍从 metadata JSON 读取模型空间字段。
    assert scan_call["output_fields"] == ["metadata"]


def test_v15_upsert_model_space_mismatch_fails_closed() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.model_space_batches = [
        [
            {
                "metadata": {
                    "embedding_model_alias": "other-vendor",
                    "embedding_model_version": "v2",
                    "embedding_dim": 1024,
                }
            }
        ]
    ]

    with pytest.raises(ValueError, match="model-space"):
        repository.upsert_batch(chunks)
    assert client.upsert_calls == []


# ---------------------------------------------------------------------------
# 5. V15 get_by_vector_doc_id 不请求 V14-only 字段
# ---------------------------------------------------------------------------


def test_v15_get_by_vector_doc_id_requests_v15_fields_only() -> None:
    chunk = embedded_contract_chunks()[0]
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.rows = {
        chunk.vector_doc_id: repository._contract_row_mapper.to_row(chunk)
    }

    row = repository.get_by_vector_doc_id(chunk.vector_doc_id)

    assert client.query_calls[0]["output_fields"] == V15_FIELD_NAMES
    for forbidden in V14_ONLY_TOP_LEVEL_FIELDS:
        assert forbidden not in client.query_calls[0]["output_fields"]
    assert row is not None
    assert row["vector_doc_id"] == chunk.vector_doc_id


def test_v15_get_by_vector_doc_id_without_embedding() -> None:
    chunk = embedded_contract_chunks()[0]
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.rows = {
        chunk.vector_doc_id: repository._contract_row_mapper.to_row(chunk)
    }

    row = repository.get_by_vector_doc_id(
        chunk.vector_doc_id,
        include_embedding=False,
    )

    assert "embedding" not in client.query_calls[0]["output_fields"]
    assert "embedding" not in row


# ---------------------------------------------------------------------------
# 6. V15 query_by_expert_id 使用一级 expert_id
# ---------------------------------------------------------------------------


def test_v15_query_by_expert_id_uses_top_level_expert_id() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    target_expert_id = chunks[0].expert_id
    for chunk in chunks:
        client.rows[chunk.vector_doc_id] = (
            repository._contract_row_mapper.to_row(chunk)
        )

    rows = repository.query_by_expert_id(target_expert_id)

    scan_call = client.query_iterator_calls[0]
    assert scan_call["filter"] == f"expert_id == {target_expert_id}"
    assert scan_call["output_fields"] == V15_EXPERT_STATE_FIELDS
    # 默认轻量输出：不返回 embedding / chunk_text / 整包 metadata。
    assert "embedding" not in scan_call["output_fields"]
    assert "chunk_text" not in scan_call["output_fields"]
    assert "metadata" not in scan_call["output_fields"]
    assert rows and all("metadata" not in row for row in rows)


# ---------------------------------------------------------------------------
# 7/9. V15 list_all_vector_states 保持 3 字段轻量读取
# ---------------------------------------------------------------------------


def test_v15_list_all_vector_states_reads_only_light_fields() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    for chunk in chunks:
        client.rows[chunk.vector_doc_id] = (
            repository._contract_row_mapper.to_row(chunk)
        )

    rows = repository.list_all_vector_states()

    scan_call = client.query_iterator_calls[0]
    assert scan_call["output_fields"] == V15_LIGHT_STATE_FIELDS
    assert scan_call["filter"] == "expert_id >= 0"
    # 全量 reconciliation 状态扫描不依赖 JSON：filter 不含 metadata JSON path。
    assert "metadata[" not in scan_call["filter"]
    # 明确不读取整个 metadata / embedding（Full Reconciliation 依赖轻量语义）。
    assert "metadata" not in scan_call["output_fields"]
    assert "embedding" not in scan_call["output_fields"]
    assert set(rows[0]) == set(V15_LIGHT_STATE_FIELDS)


# ---------------------------------------------------------------------------
# 8. V15 destructive compatibility 使用 V15 schema
# ---------------------------------------------------------------------------


def test_v15_destructive_guard_passes_with_v15_schema() -> None:
    collection_name = "expert_candidates_v15_contract_dev"
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(
        client,
        collection_name=collection_name,
        allowed=(collection_name,),
    )
    client.description = v15_description(repository)

    summary = repository.assert_destructive_maintenance_allowed(
        expected_dimension=1024
    )

    assert summary["authorized"] is True
    assert summary["schema_compatible"] is True
    assert summary["primary_key"] == "vector_doc_id"
    assert summary["embedding_dim"] == 1024


def test_v15_destructive_guard_rejects_v14_schema() -> None:
    """V15 guard 必须用 V15 7 字段预期判断，不能拿 V14 25 字段 schema 放行。"""

    collection_name = "expert_candidates_v15_contract_dev"
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(
        client,
        collection_name=collection_name,
        allowed=(collection_name,),
    )
    client.description = v14_description()

    with pytest.raises(ValueError, match="不兼容"):
        repository.assert_destructive_maintenance_allowed(
            expected_dimension=1024
        )


def test_v15_destructive_guard_unauthorized_fails_closed() -> None:
    collection_name = "expert_candidates_v15_contract_dev"
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client, collection_name=collection_name)
    client.description = v15_description(repository)

    with pytest.raises(RuntimeError, match="未获授权"):
        repository.assert_destructive_maintenance_allowed(
            expected_dimension=1024
        )


# ---------------------------------------------------------------------------
# 10. V14/V15 mapper/spec/index 不串线
# ---------------------------------------------------------------------------


def test_v14_v15_profiles_are_isolated() -> None:
    v14_repo = ContractMilvusVectorRepository(
        v15_settings("expert_candidates_v14_contract_dev"),
        1024,
    )
    v15_repo = make_v15_repository()

    assert v14_repo._schema_field_specs is not v15_repo._schema_field_specs
    assert v14_repo._schema_field_specs is contract_milvus_field_specs
    assert v15_repo._schema_field_specs is v15_milvus_field_specs
    assert v14_repo._row_mapper_cls is MilvusRowMapper
    assert v15_repo._row_mapper_cls is MilvusV15RowMapper
    assert v14_repo._index_specs_fn is index_specs
    assert v15_repo._index_specs_fn is index_specs_v15
    assert v14_repo._full_scan_filter == "embedding_dim >= 0"
    assert v15_repo._full_scan_filter == "expert_id >= 0"
    # V15 full-scan filter 不得依赖 metadata JSON path。
    assert "metadata" not in v15_repo._full_scan_filter
    assert "metadata[" not in v15_repo._full_scan_filter
    assert v14_repo._model_space_fields == (
        "embedding_model_alias",
        "embedding_model_version",
        "embedding_dim",
    )
    assert v15_repo._model_space_fields == ("metadata",)


def test_v15_schema_and_v14_schema_field_sets_differ() -> None:
    v15_fields = {spec.name for spec in v15_milvus_field_specs(1024)}
    v14_fields = {spec.name for spec in contract_milvus_field_specs(1024)}
    assert v15_fields == set(V15_FIELD_NAMES)
    assert len(v15_fields) == 7
    assert len(v14_fields) == 25
    assert v15_fields.isdisjoint(
        {"metadata_ext", "source_ext", "versions_ext", "chunk_type"}
    )


def test_v15_repository_uses_v15_builder_behind_initialize() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    repository.initialize(1024)
    schema = repository._build_schema(1024)
    description = schema.to_dict()
    field_names = {
        field.get("name")
        for field in description.get("fields", [])
        if isinstance(field, Mapping)
    }
    assert field_names == set(V15_FIELD_NAMES)
    assert field_names == {
        spec.name for spec in v15_milvus_field_specs(1024)
    }
    # profile builder 与 spec 一致（build_v15_schema 应产生相同字段集）。
    assert {
        spec.name for spec in v15_milvus_field_specs(1024)
    } == field_names


# ---------------------------------------------------------------------------
# V15 delete / 继承能力与 search 占位
# ---------------------------------------------------------------------------


def test_v15_delete_by_vector_doc_ids() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    for chunk in chunks:
        client.rows[chunk.vector_doc_id] = (
            repository._contract_row_mapper.to_row(chunk)
        )

    target = chunks[0].vector_doc_id
    result = repository.delete_by_vector_doc_ids(
        [target, "expert:99999:profile_summary:1:0"]
    )

    assert result.affected_count == 1
    assert target not in client.rows
    assert client.delete_calls[0]["ids"] == [
        target,
        "expert:99999:profile_summary:1:0",
    ]


def test_v15_search_not_connected_yet() -> None:
    repository = make_v15_repository()
    with pytest.raises(NotImplementedError, match="检索通路尚未接通"):
        repository.search([0.1] * 1024)