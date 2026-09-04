"""V15 精简 Contract Collection Repository 的接通测试（Fake/Mock，不连真实 Milvus）。

覆盖 Task5-3 Pivot 要求：
    1. V15 repository 持久化 profile 默认值指向 V15 符号；
    2. V15 initialize 使用 4 个字段；
    3. V15 initialize 只有 1 个索引（embedding HNSW/COSINE）；
    4. V15 upsert 使用 MilvusV15RowMapper，只产出 4 个物理字段；
    5. expert_id / content_hash / metadata_hash 已进入 metadata，不再是一级字段；
    6. V15 get 不请求历史一级字段；
    7. V15 query_by_expert_id 使用 metadata["expert_id"] JSON path 过滤；
    8. V15 list_all_vector_states 物理读取 vector_doc_id+metadata，从 metadata
       提取两个 hash，公共返回语义保持 vector_doc_id/content_hash/metadata_hash；
    9. V15 full scan filter 不引用任何不存在的 V15 scalar；
    10. V15 destructive compatibility 使用 4 字段 V15 schema；
    11. V15 model-space validation 仍从 metadata 读取。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from expert_contracts import adapt_expert_chunk_mapping
from rag_vector_store import (
    ContractEmbeddingWorker,
    EmbeddedContractPreparedChunk,
    MilvusSettings,
)
from rag_vector_store.milvus import ContractMilvusV15VectorRepository
from rag_vector_store.milvus.contract_v15 import MilvusV15RowMapper
from rag_vector_store.milvus.indexes import index_specs_v15
from rag_vector_store.milvus.schemas.contract_v15 import v15_milvus_field_specs

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)
_DEFAULT_UPSERT_RESPONSE = object()
_DEFAULT_DELETE_RESPONSE = object()

# V15 纯 4 字段。
V15_FIELD_NAMES = [
    "vector_doc_id",
    "chunk_text",
    "embedding",
    "metadata",
]
# list_all_vector_states 的公共返回语义（物理读取 vector_doc_id+metadata 后提取）。
V15_LIGHT_STATE_FIELDS = ["vector_doc_id", "content_hash", "metadata_hash"]
# V15 一级物理字段：禁止出现在 flat metadata JSON 中（与 contract_v15 约束一致）。
V15_PHYSICAL_FIELDS = frozenset(V15_FIELD_NAMES) - {"metadata"}
# 已从一级字段迁移进 flat metadata JSON 的字段。
V15_MOVED_TO_METADATA = ("expert_id", "content_hash", "metadata_hash")
# V14-only 一级字段：V15 中这些字段必须全部进入 flat metadata JSON。
V14_ONLY_TOP_LEVEL_FIELDS = (
    "expert_id",
    "content_hash",
    "metadata_hash",
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
    """记录式 V15 Fake：支持创建/描述/upsert/query/delete/iterator。

    ``query_iterator`` 支持 V15 实际使用的两个过滤表达式：
        - ``metadata["expert_id"] == N``（JSON path）
        - ``vector_doc_id != ""``（PK 恒真全量扫描）
    """

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
        self.row_count_override: int | None = None
        self.last_query_iterator: FakeQueryIterator | None = None

    def has_collection(self, collection_name: str) -> bool:
        return self.exists

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        if self.description is None:
            raise AssertionError("测试未配置 Collection description")
        return self.description

    def get_collection_stats(self, collection_name: str) -> dict[str, int]:
        if self.row_count_override is not None:
            return {"row_count": self.row_count_override}
        if self.model_space_batches:
            first = self.model_space_batches[0]
            if isinstance(first, list):
                return {"row_count": len(first)}
        return {"row_count": len(self.rows)}

    def prepare_index_params(self) -> FakeIndexParams:
        return self.index_params

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)
        self.exists = True

    def list_indexes(self, collection_name: str) -> list[str]:
        return ["embedding_vector"]

    def describe_index(
        self,
        collection_name: str,
        index_name: str,
    ) -> dict[str, Any]:
        assert index_name == "embedding_vector"
        return {
            "field_name": "embedding",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "M": "16",
            "efConstruction": "200",
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
        # model-space 采样校验：业务代码读取 vector_doc_id + metadata。
        # 优先返回测试显式配置的 model_space_batches[0]，保持与旧
        # query_iterator 测试数据准备方式兼容。
        output_fields = kwargs.get("output_fields")
        if (
            output_fields == ["vector_doc_id", "metadata"]
            and self.model_space_batches
        ):
            sampled = self.model_space_batches[0]
            if isinstance(sampled, list):
                return sampled
        ids = kwargs.get("ids")
        if ids is None:
            rows = list(self.rows.values())
        else:
            rows = [
                self.rows[vector_doc_id]
                for vector_doc_id in ids
                if vector_doc_id in self.rows
            ]
        if output_fields:
            rows = [
                {field: row.get(field) for field in output_fields if field in row}
                for row in rows
            ]
        return rows

    def query_iterator(self, **kwargs: Any) -> FakeQueryIterator:
        self.query_iterator_calls.append(kwargs)
        rows = list(self.rows.values())
        filter_expression = kwargs.get("filter", "")
        if filter_expression.startswith("metadata["):
            # metadata["expert_id"] == N
            expert_id = int(filter_expression.split(" == ")[1])
            rows = [
                row
                for row in rows
                if isinstance(row.get("metadata"), Mapping)
                and row["metadata"].get("expert_id") == expert_id
            ]
        elif filter_expression == 'vector_doc_id != ""':
            # PK 恒真全量扫描：所有行的 PK 均为非空字符串
            rows = [row for row in rows if row.get("vector_doc_id")]
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


def incompatible_description() -> dict[str, Any]:
    """构造一个与 V15 4 字段 Schema 不兼容的 Collection description。

    在 V15 的 4 个字段基础上额外加入一级 ``expert_id``（历史 V14 标量投影），
    使 V15 compatibility gate 检测到 unexpected_fields 并 fail-closed。
    """

    fields = [
        {"name": "vector_doc_id", "type": "VARCHAR", "is_primary": True,
         "nullable": False, "params": {"max_length": 2048}},
        {"name": "chunk_text", "type": "VARCHAR", "is_primary": False,
         "nullable": False, "params": {"max_length": 8192}},
        {"name": "embedding", "type": "FLOAT_VECTOR", "is_primary": False,
         "nullable": False, "params": {"dim": 1024}},
        {"name": "metadata", "type": "JSON", "is_primary": False,
         "nullable": False, "params": {}},
        # 额外一级字段：V15 不允许，触发 unexpected_fields
        {"name": "expert_id", "type": "INT64", "is_primary": False,
         "nullable": False, "params": {}},
    ]
    return {
        "auto_id": False,
        "enable_dynamic_field": False,
        "fields": fields,
    }


def _seed_rows(
    repository: ContractMilvusV15VectorRepository,
    client: FakeV15MilvusClient,
    chunks: Sequence[EmbeddedContractPreparedChunk],
) -> None:
    for chunk in chunks:
        client.rows[chunk.vector_doc_id] = (
            repository._contract_row_mapper.to_row(chunk)
        )


# ---------------------------------------------------------------------------
# 1. V14 默认 profile 与行为不变
# ---------------------------------------------------------------------------


def test_v15_repository_profile_defaults_point_to_v15_symbols() -> None:
    """V15 Repository 的持久化 profile 必须指向 V15 符号（不串线）。"""

    repository = make_v15_repository()
    assert repository.profile.schema_field_specs is v15_milvus_field_specs
    assert repository.profile.row_mapper_cls is MilvusV15RowMapper
    assert repository.profile.index_specs_fn is index_specs_v15
    assert repository.profile.full_scan_filter == 'vector_doc_id != ""'
    assert repository.profile.model_space_fields == ("metadata",)
    assert isinstance(repository._contract_row_mapper, MilvusV15RowMapper)


def test_v15_repository_does_not_expose_raw_milvus_client() -> None:
    """业务调用方不能绕过 Repository 直接取得原始 PyMilvus Client。"""

    repository = make_v15_repository()

    assert not hasattr(repository, "client")


def test_v15_repository_profile_is_public_and_read_only() -> None:
    repository = make_v15_repository()

    assert repository.profile.owner_field == "expert_id"
    with pytest.raises(FrozenInstanceError):
        repository.profile.owner_field = "resume_id"  # type: ignore[misc]


def test_v15_repository_default_refuses_to_create_missing_collection() -> None:
    client = FakeV15MilvusClient(exists=False)
    repository = ContractMilvusV15VectorRepository(
        v15_settings(),
        1024,
        client=client,
    )

    with pytest.raises(RuntimeError, match="禁止自动创建"):
        repository.initialize(1024)

    assert client.create_calls == []


# ---------------------------------------------------------------------------
# 2/3. V15 initialize：4 字段、dynamic=False、1 索引
# ---------------------------------------------------------------------------


def test_v15_initialize_creates_4_field_schema() -> None:
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
    # 精确 4 字段
    assert len(fields) == 4


def test_v15_initialize_declares_only_vector_index() -> None:
    client = FakeV15MilvusClient(exists=False)
    repository = make_v15_repository(client)
    repository.initialize(1024)

    indexes = client.index_params.calls
    assert [(item["field_name"], item["index_name"]) for item in indexes] == [
        ("embedding", "embedding_vector"),
    ]
    vector_index = indexes[0]
    assert vector_index["index_type"] == "HNSW"
    assert vector_index["metric_type"] == "COSINE"
    # 不允许任何 scalar index（含 expert_id / chunk_type / eligible_for_recommendation）。
    for item in indexes:
        assert item["index_type"] != "INVERTED"
        assert item["field_name"] not in (
            "expert_id",
            "chunk_type",
            "eligible_for_recommendation",
        )


def test_v15_initialize_existing_collection_passes_v15_gate() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    repository.initialize(1024)  # 不应抛异常
    assert client.create_calls == []


def test_v15_initialize_rejects_incompatible_collection() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = incompatible_description()
    with pytest.raises(ValueError, match="不兼容"):
        repository.initialize(1024)
    assert client.create_calls == []


# ---------------------------------------------------------------------------
# 4/5. V15 upsert 走 MilvusV15RowMapper，只产出 4 个物理字段
# ---------------------------------------------------------------------------


def test_v15_upsert_uses_v15_row_mapper_and_4_field_rows() -> None:
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
        # V14-only 一级字段（含 expert_id/content_hash/metadata_hash）不允许出现在 V15 行。
        for forbidden in V14_ONLY_TOP_LEVEL_FIELDS:
            assert forbidden not in row
        # flat metadata 承载其余信息（含迁移回来的 expert_id / hash）。
        assert row["metadata"]["contract_version"] == "4.1.0"
        assert "chunk_type" in row["metadata"]
        assert "source_table" in row["metadata"]
        assert "chunk_policy_version" in row["metadata"]
        assert "expert_id" in row["metadata"]
        assert "content_hash" in row["metadata"]
        assert "metadata_hash" in row["metadata"]
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
    assert row["chunk_text"] == chunk.chunk_text
    # expert_id / hash 已进入 flat metadata，值原样透传（不重算）
    assert row["metadata"]["expert_id"] == chunk.expert_id
    assert row["metadata"]["content_hash"] == chunk.content_hash
    assert row["metadata"]["metadata_hash"] == chunk.metadata_hash


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

    scan_call = client.query_calls[0]
    # 小 Collection 直接检查全部 Entity。
    assert scan_call["filter"] == 'vector_doc_id != ""'
    # model-space validation 仍从 metadata JSON 读取模型空间字段。
    assert scan_call["output_fields"] == ["vector_doc_id", "metadata"]


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


def test_v15_large_collection_uses_random_model_space_sample() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.row_count_override = 100_000
    client.model_space_batches = [
        [
            {
                "vector_doc_id": "expert:1:profile_summary:1:0",
                "metadata": {
                    "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
                    "embedding_model_version": "v1",
                    "embedding_dim": 1024,
                },
            }
        ]
    ]

    repository.upsert_batch(chunks)

    scan_call = client.query_calls[0]
    assert scan_call["filter"].startswith("RANDOM_SAMPLE(")
    assert scan_call["limit"] == 500
    assert scan_call["output_fields"] == ["vector_doc_id", "metadata"]


# ---------------------------------------------------------------------------
# 6. V15 get_by_vector_doc_id 不请求 V14-only 字段
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


def test_v15_get_many_batches_primary_key_queries_by_200() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    vector_doc_ids = [
        f"expert:1:profile_summary:{index}:0"
        for index in range(401)
    ]

    result = repository.get_many_by_vector_doc_ids(
        vector_doc_ids,
        include_embedding=False,
    )

    assert result == {}
    assert [len(call["ids"]) for call in client.query_calls] == [200, 200, 1]
    assert all(
        "embedding" not in call["output_fields"]
        for call in client.query_calls
    )


# ---------------------------------------------------------------------------
# 7. V15 query_by_owner_id 使用 Profile 的 metadata owner JSON path
# ---------------------------------------------------------------------------


def test_v15_query_by_owner_id_uses_metadata_json_path() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    _seed_rows(repository, client, chunks)
    target_expert_id = chunks[0].expert_id

    rows = repository.query_by_owner_id(target_expert_id)

    scan_call = client.query_iterator_calls[0]
    assert scan_call["filter"] == f'metadata["expert_id"] == {target_expert_id}'
    # 默认只返回 vector_doc_id（通用 owner 列表接口的唯一消费字段）
    assert scan_call["output_fields"] == ["vector_doc_id"]
    # 不请求 embedding / chunk_text / 整包 metadata
    assert "embedding" not in scan_call["output_fields"]
    assert "chunk_text" not in scan_call["output_fields"]
    assert "metadata" not in scan_call["output_fields"]
    # 返回行始终包含 vector_doc_id（公共 API 语义）
    assert rows
    assert all("vector_doc_id" in row for row in rows)


def test_v15_query_by_owner_id_rejects_top_level_hash_fields() -> None:
    repository = make_v15_repository()
    with pytest.raises(ValueError, match="一级字段"):
        repository.query_by_owner_id(
            1,
            output_fields=["vector_doc_id", "content_hash"],
        )
    with pytest.raises(ValueError, match="一级字段"):
        repository.query_by_owner_id(
            1,
            output_fields=["expert_id"],
        )


def test_v15_query_by_expert_id_warns_and_delegates() -> None:
    repository = make_v15_repository(FakeV15MilvusClient(exists=True))

    with pytest.warns(DeprecationWarning, match="query_by_expert_id 已弃用"):
        rows = repository.query_by_expert_id(1)

    assert rows == []


# ---------------------------------------------------------------------------
# 8/9. V15 list_all_vector_states：物理读取 metadata，提取 hash
# ---------------------------------------------------------------------------


def test_v15_list_all_vector_states_extracts_hashes_from_metadata() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    _seed_rows(repository, client, chunks)

    rows = repository.list_all_vector_states()

    scan_call = client.query_iterator_calls[0]
    # 物理读取 vector_doc_id + metadata（不再请求一级 hash scalar）
    assert scan_call["output_fields"] == ["vector_doc_id", "metadata"]
    assert scan_call["filter"] == 'vector_doc_id != ""'
    # 公共返回语义保持：vector_doc_id / content_hash / metadata_hash
    assert set(rows[0]) == set(V15_LIGHT_STATE_FIELDS)
    assert {row["vector_doc_id"] for row in rows} == {
        chunk.vector_doc_id for chunk in chunks
    }
    by_id = {row["vector_doc_id"]: row for row in rows}
    for chunk in chunks:
        assert by_id[chunk.vector_doc_id]["content_hash"] == chunk.content_hash
        assert by_id[chunk.vector_doc_id]["metadata_hash"] == chunk.metadata_hash


def test_v15_list_all_vector_states_does_not_request_non_existent_scalars() -> None:
    """full scan 不引用任何不存在的 V15 scalar（filter 与 output_fields 均是）。"""
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    _seed_rows(repository, client, chunks)

    repository.list_all_vector_states()

    scan_call = client.query_iterator_calls[0]
    assert "expert_id" not in scan_call["filter"]
    assert "embedding_dim" not in scan_call["filter"]
    assert "content_hash" not in scan_call["output_fields"]
    assert "metadata_hash" not in scan_call["output_fields"]
    assert "expert_id" not in scan_call["output_fields"]


def test_v15_list_all_vector_states_missing_content_hash_fails_closed() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    _seed_rows(repository, client, chunks)
    # 破坏一个行的 metadata：删掉 content_hash
    target = chunks[0].vector_doc_id
    bad_metadata = dict(client.rows[target]["metadata"])
    del bad_metadata["content_hash"]
    client.rows[target]["metadata"] = bad_metadata

    with pytest.raises(RuntimeError, match="content_hash"):
        repository.list_all_vector_states()


def test_v15_list_all_vector_states_missing_metadata_hash_fails_closed() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    _seed_rows(repository, client, chunks)
    target = chunks[0].vector_doc_id
    bad_metadata = dict(client.rows[target]["metadata"])
    del bad_metadata["metadata_hash"]
    client.rows[target]["metadata"] = bad_metadata

    with pytest.raises(RuntimeError, match="metadata_hash"):
        repository.list_all_vector_states()


def test_v15_list_all_vector_states_wrong_hash_type_fails_closed() -> None:
    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    _seed_rows(repository, client, chunks)
    target = chunks[0].vector_doc_id
    bad_metadata = dict(client.rows[target]["metadata"])
    bad_metadata["content_hash"] = 12345  # 类型错误
    client.rows[target]["metadata"] = bad_metadata

    with pytest.raises(RuntimeError, match="content_hash"):
        repository.list_all_vector_states()


def test_v15_list_all_vector_states_rejects_top_level_hash_fields() -> None:
    repository = make_v15_repository()
    with pytest.raises(ValueError, match="一级字段"):
        repository.list_all_vector_states(
            output_fields=["vector_doc_id", "content_hash"],
        )
    with pytest.raises(ValueError, match="一级字段"):
        repository.list_all_vector_states(output_fields=["metadata_hash"])


def test_v15_list_all_vector_states_rejects_duplicate_ids() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.model_space_batches = [
        [
            {
                "vector_doc_id": "expert:1:profile_summary:1:0",
                "metadata": {"content_hash": "a" * 64, "metadata_hash": "b" * 64},
            },
            {
                "vector_doc_id": "expert:1:profile_summary:1:0",
                "metadata": {"content_hash": "c" * 64, "metadata_hash": "d" * 64},
            },
        ]
    ]

    with pytest.raises(RuntimeError, match="重复 vector_doc_id"):
        repository.list_all_vector_states()


def test_v15_list_all_vector_activity_states_partitions_by_is_active() -> None:
    """活动状态扫描：返回行含严格布尔 is_active，active/inactive 可分区。"""

    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.model_space_batches = [
        [
            {
                "vector_doc_id": "expert:1:profile_summary:a:0",
                "metadata": {
                    "content_hash": "a" * 64,
                    "metadata_hash": "b" * 64,
                    "is_active": True,
                },
            },
            {
                "vector_doc_id": "expert:1:profile_summary:b:0",
                "metadata": {
                    "content_hash": "c" * 64,
                    "metadata_hash": "d" * 64,
                    "is_active": False,
                },
            },
        ]
    ]

    rows = repository.list_all_vector_activity_states()

    scan_call = client.query_iterator_calls[0]
    assert scan_call["output_fields"] == ["vector_doc_id", "metadata"]
    assert scan_call["filter"] == 'vector_doc_id != ""'
    by_id = {row["vector_doc_id"]: row for row in rows}
    assert by_id["expert:1:profile_summary:a:0"]["is_active"] is True
    assert by_id["expert:1:profile_summary:b:0"]["is_active"] is False
    active_ids = {r["vector_doc_id"] for r in rows if r["is_active"]}
    inactive_ids = {r["vector_doc_id"] for r in rows if not r["is_active"]}
    assert active_ids == {"expert:1:profile_summary:a:0"}
    assert inactive_ids == {"expert:1:profile_summary:b:0"}


def test_v15_list_all_vector_states_still_returns_inactive_entities() -> None:
    """全量语义未被破坏：list_all_vector_states 仍返回全部实体（含 inactive），
    且公共返回语义保持轻量三字段（不含 is_active）。"""

    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.model_space_batches = [
        [
            {
                "vector_doc_id": "expert:1:profile_summary:a:0",
                "metadata": {
                    "content_hash": "a" * 64,
                    "metadata_hash": "b" * 64,
                    "is_active": True,
                },
            },
            {
                "vector_doc_id": "expert:1:profile_summary:b:0",
                "metadata": {
                    "content_hash": "c" * 64,
                    "metadata_hash": "d" * 64,
                    "is_active": False,
                },
            },
        ]
    ]

    rows = repository.list_all_vector_states()

    # 全量语义：inactive 实体同样返回，不做任何过滤
    assert {row["vector_doc_id"] for row in rows} == {
        "expert:1:profile_summary:a:0",
        "expert:1:profile_summary:b:0",
    }
    # 公共返回语义保持：仅轻量三字段，无 is_active 泄漏
    for row in rows:
        assert set(row) == set(V15_LIGHT_STATE_FIELDS)


def test_v15_list_all_vector_activity_states_missing_is_active_fails_closed() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.model_space_batches = [
        [
            {
                "vector_doc_id": "expert:1:profile_summary:a:0",
                "metadata": {"content_hash": "a" * 64, "metadata_hash": "b" * 64},
            }
        ]
    ]

    with pytest.raises(RuntimeError, match="is_active"):
        repository.list_all_vector_activity_states()


def test_v15_list_all_vector_activity_states_non_bool_is_active_fails_closed() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.model_space_batches = [
        [
            {
                "vector_doc_id": "expert:1:profile_summary:a:0",
                "metadata": {
                    "content_hash": "a" * 64,
                    "metadata_hash": "b" * 64,
                    "is_active": 1,  # 非 bool（JSON int）
                },
            }
        ]
    ]

    with pytest.raises(RuntimeError, match="is_active"):
        repository.list_all_vector_activity_states()


def test_v15_list_all_vector_activity_states_rejects_duplicate_ids() -> None:
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    client.model_space_batches = [
        [
            {
                "vector_doc_id": "expert:1:profile_summary:dup:0",
                "metadata": {
                    "content_hash": "a" * 64,
                    "metadata_hash": "b" * 64,
                    "is_active": True,
                },
            },
            {
                "vector_doc_id": "expert:1:profile_summary:dup:0",
                "metadata": {
                    "content_hash": "c" * 64,
                    "metadata_hash": "d" * 64,
                    "is_active": False,
                },
            },
        ]
    ]

    with pytest.raises(RuntimeError, match="重复 vector_doc_id"):
        repository.list_all_vector_activity_states()


# ---------------------------------------------------------------------------
# 10. V15 destructive compatibility 使用 4 字段 V15 schema
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


def test_v15_destructive_guard_rejects_incompatible_schema() -> None:
    """V15 guard 必须用 V15 4 字段预期判断，不能拿不兼容 schema 放行。"""

    collection_name = "expert_candidates_v15_contract_dev"
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(
        client,
        collection_name=collection_name,
        allowed=(collection_name,),
    )
    client.description = incompatible_description()

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
# 12. V14/V15 mapper/spec/index 不串线
# ---------------------------------------------------------------------------


def test_v15_profile_symbols_are_stable() -> None:
    """V15 Repository 的持久化 profile 符号稳定，full-scan filter 不引用
    不存在的 V15 scalar。"""

    v15_repo = make_v15_repository()

    assert v15_repo.profile.schema_field_specs is v15_milvus_field_specs
    assert v15_repo.profile.row_mapper_cls is MilvusV15RowMapper
    assert v15_repo.profile.index_specs_fn is index_specs_v15
    assert v15_repo.profile.full_scan_filter == 'vector_doc_id != ""'
    # V15 full-scan filter 不引用任何不存在的 V15 scalar。
    assert "metadata" not in v15_repo.profile.full_scan_filter
    assert "metadata[" not in v15_repo.profile.full_scan_filter
    assert "expert_id" not in v15_repo.profile.full_scan_filter
    assert "embedding_dim" not in v15_repo.profile.full_scan_filter
    assert v15_repo.profile.model_space_fields == ("metadata",)


def test_v15_schema_field_set_is_4_fields() -> None:
    v15_fields = {spec.name for spec in v15_milvus_field_specs(1024)}
    assert v15_fields == set(V15_FIELD_NAMES)
    assert len(v15_fields) == 4
    # 历史 V14 一级字段（metadata_ext / source_ext / versions_ext / chunk_type /
    # expert_id）不应出现在 V15 精简 Schema 中。
    assert v15_fields.isdisjoint(
        {"metadata_ext", "source_ext", "versions_ext", "chunk_type", "expert_id"}
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
# V15 不暴露未实现的 search 能力
# ---------------------------------------------------------------------------


def test_v15_repository_does_not_expose_unimplemented_search() -> None:
    repository = make_v15_repository()
    assert not hasattr(repository, "search")
    assert not hasattr(repository, "supports_search")


# ---------------------------------------------------------------------------
# 模型空间校验缓存：V15 Repository 复用 base 缓存逻辑（同 model space 只扫描一次）
# ---------------------------------------------------------------------------


def test_v15_model_space_cache_skips_scan_on_second_upsert() -> None:
    """V15 Repository 与 V14 共享同一缓存实现，第二次 upsert 跳过全表扫描。"""

    chunks = embedded_contract_chunks()
    client = FakeV15MilvusClient(exists=True)
    repository = make_v15_repository(client)
    client.description = v15_description(repository)
    _seed_rows(repository, client, chunks)

    repository.upsert_batch(chunks)
    scans_before = sum(
        1 for c in client.query_calls
        if c.get("filter") == 'vector_doc_id != ""'
    )
    assert repository._validated_model_space is not None
    assert repository._validated_model_space["embedding_model_alias"] == (
        "alibaba-bailian:text-embedding-v4"
    )

    # 第二次：同 V15 Repository、同 model space，应命中缓存跳过采样查询。
    repository.upsert_batch(chunks)
    scans_after = sum(
        1 for c in client.query_calls
        if c.get("filter") == 'vector_doc_id != ""'
    )
    assert scans_before == 1
    assert scans_after == 1, "V15 第二次 upsert 不应再触发模型空间采样查询"
    assert len(client.upsert_calls) == 2


def test_v15_model_space_cache_isolated_per_instance() -> None:
    """不同 V15 Repository 实例的 model-space 缓存各自独立，互不串跳。

    用两个不同 model space 的 V15 实例验证实例级缓存隔离。
    """

    chunks = embedded_contract_chunks()
    other_chunks = [
        c.model_copy(update={"embedding_model_alias": "other-model"})
        for c in chunks
    ]

    client_a = FakeV15MilvusClient(exists=True)
    repo_a = make_v15_repository(client_a)
    client_a.description = v15_description(repo_a)

    client_b = FakeV15MilvusClient(exists=True)
    repo_b = make_v15_repository(client_b)
    client_b.description = v15_description(repo_b)

    repo_a.upsert_batch(chunks)
    repo_b.upsert_batch(other_chunks)

    # 两个实例各自独立缓存了不同的 model space。
    assert repo_a._validated_model_space is not None
    assert repo_b._validated_model_space is not None
    assert repo_a._validated_model_space["embedding_model_alias"] == (
        "alibaba-bailian:text-embedding-v4"
    )
    assert repo_b._validated_model_space["embedding_model_alias"] == "other-model"
    assert repo_a._validated_model_space != repo_b._validated_model_space
