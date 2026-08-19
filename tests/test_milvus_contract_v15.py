"""V15 精简 Milvus Schema / Entity / RowMapper 单元测试。

覆盖：
    - V15 字段数量 = 4、精确字段集合、PK / auto_id / dynamic / dim；
    - V15 index 只有 embedding 向量索引（无任何 scalar index）；
    - to_entity 只产出 4 个物理字段，metadata 为单层 flat JSON；
    - expert_id / content_hash / metadata_hash 已进入 metadata，不再是一级字段；
    - 一级字段不重复出现在 metadata；namespace 键被禁止；key 冲突 fail-closed；
    - null / [] / Decimal / datetime 序列化语义；source / versions 平铺；
    - from_row round-trip：embedding、hash、update_metadata 所需字段完整恢复；
    - v14 现有映射（MilvusRowMapper）不受影响。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pymilvus import DataType

from expert_contracts import adapt_expert_chunk_mapping
from rag_vector_store import ContractEmbeddingWorker, EmbeddedContractPreparedChunk
from rag_vector_store.config import MilvusSettings
from rag_vector_store.milvus import (
    CONTRACT_SCALAR_INDEX_FIELDS,
    MilvusRowMapper,
    MilvusV15Entity,
    MilvusV15RowMapper,
    index_specs_v15,
    v15_milvus_field_specs,
)
from rag_vector_store.milvus.contract_v15 import (
    _V15_NAMESPACE_KEYS,
    _V15_PHYSICAL_KEYS,
    _merge_flat_metadata,
)
from rag_vector_store.milvus.schemas.contract_v15 import (
    V15_CHUNK_TEXT_MAX_BYTES,
    V15_SCALAR_INDEX_FIELDS,
    V15_VECTOR_DOC_ID_MAX_BYTES,
    build_v15_schema,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)

# V15 纯 4 字段：expert_id / content_hash / metadata_hash 已进入 flat metadata。
V15_FIELD_NAMES = (
    "vector_doc_id",
    "chunk_text",
    "embedding",
    "metadata",
)
# 从一级字段迁移进 metadata 的字段。
V15_MOVED_TO_METADATA = ("expert_id", "content_hash", "metadata_hash")


class FakeEmbeddingModel:
    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


def v15_embedded_chunks() -> list[EmbeddedContractPreparedChunk]:
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


def v15_settings() -> MilvusSettings:
    return MilvusSettings(
        collection_name="expert_candidates_v15_contract_dev",
        index_type="HNSW",
        metric_type="COSINE",
        index_params={"M": 16, "efConstruction": 200},
    )


# ---------- Schema：字段数量 / 精确集合 / PK / auto_id / dynamic / dim ----------


def test_v15_field_count_is_four() -> None:
    specs = v15_milvus_field_specs(1024)
    assert len(specs) == 4


def test_v15_field_set_exact() -> None:
    specs = v15_milvus_field_specs(1024)
    assert tuple(spec.name for spec in specs) == V15_FIELD_NAMES
    types = {spec.name: spec.datatype for spec in specs}
    assert types == {
        "vector_doc_id": "VARCHAR",
        "chunk_text": "VARCHAR",
        "embedding": "FLOAT_VECTOR",
        "metadata": "JSON",
    }
    # 不再存在 expert_id / content_hash / metadata_hash 一级字段
    for moved in V15_MOVED_TO_METADATA:
        assert moved not in types


def test_v15_schema_pk_auto_id_dynamic_dim() -> None:
    schema = build_v15_schema(v15_settings(), 1024)
    assert schema.auto_id is False
    assert schema.enable_dynamic_field is False
    fields = {field.name: field for field in schema.fields}
    assert fields["vector_doc_id"].is_primary is True
    assert fields["vector_doc_id"].max_length == V15_VECTOR_DOC_ID_MAX_BYTES
    assert fields["embedding"].dtype == DataType.FLOAT_VECTOR
    assert fields["embedding"].dim == 1024
    assert fields["metadata"].dtype == DataType.JSON
    assert len(schema.fields) == 4


def test_v15_schema_rejects_missing_dimension() -> None:
    with pytest.raises(ValueError, match="vector dimension is not configured"):
        build_v15_schema(v15_settings(), None)


# ---------- Index：只有 embedding 向量索引 ----------


def test_v15_index_specs_only_vector_index() -> None:
    specs = index_specs_v15(v15_settings())
    assert len(specs) == 1
    assert [spec.field_name for spec in specs] == ["embedding"]
    assert specs[0].index_type == "HNSW"
    assert specs[0].metric_type == "COSINE"
    assert specs[0].index_name == "embedding_vector"
    # 不再有 expert_id INVERTED / 任何 scalar index
    assert V15_SCALAR_INDEX_FIELDS == ()


def test_v15_index_specs_do_not_include_any_scalar_field() -> None:
    v15_names = {spec.field_name for spec in index_specs_v15(v15_settings())}
    for field in CONTRACT_SCALAR_INDEX_FIELDS:
        assert field not in v15_names
    for moved in V15_MOVED_TO_METADATA:
        assert moved not in v15_names


# ---------- to_entity：4 个物理字段 + flat metadata ----------


def test_v15_to_entity_only_four_physical_fields() -> None:
    chunk = v15_embedded_chunks()[1]
    entity = MilvusV15RowMapper().to_entity(chunk)

    assert isinstance(entity, MilvusV15Entity)
    assert tuple(entity.model_dump().keys()) == V15_FIELD_NAMES
    assert entity.vector_doc_id == chunk.vector_doc_id
    assert entity.chunk_text == chunk.chunk_text
    assert entity.embedding == chunk.embedding
    # expert_id / content_hash / metadata_hash 已进入 metadata，不再是实体属性
    assert not hasattr(entity, "expert_id")
    assert not hasattr(entity, "content_hash")
    assert not hasattr(entity, "metadata_hash")


def test_v15_metadata_is_flat_json() -> None:
    chunk = v15_embedded_chunks()[1]
    entity = MilvusV15RowMapper().to_entity(chunk)

    flat = entity.metadata
    # 单层 flat：任何 value 都不是嵌套 dict（list / None / 标量允许）
    for key, value in flat.items():
        assert not isinstance(value, Mapping), f"metadata 必须单层 flat: {key!r} 是 dict"


def test_v15_physical_fields_not_duplicated_in_metadata() -> None:
    chunk = v15_embedded_chunks()[1]
    entity = MilvusV15RowMapper().to_entity(chunk)
    assert not _V15_PHYSICAL_KEYS.intersection(entity.metadata.keys())


def test_v15_metadata_no_namespace_keys() -> None:
    chunk = v15_embedded_chunks()[1]
    entity = MilvusV15RowMapper().to_entity(chunk)
    assert not _V15_NAMESPACE_KEYS.intersection(entity.metadata.keys())


def test_v15_moved_fields_in_metadata() -> None:
    """expert_id / content_hash / metadata_hash 进入 metadata，值原样透传。"""
    chunk = v15_embedded_chunks()[1]
    flat = MilvusV15RowMapper().to_entity(chunk).metadata
    assert flat["expert_id"] == chunk.expert_id
    assert flat["content_hash"] == chunk.content_hash
    assert flat["metadata_hash"] == chunk.metadata_hash


def test_v15_metadata_required_top_level_fields_present() -> None:
    chunk = v15_embedded_chunks()[1]  # employment
    flat = MilvusV15RowMapper().to_entity(chunk).metadata
    # 顶层 Contract 的非物理字段平铺进 flat metadata（不含 metadata_hash 自身计算）。
    for key in (
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
        "weight_tier",
        "is_active",
        "updated_at",
        "embedded_at",
        "embedding_dim",
        "embedding_model_alias",
        "embedding_model_version",
    ):
        assert key in flat, f"flat metadata 缺少 {key!r}"
    # per-type：employment 不输出 profile 专属字段，禁止占位。
    for profile_only in (
        "eligible_for_recommendation",
        "job_status_norm",
        "abroad_flag_norm",
        "work_years",
        "function_codes",
        "school_names_raw",
        "major_names_raw",
    ):
        assert profile_only not in flat, (
            f"employment flat metadata 不应含 profile 专属字段 {profile_only!r}"
        )
    assert flat["embedding_dim"] == len(chunk.embedding)
    assert flat["embedding_model_alias"] == chunk.embedding_model_alias
    assert flat["embedding_model_version"] == chunk.embedding_model_version


def test_v15_metadata_source_versions_flattened() -> None:
    chunk = v15_embedded_chunks()[1]
    flat = MilvusV15RowMapper().to_entity(chunk).metadata
    for key in (
        "source_table",
        "source_field",
        "source_field_id",
        "source_version",
        "source_updated_at",
        "source_hash",
        "chunk_policy_version",
        "metadata_schema_version",
    ):
        assert key in flat, f"flat metadata 缺少 {key!r}"
    assert flat["source_table"] == chunk.source.source_table
    assert flat["source_field_id"] == chunk.source.source_field_id
    assert flat["source_hash"] == chunk.source.source_hash
    assert flat["chunk_policy_version"] == chunk.versions.chunk_policy_version
    assert flat["metadata_schema_version"] == chunk.versions.metadata_schema_version


def test_v15_metadata_business_fields_flattened() -> None:
    chunk = v15_embedded_chunks()[1]
    flat = MilvusV15RowMapper().to_entity(chunk).metadata
    assert flat["company_name_raw"] == chunk.metadata.company_name_raw
    assert flat["is_current"] == chunk.metadata.is_current
    assert flat["industry_codes"] == chunk.metadata.industry_codes


def test_v15_null_and_empty_array_preserved() -> None:
    chunk = v15_embedded_chunks()[2]  # project：大量 null / 空数组
    flat = MilvusV15RowMapper().to_entity(chunk).metadata
    # per-type：project 不输出 profile/employment 专属字段，禁止占位。
    for other_type_field in (
        "role_code",
        "role_level",
        "role_name_raw",
        "job_status_norm",
        "abroad_flag_norm",
        "work_years",
        "is_current",
        "function_codes",
        "school_names_raw",
        "major_names_raw",
        "eligible_for_recommendation",
    ):
        assert other_type_field not in flat, (
            f"project flat metadata 不应含其他类型字段 {other_type_field!r}"
        )
    # 公共字段与 project 专属字段的 null / 空数组语义仍保留。
    assert flat["industry_codes"] == chunk.metadata.industry_codes
    assert flat["topic_codes"] == chunk.metadata.topic_codes
    assert flat["company_name_raw"] is None or isinstance(flat["company_name_raw"], str)
    assert flat["project_name"] == chunk.metadata.project_name


def test_v15_decimal_work_years_serialized_as_string() -> None:
    chunk = v15_embedded_chunks()[0]  # profile：work_years=12.00
    flat = MilvusV15RowMapper().to_entity(chunk).metadata
    assert flat["work_years"] == "12.00"
    assert isinstance(flat["work_years"], str)


def test_v15_datetime_iso_strings_preserved() -> None:
    chunk = v15_embedded_chunks()[0]
    flat = MilvusV15RowMapper().to_entity(chunk).metadata
    # datetime 保持 ISO 8601 字符串语义，不做对象转换
    assert isinstance(flat["updated_at"], str)
    assert isinstance(flat["embedded_at"], str)
    assert flat["updated_at"] == chunk.updated_at
    assert flat["embedded_at"] == chunk.embedded_at


# ---------- flat 合并冲突保护：fail-closed ----------


def test_v15_merge_rejects_duplicate_key() -> None:
    with pytest.raises(ValueError, match="key 冲突"):
        _merge_flat_metadata(
            (
                ("来源A", {"chunk_type": "employment"}),
                ("来源B", {"chunk_type": "project"}),
            )
        )


def test_v15_merge_rejects_physical_key() -> None:
    with pytest.raises(ValueError, match="一级字段"):
        _merge_flat_metadata((("来源A", {"vector_doc_id": "expert:1:profile_summary:1:0"}),))


def test_v15_merge_rejects_namespace_key() -> None:
    with pytest.raises(ValueError, match="namespace 键"):
        _merge_flat_metadata((("来源A", {"source": {"source_table": "experts"}}),))


def test_v15_merge_no_conflict_in_real_contract() -> None:
    chunk = v15_embedded_chunks()[0]
    top_level = chunk.model_dump(mode="json")
    top_level_flat = {
        key: value
        for key, value in top_level.items()
        if key not in _V15_PHYSICAL_KEYS and key not in ("metadata", "source", "versions")
    }
    flat = _merge_flat_metadata(
        (
            ("contract 顶层", top_level_flat),
            ("业务 metadata", chunk.metadata.model_dump(mode="json")),
            ("source", chunk.source.model_dump(mode="json")),
            ("versions", chunk.versions.model_dump(mode="json")),
        )
    )
    assert flat  # 真实 Contract 无冲突，合并成功


# ---------- from_row：round-trip / update_metadata 依赖 ----------


def test_v15_from_row_round_trip_full() -> None:
    for chunk in v15_embedded_chunks():
        row = MilvusV15RowMapper().to_row(chunk)
        restored = MilvusV15RowMapper().from_row(row)
        assert restored == chunk


def test_v15_from_row_embedding_round_trip() -> None:
    chunk = v15_embedded_chunks()[1]
    restored = MilvusV15RowMapper().from_row(
        MilvusV15RowMapper().to_row(chunk)
    )
    assert restored.embedding == chunk.embedding
    assert restored.embedding_dim == chunk.embedding_dim
    assert restored.embedding_model_alias == chunk.embedding_model_alias
    assert restored.embedding_model_version == chunk.embedding_model_version
    assert restored.embedded_at == chunk.embedded_at


def test_v15_from_row_hashes_round_trip() -> None:
    chunk = v15_embedded_chunks()[1]
    restored = MilvusV15RowMapper().from_row(
        MilvusV15RowMapper().to_row(chunk)
    )
    assert restored.content_hash == chunk.content_hash
    assert restored.metadata_hash == chunk.metadata_hash
    assert restored.vector_doc_id == chunk.vector_doc_id


def test_v15_from_row_supports_update_metadata_fields() -> None:
    """update_metadata 依赖 from_row 还原的完整对象：旧向量 + 模型空间字段。"""
    chunk = v15_embedded_chunks()[1]
    restored = MilvusV15RowMapper().from_row(
        MilvusV15RowMapper().to_row(chunk)
    )
    # 安全门所需字段（_metadata_gate_error / EmbeddedContractPreparedChunk 重建）
    assert restored.embedding
    assert restored.embedding_dim == 1024
    assert restored.embedding_model_alias == "alibaba-bailian:text-embedding-v4"
    assert restored.embedding_model_version == "v1"
    assert restored.embedded_at
    # 重建新 Contract 所需字段
    assert restored.metadata is not None
    assert restored.source is not None
    assert restored.versions is not None
    assert restored.is_active == chunk.is_active
    assert restored.updated_at == chunk.updated_at


def test_v15_from_row_missing_metadata_raises() -> None:
    with pytest.raises(TypeError, match="metadata JSON"):
        MilvusV15RowMapper().from_row(
            {"vector_doc_id": "expert:1:profile_summary:1:0", "chunk_text": "x"}
        )


def test_v15_from_row_missing_flat_key_fails_closed() -> None:
    row = MilvusV15RowMapper().to_row(v15_embedded_chunks()[0])
    row["metadata"] = dict(row["metadata"])
    del row["metadata"]["source_table"]  # 必需字段缺失
    with pytest.raises(Exception) as excinfo:
        MilvusV15RowMapper().from_row(row)
    assert "source_table" in str(excinfo.value)


# ---------- 边界 / 类型 / v14 兼容 ----------


def test_v15_to_entity_rejects_non_embedded_chunk() -> None:
    with pytest.raises(TypeError, match="只接受 EmbeddedContractPreparedChunk"):
        MilvusV15RowMapper().to_entity(object())  # type: ignore[arg-type]


def test_v15_to_rows_rejects_duplicate_pk() -> None:
    chunks = v15_embedded_chunks()
    with pytest.raises(ValueError, match="重复 vector_doc_id"):
        MilvusV15RowMapper().to_rows([chunks[0], chunks[0]])


def test_v15_entity_rejects_non_finite_embedding() -> None:
    with pytest.raises(ValueError, match="有限浮点数"):
        MilvusV15Entity(
            vector_doc_id="expert:1:profile_summary:1:0",
            chunk_text="x",
            embedding=[float("nan"), 0.1],
            metadata={},
        )


def test_v15_entity_rejects_extra_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MilvusV15Entity(
            vector_doc_id="expert:1:profile_summary:1:0",
            chunk_text="x",
            embedding=[0.1],
            metadata={},
            expert_id=1,  # type: ignore[call-arg]
        )


def test_v15_storage_limits_reject_oversized_chunk_text() -> None:
    chunk = v15_embedded_chunks()[0]
    too_long = chunk.model_copy(
        update={"chunk_text": "长" * V15_CHUNK_TEXT_MAX_BYTES}
    )
    with pytest.raises(ValueError, match="超过 Milvus VARCHAR"):
        MilvusV15RowMapper().to_entity(too_long)


def test_v15_metadata_json_size_limit() -> None:
    chunk = v15_embedded_chunks()[0]
    # 手工构造超大 metadata 注入 Entity 校验（VARCHAR 检查不涉及 metadata）
    entity = MilvusV15RowMapper().to_entity(chunk)
    big_metadata = dict(entity.metadata)
    big_metadata["huge"] = "x" * 70000
    oversized = MilvusV15Entity(
        vector_doc_id=entity.vector_doc_id,
        chunk_text=entity.chunk_text,
        embedding=entity.embedding,
        metadata=big_metadata,
    )
    with pytest.raises(ValueError, match="超过 Milvus JSON 65536 bytes"):
        MilvusV15RowMapper._validate_storage_limits(oversized)


def test_v14_row_mapper_unaffected() -> None:
    """v14 映射行为不被 V15 改动破坏。"""
    from tests.test_contract_milvus_repository import embedded_contract_chunks

    chunk = embedded_contract_chunks()[0]
    entity = MilvusRowMapper().to_entity(chunk)
    assert entity.metadata_ext == chunk.metadata.model_dump(mode="json")
    assert entity.source_ext == chunk.source.model_dump(mode="json")
    assert entity.versions_ext == chunk.versions.model_dump(mode="json")
    assert tuple(entity.model_dump().keys())[:3] == (
        "vector_doc_id",
        "expert_id",
        "serial_number",
    )
