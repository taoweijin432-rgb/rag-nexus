"""正式 Contract Collection Schema 定义。

这里是当前唯一正式业务 Collection 的字段基线：输入字段、模型输出、标量投影
和三组 JSON extension 共 40 个字段。函数只返回声明或构造 SDK Schema，不负责
连接、创建、迁移或修改已存在 Collection。
"""

from __future__ import annotations

from typing import Any

from ...config import MilvusSettings
from .builder import build_schema_from_specs
from .specs import MilvusFieldSpec

CONTRACT_VECTOR_DIMENSION = 1024
CONTRACT_VECTOR_DOC_ID_MAX_BYTES = 2048


def contract_milvus_field_specs(
    vector_dimension: int = CONTRACT_VECTOR_DIMENSION,
) -> tuple[MilvusFieldSpec, ...]:
    """返回无损承载正式 Chunk 契约输出所需的 40 个字段定义。

    维度由调用方传入并写入向量字段；其余字段必须保持稳定，Repository 的
    Compatibility Gate 会在任何写入前逐项核对。
    """

    return (
        MilvusFieldSpec(
            "vector_doc_id",
            "VARCHAR",
            primary_key=True,
            max_length=CONTRACT_VECTOR_DOC_ID_MAX_BYTES,
        ),
        MilvusFieldSpec("chunk_id", "VARCHAR", max_length=1024),
        MilvusFieldSpec("expert_id", "INT64"),
        MilvusFieldSpec("serial_number", "VARCHAR", nullable=True, max_length=256),
        MilvusFieldSpec("chunk_type", "VARCHAR", max_length=32),
        MilvusFieldSpec("chunk_subtype", "VARCHAR", nullable=True, max_length=32),
        MilvusFieldSpec("chunk_index", "INT64"),
        MilvusFieldSpec("chunk_total", "INT64"),
        MilvusFieldSpec("chunk_text", "VARCHAR", max_length=8192),
        MilvusFieldSpec("char_count", "INT64"),
        MilvusFieldSpec("content_hash", "VARCHAR", max_length=64),
        MilvusFieldSpec("embedding", "FLOAT_VECTOR", dim=vector_dimension),
        MilvusFieldSpec("embedding_dim", "INT64"),
        MilvusFieldSpec("embedding_model_alias", "VARCHAR", max_length=512),
        MilvusFieldSpec("embedding_model_version", "VARCHAR", max_length=512),
        MilvusFieldSpec("company_name_norm", "VARCHAR", nullable=True, max_length=2048),
        MilvusFieldSpec(
            "company_normalization_status",
            "VARCHAR",
            nullable=True,
            max_length=32,
        ),
        MilvusFieldSpec("role_code", "VARCHAR", nullable=True, max_length=512),
        MilvusFieldSpec("role_level", "VARCHAR", nullable=True, max_length=256),
        MilvusFieldSpec("start_month", "VARCHAR", nullable=True, max_length=7),
        MilvusFieldSpec("end_month", "VARCHAR", nullable=True, max_length=7),
        MilvusFieldSpec("is_current", "BOOL", nullable=True),
        MilvusFieldSpec("origin_type", "VARCHAR", max_length=32),
        MilvusFieldSpec("verification_status", "VARCHAR", max_length=32),
        MilvusFieldSpec("weight_tier", "VARCHAR", max_length=4),
        MilvusFieldSpec("is_active", "BOOL"),
        MilvusFieldSpec("source_table", "VARCHAR", max_length=512),
        MilvusFieldSpec("source_field_id", "VARCHAR", max_length=256),
        MilvusFieldSpec("source_item_key", "VARCHAR", max_length=512),
        MilvusFieldSpec("source_field", "VARCHAR", max_length=512),
        MilvusFieldSpec("source_hash", "VARCHAR", max_length=64),
        MilvusFieldSpec("contract_version", "VARCHAR", max_length=32),
        MilvusFieldSpec("chunk_policy_version", "VARCHAR", max_length=128),
        MilvusFieldSpec("metadata_schema_version", "VARCHAR", max_length=128),
        MilvusFieldSpec("data_version", "VARCHAR", max_length=256),
        MilvusFieldSpec("updated_at", "VARCHAR", max_length=64),
        MilvusFieldSpec("embedded_at", "VARCHAR", max_length=64),
        MilvusFieldSpec("metadata_ext", "JSON"),
        MilvusFieldSpec("source_ext", "JSON"),
        MilvusFieldSpec("versions_ext", "JSON"),
    )


CONTRACT_SCALAR_INDEX_FIELDS = (
    "expert_id",
    "chunk_type",
    "company_name_norm",
    "role_code",
    "role_level",
    "start_month",
    "end_month",
    "is_current",
    "verification_status",
    "weight_tier",
    "is_active",
)


def build_contract_schema(
    settings: MilvusSettings,
    vector_dimension: int | None,
) -> Any:
    """构造正式 Contract Collection Schema，不执行 Collection 操作。"""

    if vector_dimension is None:
        raise ValueError("vector dimension is not configured")
    return build_schema_from_specs(contract_milvus_field_specs(vector_dimension))
