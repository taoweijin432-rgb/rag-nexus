"""Milvus Collection Schema 的定义与构建入口。

V14 25 字段 Schema 已退役；当前唯一正式物理 Schema 为 V15 精简 4 字段形态
（``contract_v15``）。通用字段定义类型 ``MilvusFieldSpec`` 与构建器
``build_schema_from_specs`` 仍由 ``specs`` / ``builder`` 提供，供 V15 复用。
"""

from .contract_v15 import (
    V15_CHUNK_TEXT_MAX_BYTES,
    V15_CONTRACT_VECTOR_DIMENSION,
    V15_VECTOR_DOC_ID_MAX_BYTES,
    build_v15_schema,
    v15_milvus_field_specs,
)
from .specs import MilvusFieldSpec

__all__ = [
    "V15_CHUNK_TEXT_MAX_BYTES",
    "V15_CONTRACT_VECTOR_DIMENSION",
    "V15_VECTOR_DOC_ID_MAX_BYTES",
    "MilvusFieldSpec",
    "build_v15_schema",
    "v15_milvus_field_specs",
]
