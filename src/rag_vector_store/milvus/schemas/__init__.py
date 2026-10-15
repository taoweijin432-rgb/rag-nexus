"""Milvus Collection Schema 的定义与构建入口。

V14 25 字段 Schema 已退役；V15 精简 4 字段仍是默认生产形态，Expert V16
12 字段形态仅在构造配置显式选择后启用。通用字段定义类型
``MilvusFieldSpec`` 与构建器由 ``specs`` / ``builder`` 提供。
"""

from .contract_v15 import (
    V15_CHUNK_TEXT_MAX_BYTES,
    V15_CONTRACT_VECTOR_DIMENSION,
    V15_VECTOR_DOC_ID_MAX_BYTES,
    build_v15_schema,
    v15_milvus_field_specs,
)
from .contract_v16 import (
    V16_BM25_FUNCTION_NAME,
    V16_CHUNK_TEXT_MAX_BYTES,
    V16_VECTOR_DOC_ID_MAX_BYTES,
    build_v16_schema,
    v16_milvus_field_specs,
)
from .specs import MilvusFieldSpec

__all__ = [
    "V15_CHUNK_TEXT_MAX_BYTES",
    "V15_CONTRACT_VECTOR_DIMENSION",
    "V15_VECTOR_DOC_ID_MAX_BYTES",
    "V16_BM25_FUNCTION_NAME",
    "V16_CHUNK_TEXT_MAX_BYTES",
    "V16_VECTOR_DOC_ID_MAX_BYTES",
    "MilvusFieldSpec",
    "build_v15_schema",
    "build_v16_schema",
    "v15_milvus_field_specs",
    "v16_milvus_field_specs",
]
