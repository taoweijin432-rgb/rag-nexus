"""三种 Milvus Collection Schema 的定义与构建入口。"""

from .contract import (
    CONTRACT_SCALAR_INDEX_FIELDS,
    CONTRACT_VECTOR_DIMENSION,
    CONTRACT_VECTOR_DOC_ID_MAX_BYTES,
    build_contract_schema,
    contract_milvus_field_specs,
)
from .specs import MilvusFieldSpec

__all__ = [
    "CONTRACT_SCALAR_INDEX_FIELDS",
    "CONTRACT_VECTOR_DIMENSION",
    "CONTRACT_VECTOR_DOC_ID_MAX_BYTES",
    "MilvusFieldSpec",
    "build_contract_schema",
    "contract_milvus_field_specs",
]
