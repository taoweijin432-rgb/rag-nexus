"""Milvus Schema、Index 和兼容性边界。"""

from .compatibility import (
    CollectionCompatibilityResult,
    compare_contract_indexes,
    compare_model_space,
    compare_schema_specs,
)
from .contract import (
    CONTRACT_SCALAR_INDEX_FIELDS,
    CONTRACT_VECTOR_DIMENSION,
    CONTRACT_VECTOR_DOC_ID_MAX_BYTES,
    MilvusExpertChunkEntity,
    MilvusFieldSpec,
    MilvusRowMapper,
    contract_milvus_field_specs,
)
from .indexes import IndexSpec, build_index_params, index_specs, vector_index_spec
from .repository import ContractMilvusVectorRepository

__all__ = [
    "CONTRACT_SCALAR_INDEX_FIELDS",
    "CONTRACT_VECTOR_DIMENSION",
    "CONTRACT_VECTOR_DOC_ID_MAX_BYTES",
    "CollectionCompatibilityResult",
    "ContractMilvusVectorRepository",
    "IndexSpec",
    "MilvusExpertChunkEntity",
    "MilvusFieldSpec",
    "MilvusRowMapper",
    "build_index_params",
    "compare_contract_indexes",
    "compare_model_space",
    "compare_schema_specs",
    "contract_milvus_field_specs",
    "index_specs",
    "vector_index_spec",
]
