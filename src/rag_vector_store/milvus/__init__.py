"""Milvus Schema、Index 和兼容性边界。"""

from .compatibility import (
    CollectionCompatibilityResult,
    compare_contract_indexes,
    compare_model_space,
    compare_schema_specs,
)
from .contract_v15 import (
    V15_CONTRACT_VECTOR_DIMENSION,
    V15_VECTOR_DOC_ID_MAX_BYTES,
    ContractV15RowMapper,
    MilvusV15Entity,
    MilvusV15RowMapper,
    ResumeMilvusV15RowMapper,
    v15_milvus_field_specs,
)
from .indexes import (
    IndexSpec,
    build_index_params,
    index_specs_v15,
    vector_index_spec,
)
from .repository import ContractMilvusRepositoryBase
from .repository_v15 import (
    EXPERT_V15_PROFILE,
    RESUME_V15_PROFILE,
    ContractMilvusV15VectorRepository,
)

__all__ = [
    "EXPERT_V15_PROFILE",
    "RESUME_V15_PROFILE",
    "V15_CONTRACT_VECTOR_DIMENSION",
    "V15_VECTOR_DOC_ID_MAX_BYTES",
    "CollectionCompatibilityResult",
    "ContractMilvusRepositoryBase",
    "ContractMilvusV15VectorRepository",
    "ContractV15RowMapper",
    "IndexSpec",
    "MilvusV15Entity",
    "MilvusV15RowMapper",
    "ResumeMilvusV15RowMapper",
    "build_index_params",
    "compare_contract_indexes",
    "compare_model_space",
    "compare_schema_specs",
    "index_specs_v15",
    "v15_milvus_field_specs",
    "vector_index_spec",
]
