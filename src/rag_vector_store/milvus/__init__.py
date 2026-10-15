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
from .contract_v16 import (
    ContractV16RowMapper,
    MilvusV16Entity,
    MilvusV16RowMapper,
    map_flat_v15_row_to_v16,
    validate_v16_mirrors,
)
from .indexes import (
    IndexSpec,
    build_index_params,
    index_specs_v15,
    index_specs_v16,
    vector_index_spec,
)
from .repository import ContractMilvusRepositoryBase
from .repository_v15 import (
    EXPERT_V15_PROFILE,
    RESUME_V15_PROFILE,
    ContractMilvusV15VectorRepository,
)
from .repository_v16 import EXPERT_V16_PROFILE, ContractMilvusV16VectorRepository

__all__ = [
    "EXPERT_V15_PROFILE",
    "EXPERT_V16_PROFILE",
    "RESUME_V15_PROFILE",
    "V15_CONTRACT_VECTOR_DIMENSION",
    "V15_VECTOR_DOC_ID_MAX_BYTES",
    "CollectionCompatibilityResult",
    "ContractMilvusRepositoryBase",
    "ContractMilvusV15VectorRepository",
    "ContractMilvusV16VectorRepository",
    "ContractV15RowMapper",
    "ContractV16RowMapper",
    "IndexSpec",
    "MilvusV15Entity",
    "MilvusV15RowMapper",
    "MilvusV16Entity",
    "MilvusV16RowMapper",
    "ResumeMilvusV15RowMapper",
    "build_index_params",
    "compare_contract_indexes",
    "compare_model_space",
    "compare_schema_specs",
    "index_specs_v15",
    "index_specs_v16",
    "map_flat_v15_row_to_v16",
    "v15_milvus_field_specs",
    "validate_v16_mirrors",
    "vector_index_spec",
]
