"""正式 Expert Chunk Contract v2 的向量化与 Milvus 公共 API。"""

from expert_contracts import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
    ExpertChunkMetadata,
    ExpertChunkSource,
    ExpertChunkVersions,
)

from .chunking import (
    EXPECTED_EMBEDDING_DIMENSION,
    ContractEmbeddingResult,
    ContractEmbeddingWorker,
    ContractIngestionError,
    ContractIngestionResult,
    ContractIngestionService,
    ContractRepository,
    ContractValidationError,
    ContractValidator,
    EmbeddingEligibilityError,
    EmbeddingEligibilityValidator,
    SkippedContractChunk,
    compare_contract_output_schema,
    expected_contract_output_fields,
)
from .config import MilvusSettings
from .embedding import EmbeddingModel
from .embedding_adapters import (
    CompanyCloudEmbeddingError,
    CompanyCloudEmbeddingModel,
    CompanyCloudEmbeddingSettings,
)
from .milvus.contract import (
    CONTRACT_SCALAR_INDEX_FIELDS,
    CONTRACT_VECTOR_DIMENSION,
    CONTRACT_VECTOR_DOC_ID_MAX_BYTES,
    MilvusExpertChunkEntity,
    MilvusFieldSpec,
    MilvusRowMapper,
    contract_milvus_field_specs,
)
from .milvus.repository import ContractMilvusVectorRepository
from .model_space import (
    FORMAL_EMBEDDING_DIMENSION,
    FORMAL_EMBEDDING_MODEL,
    FORMAL_EMBEDDING_MODEL_ALIAS,
    FORMAL_EMBEDDING_MODEL_SPACE_ID,
    FORMAL_EMBEDDING_MODEL_VERSION,
    FORMAL_EMBEDDING_PROVIDER,
    build_model_space_id,
)
from .models import MutationResult, SearchHit

__all__ = [
    "CONTRACT_SCALAR_INDEX_FIELDS",
    "CONTRACT_VECTOR_DIMENSION",
    "CONTRACT_VECTOR_DOC_ID_MAX_BYTES",
    "EXPECTED_EMBEDDING_DIMENSION",
    "EXPERT_CHUNK_CONTRACT_VERSION",
    "FORMAL_EMBEDDING_DIMENSION",
    "FORMAL_EMBEDDING_MODEL",
    "FORMAL_EMBEDDING_MODEL_ALIAS",
    "FORMAL_EMBEDDING_MODEL_SPACE_ID",
    "FORMAL_EMBEDDING_MODEL_VERSION",
    "FORMAL_EMBEDDING_PROVIDER",
    "CompanyCloudEmbeddingError",
    "CompanyCloudEmbeddingModel",
    "CompanyCloudEmbeddingSettings",
    "ContractEmbeddingResult",
    "ContractEmbeddingWorker",
    "ContractIngestionError",
    "ContractIngestionResult",
    "ContractIngestionService",
    "ContractMilvusVectorRepository",
    "ContractPreparedChunk",
    "ContractRepository",
    "ContractValidationError",
    "ContractValidator",
    "EmbeddedContractPreparedChunk",
    "EmbeddingEligibilityError",
    "EmbeddingEligibilityValidator",
    "EmbeddingModel",
    "ExpertChunkEmbeddingInput",
    "ExpertChunkMetadata",
    "ExpertChunkSource",
    "ExpertChunkVersions",
    "MilvusExpertChunkEntity",
    "MilvusFieldSpec",
    "MilvusRowMapper",
    "MilvusSettings",
    "MutationResult",
    "SearchHit",
    "SkippedContractChunk",
    "build_model_space_id",
    "compare_contract_output_schema",
    "contract_milvus_field_specs",
    "expected_contract_output_fields",
]
