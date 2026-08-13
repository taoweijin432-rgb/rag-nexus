"""正式 Contract 校验、Embedding 与 ingestion 边界。"""

from .contract_ingestion import (
    ContractBatchIngestionResult,
    ContractBatchIngestionService,
    ContractIngestionError,
    ContractIngestionResult,
    ContractIngestionService,
    ContractRepository,
)
from .contract_pipeline import (
    EXPECTED_EMBEDDING_DIMENSION,
    ContractEmbeddingResult,
    ContractEmbeddingWorker,
    ContractValidationError,
    ContractValidator,
    EmbeddingEligibilityError,
    EmbeddingEligibilityValidator,
    SkippedContractChunk,
    compare_contract_output_schema,
    expected_contract_output_fields,
)

__all__ = [
    "EXPECTED_EMBEDDING_DIMENSION",
    "ContractBatchIngestionResult",
    "ContractBatchIngestionService",
    "ContractEmbeddingResult",
    "ContractEmbeddingWorker",
    "ContractIngestionError",
    "ContractIngestionResult",
    "ContractIngestionService",
    "ContractRepository",
    "ContractValidationError",
    "ContractValidator",
    "EmbeddingEligibilityError",
    "EmbeddingEligibilityValidator",
    "SkippedContractChunk",
    "compare_contract_output_schema",
    "expected_contract_output_fields",
]
