"""向量化领域无关内核。

本包只依赖稳定的协议和基础模型，不依赖 ``expert_contracts`` 或任何具体业务
领域。专家链路通过 ``core.expert`` 提供兼容适配；未来其它领域可以提供自己的
Contract Adapter，而无需改动 Embedding/入库编排。
"""

from .adapters import VectorConsumerContractAdapter, VectorContractAdapter
from .embedding import (
    DomainEmbeddingResult,
    DomainEmbeddingWorker,
    DomainSkippedVector,
)
from .embedding_validation import validate_embedding_vector
from .flat_metadata import merge_flat_metadata
from .guards import require_adapter_input
from .hash_state import HASH_FIELDS, project_hash_fields, read_hash_field
from .ingestion import (
    DomainBatchIngestionResult,
    DomainBatchIngestionService,
    DomainIngestionError,
    DomainIngestionResult,
    DomainIngestionService,
)
from .lifecycle import (
    DomainDeletionResult,
    DomainLifecycleItem,
    DomainLifecycleResult,
    DomainVectorLifecycleService,
)
from .storage import VectorRepositoryProfile, VectorRowMapper

__all__ = [
    "HASH_FIELDS",
    "DomainBatchIngestionResult",
    "DomainBatchIngestionService",
    "DomainDeletionResult",
    "DomainEmbeddingResult",
    "DomainEmbeddingWorker",
    "DomainIngestionError",
    "DomainIngestionResult",
    "DomainIngestionService",
    "DomainLifecycleItem",
    "DomainLifecycleResult",
    "DomainSkippedVector",
    "DomainVectorLifecycleService",
    "VectorConsumerContractAdapter",
    "VectorContractAdapter",
    "VectorRepositoryProfile",
    "VectorRowMapper",
    "merge_flat_metadata",
    "project_hash_fields",
    "read_hash_field",
    "require_adapter_input",
    "validate_embedding_vector",
]
