"""向量化领域无关内核。

本包只依赖稳定的协议和基础模型，不依赖 ``expert_contracts`` 或任何具体业务
领域。领域 Adapter 位于 ``rag_vector_store.domains``；旧 ``core.expert`` 路径
仅保留一个发布周期的兼容转发。
"""

from .adapters import (
    VectorConsumerContractAdapter,
    VectorContractAdapter,
    VectorStorageAdapter,
)
from .embedding import (
    DomainEmbeddingError,
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
    "DomainEmbeddingError",
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
    "VectorStorageAdapter",
    "merge_flat_metadata",
    "project_hash_fields",
    "read_hash_field",
    "require_adapter_input",
    "validate_embedding_vector",
]


def __getattr__(name: str) -> object:
    """延迟兼容旧包级 Adapter 导入，避免普通 core import 加载领域 Contract。"""

    if name in {"ExpertContractAdapter", "ResumeContractAdapter"}:
        from .. import domains

        return getattr(domains, name)
    raise AttributeError(name)
