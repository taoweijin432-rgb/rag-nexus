"""V15 增量向量化持久化层与 Work Item 适配（Phase 0 / Phase 2）。

提供 Batch / Work Item 的 MySQL 状态持久化与状态机，以及 Work Item 到
正式 V15 Contract 的薄适配；不实现 Consumer、不调用 Embedding、不写 Milvus、
不扫描业务源表、不生成 chunk_text。
"""

from .contract_adapter import (
    DisableWorkItem,
    V15WorkItemContractAdapter,
    WorkItemContractAdaptation,
    WorkItemContractAdapterError,
)
from .models import (
    ALLOWED_SOURCE_TABLES,
    APPENDABLE_BATCH_STATUSES,
    BATCH_FINAL_STATUSES,
    V15BatchStatus,
    V15VectorizationBatch,
    V15VectorizationWorkItem,
    V15WorkItemCreateResult,
    V15WorkItemInput,
    V15WorkItemOperation,
    V15WorkItemStatus,
)
from .repository import (
    MySQLV15VectorizationRepository,
    V15BatchNotFoundError,
    V15PersistenceError,
    V15TransitionError,
    V15WorkItemNotFoundError,
    vector_doc_id_hash,
)

__all__ = [
    "ALLOWED_SOURCE_TABLES",
    "APPENDABLE_BATCH_STATUSES",
    "BATCH_FINAL_STATUSES",
    "DisableWorkItem",
    "MySQLV15VectorizationRepository",
    "V15BatchNotFoundError",
    "V15BatchStatus",
    "V15PersistenceError",
    "V15TransitionError",
    "V15VectorizationBatch",
    "V15VectorizationWorkItem",
    "V15WorkItemContractAdapter",
    "V15WorkItemCreateResult",
    "V15WorkItemInput",
    "V15WorkItemNotFoundError",
    "V15WorkItemOperation",
    "V15WorkItemStatus",
    "WorkItemContractAdaptation",
    "WorkItemContractAdapterError",
    "vector_doc_id_hash",
]
