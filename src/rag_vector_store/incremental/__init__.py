"""V15 增量向量化持久化层、Work Item 适配与 Consumer 业务核心（Phase 0 / 2 / 3A）。

提供 Batch / Work Item 的 MySQL 状态持久化与状态机、Work Item 到正式 V15
Contract 的薄适配，以及无状态持久化的通用 Work Item 消费核心。Consumer 只返回
结构化 ``ExecutionResult``：不调用 Embedding 实现、不做物理删除、不扫描业务源表、
不生成 chunk_text，也不新增任何状态存储端口；状态流转由后续唯一状态表适配层负责。
"""

from .consumer import (
    SourceEventOrderView,
    V15WorkItemConsumer,
    VectorMutationPort,
    WorkItemAdapterPort,
)
from .consumer_models import (
    CONSUMER_ACTION_ADAPT_DISABLE,
    CONSUMER_ACTION_ADAPT_UPSERT,
    CONSUMER_ACTION_DISABLE_STALE_TARGETS,
    CONSUMER_ACTION_DISABLE_TARGETS,
    CONSUMER_ACTION_EMBEDDING_UPSERT,
    CONSUMER_ACTION_FETCH_EXISTING,
    CONSUMER_ACTION_METADATA_ONLY_UPDATE,
    CONSUMER_ACTION_SKIP_UNCHANGED,
    CONSUMER_ACTION_SYNTHESIZE_FROM_STORED,
    ConsumerExecutionStatus,
    ExecutionResult,
)
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
    "CONSUMER_ACTION_ADAPT_DISABLE",
    "CONSUMER_ACTION_ADAPT_UPSERT",
    "CONSUMER_ACTION_DISABLE_STALE_TARGETS",
    "CONSUMER_ACTION_DISABLE_TARGETS",
    "CONSUMER_ACTION_EMBEDDING_UPSERT",
    "CONSUMER_ACTION_FETCH_EXISTING",
    "CONSUMER_ACTION_METADATA_ONLY_UPDATE",
    "CONSUMER_ACTION_SKIP_UNCHANGED",
    "CONSUMER_ACTION_SYNTHESIZE_FROM_STORED",
    "ConsumerExecutionStatus",
    "DisableWorkItem",
    "ExecutionResult",
    "MySQLV15VectorizationRepository",
    "SourceEventOrderView",
    "V15BatchNotFoundError",
    "V15BatchStatus",
    "V15PersistenceError",
    "V15TransitionError",
    "V15VectorizationBatch",
    "V15VectorizationWorkItem",
    "V15WorkItemConsumer",
    "V15WorkItemContractAdapter",
    "V15WorkItemCreateResult",
    "V15WorkItemInput",
    "V15WorkItemNotFoundError",
    "V15WorkItemOperation",
    "V15WorkItemStatus",
    "VectorMutationPort",
    "WorkItemAdapterPort",
    "WorkItemContractAdaptation",
    "WorkItemContractAdapterError",
    "vector_doc_id_hash",
]
