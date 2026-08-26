"""Canal CDC 边界模型与客户端端口。

本包只定义上游可依赖的协议边界，不依赖 Canal SDK，也不执行消费或确认。
"""

from .delete_scope import (
    CdcDeleteSourcePort,
    prepare_cdc_delete_batch,
    prepare_cdc_delete_task,
)
from .ingest import (
    CdcBatchPersistencePort,
    CdcBatchReceiver,
    CdcBatchReceiveResult,
    CdcEventIdentityProvider,
    CdcIngestError,
    CdcPersistenceConflictError,
    CdcTaskPlan,
    CdcTaskReceipt,
    CdcTaskReceiptStatus,
)
from .parser import (
    CanalBatchDecoder,
    CanalClientAdapter,
    CanalConnectorPort,
    CanalDeleteEvent,
    CanalParseIssue,
    CanalParserError,
    CanalParseResult,
    parse_canal_batch,
)
from .protocol import (
    CANAL_DESTINATION,
    CANAL_SUBSCRIPTION,
    CanalBatch,
    CanalClientPort,
    CanalColumn,
    CanalEntry,
    CanalProtocolBaseline,
    CanalProtocolNotFrozenError,
    CanalRowData,
)
from .replay import (
    CanalReplaySafetyError,
    FakeCanalClient,
    ReplayDelete,
    ReplayResult,
    replay_batch,
    replay_next_batch,
)
from .task_processor import (
    CdcDeleteTaskProcessor,
    CdcDeleteTaskRepositoryPort,
    CdcHandoffOrchestratorPort,
)

__all__ = [
    "CANAL_DESTINATION",
    "CANAL_SUBSCRIPTION",
    "CanalBatch",
    "CanalBatchDecoder",
    "CanalClientAdapter",
    "CanalClientPort",
    "CanalColumn",
    "CanalConnectorPort",
    "CanalDeleteEvent",
    "CanalEntry",
    "CanalParseIssue",
    "CanalParseResult",
    "CanalParserError",
    "CanalProtocolBaseline",
    "CanalProtocolNotFrozenError",
    "CanalReplaySafetyError",
    "CanalRowData",
    "CdcBatchPersistencePort",
    "CdcBatchReceiveResult",
    "CdcBatchReceiver",
    "CdcDeleteSourcePort",
    "CdcDeleteTaskProcessor",
    "CdcDeleteTaskRepositoryPort",
    "CdcEventIdentityProvider",
    "CdcHandoffOrchestratorPort",
    "CdcIngestError",
    "CdcPersistenceConflictError",
    "CdcTaskPlan",
    "CdcTaskReceipt",
    "CdcTaskReceiptStatus",
    "FakeCanalClient",
    "ReplayDelete",
    "ReplayResult",
    "parse_canal_batch",
    "prepare_cdc_delete_batch",
    "prepare_cdc_delete_task",
    "replay_batch",
    "replay_next_batch",
]
