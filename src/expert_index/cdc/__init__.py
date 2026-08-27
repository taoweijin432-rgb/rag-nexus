"""Canal CDC 边界模型与客户端端口。

本包只定义上游可依赖的协议边界，不依赖 Canal SDK，也不执行消费或确认。
"""

from .client import CanalPythonConnector, CanalSdkUnavailableError, build_real_canal_client
from .decoder import CanalDecodeError, decode_canal_message
from .delete_scope import (
    CdcDeleteSourcePort,
    prepare_cdc_delete_batch,
    prepare_cdc_delete_task,
)
from .finalize import (
    CdcAckPendingError,
    CdcBatchCommitRequest,
    CdcBatchCoordinator,
    CdcBatchNotReadyError,
    CdcBatchRunResult,
    CdcBinlogPosition,
    CdcCheckpointCommitResult,
    CdcCheckpointConflictError,
    CdcCheckpointRepositoryPort,
    CdcClaimedTaskProcessorPort,
    CdcEventCommitIdentity,
    CdcTaskRepositoryPort,
    build_cdc_batch_commit_request,
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
    CanalChangeEvent,
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
    "CanalChangeEvent",
    "CanalClientAdapter",
    "CanalClientPort",
    "CanalColumn",
    "CanalConnectorPort",
    "CanalDecodeError",
    "CanalDeleteEvent",
    "CanalEntry",
    "CanalParseIssue",
    "CanalParseResult",
    "CanalParserError",
    "CanalProtocolBaseline",
    "CanalProtocolNotFrozenError",
    "CanalPythonConnector",
    "CanalReplaySafetyError",
    "CanalRowData",
    "CanalSdkUnavailableError",
    "CdcAckPendingError",
    "CdcBatchCommitRequest",
    "CdcBatchCoordinator",
    "CdcBatchNotReadyError",
    "CdcBatchPersistencePort",
    "CdcBatchReceiveResult",
    "CdcBatchReceiver",
    "CdcBatchRunResult",
    "CdcBinlogPosition",
    "CdcCheckpointCommitResult",
    "CdcCheckpointConflictError",
    "CdcCheckpointRepositoryPort",
    "CdcClaimedTaskProcessorPort",
    "CdcDeleteSourcePort",
    "CdcDeleteTaskProcessor",
    "CdcDeleteTaskRepositoryPort",
    "CdcEventCommitIdentity",
    "CdcEventIdentityProvider",
    "CdcHandoffOrchestratorPort",
    "CdcIngestError",
    "CdcPersistenceConflictError",
    "CdcTaskPlan",
    "CdcTaskReceipt",
    "CdcTaskReceiptStatus",
    "CdcTaskRepositoryPort",
    "FakeCanalClient",
    "ReplayDelete",
    "ReplayResult",
    "build_cdc_batch_commit_request",
    "build_real_canal_client",
    "decode_canal_message",
    "parse_canal_batch",
    "prepare_cdc_delete_batch",
    "prepare_cdc_delete_task",
    "replay_batch",
    "replay_next_batch",
]


def __getattr__(name: str):
    """按需加载运行时组装器，避免 Repository 初始化时形成循环导入。"""

    if name in {"CdcRuntime", "CdcRuntimeDisabledError", "build_cdc_runtime"}:
        from .runtime import CdcRuntime, CdcRuntimeDisabledError, build_cdc_runtime

        return {
            "CdcRuntime": CdcRuntime,
            "CdcRuntimeDisabledError": CdcRuntimeDisabledError,
            "build_cdc_runtime": build_cdc_runtime,
        }[name]
    raise AttributeError(name)


__all__.extend(["CdcRuntime", "CdcRuntimeDisabledError", "build_cdc_runtime"])
