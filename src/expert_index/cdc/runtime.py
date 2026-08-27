"""CDC 真实 Client 到现有任务编排的组装入口。

该模块只负责依赖组装和单 batch 调用，不创建批次、不初始化 Checkpoint，也不改变
Source/Milvus。业务处理顺序由 :class:`CdcBatchCoordinator` 统一保证。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..handoff_factory import build_incremental_handoff_runtime
from ..repositories.cdc_checkpoint import MySQLCdcCheckpointRepository
from ..repositories.cdc_event import MySQLCdcBatchPersistence
from .finalize import CdcBatchCoordinator, CdcBatchRunResult
from .ingest import CdcBatchReceiver, CdcEventIdentityProvider
from .protocol import CanalClientPort, CanalProtocolBaseline


class CdcRuntimeDisabledError(RuntimeError):
    """表示 CDC_ENABLED 未打开，拒绝组装真实消费运行时。"""


@dataclass(frozen=True)
class CdcRuntime:
    """持有真实 Canal Client 与既有 CDC/P4/P5/P6 编排器。"""

    client: CanalClientPort
    coordinator: CdcBatchCoordinator

    def consume_once(self, *, batch_size: int, local_batch_id: int) -> CdcBatchRunResult:
        """读取并处理一个 batch；成功后由 Coordinator 按顺序 ACK。"""

        self.client.connect()
        try:
            return self.coordinator.consume_once(
                self.client,
                batch_size=batch_size,
                local_batch_id=local_batch_id,
            )
        finally:
            self.client.disconnect()


def build_cdc_runtime(
    settings: Settings,
    *,
    client: CanalClientPort,
    protocol_baseline: CanalProtocolBaseline,
    identity_provider: CdcEventIdentityProvider,
    max_attempts: int,
    retry_delay_seconds: int,
    task_lease_seconds: int,
    vector_service: object | None = None,
) -> CdcRuntime:
    """将真实 Client 接入已有 CDC Event/Task、P4/P5/P6 和 Checkpoint 组件。

    ``protocol_baseline`` 由 Probe/部署配置显式提供；本函数不猜测 Canal 身份或
    Offset。``client`` 可由 :func:`build_real_canal_client` 构造，也可在测试中注入 Fake。
    """

    # Protocol 默认不启用 runtime_checkable，因此使用能力校验避免把 Fake/SDK
    # 适配器限制在同一个具体实现类。
    for method in ("connect", "disconnect", "fetch_without_ack", "rollback", "ack"):
        if not callable(getattr(client, method, None)):
            raise TypeError(f"client 缺少 Canal Client 方法: {method}")
    if not isinstance(protocol_baseline, CanalProtocolBaseline):
        raise TypeError("protocol_baseline 必须是 CanalProtocolBaseline")
    if not settings.cdc_runtime.enabled:
        raise CdcRuntimeDisabledError("CDC_ENABLED=false，拒绝组装真实 Canal 消费运行时")
    settings.validate_index_write_boundary()
    handoff = build_incremental_handoff_runtime(
        settings,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        vector_service=vector_service,
    )
    receiver = CdcBatchReceiver(
        identity_provider=identity_provider,
        policy_version=settings.chunk_policy_version,
    )
    coordinator = CdcBatchCoordinator(
        protocol_baseline=protocol_baseline,
        receiver=receiver,
        persistence=MySQLCdcBatchPersistence(settings.index_db),
        task_repository=handoff.task_repo,
        task_processor=handoff.cdc_delete_processor,
        checkpoint_repository=MySQLCdcCheckpointRepository(settings.index_db),
        task_lease_seconds=task_lease_seconds,
    )
    return CdcRuntime(client=client, coordinator=coordinator)


__all__ = ["CdcRuntime", "CdcRuntimeDisabledError", "build_cdc_runtime"]
