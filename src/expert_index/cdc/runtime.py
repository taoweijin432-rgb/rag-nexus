"""CDC 真实 Client 到现有任务编排的组装入口。

该模块只负责依赖组装和 Canal 会话生命周期，不创建批次、不初始化 Checkpoint，也不改变
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
from .protocol import (
    CANAL_DESTINATION,
    CANAL_SERVER_VERSION,
    CANAL_SUBSCRIPTION,
    CanalClientPort,
    CanalProtocolBaseline,
)


class CdcRuntimeDisabledError(RuntimeError):
    """表示 CDC_ENABLED 未打开，拒绝组装真实消费运行时。"""


def build_cdc_protocol_baseline(settings: Settings) -> CanalProtocolBaseline:
    """将显式部署配置转换为 CDC 协议门禁模型。

    ``production_ready`` 只有在真实 Probe、源连接和 ACK 测试均由部署人员确认后
    才会成立；本函数不会根据连接成功或默认值自动放行。
    """

    config = settings.cdc_runtime
    unresolved: list[str] = []
    if config.server_version != CANAL_SERVER_VERSION:
        unresolved.append("canal_server_version_mismatch")
    if config.server_mode.strip().casefold() != "tcp":
        unresolved.append("canal_server_mode_unconfirmed")
    if config.destination != CANAL_DESTINATION:
        unresolved.append("canal_destination_mismatch")
    if config.subscription != CANAL_SUBSCRIPTION:
        unresolved.append("canal_subscription_mismatch")
    if config.tcp_port != config.port:
        unresolved.append("canal_port_mismatch")
    if not config.source_connected:
        unresolved.append("source_connection_unconfirmed")
    if config.probe_batch_id < 1 or config.probe_entry_count < 1:
        unresolved.append("real_probe_batch_unconfirmed")
    if not config.ack_confirmed:
        unresolved.append("ack_replay_unconfirmed")
    if not config.protocol_ready:
        unresolved.append("protocol_gate_disabled")
    return CanalProtocolBaseline(
        server_version=config.server_version,
        server_mode=config.server_mode,
        tcp_port=config.tcp_port,
        destination=config.destination,
        subscription=config.subscription,
        source_connected=config.source_connected,
        probe_batch_id=config.probe_batch_id,
        probe_entry_count=config.probe_entry_count,
        ack_called=config.ack_confirmed,
        production_ready=not unresolved,
        unresolved=tuple(unresolved),
    )


@dataclass
class CdcRuntime:
    """持有真实 Canal Client 与既有 CDC/P4/P5/P6 编排器。"""

    client: CanalClientPort
    coordinator: CdcBatchCoordinator
    _connected: bool = False

    def start(self) -> None:
        """建立并保留一个 Canal 会话，供 Worker 连续轮询复用。"""

        if self._connected:
            return
        self.client.connect()
        self._connected = True

    def stop(self) -> None:
        """释放 Canal 会话；重复调用不会产生额外断开请求。"""

        if not self._connected:
            return
        try:
            self.client.disconnect()
        finally:
            self._connected = False

    def consume_once(self, *, batch_size: int, local_batch_id: int) -> CdcBatchRunResult:
        """读取并处理一个 batch；成功后由 Coordinator 按顺序 ACK。"""

        self.start()
        try:
            return self.coordinator.consume_once(
                self.client,
                batch_size=batch_size,
                local_batch_id=local_batch_id,
            )
        except Exception:
            # 传输或业务异常后丢弃当前会话，下一轮重新握手和订阅；未确认 batch
            # 的 rollback 仍由 Coordinator/Adapter 在异常边界内负责。
            self.stop()
            raise


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


__all__ = [
    "CdcRuntime",
    "CdcRuntimeDisabledError",
    "build_cdc_protocol_baseline",
    "build_cdc_runtime",
]
