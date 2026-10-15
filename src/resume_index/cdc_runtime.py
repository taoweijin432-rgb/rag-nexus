"""Resume Canal CDC 运行时组装。

本模块只提供 Resume 专用的协议门禁和依赖注入：公共 CDC Parser、Event/Task
持久化、Lease/Retry/DLQ、Checkpoint、ACK/rollback 与向量收敛仍由
``expert_index.cdc`` 中的数据集无关组件负责。CDC 只补充物理 DELETE；INSERT、
UPDATE 和逻辑删除继续由 Resume Poll 处理。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from expert_index.cdc import (
    CanalChangeEvent,
    CanalClientPort,
    CanalProtocolBaseline,
    CdcBatchRunResult,
    CdcRuntime,
    CdcRuntimeDisabledError,
    build_cdc_event_id,
    build_cdc_protocol_baseline,
    build_cdc_runtime,
)
from expert_index.scanner import IncrementalScanDisabledError
from rag_vector_store import MilvusSettings, ResumeVectorIngestionService, VectorIngestionService

from .config import (
    RESUME_CANAL_DESTINATION,
    RESUME_CANAL_SUBSCRIPTION,
    ResumeSettings,
)
from .dataset import RESUME_SPEC
from .preflight import ResumeSourcePreflightResult, run_resume_source_preflight
from .runtime import resolve_resume_vector_service
from .source import MySQLResumeSourceAdapter


@dataclass(frozen=True)
class ResumeCdcRuntime:
    """已通过 Resume 来源预检并绑定 Resume 契约的 CDC 运行时。"""

    runtime: CdcRuntime
    preflight: ResumeSourcePreflightResult

    def start(self) -> None:
        """建立 Resume Canal 会话；重复调用由公共运行时保证幂等。"""

        self.runtime.start()

    def stop(self) -> None:
        """释放 Resume Canal 会话；不修改已确认业务水位。"""

        self.runtime.stop()

    def consume_once(self, *, batch_size: int, local_batch_id: int) -> CdcBatchRunResult:
        """消费一个 Canal batch，并沿公共协调器完成处理、Checkpoint 与 ACK。"""

        return self.runtime.consume_once(
            batch_size=batch_size,
            local_batch_id=local_batch_id,
        )


def build_resume_cdc_event_id(event: CanalChangeEvent) -> str:
    """用 Resume destination 命名空间生成稳定事件身份。"""

    return build_cdc_event_id(event, destination=RESUME_CANAL_DESTINATION)


def build_resume_cdc_protocol_baseline(
    settings: ResumeSettings,
) -> CanalProtocolBaseline:
    """按冻结的 Resume destination 和四表订阅构造协议门禁。"""

    if not isinstance(settings, ResumeSettings):
        raise TypeError("settings 必须是 ResumeSettings")
    settings.validate_isolation_boundaries()
    return build_cdc_protocol_baseline(
        settings.to_processing_settings(),
        expected_destination=RESUME_CANAL_DESTINATION,
        expected_subscription=RESUME_CANAL_SUBSCRIPTION,
    )


def build_resume_cdc_runtime(
    settings: ResumeSettings,
    *,
    client: CanalClientPort,
    protocol_baseline: CanalProtocolBaseline,
    max_attempts: int,
    retry_delay_seconds: int,
    task_lease_seconds: int,
    identity_provider: Callable[[CanalChangeEvent], str] = build_resume_cdc_event_id,
    source: MySQLResumeSourceAdapter | None = None,
    vector_service: VectorIngestionService | ResumeVectorIngestionService | None = None,
    base_milvus_settings: MilvusSettings | None = None,
    preflight_runner: Callable[[ResumeSettings], ResumeSourcePreflightResult] = (
        run_resume_source_preflight
    ),
) -> ResumeCdcRuntime:
    """按 fail-closed 顺序组装 Resume 物理 DELETE CDC 全链路。

    :param settings: Resume 独立 Source、Index、Collection、Poll 与 CDC 配置。
    :param client: 已绑定 Resume destination/subscription 的 Canal Client 端口。
    :param protocol_baseline: 真实 Probe 和 ACK/重投验证得到的协议门禁。
    :param max_attempts: Task 自动处理尝试上限，由 CLI 显式传入。
    :param retry_delay_seconds: 可重试失败的冷却秒数，由 CLI 显式传入。
    :param task_lease_seconds: 单次 Task 处理租约秒数，由 CLI 显式传入。
    :param identity_provider: Resume CDC 稳定事件身份生成器；测试可注入。
    :param source: 可选 Resume 只读 Source Adapter；生产默认创建。
    :param vector_service: 可选已绑定 Resume Collection/Profile 的向量服务。
    :param base_milvus_settings: 未注入向量服务时使用的 Milvus 连接基线。
    :param preflight_runner: 四张 Resume 来源表的只读结构预检。
    """

    if not isinstance(settings, ResumeSettings):
        raise TypeError("settings 必须是 ResumeSettings")
    settings.validate_isolation_boundaries()
    if not settings.cdc_runtime.enabled:
        raise CdcRuntimeDisabledError(
            "RESUME_CDC_ENABLED=false，拒绝组装真实 Resume Canal 消费运行时"
        )
    if not settings.incremental_scan.enabled:
        raise IncrementalScanDisabledError(
            "RESUME_INCREMENTAL_ENABLED 未开启，Resume CDC 无法复用增量任务处理链"
        )
    if not isinstance(protocol_baseline, CanalProtocolBaseline):
        raise TypeError("protocol_baseline 必须是 CanalProtocolBaseline")
    if protocol_baseline.destination != RESUME_CANAL_DESTINATION:
        raise ValueError("Resume CDC 协议门禁 destination 不匹配")
    if protocol_baseline.subscription != RESUME_CANAL_SUBSCRIPTION:
        raise ValueError("Resume CDC 协议门禁 subscription 不匹配")
    if not callable(identity_provider):
        raise TypeError("identity_provider 必须可调用")

    # 来源结构先于 Embedding、Milvus 和 Index Repository 组装验证，避免半启动。
    preflight = preflight_runner(settings)
    resolved_vector_service = resolve_resume_vector_service(
        settings,
        vector_service=vector_service,
        base_milvus_settings=base_milvus_settings,
    )
    resolved_source = source or MySQLResumeSourceAdapter(settings.source_db)
    runtime = build_cdc_runtime(
        settings.to_processing_settings(),
        client=client,
        protocol_baseline=protocol_baseline,
        identity_provider=identity_provider,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        task_lease_seconds=task_lease_seconds,
        vector_service=resolved_vector_service,
        dataset_spec=RESUME_SPEC,
        source=resolved_source,
        source_schema=settings.source_db.database,
        destination=RESUME_CANAL_DESTINATION,
    )
    return ResumeCdcRuntime(runtime=runtime, preflight=preflight)


__all__ = [
    "ResumeCdcRuntime",
    "build_resume_cdc_event_id",
    "build_resume_cdc_protocol_baseline",
    "build_resume_cdc_runtime",
]
