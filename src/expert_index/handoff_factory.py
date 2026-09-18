"""P5.4 生产依赖组装：把 P4/P5.1/P5.2/P5.3A 组件接成 Handoff 闭环。

本模块只做**被动组装**：返回一个 :class:`IncrementalHandoffRuntime`，里面持有
跑正式增量 Handoff 闭环所需的全部组件实例。**不启动后台 Worker、不轮询、不认领
任务、不调用 Consumer、不访问数据库或 Milvus**——这些生命周期动作属于 P5.7
批次协调器，本层不实现。

组装顺序（fail-closed 前置校验）：

1. ``settings.incremental_scan.enabled`` 为 false 时立即抛
   :class:`IncrementalScanDisabledError`（复用 P3 已有异常），不构造任何下游依赖；
2. :meth:`Settings.validate_index_write_boundary` 校验 Index/Source 物理 Schema 隔离；
3. ``max_attempts``/``retry_delay_seconds`` 显式校验；
4. 缺失 ``vector_service`` 时才惰性调用
   ``create_vector_ingestion_service()``，**绝不调用
   ``initialize()``**（本层只组装、不连库/不连 Milvus）。

依赖注入边界（不重复实现业务判断）：

- ``MySQLSourceAdapter(settings.source_db)``：来源只读事实（P4.3）；
- ``ChunkFactory(settings)``：Chunk 构建器（P4.4）；
- ``MySQLSourceSyncTaskRepository(settings.index_db)`` / ``MySQLArtifactRepository``：
  P5.3A 公共持久化；
- :class:`PreparationCoordinator`：P4 外层协调器，复用同一 ``task_repo`` /
  ``source`` / ``chunk_factory``，续租与阶段推进由其内部以 Repository 数据库时间
  权威判定；``lease_seconds`` 复用 ``settings.incremental_scan.lease_seconds``
  （与 P3 claim/renew 同一配置），不新增默认值或新配置；
- ``ScopeHandoffBuilder(settings.timezone)``：P5.1 内存构建器；
- ``EmbeddingEligibilityValidator().get_skip_reason`` 注入
  :class:`ContractConsumerCore`（**禁止 None**），Eligibility 规则只有一处实现；
- ``ScopeConsumer(core, state_port=vector_service, receipt_sink=...)``：P5.2 范围收敛；
- :class:`HandoffOrchestrator`：P5.3B 编排闭环（只接 ``validated`` 主路径，
  ``contract_ready``/``handed_off`` 仅供崩溃恢复）；
- :class:`CdcDeleteTaskProcessor`：P7.6 复用同一 Task Repository、Source、
  ChunkFactory 和 HandoffOrchestrator 处理已领取的 CDC Task。

Receipt sink 是一个把 :class:`ScopeHandoffReceipt` 构造器适配为
``make_receipt(**kwargs)`` 的薄包装，只做 ``impact_scope`` 类型校验与构造，不做任何
业务判断（计数、状态、收敛证据全部由 ScopeConsumer 决定）。

绝不：

- 调用 ``vector_service.repository.initialize()`` 或任何 Milvus/Embedding 实际连接；
- 启动 Worker / 轮询 / checkpoint 推进（属 P5.7）；
- 复制 Eligibility / 三 Hash / stale 计算等下游业务逻辑；
- 引入 V15 Work Item/Batch、CheckpointManager 或向量物理删除。
"""

from __future__ import annotations

from dataclasses import dataclass

from .cdc.task_processor import CdcDeleteTaskProcessor
from .chunk_builder import ChunkFactory
from .config import Settings
from .dataset import EXPERT_SPEC, DatasetSpec
from .handoff import ScopeHandoffBuilder, ScopeHandoffReceipt
from .handoff_orchestrator import HandoffOrchestrator
from .models import IncrementalImpactScope
from .preparation_coordinator import PreparationCoordinator
from .repositories.artifact import MySQLArtifactRepository
from .repositories.scope_fence import MySQLScopeExecutionFence
from .repositories.sync_task import MySQLSourceSyncTaskRepository
from .scanner import IncrementalScanDisabledError
from .source import MySQLSourceAdapter


@dataclass(frozen=True)
class IncrementalHandoffRuntime:
    """正式增量 Handoff 闭环组件集合（被动组装结果）。

    持有跑闭环所需的全部组件实例。正式调用顺序（P5.7 批次协调器执行）：

    1. 新领取任务先调用 :meth:`PreparationCoordinator.prepare` —— 由 Coordinator
       负责续租与单调节段推进（candidate→source_read→normalized→chunk_built→
       validated），返回 ``validated`` 的内存 :class:`PreparedChunkSet`；
    2. 把 validated 的 PreparedChunkSet 交给 :meth:`HandoffOrchestrator.process`
       完成 Artifact→Consumer→回执→终态写入；
    3. ``contract_ready`` / ``handed_off`` 阶段仅用于 P5 崩溃恢复（Artifact 已落库
       后进程中断），由 :meth:`HandoffOrchestrator.process` 从已有 Artifact 重建
       Request，不重新调用 Coordinator/Builder。

    已领取的 CDC Task 调用
    :meth:`CdcDeleteTaskProcessor.process_claimed`，由它复用同一 Handoff
    编排器；不经过 Poll 的 ``PreparationCoordinator``。

    本结构不暴露任何后台入口、不轮询、不认领任务、不推进 checkpoint。
    """

    source: object
    chunk_factory: ChunkFactory
    task_repo: MySQLSourceSyncTaskRepository
    artifact_repo: MySQLArtifactRepository
    builder: ScopeHandoffBuilder
    preparation_coordinator: PreparationCoordinator
    orchestrator: HandoffOrchestrator
    cdc_delete_processor: CdcDeleteTaskProcessor | None
    # 以下两项暴露供 P5.7 续租/重放/诊断使用；本层不直接调用它们的后台方法。
    vector_service: object
    scope_consumer: object


class _ReceiptSinkAdapter:
    """把 ``ScopeHandoffReceipt`` 构造器适配为 ``ScopeReceiptSink`` 接口。

    只做 ``impact_scope`` 类型校验与 Receipt 构造，不做计数/状态/收敛判断。
    生产由 :class:`ScopeConsumer` 调用 ``make_receipt`` 组装真实回执。
    """

    def __init__(self, receipt_cls: type[ScopeHandoffReceipt]) -> None:
        self._receipt_cls = receipt_cls

    def make_receipt(self, **kwargs: object) -> ScopeHandoffReceipt:
        impact_scope = kwargs.get("impact_scope")
        if not isinstance(impact_scope, IncrementalImpactScope):
            raise TypeError("impact_scope 必须是 IncrementalImpactScope")
        # 其余字段直接透传；ScopeHandoffReceipt 自身做完整校验。
        return self._receipt_cls(**kwargs)  # type: ignore[arg-type]


def build_incremental_handoff_runtime(
    settings: Settings,
    *,
    vector_service: object | None = None,
    max_attempts: int,
    retry_delay_seconds: int,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
    source: object | None = None,
    include_cdc_delete_processor: bool = True,
) -> IncrementalHandoffRuntime:
    """组装正式增量 Handoff 闭环组件集合。

    :param settings: 已由 ``Settings.from_env`` 构造的配置对象。
    :param vector_service: 可选的已构造 ``VectorIngestionService`` 实例（测试注入）。
        为 ``None`` 时惰性调用 ``create_vector_ingestion_service()``，
        **绝不调用 ``initialize()``**。
    :param max_attempts: ``HandoffOrchestrator`` 的可重试上限，显式传入不猜默认。
    :param retry_delay_seconds: ``HandoffOrchestrator`` 的重试冷却秒数，显式传入。
    :return: :class:`IncrementalHandoffRuntime`，持有全部组件实例。
    :raises IncrementalScanDisabledError: ``INCREMENTAL_ENABLED=false`` 时。
    :raises ValueError: 重试参数非法，或 Index/Source 物理 Schema 未隔离。
    """

    # 1) 增量入口开关：未显式开启时拒绝组装任何下游依赖。
    if not settings.incremental_scan.enabled:
        raise IncrementalScanDisabledError(
            "INCREMENTAL_ENABLED 未开启，增量 Handoff 闭环组装入口保持禁用"
        )

    # 2) Index/Source 物理 Schema 隔离（复用 Settings 校验）。
    settings.validate_index_write_boundary()

    # 3) 重试参数显式校验（与 HandoffOrchestrator 自身校验对齐）。
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 0
    ):
        raise ValueError("max_attempts 必须是非负整数")
    if (
        isinstance(retry_delay_seconds, bool)
        or not isinstance(retry_delay_seconds, int)
        or retry_delay_seconds < 1
    ):
        raise ValueError("retry_delay_seconds 必须是正整数")

    # 4) 缺失 vector_service 时才惰性构造；绝不调用 initialize()（本层只组装、不连库）。
    if vector_service is None:
        # 局部导入，避免顶层形成 expert_index ↔ rag_vector_store 强耦合；
        # create_vector_ingestion_service 内部会从环境读取 Milvus/Embedding 配置，
        # 缺失时由其自身 fail-closed，本层不补默认。
        from rag_vector_store import create_vector_ingestion_service

        vector_service = create_vector_ingestion_service()

    # 局部导入下游业务核心（避免顶层环依赖）。
    from rag_vector_store.chunking.contract_pipeline import EmbeddingEligibilityValidator
    from rag_vector_store.incremental.consumer_core import ContractConsumerCore
    from rag_vector_store.incremental.scope_consumer import ScopeConsumer

    # 6) 来源/构建器/持久化（P4.3 / P4.4 / P5.3A）。
    resolved_source = source or MySQLSourceAdapter(settings.source_db)
    chunk_factory = ChunkFactory(settings, dataset_spec=dataset_spec)
    task_repo = MySQLSourceSyncTaskRepository(settings.index_db)
    artifact_repo = MySQLArtifactRepository(settings.index_db)
    scope_fence = MySQLScopeExecutionFence(settings.index_db)
    builder = ScopeHandoffBuilder(settings.timezone, dataset_spec=dataset_spec)

    # 7) PreparationCoordinator（P4 外层协调器）：复用同一 task_repo/source/chunk_factory；
    #    lease_seconds 复用 settings.incremental_scan.lease_seconds（与 P3 claim/renew
    #    同一配置），不新增默认值或新配置。真正续租有效性由 Repository 以
    #    MySQL CURRENT_TIMESTAMP(6) + attempt_token + rowcount 权威判定。
    preparation_coordinator = PreparationCoordinator(
        task_repo,
        resolved_source,
        chunk_factory,
        lease_seconds=settings.incremental_scan.lease_seconds,
    )

    # 8) Eligibility 必须注入正式实现，禁止 None（避免静默放行 ineligible desired）。
    eligibility = EmbeddingEligibilityValidator()
    consumer_core = ContractConsumerCore(
        vector_store=vector_service,  # type: ignore[arg-type]
        eligibility_skip_of=eligibility.get_skip_reason,
    )

    # 9) ScopeConsumer（P5.2）：state_port 与 core 共用同一 vector_service。
    receipt_sink = _ReceiptSinkAdapter(ScopeHandoffReceipt)
    scope_consumer = ScopeConsumer(
        core=consumer_core,
        state_port=vector_service,  # type: ignore[arg-type]
        receipt_sink=receipt_sink,
        scope_source_tables={
            "profile": dataset_spec.profile_source_label,
            "employment": dataset_spec.employment_table,
            "project": dataset_spec.project_table,
        },
    )

    # 10) HandoffOrchestrator（P5.3B）。
    orchestrator = HandoffOrchestrator(
        task_repo=task_repo,
        artifact_repo=artifact_repo,
        builder=builder,
        consumer=scope_consumer,  # type: ignore[arg-type]
        scope_fence=scope_fence,
        lease_seconds=settings.incremental_scan.lease_seconds,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )

    # 11) P7.6 CDC Task：复用同一 Handoff/P6 失败恢复，不提交 CDC Checkpoint，
    #     也不调用 Canal ACK。
    cdc_delete_processor = None
    if include_cdc_delete_processor:
        if not callable(
            getattr(resolved_source, "read_cdc_education_delete_facts", None)
        ):
            raise TypeError("CDC Delete Processor 的 Source 缺少教育删除事实读取能力")
        cdc_delete_processor = CdcDeleteTaskProcessor(
            task_repository=task_repo,
            source=resolved_source,
            factory=chunk_factory,
            orchestrator=orchestrator,
            lease_seconds=settings.incremental_scan.lease_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            dataset_spec=dataset_spec,
        )

    return IncrementalHandoffRuntime(
        source=resolved_source,
        chunk_factory=chunk_factory,
        task_repo=task_repo,
        artifact_repo=artifact_repo,
        builder=builder,
        preparation_coordinator=preparation_coordinator,
        orchestrator=orchestrator,
        cdc_delete_processor=cdc_delete_processor,
        vector_service=vector_service,
        scope_consumer=scope_consumer,
    )


__all__ = [
    "IncrementalHandoffRuntime",
    "build_incremental_handoff_runtime",
]
