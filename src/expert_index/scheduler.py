"""P6.2 单轮增量任务调度入口。

本模块只负责把已经由 P3 扫描出的 Batch Source 按固定顺序交给
``SourceBatchCoordinator``。任务领取、退避时间、过期 lease 回收和阶段恢复仍由已有
Repository/Coordinator 完成，调度器不复制状态机，也不启动常驻线程。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .models import PERSISTED_SOURCE_TABLES
from .observability import IncrementalMetrics
from .scanner import INCREMENTAL_SOURCE_TABLE_ORDER
from .source_batch_coordinator import SourceBatchRunResult

__all__ = [
    "IncrementalRoundResult",
    "IncrementalRoundScheduler",
    "SourceBatchExecutionPlan",
]

_SAFE_OWNER = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class SourceBatchExecutionPlan:
    """描述一次单来源 Batch Source 执行所需的持久化身份。"""

    source_table: str
    batch_id: int
    lease_owner: str
    checkpoint_lock_version: int

    def __post_init__(self) -> None:
        """校验调度计划只包含合法来源和安全状态定位字段。"""

        if self.source_table not in PERSISTED_SOURCE_TABLES:
            raise ValueError(f"不支持的来源表: {self.source_table!r}")
        if (
            isinstance(self.batch_id, bool)
            or not isinstance(self.batch_id, int)
            or self.batch_id < 1
        ):
            raise ValueError("batch_id 必须是正整数")
        if (
            not isinstance(self.lease_owner, str)
            or not 1 <= len(self.lease_owner) <= 128
            or self.lease_owner != self.lease_owner.strip()
            or _SAFE_OWNER.fullmatch(self.lease_owner) is None
        ):
            raise ValueError("lease_owner 必须是安全的非空标识")
        if (
            isinstance(self.checkpoint_lock_version, bool)
            or not isinstance(self.checkpoint_lock_version, int)
            or self.checkpoint_lock_version < 0
        ):
            raise ValueError("checkpoint_lock_version 必须是非负整数")


@dataclass(frozen=True)
class IncrementalRoundResult:
    """一次调度轮次的无 PII 数量摘要。"""

    planned_source_count: int
    completed_source_count: int
    failed_source_count: int
    claimed_count: int
    success_count: int
    retryable_count: int
    rejected_count: int
    lease_conflict_count: int
    checkpoint_committed_count: int
    checkpoint_already_committed_count: int
    outcomes: tuple[SourceBatchRunResult, ...]


class IncrementalRoundScheduler:
    """执行一次有限来源计划，不提供后台循环或并发 Worker。"""

    def __init__(
        self,
        source_batch_coordinator,
        *,
        metrics: IncrementalMetrics | None = None,
        source_table_order: tuple[str, ...] = INCREMENTAL_SOURCE_TABLE_ORDER,
    ) -> None:
        """注入单来源协调器和可选的进程内指标记录器。"""

        if not callable(getattr(source_batch_coordinator, "run_claimed_source_batch", None)):
            raise TypeError("source_batch_coordinator 缺少 run_claimed_source_batch")
        self._coordinator = source_batch_coordinator
        if metrics is not None and not isinstance(metrics, IncrementalMetrics):
            raise TypeError("metrics 必须是 IncrementalMetrics 或 None")
        if (
            not source_table_order
            or len(set(source_table_order)) != len(source_table_order)
            or any(table not in PERSISTED_SOURCE_TABLES for table in source_table_order)
        ):
            raise ValueError("source_table_order 必须是已冻结且不重复的来源表顺序")
        self._metrics = metrics
        self._source_table_order = source_table_order

    def run_once(
        self,
        plans: Iterable[SourceBatchExecutionPlan],
    ) -> IncrementalRoundResult:
        """按四表固定顺序执行一轮，并汇总每个来源的安全计数。

        ``SourceBatchCoordinator`` 内部固定单条 claim；普通 retry 和过期 processing
        的新 ``attempt_token`` 由 Task Repository 生成，当前调度器不会复用旧 token。
        某个来源失败只结束该来源，本轮继续执行其他已提供的独立来源计划。
        """

        normalized = tuple(plans)
        if any(not isinstance(plan, SourceBatchExecutionPlan) for plan in normalized):
            raise TypeError("plans 必须全部是 SourceBatchExecutionPlan")
        if len({plan.source_table for plan in normalized}) != len(normalized):
            raise ValueError("同一轮不能重复调度同一来源表")
        order = {source: index for index, source in enumerate(self._source_table_order)}
        unknown = {plan.source_table for plan in normalized} - set(order)
        if unknown:
            raise ValueError(f"调度计划包含当前数据集之外的来源表: {sorted(unknown)}")
        ordered = tuple(sorted(normalized, key=lambda plan: order[plan.source_table]))
        outcomes: list[SourceBatchRunResult] = []
        for plan in ordered:
            try:
                outcome = self._coordinator.run_claimed_source_batch(
                    plan.source_table,
                    plan.batch_id,
                    plan.lease_owner,
                    checkpoint_lock_version=plan.checkpoint_lock_version,
                    task_claim_limit=1,
                )
            except Exception as error:  # noqa: BLE001 - 单来源失败隔离到摘要
                # 不暴露数据库/来源正文；异常类名足够用于聚合和后续诊断。
                outcome = _failed_source_result(plan, type(error).__name__)
            outcomes.append(outcome)
        result = _summarize(outcomes)
        if self._metrics is not None:
            self._metrics.record_round(result)
        return result


def _failed_source_result(
    plan: SourceBatchExecutionPlan,
    error_type: str,
) -> SourceBatchRunResult:
    """为单来源调度异常构造无 PII 失败摘要。"""

    return SourceBatchRunResult(
        source_table=plan.source_table,
        batch_id=plan.batch_id,
        candidate_count=0,
        claimed_count=0,
        success_count=0,
        retryable_count=0,
        rejected_count=0,
        lease_conflict_count=0,
        checkpoint_committed=False,
        checkpoint_already_committed=False,
        checkpoint_skipped_count=None,
        error=error_type,
    )


def _summarize(outcomes: list[SourceBatchRunResult]) -> IncrementalRoundResult:
    """把单来源结果汇总为稳定数量摘要。"""

    return IncrementalRoundResult(
        planned_source_count=len(outcomes),
        completed_source_count=sum(
            outcome.error is None for outcome in outcomes
        ),
        failed_source_count=sum(outcome.error is not None for outcome in outcomes),
        claimed_count=sum(outcome.claimed_count for outcome in outcomes),
        success_count=sum(outcome.success_count for outcome in outcomes),
        retryable_count=sum(outcome.retryable_count for outcome in outcomes),
        rejected_count=sum(outcome.rejected_count for outcome in outcomes),
        lease_conflict_count=sum(outcome.lease_conflict_count for outcome in outcomes),
        checkpoint_committed_count=sum(
            outcome.checkpoint_committed for outcome in outcomes
        ),
        checkpoint_already_committed_count=sum(
            outcome.checkpoint_already_committed for outcome in outcomes
        ),
        outcomes=tuple(outcomes),
    )
