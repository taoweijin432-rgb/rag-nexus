"""普通 Poll 增量的一次性总编排入口。

本模块只负责把既有 Scanner、SourceBatchCoordinator、P4/P5 Handoff 和 P6 Scheduler
接成一轮执行；业务事实、Hash、Embedding、Milvus 和 Checkpoint 规则继续由现有组件
负责。它不启动后台线程，也不接入 CDC 或对账。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .checkpoint import CheckpointManager
from .config import Settings
from .dataset import EXPERT_SPEC, DatasetSpec
from .handoff_factory import build_incremental_handoff_runtime
from .models import BatchSourceScan
from .scanner import (
    IncrementalScanCycleResult,
    build_incremental_scan_coordinator,
)
from .scheduler import (
    IncrementalRoundResult,
    IncrementalRoundScheduler,
    SourceBatchExecutionPlan,
)
from .source_batch_coordinator import SourceBatchCoordinator


class IncrementalRuntimeAssemblyError(RuntimeError):
    """表示扫描结果不能安全转换为可执行的来源批次计划。"""


class IncrementalScanPort(Protocol):
    """总编排入口依赖的单轮四表扫描端口。"""

    def scan_all_sources(
        self,
        worker_id: str,
        *,
        cycle_id: str,
    ) -> IncrementalScanCycleResult: ...


class IncrementalSchedulerPort(Protocol):
    """总编排入口依赖的已有单轮任务执行端口。"""

    def run_once(self, plans: tuple[SourceBatchExecutionPlan, ...]) -> IncrementalRoundResult: ...


@dataclass(frozen=True)
class IncrementalOnceResult:
    """保存一次 Poll 扫描和任务执行的安全摘要。"""

    scan_cycle: IncrementalScanCycleResult
    execution: IncrementalRoundResult

    def to_summary(self) -> dict[str, object]:
        """转换为不含来源正文、游标详情或凭据的 CLI 输出。"""

        scan_outcomes: list[dict[str, object]] = []
        for outcome in self.scan_cycle.outcomes:
            result = outcome.result
            scan_outcomes.append(
                {
                    "source_table": outcome.source_table,
                    "succeeded": outcome.succeeded,
                    "failure_type": outcome.failure_type,
                    "recovered": result.recovered if result else False,
                    "page_count": result.page_count if result else 0,
                    "scheduled_candidate_count": (
                        result.scheduled_candidate_count if result else 0
                    ),
                    "lease_released": result.lease_released if result else False,
                }
            )
        return {
            "cycle_id": self.scan_cycle.cycle_id,
            "scan": {
                "succeeded_source_count": self.scan_cycle.succeeded_source_count,
                "failed_source_count": self.scan_cycle.failed_source_count,
                "outcomes": scan_outcomes,
            },
            "execution": {
                "planned_source_count": self.execution.planned_source_count,
                "completed_source_count": self.execution.completed_source_count,
                "failed_source_count": self.execution.failed_source_count,
                "claimed_count": self.execution.claimed_count,
                "success_count": self.execution.success_count,
                "retryable_count": self.execution.retryable_count,
                "rejected_count": self.execution.rejected_count,
                "lease_conflict_count": self.execution.lease_conflict_count,
                "checkpoint_committed_count": self.execution.checkpoint_committed_count,
                "checkpoint_already_committed_count": (
                    self.execution.checkpoint_already_committed_count
                ),
            },
        }


class IncrementalRuntime:
    """执行一次普通 Poll 的扫描、任务处理和安全水位提交。"""

    def __init__(
        self,
        scanner: IncrementalScanPort,
        scheduler: IncrementalSchedulerPort,
    ) -> None:
        """注入已组装的扫描器和调度器；构造过程不访问数据库或下游。"""

        if not callable(getattr(scanner, "scan_all_sources", None)):
            raise TypeError("scanner 缺少 scan_all_sources")
        if not callable(getattr(scheduler, "run_once", None)):
            raise TypeError("scheduler 缺少 run_once")
        self.scanner = scanner
        self.scheduler = scheduler

    def run_once(self, worker_id: str, *, cycle_id: str) -> IncrementalOnceResult:
        """执行一个四表周期，并将成功扫描范围继续交给既有单轮 Scheduler。"""

        scan_cycle = self.scanner.scan_all_sources(worker_id, cycle_id=cycle_id)
        plans = tuple(
            self._execution_plan(outcome.result)
            for outcome in scan_cycle.outcomes
            if outcome.result and outcome.result.batch_source is not None
        )
        execution = self.scheduler.run_once(plans)
        return IncrementalOnceResult(scan_cycle=scan_cycle, execution=execution)

    @staticmethod
    def _execution_plan(result) -> SourceBatchExecutionPlan:
        """把扫描结果转换为执行计划，并校验 lease owner 没有丢失。"""

        if result.batch_source is None:
            raise IncrementalRuntimeAssemblyError(
                "无批次范围的扫描结果不应进入执行计划"
            )
        if result.lease_owner is None:
            raise IncrementalRuntimeAssemblyError(
                "存在 Batch Source 但缺少来源 lease owner"
            )
        scan: BatchSourceScan = result.batch_source
        return SourceBatchExecutionPlan(
            source_table=scan.source_table,
            batch_id=scan.batch_id,
            lease_owner=result.lease_owner,
            checkpoint_lock_version=scan.checkpoint_lock_version,
        )


def build_incremental_runtime(
    settings: Settings,
    *,
    max_attempts: int,
    retry_delay_seconds: int,
    task_lease_seconds: int,
    vector_service: object | None = None,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
    source: object | None = None,
    include_cdc_delete_processor: bool = True,
) -> IncrementalRuntime:
    """组装普通 Poll 总编排入口，所有重试和 lease 参数必须显式传入。"""

    scanner = build_incremental_scan_coordinator(
        settings,
        dataset_spec=dataset_spec,
        source_adapter=source,
    )
    handoff = build_incremental_handoff_runtime(
        settings,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        vector_service=vector_service,
        dataset_spec=dataset_spec,
        source=source,
        include_cdc_delete_processor=include_cdc_delete_processor,
    )
    coordinator = SourceBatchCoordinator(
        handoff.preparation_coordinator,
        handoff.orchestrator,
        CheckpointManager(settings.index_db),
        scanner.scan_service.source_state_repository,
        handoff.task_repo,
        task_lease_seconds=task_lease_seconds,
        source_lease_seconds=settings.incremental_scan.lease_seconds,
    )
    return IncrementalRuntime(
        scanner,
        IncrementalRoundScheduler(
            coordinator,
            source_table_order=dataset_spec.source_tables,
        ),
    )


__all__ = [
    "IncrementalOnceResult",
    "IncrementalRuntime",
    "IncrementalRuntimeAssemblyError",
    "build_incremental_runtime",
]
