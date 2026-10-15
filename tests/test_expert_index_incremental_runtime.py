"""普通 Poll 增量总编排入口的最小接线测试。"""

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from expert_index.cli import build_parser
from expert_index.config import DatabaseConfig, DatabaseRole, IncrementalScanConfig, Settings
from expert_index.incremental_runtime import IncrementalRuntime, build_incremental_runtime
from expert_index.models import BatchSourceScan, SourceScanCursor
from expert_index.scanner import (
    IncrementalScanCycleResult,
    IncrementalScanResult,
    IncrementalSourceScanOutcome,
)
from expert_index.scheduler import IncrementalRoundResult, SourceBatchExecutionPlan


@dataclass
class _Scanner:
    result: IncrementalScanCycleResult

    def scan_all_sources(self, _worker_id: str, *, cycle_id: str) -> IncrementalScanCycleResult:
        assert cycle_id == self.result.cycle_id
        return self.result


class _Scheduler:
    def __init__(self) -> None:
        self.plans: tuple[SourceBatchExecutionPlan, ...] = ()

    def run_once(self, plans: tuple[SourceBatchExecutionPlan, ...]) -> IncrementalRoundResult:
        self.plans = plans
        return IncrementalRoundResult(1, 1, 0, 1, 1, 0, 0, 0, 1, 0, ())


def test_runtime_composes_scanned_batch_into_scheduler_plan() -> None:
    """扫描得到的 Batch Source 必须沿用同一 lease owner 进入任务调度。"""

    cursor = SourceScanCursor(datetime(2026, 8, 13, tzinfo=timezone.utc), 0)
    scan = BatchSourceScan(
        batch_id=9,
        source_table="experts",
        start_cursor=cursor,
        current_cursor=cursor,
        upper_bound=SourceScanCursor(datetime(2026, 8, 14, tzinfo=timezone.utc), 1),
        checkpoint_lock_version=3,
    )
    cycle = IncrementalScanCycleResult(
        "cycle-1",
        (
            IncrementalSourceScanOutcome(
                "experts",
                IncrementalScanResult(
                    "experts", scan, False, 1, 2, False, "worker.cycle.experts"
                ),
            ),
        ),
    )
    scheduler = _Scheduler()

    result = IncrementalRuntime(_Scanner(cycle), scheduler).run_once(
        "worker", cycle_id="cycle-1"
    )

    assert result.execution.success_count == 1
    assert scheduler.plans == (
        SourceBatchExecutionPlan("experts", 9, "worker.cycle.experts", 3),
    )


def test_cli_exposes_one_shot_incremental_command() -> None:
    """正式入口必须要求 worker、周期和冻结的重试/租约参数。"""

    args = build_parser().parse_args(
        [
            "incremental-once",
            "--worker-id",
            "worker-1",
            "--cycle-id",
            "cycle-1",
            "--max-attempts",
            "4",
            "--retry-delay-seconds",
            "30",
            "--task-lease-seconds",
            "180",
        ]
    )

    assert args.command == "incremental-once"
    assert args.task_lease_seconds == 180
    assert not hasattr(args, "worker_count")


def test_runtime_builder_keeps_disabled_scan_before_downstream_assembly() -> None:
    """关闭增量开关时，入口必须在构造 Embedding/Milvus 前拒绝执行。"""

    source = DatabaseConfig("source", 3306, "reader", "secret", "expert", DatabaseRole.SOURCE)
    index = DatabaseConfig("index", 3306, "writer", "secret", "expert_index", DatabaseRole.INDEX)
    settings = Settings(
        source_db=source,
        index_db=index,
        incremental_scan=IncrementalScanConfig(enabled=False),
    )

    with pytest.raises(RuntimeError, match="INCREMENTAL_ENABLED"):
        build_incremental_runtime(
            settings,
            max_attempts=4,
            retry_delay_seconds=30,
            task_lease_seconds=180,
        )
