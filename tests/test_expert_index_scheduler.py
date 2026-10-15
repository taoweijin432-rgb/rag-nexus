"""P6.2 单轮调度入口的最小测试。"""

import threading
from dataclasses import dataclass

import pytest

from expert_index.scheduler import IncrementalRoundScheduler, SourceBatchExecutionPlan
from expert_index.source_batch_coordinator import SourceBatchRunResult


def _result(source_table: str, batch_id: int) -> SourceBatchRunResult:
    return SourceBatchRunResult(
        source_table=source_table,
        batch_id=batch_id,
        candidate_count=1,
        claimed_count=1,
        success_count=1,
        retryable_count=0,
        rejected_count=0,
        lease_conflict_count=0,
        checkpoint_committed=True,
        checkpoint_already_committed=False,
        checkpoint_skipped_count=0,
        error=None,
    )


@dataclass
class FakeCoordinator:
    calls: list[tuple] | None = None

    def __post_init__(self) -> None:
        self.calls = []

    def run_claimed_source_batch(self, source_table, batch_id, lease_owner, *, checkpoint_lock_version, task_claim_limit):
        assert task_claim_limit == 1
        self.calls.append((source_table, batch_id, lease_owner, checkpoint_lock_version))
        return _result(source_table, batch_id)


def test_run_once_uses_stable_source_order_and_aggregates() -> None:
    coordinator = FakeCoordinator()
    scheduler = IncrementalRoundScheduler(coordinator)
    result = scheduler.run_once(
        [
            SourceBatchExecutionPlan("experts_project_experience", 3, "w", 0),
            SourceBatchExecutionPlan("experts", 1, "w", 0),
        ]
    )
    assert [call[0] for call in coordinator.calls] == [
        "experts",
        "experts_project_experience",
    ]
    assert result.planned_source_count == 2
    assert result.success_count == 2
    assert result.checkpoint_committed_count == 2


def test_run_once_rejects_duplicate_source_plan() -> None:
    scheduler = IncrementalRoundScheduler(FakeCoordinator())
    plan = SourceBatchExecutionPlan("experts", 1, "w", 0)
    with pytest.raises(ValueError):
        scheduler.run_once([plan, plan])


def test_run_once_can_process_distinct_sources_concurrently_and_keep_result_order() -> None:
    """来源表并发不应改变结果的固定表顺序。"""

    barrier = threading.Barrier(2)

    class ConcurrentCoordinator:
        def run_claimed_source_batch(
            self,
            source_table,
            batch_id,
            lease_owner,
            *,
            checkpoint_lock_version,
            task_claim_limit,
        ):
            assert task_claim_limit == 1
            barrier.wait(timeout=2)
            return _result(source_table, batch_id)

    scheduler = IncrementalRoundScheduler(
        ConcurrentCoordinator(),
        worker_count=2,
    )
    result = scheduler.run_once(
        [
            SourceBatchExecutionPlan("experts_work_experience", 2, "work", 0),
            SourceBatchExecutionPlan("experts", 1, "profile", 0),
        ]
    )

    assert result.completed_source_count == 2
    assert [outcome.source_table for outcome in result.outcomes] == [
        "experts",
        "experts_work_experience",
    ]


@pytest.mark.parametrize("worker_count", [True, 0, 5])
def test_scheduler_rejects_invalid_source_worker_count(worker_count: object) -> None:
    with pytest.raises(ValueError, match="worker_count"):
        IncrementalRoundScheduler(
            FakeCoordinator(),
            worker_count=worker_count,  # type: ignore[arg-type]
        )
