"""P6.4 最小指标和交付门禁测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from expert_index.observability import (
    IncrementalMetrics,
    MetricsInputError,
    evaluate_delivery_gate,
)


def test_metrics_record_round_counts_without_business_payload() -> None:
    metrics = IncrementalMetrics()
    metrics.record_round(
        SimpleNamespace(
            claimed_count=2,
            success_count=1,
            retryable_count=1,
            rejected_count=0,
            lease_conflict_count=0,
            failed_source_count=0,
            checkpoint_committed_count=1,
            checkpoint_already_committed_count=0,
        )
    )
    assert metrics.snapshot()["task_claimed"] == 2
    assert metrics.snapshot()["task_retry"] == 1


def test_metrics_reject_unsafe_label() -> None:
    with pytest.raises(MetricsInputError):
        IncrementalMetrics().increment("task_claimed", labels={"payload": "secret"})


def test_delivery_gate_stays_closed_until_all_facts_are_ready() -> None:
    report = evaluate_delivery_gate(
        required_migrations=("012", "013", "014", "015", "016"),
        applied_migrations=("012", "013", "014", "015"),
        source_index_isolated=True,
        contract_frozen=True,
        targeted_tests_passed=True,
        incremental_enabled=False,
        retry_policy_frozen=True,
        runtime_budgets_frozen=True,
    )
    assert report.ready is False
    assert report.blockers == ("missing_migrations:016",)
