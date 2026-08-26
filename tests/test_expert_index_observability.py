"""P6.4 最小指标和交付门禁测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from expert_index.observability import (
    IncrementalMetrics,
    MetricsInputError,
    evaluate_cdc_delivery_gate,
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


def test_cdc_delivery_gate_separates_code_delivery_from_production_activation() -> None:
    """迁移和回放通过后，真实协议未冻结仍必须阻止 CDC 启用。"""

    report = evaluate_cdc_delivery_gate(
        required_migrations=("017", "018"),
        applied_migrations=("017", "018"),
        source_index_isolated=True,
        fixtures_passed=True,
        failure_recovery_passed=True,
        protocol_frozen=False,
        cdc_enabled=False,
    )

    assert report.ready is False
    assert report.blockers == ("canal_production_protocol_not_frozen",)

    ready_report = evaluate_cdc_delivery_gate(
        required_migrations=("017", "018"),
        applied_migrations=("017", "018"),
        source_index_isolated=True,
        fixtures_passed=True,
        failure_recovery_passed=True,
        protocol_frozen=True,
        cdc_enabled=False,
    )
    assert ready_report.ready is True


def test_cdc_metrics_only_expose_safe_status_and_counts() -> None:
    """CDC 指标不输出 batch ID、Event 身份或业务字段。"""

    metrics = IncrementalMetrics()
    metrics.record_cdc_batch(
        SimpleNamespace(status="not_ready", task_count=2, checkpoint=None)
    )

    assert metrics.snapshot() == {
        "cdc_batch{cdc_status=not_ready}": 1,
        "cdc_task{cdc_status=not_ready}": 2,
    }
