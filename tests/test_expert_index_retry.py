"""P6.1 失败分类的最小契约测试。"""

import pytest

from expert_index.retry import classify_failure


def test_rejected_is_always_dead_without_matching_message() -> None:
    result = classify_failure(
        error_stage="receipt",
        error_code="receipt_invalid",
        downstream_status="rejected",
        exception_type="ValueError",
        retryable=True,
    )
    assert result.disposition == "dead"
    assert result.reason_code == "receipt_invalid"


def test_failed_uses_explicit_retryable_flag() -> None:
    result = classify_failure(
        error_stage="scope_consumer",
        error_code="consumer_timeout",
        downstream_status="failed",
        exception_type="TimeoutError",
        retryable=True,
    )
    assert result.disposition == "retryable"


def test_classifier_rejects_unsafe_or_success_inputs() -> None:
    with pytest.raises(ValueError):
        classify_failure(
            error_stage="receipt\nsecret",
            error_code="receipt_invalid",
            downstream_status="rejected",
            exception_type=None,
            retryable=False,
        )
    with pytest.raises(ValueError):
        classify_failure(
            error_stage="receipt",
            error_code="receipt_invalid",
            downstream_status="committed",
            exception_type=None,
            retryable=False,
        )
