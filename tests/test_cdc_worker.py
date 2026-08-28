from __future__ import annotations

from dataclasses import dataclass

import pytest

from expert_index.cdc import (
    CdcAckPendingError,
    CdcBatchRunResult,
    CdcWorker,
    CdcWorkerConfig,
)


@dataclass
class _Runtime:
    results: list[object]

    def consume_once(self, *, batch_size: int, local_batch_id: int) -> CdcBatchRunResult:
        assert batch_size == 2
        assert local_batch_id > 0
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _result(status: str) -> CdcBatchRunResult:
    return CdcBatchRunResult(status, 10, 1, 0)


def test_worker_waits_for_empty_and_counts_acked_batch() -> None:
    runtime = _Runtime([_result("empty"), _result("acked")])
    waits: list[float] = []
    worker = CdcWorker(
        runtime,
        iter((1, 2)).__next__,
        config=CdcWorkerConfig(batch_size=2, idle_sleep_seconds=3, max_cycles=2),
        wait=lambda seconds: waits.append(seconds) or False,
    )

    summary = worker.run()

    assert summary.cycles == 2
    assert summary.empty_count == 1
    assert summary.acked_count == 1
    assert summary.error_count == 0
    assert summary.stop_reason == "max_cycles"
    assert waits == [3]


def test_worker_reconnects_and_retries_ack_pending_batch() -> None:
    runtime = _Runtime([CdcAckPendingError(10, object()), _result("acked")])
    waits: list[float] = []
    worker = CdcWorker(
        runtime,
        lambda: 1,
        config=CdcWorkerConfig(batch_size=2, max_cycles=2),
        wait=lambda seconds: waits.append(seconds) or False,
    )

    summary = worker.run()

    assert summary.cycles == 2
    assert summary.error_count == 1
    assert summary.acked_count == 1
    assert summary.stop_reason == "max_cycles"
    assert summary.last_error_type == "CdcAckPendingError"
    assert waits == [10]


def test_worker_retries_transient_error_with_backoff() -> None:
    runtime = _Runtime([RuntimeError("temporary"), _result("acked")])
    waits: list[float] = []
    worker = CdcWorker(
        runtime,
        lambda: 7,
        config=CdcWorkerConfig(batch_size=2, error_sleep_seconds=4, max_cycles=2),
        wait=lambda seconds: waits.append(seconds) or False,
    )

    summary = worker.run()

    assert summary.cycles == 2
    assert summary.error_count == 1
    assert summary.acked_count == 1
    assert summary.last_error_type == "RuntimeError"
    assert waits == [4]


@pytest.mark.parametrize("bad_id", [0, -1, True, "1"])
def test_worker_rejects_invalid_local_batch_id(bad_id: object) -> None:
    runtime = _Runtime([])
    worker = CdcWorker(
        runtime,
        lambda: bad_id,
        config=CdcWorkerConfig(max_cycles=1),
        wait=lambda _seconds: False,
    )

    summary = worker.run()

    assert summary.error_count == 1
    assert summary.last_error_type in {"ValueError", "TypeError"}
