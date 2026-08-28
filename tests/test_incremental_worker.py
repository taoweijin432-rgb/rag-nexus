from __future__ import annotations

from dataclasses import dataclass

import pytest

from expert_index.incremental_worker import IncrementalWorker, IncrementalWorkerConfig


@dataclass(frozen=True)
class _Execution:
    failed_source_count: int = 0


@dataclass(frozen=True)
class _ScanCycle:
    failed_source_count: int = 0


@dataclass(frozen=True)
class _Result:
    execution: _Execution
    scan_cycle: _ScanCycle = _ScanCycle()


class _Runtime:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls: list[tuple[str, str]] = []

    def run_once(self, worker_id: str, *, cycle_id: str) -> _Result:
        self.calls.append((worker_id, cycle_id))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_worker_runs_cycles_and_uses_failure_backoff() -> None:
    runtime = _Runtime([_Result(_Execution()), _Result(_Execution(1))])
    waits: list[float] = []
    worker = IncrementalWorker(
        runtime,
        "poll-worker",
        config=IncrementalWorkerConfig(interval_seconds=5, error_sleep_seconds=2, max_cycles=2),
        wait=lambda seconds: waits.append(seconds) or False,
    )

    summary = worker.run()

    assert summary.cycles == 2
    assert summary.successful_cycles == 1
    assert summary.failed_cycles == 1
    assert summary.error_count == 0
    assert summary.stop_reason == "max_cycles"
    assert waits == [5, 2]
    assert all(worker_id == "poll-worker" for worker_id, _ in runtime.calls)


def test_worker_retries_runtime_exception() -> None:
    runtime = _Runtime([RuntimeError("temporary"), _Result(_Execution())])
    waits: list[float] = []
    worker = IncrementalWorker(
        runtime,
        "poll-worker",
        config=IncrementalWorkerConfig(error_sleep_seconds=3, max_cycles=2),
        wait=lambda seconds: waits.append(seconds) or False,
    )

    summary = worker.run()

    assert summary.error_count == 1
    assert summary.successful_cycles == 1
    assert summary.last_error_type == "RuntimeError"
    assert waits == [3, 60]


def test_worker_uses_failure_backoff_when_scan_fails() -> None:
    runtime = _Runtime([_Result(_Execution(), _ScanCycle(1))])
    waits: list[float] = []
    worker = IncrementalWorker(
        runtime,
        "poll-worker",
        config=IncrementalWorkerConfig(error_sleep_seconds=3, max_cycles=1),
        wait=lambda seconds: waits.append(seconds) or False,
    )

    summary = worker.run()

    assert summary.failed_cycles == 1
    assert summary.successful_cycles == 0
    assert waits == [3]


@pytest.mark.parametrize("bad_interval", [-1, float("inf"), True])
def test_worker_config_rejects_invalid_interval(bad_interval: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        IncrementalWorkerConfig(interval_seconds=bad_interval)  # type: ignore[arg-type]
