"""确保真实 CLI 接入信号处理，且当前轮次收口后停止。"""
from __future__ import annotations

import json
import signal
from types import SimpleNamespace

from expert_index import cli
from expert_index.incremental_worker import IncrementalWorkerConfig, run_incremental_worker


def test_cli_worker_uses_signal_aware_entry(monkeypatch, capsys):
    from tests.test_incremental_handoff_fake_e2e import _settings

    monkeypatch.setattr(cli.Settings, "from_env", lambda: _settings())
    runtime = object()
    calls = []
    monkeypatch.setattr(cli, "build_incremental_runtime", lambda *a, **kw: runtime)

    def run(actual, worker_id, *, config):
        calls.append((actual, worker_id, config))
        from expert_index.incremental_worker import IncrementalWorkerSummary
        return IncrementalWorkerSummary(1, 1, 0, 0, "stop_requested")

    monkeypatch.setattr(cli, "run_incremental_worker", run)
    cli.main(["incremental-worker", "--worker-id", "poll-test", "--max-attempts", "3",
              "--retry-delay-seconds", "1", "--task-lease-seconds", "120"])
    assert calls[0][0] is runtime and calls[0][1] == "poll-test"
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "stop_requested"


def test_sigterm_finishes_current_round_before_stopping(monkeypatch):
    handlers, operations = {}, []
    monkeypatch.setattr(signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler))

    class Runtime:
        def run_once(self, worker_id, *, cycle_id):
            operations.append("started")
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            operations.append("checkpoint_committed")
            return SimpleNamespace(scan_cycle=SimpleNamespace(failed_source_count=0),
                                   execution=SimpleNamespace(failed_source_count=0))

    result = run_incremental_worker(Runtime(), "poll-test",
                                    config=IncrementalWorkerConfig(max_cycles=3))
    assert operations == ["started", "checkpoint_committed"]
    assert result.cycles == result.successful_cycles == 1
    assert result.stop_reason == "stop_requested"
