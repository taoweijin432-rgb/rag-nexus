"""清理后的正式协议和本地工具边界回归，不连接外部系统。"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from expert_index.source_batch_coordinator import PreparationFailureHandler
from scripts.sql_files import load_sql_statements
from tests import test_incremental_handoff_fake_e2e as e2e


def test_preparation_rejects_repository_without_atomic_dlq_protocol():
    legacy = SimpleNamespace(mark_source_sync_task_rejected=lambda: None,
                             mark_source_sync_task_retryable=lambda: None)
    with pytest.raises(TypeError, match="with_dlq"):
        PreparationFailureHandler(legacy, max_attempts=2, retry_delay_seconds=1)


@pytest.mark.parametrize("method", ["mark_source_sync_task_rejected_with_dlq",
                                    "mark_source_sync_task_retryable_with_dlq"])
def test_handoff_rejects_incomplete_atomic_protocol(method):
    repository = e2e.FakeTaskRepo(state=e2e._task())
    setattr(repository, method, None)
    with pytest.raises(TypeError, match="正式原子接口"):
        e2e._build_orchestrator(e2e.FakeVectorStore(), repository, e2e.FakeArtifactRepo())


def test_sql_file_loader_preserves_statement_order_and_comments(tmp_path):
    path = tmp_path / "migration.sql"
    path.write_text("-- heading\nCREATE TABLE sample (id INT);\n-- next\nSELECT 1;\n")
    assert load_sql_statements(path) == ["CREATE TABLE sample (id INT)", "SELECT 1"]
    for name in ("source_lease", "source_scan", "task_lease", "checkpoint_commit"):
        module = importlib.import_module("scripts.verify_p3_" + name)
        assert module.load_sql_statements is load_sql_statements
