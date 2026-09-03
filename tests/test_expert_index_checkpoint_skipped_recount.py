"""P5.6 CheckpointManager ``skipped_count`` 确定性重算测试。

验证 skipped_count 不再直接读 Batch Source 现有值，而是由 CheckpointManager 在锁定
全部成功任务时按冻结的 ``NO_WRITE_RESULT_CODES`` 重新统计，与 success_count 一起
原子写入 Batch Source（frozen #4/#5/#6）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.checkpoint import CheckpointManager
from expert_index.config import DatabaseConfig, DatabaseRole

START_TIME = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)
UPPER_TIME = datetime(2026, 8, 25, 8, 0, tzinfo=timezone.utc)
SUCCESS_TIME = datetime(2026, 8, 25, 9, 0, tzinfo=timezone.utc)
LEASE_UNTIL = datetime(2026, 8, 25, 9, 5, tzinfo=timezone.utc)
SOURCE_TABLE = "experts_work_experience"
BATCH_ID = 901
LEASE_OWNER = "checkpoint-worker-1"


class ScriptedCursor:
    def __init__(
        self,
        *,
        one_results: list[dict[str, Any]] | None = None,
        all_results: list[list[dict[str, Any]]] | None = None,
        rowcounts: list[int] | None = None,
    ) -> None:
        self.one_results = list(one_results or [])
        self.all_results = list(all_results or [])
        self.rowcounts = list(rowcounts or [])
        self.rowcount = 0
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append((" ".join(sql.split()), params))
        self.rowcount = self.rowcounts.pop(0) if self.rowcounts else 0

    def fetchone(self) -> dict[str, Any]:
        return self.one_results.pop(0)

    def fetchall(self) -> list[dict[str, Any]]:
        return self.all_results.pop(0)


class ScriptedConnection:
    def __init__(self, cursor: ScriptedCursor) -> None:
        self.cursor_instance = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> ScriptedCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def _manager() -> CheckpointManager:
    config = DatabaseConfig(
        "127.0.0.1", 3306, "test", "test", "index_test", role=DatabaseRole.INDEX,
    )
    return CheckpointManager(config)


def _patch_connection(monkeypatch: pytest.MonkeyPatch, connection: ScriptedConnection) -> None:
    monkeypatch.setattr(
        "expert_index.checkpoint.pymysql.connect", lambda **_kwargs: connection,
    )


def _checkpoint_row() -> dict[str, Any]:
    return {
        "source_table": SOURCE_TABLE, "last_updated_at": START_TIME, "last_id": 0,
        "last_batch_id": None, "last_success_time": None, "lock_version": 0,
        "lease_owner": LEASE_OWNER, "lease_until": LEASE_UNTIL, "lease_is_active": 1,
    }


def _committed_checkpoint_row() -> dict[str, Any]:
    return {
        "source_table": SOURCE_TABLE, "last_updated_at": UPPER_TIME, "last_id": 902,
        "last_batch_id": BATCH_ID, "last_success_time": SUCCESS_TIME, "lock_version": 1,
        "lease_owner": None, "lease_until": None, "lease_is_active": 0,
    }


def _batch_source_row(candidate_count: int = 4) -> dict[str, Any]:
    return {
        "id": 1001, "batch_id": BATCH_ID, "source_table": SOURCE_TABLE,
        "start_updated_at": START_TIME, "start_id": 0,
        "cursor_updated_at": UPPER_TIME, "cursor_id": 902,
        "upper_bound_updated_at": UPPER_TIME, "upper_bound_id": 902,
        "checkpoint_lock_version": 0, "status": "running",
        "candidate_count": candidate_count, "success_count": 0,
        "failed_count": 0, "skipped_count": 0,
        "started_at": UPPER_TIME, "finished_at": None,
    }


def _batch_row(candidate_count: int = 4) -> dict[str, Any]:
    return {
        "id": BATCH_ID, "run_mode": "incremental", "status": "running",
        "cursor_updated_at": None, "cursor_id": None,
        "candidate_count": 0, "success_count": 0, "failed_count": 0,
        "error_summary": None, "finished_at": None,
    }


def _success_task(task_id: int, result_code: str) -> dict[str, Any]:
    return {
        "id": task_id, "processing_stage": "completed", "sync_status": "success",
        "result_code": result_code, "attempt_token": None, "lease_until": None,
        "completed_at": SUCCESS_TIME,
    }


def _cursor_with_tasks(tasks: list[dict[str, Any]], candidate_count: int = 4) -> ScriptedCursor:
    return ScriptedCursor(
        one_results=[
            _checkpoint_row(), _batch_source_row(candidate_count), _batch_row(candidate_count),
            _committed_checkpoint_row(),
        ],
        all_results=[tasks],
        rowcounts=[0, 0, 0, 0, 1, 1, 1, 0],
    )


# ---------------------------------------------------------------------------
# skipped_count 确定性重算（frozen #5：NO_WRITE_RESULT_CODES = {"no_action_unchanged"}）
# ---------------------------------------------------------------------------

def test_skipped_count_recomputed_from_no_action_unchanged_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只有 no_action_unchanged 计入 skipped_count。"""

    tasks = [
        _success_task(801, "embedded"),
        _success_task(802, "no_action_unchanged"),
        _success_task(803, "metadata_updated"),
        _success_task(804, "embedded"),
    ]
    cursor = _cursor_with_tasks(tasks)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE, BATCH_ID, LEASE_OWNER, checkpoint_lock_version=0,
    )

    assert result.skipped_count == 1
    # batch_source UPDATE 的第二个参数是重算的 skipped_count；第一个是 success_count。
    assert cursor.executed[4][1][1] == 1


def test_skipped_count_zero_when_no_no_action_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """全部有写入效果 → skipped_count=0。"""

    tasks = [
        _success_task(801, "embedded"),
        _success_task(802, "metadata_updated"),
        _success_task(803, "disabled"),
        _success_task(804, "mixed_mutation"),
    ]
    cursor = _cursor_with_tasks(tasks)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE, BATCH_ID, LEASE_OWNER, checkpoint_lock_version=0,
    )

    assert result.skipped_count == 0


def test_skipped_count_counts_all_no_action_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """多条 no_action_unchanged 全部计入。"""

    tasks = [
        _success_task(801, "no_action_unchanged"),
        _success_task(802, "no_action_unchanged"),
        _success_task(803, "embedded"),
        _success_task(804, "no_action_unchanged"),
    ]
    cursor = _cursor_with_tasks(tasks)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE, BATCH_ID, LEASE_OWNER, checkpoint_lock_version=0,
    )

    assert result.skipped_count == 3
    assert cursor.executed[4][1][1] == 3


# ---------------------------------------------------------------------------
# frozen #4：recovered_unchanged 不计 skipped
# ---------------------------------------------------------------------------

def test_recovered_unchanged_not_counted_as_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """recovered_unchanged 属幂等重放已收敛，不计入 skipped_count。"""

    tasks = [
        _success_task(801, "recovered_unchanged"),
        _success_task(802, "recovered_unchanged"),
        _success_task(803, "embedded"),
        _success_task(804, "no_action_unchanged"),
    ]
    cursor = _cursor_with_tasks(tasks)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE, BATCH_ID, LEASE_OWNER, checkpoint_lock_version=0,
    )

    assert result.skipped_count == 1


# ---------------------------------------------------------------------------
# frozen #6：expert_deleted/no_chunkable_content 不计 skipped
# ---------------------------------------------------------------------------

def test_expert_deleted_not_counted_as_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """expert_deleted 是 P4 零写入业务终态，不在 NO_WRITE_RESULT_CODES 内。"""

    tasks = [
        _success_task(801, "expert_deleted"),
        _success_task(802, "no_chunkable_content"),
        _success_task(803, "embedded"),
        _success_task(804, "no_action_unchanged"),
    ]
    cursor = _cursor_with_tasks(tasks)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE, BATCH_ID, LEASE_OWNER, checkpoint_lock_version=0,
    )

    assert result.skipped_count == 1


def test_p4_business_result_codes_not_in_no_write_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """expert_deleted/no_chunkable_content 单独构成批次时 skipped_count=0。"""

    tasks = [
        _success_task(801, "expert_deleted"),
        _success_task(802, "no_chunkable_content"),
        _success_task(803, "expert_deleted"),
        _success_task(804, "no_chunkable_content"),
    ]
    cursor = _cursor_with_tasks(tasks)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE, BATCH_ID, LEASE_OWNER, checkpoint_lock_version=0,
    )

    assert result.skipped_count == 0


# ---------------------------------------------------------------------------
# 原 Batch Source 现有 skipped_count 值被覆盖（不再透传）
# ---------------------------------------------------------------------------

def test_batch_source_stored_skipped_count_is_overwritten_by_recomputed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch Source 行里的 skipped_count 旧值不参与，重算值原子写入。"""

    tasks = [
        _success_task(801, "embedded"),
        _success_task(802, "embedded"),
        _success_task(803, "embedded"),
        _success_task(804, "embedded"),
    ]
    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(),
            {**_batch_source_row(), "skipped_count": 99},  # 旧值 99 应被忽略
            _batch_row(),
            _committed_checkpoint_row(),
        ],
        all_results=[tasks],
        rowcounts=[0, 0, 0, 0, 1, 1, 1, 0],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE, BATCH_ID, LEASE_OWNER, checkpoint_lock_version=0,
    )

    assert result.skipped_count == 0
    assert cursor.executed[4][1][1] == 0
