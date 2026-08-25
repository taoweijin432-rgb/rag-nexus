"""增量来源批次安全 Checkpoint 提交事务测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.checkpoint import (
    CheckpointManager,
    SourceBatchCommitConflictError,
    SourceBatchNotReadyError,
)
from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.models import SourceScanCursor

START_TIME = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)
UPPER_TIME = datetime(2026, 8, 25, 8, 0, tzinfo=timezone.utc)
SUCCESS_TIME = datetime(2026, 8, 25, 9, 0, tzinfo=timezone.utc)
LEASE_UNTIL = datetime(2026, 8, 25, 9, 5, tzinfo=timezone.utc)
SOURCE_TABLE = "experts_work_experience"
BATCH_ID = 901
LEASE_OWNER = "checkpoint-worker-1"


class ScriptedCursor:
    """按 SQL 执行顺序返回预设结果并记录 rowcount。"""

    def __init__(
        self,
        *,
        one_results: list[dict[str, Any] | None],
        all_results: list[list[dict[str, Any]]],
        rowcounts: list[int] | None = None,
    ) -> None:
        self.one_results = list(one_results)
        self.all_results = list(all_results)
        self.rowcounts = list(rowcounts or [])
        self.rowcount = 0
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        """记录压缩后的 SQL，并切换本次 UPDATE 的 rowcount。"""

        self.executed.append((" ".join(sql.split()), params))
        self.rowcount = self.rowcounts.pop(0) if self.rowcounts else 0

    def fetchone(self) -> dict[str, Any] | None:
        return self.one_results.pop(0)

    def fetchall(self) -> list[dict[str, Any]]:
        return self.all_results.pop(0)


class ScriptedConnection:
    """记录 Checkpoint Manager 的提交、回滚和连接关闭。"""

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
    """构造只允许连接虚构 Index Schema 的 Checkpoint Manager。"""

    return CheckpointManager(
        DatabaseConfig(
            "127.0.0.1",
            3306,
            "test",
            "test",
            "index_test",
            role=DatabaseRole.INDEX,
        )
    )


def _patch_connection(
    monkeypatch: pytest.MonkeyPatch,
    connection: ScriptedConnection,
) -> None:
    """让一次安全提交使用指定的脚本化连接。"""

    monkeypatch.setattr(
        "expert_index.checkpoint.pymysql.connect",
        lambda **_kwargs: connection,
    )


def _checkpoint_row(**changes: object) -> dict[str, Any]:
    """构造尚未推进且持有有效来源 lease 的 Checkpoint 行。"""

    values: dict[str, Any] = {
        "source_table": SOURCE_TABLE,
        "last_updated_at": START_TIME,
        "last_id": 0,
        "last_batch_id": None,
        "last_success_time": None,
        "lock_version": 0,
        "lease_owner": LEASE_OWNER,
        "lease_until": LEASE_UNTIL,
        "lease_is_active": 1,
    }
    values.update(changes)
    return values


def _batch_source_row(**changes: object) -> dict[str, Any]:
    """构造已封口但尚未提交的 Batch Source 行。"""

    values: dict[str, Any] = {
        "id": 1001,
        "batch_id": BATCH_ID,
        "source_table": SOURCE_TABLE,
        "start_updated_at": START_TIME,
        "start_id": 0,
        "cursor_updated_at": UPPER_TIME,
        "cursor_id": 902,
        "upper_bound_updated_at": UPPER_TIME,
        "upper_bound_id": 902,
        "checkpoint_lock_version": 0,
        "status": "running",
        "candidate_count": 2,
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "started_at": UPPER_TIME,
        "finished_at": None,
    }
    values.update(changes)
    return values


def _batch_row(**changes: object) -> dict[str, Any]:
    """构造与 Batch Source 一一对应的 incremental 批次主记录。"""

    values: dict[str, Any] = {
        "id": BATCH_ID,
        "run_mode": "incremental",
        "status": "running",
        "cursor_updated_at": None,
        "cursor_id": None,
        "candidate_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "error_summary": None,
        "finished_at": None,
    }
    values.update(changes)
    return values


def _success_tasks() -> list[dict[str, Any]]:
    """构造两条无残留 attempt lease 的可靠 success 任务证据。"""

    return [
        {
            "id": 801,
            "processing_stage": "completed",
            "sync_status": "success",
            "result_code": "downstream_applied",
            "attempt_token": None,
            "lease_until": None,
            "completed_at": SUCCESS_TIME,
        },
        {
            "id": 802,
            "processing_stage": "completed",
            "sync_status": "success",
            "result_code": "hash_unchanged",
            "attempt_token": None,
            "lease_until": None,
            "completed_at": SUCCESS_TIME,
        },
    ]


def _committed_checkpoint_row() -> dict[str, Any]:
    """构造首次提交 UPDATE 后的 Checkpoint 查询行。"""

    return _checkpoint_row(
        last_updated_at=UPPER_TIME,
        last_id=902,
        last_batch_id=BATCH_ID,
        last_success_time=SUCCESS_TIME,
        lock_version=1,
        lease_owner=None,
        lease_until=None,
        lease_is_active=0,
    )


def _commit_cursor(
    *,
    checkpoint: dict[str, Any] | None = None,
    batch_source: dict[str, Any] | None = None,
    batch: dict[str, Any] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    rowcounts: list[int] | None = None,
) -> ScriptedCursor:
    """构造一次首次提交所需的完整 SQL 结果序列。"""

    return ScriptedCursor(
        one_results=[
            checkpoint or _checkpoint_row(),
            batch_source or _batch_source_row(),
            batch or _batch_row(),
            _committed_checkpoint_row(),
        ],
        all_results=[tasks if tasks is not None else _success_tasks()],
        rowcounts=rowcounts or [0, 0, 0, 0, 1, 1, 1, 0],
    )


def test_try_commit_source_batch_atomically_closes_all_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全部 success 后应在一个事务内收口两级批次并推进安全水位。"""

    cursor = _commit_cursor()
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE,
        BATCH_ID,
        LEASE_OWNER,
        checkpoint_lock_version=0,
    )

    assert result.already_committed is False
    assert result.candidate_count == 2
    assert result.checkpoint.last_updated_at == UPPER_TIME
    assert result.checkpoint.last_id == 902
    assert result.checkpoint.last_batch_id == BATCH_ID
    assert result.checkpoint.lock_version == 1
    assert result.checkpoint.lease_owner is None
    assert ["FOR UPDATE" in sql for sql, _ in cursor.executed[:4]] == [True] * 4
    assert "ORDER BY id ASC FOR UPDATE" in cursor.executed[3][0]
    assert "status='success'" in cursor.executed[4][0]
    assert cursor.executed[4][1] == (1001, BATCH_ID, SOURCE_TABLE, 0)
    assert cursor.executed[5][1] == (
        UPPER_TIME,
        902,
        2,
        2,
        BATCH_ID,
    )
    assert cursor.executed[6][1] == (
        UPPER_TIME,
        902,
        BATCH_ID,
        SOURCE_TABLE,
        START_TIME,
        0,
        0,
        LEASE_OWNER,
    )
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True


@pytest.mark.parametrize(
    "sync_status",
    ["pending", "processing", "failed", "retrying", "dead"],
)
def test_try_commit_source_batch_rejects_any_non_success_task(
    monkeypatch: pytest.MonkeyPatch,
    sync_status: str,
) -> None:
    """任意未成功状态都必须在写批次或 Checkpoint 前回滚。"""

    tasks = _success_tasks()
    tasks[1] = dict(tasks[1], sync_status=sync_status)
    cursor = _commit_cursor(tasks=tasks)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceBatchNotReadyError, match=sync_status):
        _manager().try_commit_source_batch(
            SOURCE_TABLE,
            BATCH_ID,
            LEASE_OWNER,
            checkpoint_lock_version=0,
        )

    assert len(cursor.executed) == 4
    assert connection.commit_count == 0
    assert connection.rollback_count == 1


def test_try_commit_source_batch_rejects_task_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任务数与扫描候选数不一致时不能推进安全水位。"""

    cursor = _commit_cursor(tasks=_success_tasks()[:1])
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceBatchNotReadyError, match="任务总数"):
        _manager().try_commit_source_batch(
            SOURCE_TABLE,
            BATCH_ID,
            LEASE_OWNER,
            checkpoint_lock_version=0,
        )

    assert len(cursor.executed) == 4
    assert connection.rollback_count == 1


@pytest.mark.parametrize(
    "task_change",
    [
        {"result_code": None},
        {"result_code": "contains spaces"},
        {"processing_stage": "handed_off"},
        {"completed_at": None},
        {"attempt_token": "stale-attempt", "lease_until": LEASE_UNTIL},
    ],
)
def test_try_commit_source_batch_rejects_incomplete_success_evidence(
    monkeypatch: pytest.MonkeyPatch,
    task_change: dict[str, object],
) -> None:
    """success 缺少安全结果码或残留 attempt lease 时不能提交。"""

    tasks = _success_tasks()
    tasks[0] = dict(tasks[0], **task_change)
    cursor = _commit_cursor(tasks=tasks)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceBatchNotReadyError):
        _manager().try_commit_source_batch(
            SOURCE_TABLE,
            BATCH_ID,
            LEASE_OWNER,
            checkpoint_lock_version=0,
        )

    assert len(cursor.executed) == 4
    assert connection.rollback_count == 1


def test_try_commit_source_batch_requires_sealed_fixed_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """临时游标未到上界时即使任务全 success 也不能提交。"""

    batch_source = _batch_source_row(
        cursor_updated_at=UPPER_TIME,
        cursor_id=901,
    )
    cursor = _commit_cursor(batch_source=batch_source)
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceBatchNotReadyError, match="尚未到达固定上界"):
        _manager().try_commit_source_batch(
            SOURCE_TABLE,
            BATCH_ID,
            LEASE_OWNER,
            checkpoint_lock_version=0,
        )

    assert connection.rollback_count == 1


def test_try_commit_source_batch_rejects_invalid_skipped_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未冻结分类规则前也不能接受超过候选数的 skipped_count。"""

    cursor = _commit_cursor(batch_source=_batch_source_row(skipped_count=3))
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceBatchCommitConflictError, match="skipped_count"):
        _manager().try_commit_source_batch(
            SOURCE_TABLE,
            BATCH_ID,
            LEASE_OWNER,
            checkpoint_lock_version=0,
        )

    assert connection.rollback_count == 1


@pytest.mark.parametrize(
    "checkpoint_change",
    [
        {"lease_owner": "other-worker"},
        {"lease_is_active": 0},
        {"lock_version": 1},
        {"last_id": 1},
    ],
)
def test_try_commit_source_batch_rejects_stale_checkpoint_owner_or_version(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_change: dict[str, object],
) -> None:
    """旧 owner、过期 lease、版本变化和起点变化都必须 fail-closed。"""

    cursor = _commit_cursor(checkpoint=_checkpoint_row(**checkpoint_change))
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceBatchCommitConflictError):
        _manager().try_commit_source_batch(
            SOURCE_TABLE,
            BATCH_ID,
            LEASE_OWNER,
            checkpoint_lock_version=0,
        )

    assert connection.rollback_count == 1


def test_try_commit_source_batch_rolls_back_all_updates_on_checkpoint_cas_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最后一步 Checkpoint CAS 失败时前两个状态更新也必须回滚。"""

    cursor = _commit_cursor(rowcounts=[0, 0, 0, 0, 1, 1, 0])
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceBatchCommitConflictError, match="乐观锁"):
        _manager().try_commit_source_batch(
            SOURCE_TABLE,
            BATCH_ID,
            LEASE_OWNER,
            checkpoint_lock_version=0,
        )

    assert len(cursor.executed) == 7
    assert connection.commit_count == 0
    assert connection.rollback_count == 1


def test_try_commit_source_batch_is_idempotent_after_committed_response_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相同批次已完整提交时应只核验证据并返回，不要求残留 lease。"""

    checkpoint = _committed_checkpoint_row()
    batch_source = _batch_source_row(
        status="success",
        success_count=2,
        finished_at=SUCCESS_TIME,
    )
    batch = _batch_row(
        status="success",
        cursor_updated_at=UPPER_TIME,
        cursor_id=902,
        candidate_count=2,
        success_count=2,
        finished_at=SUCCESS_TIME,
    )
    cursor = ScriptedCursor(
        one_results=[checkpoint, batch_source, batch],
        all_results=[_success_tasks()],
        rowcounts=[0, 0, 0, 0],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    result = _manager().try_commit_source_batch(
        SOURCE_TABLE,
        BATCH_ID,
        "old-response-owner",
        checkpoint_lock_version=0,
    )

    assert result.already_committed is True
    assert result.checkpoint.last_batch_id == BATCH_ID
    assert len(cursor.executed) == 4
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


@pytest.mark.parametrize(
    ("source_table", "batch_id", "lease_owner", "lock_version"),
    [
        ("unknown", BATCH_ID, LEASE_OWNER, 0),
        (SOURCE_TABLE, 0, LEASE_OWNER, 0),
        (SOURCE_TABLE, BATCH_ID, "", 0),
        (SOURCE_TABLE, BATCH_ID, LEASE_OWNER, -1),
    ],
)
def test_try_commit_source_batch_rejects_invalid_arguments_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
    source_table: str,
    batch_id: int,
    lease_owner: str,
    lock_version: int,
) -> None:
    """非法表名、批次、owner 或版本不得触发 Index MySQL 连接。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.checkpoint.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError):
        _manager().try_commit_source_batch(
            source_table,
            batch_id,
            lease_owner,
            checkpoint_lock_version=lock_version,
        )


def test_commit_result_exposes_exact_composite_cursor() -> None:
    """测试常量应能构造提交目标的精确复合位置。"""

    assert SourceScanCursor(UPPER_TIME, 902) > SourceScanCursor(START_TIME, 0)
