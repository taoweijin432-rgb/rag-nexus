"""Checkpoint lease 与未完成来源批次恢复的 Repository 测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.models import BatchSourceScan, SourceCheckpoint, SourceScanCursor
from expert_index.repositories import (
    BatchSourceRecoveryConflictError,
    MySQLSourceStateRepository,
    SourceCheckpointNotInitializedError,
    SourceLeaseConflictError,
)

CHECKPOINT_TIME = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)
LEASE_UNTIL = datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc)
EXTENDED_LEASE_UNTIL = datetime(2026, 8, 24, 13, 5, tzinfo=timezone.utc)


class ScriptedCursor:
    """按执行顺序返回预设结果，并记录 SQL、参数和 rowcount。"""

    def __init__(
        self,
        *,
        one_results: list[dict[str, Any] | None] | None = None,
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
        """记录压缩 SQL，并切换为本次执行对应的 rowcount。"""

        self.executed.append((" ".join(sql.split()), params))
        self.rowcount = self.rowcounts.pop(0) if self.rowcounts else 0

    def fetchone(self) -> dict[str, Any] | None:
        return self.one_results.pop(0)

    def fetchall(self) -> list[dict[str, Any]]:
        return self.all_results.pop(0)


class ScriptedConnection:
    """记录 Repository 事务提交、回滚和连接关闭。"""

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


def _repository() -> MySQLSourceStateRepository:
    """创建不会主动连接真实数据库的 Index Repository。"""

    return MySQLSourceStateRepository(
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
    """让一次来源状态操作使用指定 Scripted Connection。"""

    monkeypatch.setattr(
        "expert_index.repositories.source_state.pymysql.connect",
        lambda **_kwargs: connection,
    )


def _checkpoint_row(
    *,
    lease_owner: str | None = None,
    lease_until: datetime | None = None,
    lease_is_active: int = 0,
) -> dict[str, Any]:
    """构造一行完整 Checkpoint 查询结果。"""

    return {
        "source_table": "experts",
        "last_updated_at": CHECKPOINT_TIME,
        "last_id": 0,
        "last_batch_id": None,
        "last_success_time": None,
        "lock_version": 0,
        "lease_owner": lease_owner,
        "lease_until": lease_until,
        "lease_is_active": lease_is_active,
    }


def _batch_source_row(*, batch_id: int = 901) -> dict[str, Any]:
    """构造一条可恢复的固定来源扫描范围。"""

    return {
        "id": 11,
        "batch_id": batch_id,
        "source_table": "experts",
        "start_updated_at": CHECKPOINT_TIME,
        "start_id": 0,
        "cursor_updated_at": datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc),
        "cursor_id": 100,
        "upper_bound_updated_at": datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc),
        "upper_bound_id": 300,
        "checkpoint_lock_version": 0,
        "status": "running",
        "candidate_count": 100,
        "success_count": 80,
        "failed_count": 1,
        "skipped_count": 19,
        "started_at": datetime(2026, 8, 24, 8, 1, tzinfo=timezone.utc),
        "finished_at": None,
    }


def test_acquire_source_lease_uses_database_clock_and_returns_persisted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空闲 Checkpoint 应在行锁内认领，并返回数据库计算的 lease_until。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(),
            _checkpoint_row(
                lease_owner="scanner-attempt-1",
                lease_until=LEASE_UNTIL,
            ),
        ],
        rowcounts=[1, 1, 1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    checkpoint = _repository().acquire_source_lease(
        "experts",
        "scanner-attempt-1",
        lease_seconds=60,
    )

    assert checkpoint == SourceCheckpoint(
        source_table="experts",
        last_updated_at=CHECKPOINT_TIME,
        last_id=0,
        lease_owner="scanner-attempt-1",
        lease_until=LEASE_UNTIL,
    )
    lock_sql, lock_params = cursor.executed[0]
    update_sql, update_params = cursor.executed[1]
    assert "CURRENT_TIMESTAMP(6)" in lock_sql
    assert "FOR UPDATE" in lock_sql
    assert lock_params == ("experts",)
    assert "TIMESTAMPADD( SECOND, %s, CURRENT_TIMESTAMP(6) )" in update_sql
    assert "lease_until <= CURRENT_TIMESTAMP(6)" in update_sql
    assert update_params == ("scanner-attempt-1", 60, "experts")
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True


def test_acquire_source_lease_rejects_active_lease_even_for_same_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有效 lease 必须走显式续租入口，不能被 acquire 覆盖。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner="scanner-attempt-1",
                lease_until=LEASE_UNTIL,
                lease_is_active=1,
            )
        ],
        rowcounts=[1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceLeaseConflictError, match="仍在有效期内"):
        _repository().acquire_source_lease(
            "experts",
            "scanner-attempt-1",
            lease_seconds=60,
        )

    assert len(cursor.executed) == 1
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.closed is True


def test_acquire_source_lease_requires_initialized_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少显式初始水位时不能自动创建或猜测 Checkpoint。"""

    cursor = ScriptedCursor(one_results=[None], rowcounts=[0])
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceCheckpointNotInitializedError, match="尚未初始化"):
        _repository().acquire_source_lease(
            "experts",
            "scanner-attempt-1",
            lease_seconds=60,
        )

    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.closed is True


def test_renew_source_lease_requires_matching_active_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有效 owner 可以续租，且新到期时间必须来自 Index MySQL。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner="scanner-attempt-1",
                lease_until=LEASE_UNTIL,
                lease_is_active=1,
            ),
            _checkpoint_row(
                lease_owner="scanner-attempt-1",
                lease_until=EXTENDED_LEASE_UNTIL,
            ),
        ],
        rowcounts=[1, 1, 1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    checkpoint = _repository().renew_source_lease(
        "experts",
        "scanner-attempt-1",
        lease_seconds=120,
    )

    assert checkpoint.lease_until == EXTENDED_LEASE_UNTIL
    update_sql, update_params = cursor.executed[1]
    assert "lease_owner=%s" in update_sql
    assert "lease_until > CURRENT_TIMESTAMP(6)" in update_sql
    assert update_params == (120, "experts", "scanner-attempt-1")
    assert connection.commit_count == 1


@pytest.mark.parametrize(
    ("persisted_owner", "lease_is_active"),
    [("new-owner", 1), ("scanner-attempt-1", 0)],
)
def test_renew_source_lease_rejects_old_or_expired_owner(
    monkeypatch: pytest.MonkeyPatch,
    persisted_owner: str,
    lease_is_active: int,
) -> None:
    """旧 owner 和已过期 owner 都不能写入新的到期时间。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner=persisted_owner,
                lease_until=LEASE_UNTIL,
                lease_is_active=lease_is_active,
            )
        ],
        rowcounts=[1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceLeaseConflictError, match="不再拥有"):
        _repository().renew_source_lease(
            "experts",
            "scanner-attempt-1",
            lease_seconds=60,
        )

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


def test_release_source_lease_clears_only_matching_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """owner 匹配时可释放 lease，但不能改变安全游标和 lock_version。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner="scanner-attempt-1",
                lease_until=LEASE_UNTIL,
                lease_is_active=1,
            ),
            _checkpoint_row(),
        ],
        rowcounts=[1, 1, 1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    checkpoint = _repository().release_source_lease(
        "experts",
        "scanner-attempt-1",
    )

    assert checkpoint.last_updated_at == CHECKPOINT_TIME
    assert checkpoint.last_id == 0
    assert checkpoint.lock_version == 0
    assert checkpoint.lease_owner is None
    assert checkpoint.lease_until is None
    update_sql, update_params = cursor.executed[1]
    assert "SET lease_owner=NULL, lease_until=NULL" in update_sql
    assert update_params == ("experts", "scanner-attempt-1")
    assert connection.commit_count == 1


@pytest.mark.parametrize(
    ("persisted_owner", "lease_is_active"),
    [("scanner-attempt-2", 1), ("scanner-attempt-1", 0)],
)
def test_release_source_lease_rejects_previous_or_expired_owner(
    monkeypatch: pytest.MonkeyPatch,
    persisted_owner: str,
    lease_is_active: int,
) -> None:
    """旧 owner 和 lease 已过期的 owner 都不能释放当前租约字段。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner=persisted_owner,
                lease_until=LEASE_UNTIL,
                lease_is_active=lease_is_active,
            )
        ],
        rowcounts=[1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceLeaseConflictError, match="不再拥有"):
        _repository().release_source_lease("experts", "scanner-attempt-1")

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


def test_get_recoverable_batch_source_scan_returns_unique_unfinished_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有效 lease owner 应恢复原批次范围、临时游标和计数。"""

    row = _batch_source_row()
    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner="scanner-attempt-2",
                lease_until=LEASE_UNTIL,
                lease_is_active=1,
            )
        ],
        all_results=[[row]],
        rowcounts=[1, 1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    scan = _repository().get_recoverable_batch_source_scan(
        "experts",
        "scanner-attempt-2",
    )

    assert scan == BatchSourceScan(
        batch_id=901,
        source_table="experts",
        start_cursor=SourceScanCursor(CHECKPOINT_TIME, 0),
        current_cursor=SourceScanCursor(
            datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc),
            100,
        ),
        upper_bound=SourceScanCursor(
            datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc),
            300,
        ),
        checkpoint_lock_version=0,
        status="running",
        candidate_count=100,
        success_count=80,
        failed_count=1,
        skipped_count=19,
        started_at=datetime(2026, 8, 24, 8, 1, tzinfo=timezone.utc),
    )
    query_sql, query_params = cursor.executed[1]
    assert "source_table=%s" in query_sql
    assert "status IN ('pending', 'running', 'failed')" in query_sql
    assert "ORDER BY id ASC LIMIT 2 FOR UPDATE" in query_sql
    assert query_params == ("experts",)
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


def test_get_recoverable_batch_source_scan_returns_none_without_unfinished_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有未完成范围时返回 None，供后续 Scanner 决定是否创建新批次。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner="scanner-attempt-2",
                lease_until=LEASE_UNTIL,
                lease_is_active=1,
            )
        ],
        all_results=[[]],
        rowcounts=[1, 0],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    assert (
        _repository().get_recoverable_batch_source_scan(
            "experts",
            "scanner-attempt-2",
        )
        is None
    )
    assert connection.commit_count == 1


def test_get_recoverable_batch_source_scan_rejects_overlapping_unfinished_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一来源存在两条未完成范围时必须停止，不能任意恢复其中一条。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner="scanner-attempt-2",
                lease_until=LEASE_UNTIL,
                lease_is_active=1,
            )
        ],
        all_results=[[_batch_source_row(), _batch_source_row(batch_id=902)]],
        rowcounts=[1, 2],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(BatchSourceRecoveryConflictError, match="多个未完成"):
        _repository().get_recoverable_batch_source_scan(
            "experts",
            "scanner-attempt-2",
        )

    assert connection.commit_count == 0
    assert connection.rollback_count == 1


def test_get_recoverable_batch_source_scan_rejects_stale_checkpoint_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未完成范围的起点版本与安全 Checkpoint 不一致时必须停止恢复。"""

    checkpoint = _checkpoint_row(
        lease_owner="scanner-attempt-2",
        lease_until=LEASE_UNTIL,
        lease_is_active=1,
    )
    checkpoint["lock_version"] = 1
    cursor = ScriptedCursor(
        one_results=[checkpoint],
        all_results=[[_batch_source_row()]],
        rowcounts=[1, 1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(BatchSourceRecoveryConflictError, match="安全 Checkpoint 不一致"):
        _repository().get_recoverable_batch_source_scan(
            "experts",
            "scanner-attempt-2",
        )

    assert connection.commit_count == 0
    assert connection.rollback_count == 1


def test_get_recoverable_batch_source_scan_requires_current_active_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无效 owner 不能读取并锁定可恢复范围。"""

    cursor = ScriptedCursor(
        one_results=[
            _checkpoint_row(
                lease_owner="scanner-attempt-3",
                lease_until=LEASE_UNTIL,
                lease_is_active=1,
            )
        ],
        rowcounts=[1],
    )
    connection = ScriptedConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceLeaseConflictError, match="不再拥有"):
        _repository().get_recoverable_batch_source_scan(
            "experts",
            "scanner-attempt-2",
        )

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("acquire_source_lease", ("unknown", "owner"), {"lease_seconds": 60}),
        ("acquire_source_lease", ("experts", ""), {"lease_seconds": 60}),
        ("acquire_source_lease", ("experts", "owner"), {"lease_seconds": 0}),
        ("renew_source_lease", ("experts", "owner"), {"lease_seconds": True}),
        ("release_source_lease", ("experts", "x" * 129), {}),
    ],
)
def test_lease_methods_reject_invalid_arguments_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    """非法表名、owner 和时长都不能触发 Index MySQL 连接。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.source_state.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError):
        getattr(_repository(), method_name)(*args, **kwargs)
