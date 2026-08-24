"""来源同步任务 Repository 的幂等创建和查询测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.models import NewSourceSyncTask, SourceSyncTask
from expert_index.repositories import MySQLSourceSyncTaskRepository


class RecordingCursor:
    """记录 SQL，并返回测试指定的查询结果。"""

    def __init__(
        self,
        *,
        one: dict[str, Any] | None = None,
        lastrowid: int = 801,
    ) -> None:
        self.one = one
        self.lastrowid = lastrowid
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self) -> dict[str, Any] | None:
        return self.one


class RecordingConnection:
    """记录同步任务方法的事务和连接关闭行为。"""

    def __init__(self, cursor: RecordingCursor) -> None:
        self.cursor_instance = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> RecordingCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def _repository() -> MySQLSourceSyncTaskRepository:
    """构造使用虚构连接参数的同步任务 Repository。"""

    config = DatabaseConfig(
        "127.0.0.1",
        3306,
        "test",
        "test",
        "index_test",
        role=DatabaseRole.INDEX,
    )
    return MySQLSourceSyncTaskRepository(config)


def _new_task(**changes: object) -> NewSourceSyncTask:
    """构造一条符合 012 身份约束的待创建任务。"""

    values: dict[str, object] = {
        "dedupe_key": "a" * 64,
        "batch_id": 701,
        "event_source": "poll",
        "source_table": "experts_work_experience",
        "source_record_id": 901,
        "source_updated_at": datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc),
        "expert_id": 101,
        "operation_type": "update",
        "target_type": "employment",
        "policy_version": "1.3",
    }
    values.update(changes)
    return NewSourceSyncTask(**values)  # type: ignore[arg-type]


def _patch_connection(
    monkeypatch: pytest.MonkeyPatch,
    connection: RecordingConnection,
) -> None:
    """让一次同步任务操作使用指定 Recording Connection。"""

    monkeypatch.setattr(
        "expert_index.repositories.sync_task.pymysql.connect",
        lambda **_kwargs: connection,
    )


def test_create_source_sync_task_is_idempotent_by_cross_scan_dedupe_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复扫描应在核验冻结身份后返回 dedupe_key 对应的原任务 ID。"""

    cursor = RecordingCursor(one={"id": 801}, lastrowid=801)
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _new_task()
    task_id = _repository().create_source_sync_task(task)

    insert_sql, insert_params = cursor.executed[0]
    identity_sql, identity_params = cursor.executed[1]
    assert task_id == 801
    assert "dedupe_key, batch_id, event_source" in insert_sql
    assert "'candidate', 'pending', 0" in insert_sql
    assert "ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)" in insert_sql
    assert insert_params == (
        task.dedupe_key,
        701,
        "poll",
        "experts_work_experience",
        901,
        task.source_updated_at,
        101,
        "update",
        "employment",
        "1.3",
    )
    assert "source_updated_at <=> %s" in identity_sql
    assert "batch_id" not in identity_sql
    assert identity_params == (
        task.dedupe_key,
        "poll",
        "experts_work_experience",
        901,
        task.source_updated_at,
        "update",
        "employment",
        "1.3",
    )
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True


def test_create_source_sync_task_requires_domain_contract_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非 NewSourceSyncTask 对象不应访问数据库。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.sync_task.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(TypeError, match="task 必须是 NewSourceSyncTask"):
        _repository().create_source_sync_task(object())  # type: ignore[arg-type]


def test_create_source_sync_task_rolls_back_mismatched_unique_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧批次唯一键命中不同身份时必须回滚，不得误报幂等成功。"""

    cursor = RecordingCursor(one=None)
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(RuntimeError, match="未找到匹配的 dedupe_key 身份"):
        _repository().create_source_sync_task(_new_task())

    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.closed is True


def test_get_source_sync_task_returns_complete_domain_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查询结果应完整映射为 SourceSyncTask。"""

    created_at = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    updated_at = datetime(2026, 8, 21, 10, 2, tzinfo=timezone.utc)
    started_at = datetime(2026, 8, 21, 10, 1, tzinfo=timezone.utc)
    lease_until = datetime(2026, 8, 21, 10, 3, tzinfo=timezone.utc)
    cursor = RecordingCursor(
        one={
            "id": 801,
            "dedupe_key": "b" * 64,
            "batch_id": 701,
            "event_source": "poll",
            "source_table": "experts_project_experience",
            "source_record_id": 901,
            "source_updated_at": created_at,
            "expert_id": 101,
            "operation_type": "insert",
            "target_type": "project",
            "processing_stage": "source_read",
            "sync_status": "processing",
            "retry_count": 0,
            "attempt_token": "attempt-801",
            "lease_until": lease_until,
            "next_retry_at": None,
            "policy_version": "1.3",
            "error_stage": None,
            "error_code": None,
            "error_type": None,
            "error_message": None,
            "result_code": None,
            "started_at": started_at,
            "completed_at": None,
            "created_at": created_at,
            "updated_at": updated_at,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().get_source_sync_task(801)

    assert task == SourceSyncTask(
        task_id=801,
        dedupe_key="b" * 64,
        batch_id=701,
        event_source="poll",
        source_table="experts_project_experience",
        source_record_id=901,
        source_updated_at=created_at,
        expert_id=101,
        operation_type="insert",
        target_type="project",
        processing_stage="source_read",
        sync_status="processing",
        retry_count=0,
        attempt_token="attempt-801",
        lease_until=lease_until,
        policy_version="1.3",
        created_at=created_at,
        updated_at=updated_at,
        started_at=started_at,
    )
    assert cursor.executed[0][1] == (801,)
    assert connection.commit_count == 0
    assert connection.closed is True


def test_get_source_sync_task_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不存在的任务应返回 None。"""

    cursor = RecordingCursor(one=None)
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    assert _repository().get_source_sync_task(801) is None
    assert connection.commit_count == 0
    assert connection.closed is True
