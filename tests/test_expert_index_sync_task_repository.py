"""来源同步任务 Repository 的幂等创建和查询测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.models import SourceSyncTask
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


def _patch_connection(
    monkeypatch: pytest.MonkeyPatch,
    connection: RecordingConnection,
) -> None:
    """让一次同步任务操作使用指定 Recording Connection。"""

    monkeypatch.setattr(
        "expert_index.repositories.sync_task.pymysql.connect",
        lambda **_kwargs: connection,
    )


def test_create_source_sync_task_is_idempotent_by_source_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复来源记录应返回唯一键对应的原任务 ID。"""

    cursor = RecordingCursor(lastrowid=801)
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task_id = _repository().create_source_sync_task(
        batch_id=701,
        source_table="experts_work_experience",
        source_record_id=901,
        expert_id=101,
        operation_type="update",
    )

    sql, params = cursor.executed[0]
    assert task_id == 801
    assert "VALUES (%s, %s, %s, %s, %s, 'pending', 0)" in sql
    assert "ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)" in sql
    assert params == (701, "experts_work_experience", 901, 101, "update")
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True


def test_create_source_sync_task_validates_identity_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法来源定位字段不应访问数据库。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.sync_task.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError, match="source_record_id 必须是正整数"):
        _repository().create_source_sync_task(
            batch_id=701,
            source_table="experts",
            source_record_id=0,
            expert_id=101,
            operation_type="update",
        )


def test_get_source_sync_task_returns_complete_domain_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查询结果应完整映射为 SourceSyncTask。"""

    created_at = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    started_at = datetime(2026, 8, 21, 10, 1, tzinfo=timezone.utc)
    cursor = RecordingCursor(
        one={
            "id": 801,
            "batch_id": 701,
            "source_table": "experts_project_experience",
            "source_record_id": 901,
            "expert_id": 101,
            "operation_type": "insert",
            "sync_status": "processing",
            "retry_count": 0,
            "error_stage": None,
            "error_type": None,
            "error_message": None,
            "result_code": None,
            "started_at": started_at,
            "completed_at": None,
            "created_at": created_at,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().get_source_sync_task(801)

    assert task == SourceSyncTask(
        task_id=801,
        batch_id=701,
        source_table="experts_project_experience",
        source_record_id=901,
        expert_id=101,
        operation_type="insert",
        sync_status="processing",
        retry_count=0,
        created_at=created_at,
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
