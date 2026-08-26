"""专家级 MySQL 执行栅栏的紧凑 Repository 测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.models import SourceSyncTask, impact_scope_from_target
from expert_index.repositories.scope_fence import (
    MySQLScopeExecutionFence,
    ScopeFenceUnavailableError,
)


class _Cursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, sql, params):
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None


class _Connection:
    def __init__(self, rows):
        self.cursor_value = _Cursor(rows)
        self.closed = False

    def cursor(self):
        return self.cursor_value

    def close(self):
        self.closed = True


def _task() -> SourceSyncTask:
    now = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
    return SourceSyncTask(
        task_id=11,
        dedupe_key="a" * 64,
        batch_id=7,
        event_source="poll",
        source_table="experts_work_experience",
        source_record_id=31,
        source_updated_at=now,
        expert_id=9,
        operation_type="update",
        target_type="employment",
        processing_stage="handed_off",
        sync_status="processing",
        retry_count=0,
        policy_version="1.3",
        attempt_token="tok-11",
        lease_until=now,
        created_at=now,
        updated_at=now,
    )


def _fence() -> MySQLScopeExecutionFence:
    config = DatabaseConfig(
        "127.0.0.1", 3306, "index", "user", "password", role=DatabaseRole.INDEX,
    )
    return MySQLScopeExecutionFence(config)


def test_scope_fence_serializes_and_detects_newer_overlap(monkeypatch):
    connection = _Connection([{"acquired": 1}, {"id": 11}, {"has_newer": 1}])
    monkeypatch.setattr(
        "expert_index.repositories.scope_fence.pymysql.connect",
        lambda **_: connection,
    )
    scope = impact_scope_from_target(
        target_type="employment", expert_id=9, source_field_id="31",
    )

    with _fence().hold(_task(), "tok-11", scope, timeout_seconds=30) as decision:
        assert decision.superseded is True

    statements = connection.cursor_value.executed
    assert "GET_LOCK" in statements[0][0]
    assert "newer.source_record_id=%s" in statements[2][0]
    assert "RELEASE_LOCK" in statements[3][0]
    assert connection.closed is True


def test_scope_fence_timeout_fails_closed(monkeypatch):
    connection = _Connection([{"acquired": 0}])
    monkeypatch.setattr(
        "expert_index.repositories.scope_fence.pymysql.connect",
        lambda **_: connection,
    )

    with pytest.raises(ScopeFenceUnavailableError, match="获取超时"), _fence().hold(
        _task(),
        "tok-11",
        impact_scope_from_target(target_type="expert", expert_id=9),
        timeout_seconds=1,
    ):
        pytest.fail("未取得栅栏时不得进入执行区")
    assert connection.closed is True
