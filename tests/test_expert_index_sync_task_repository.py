"""来源同步任务 Repository 的幂等创建和查询测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.models import NewSourceSyncTask, SourceSyncTask
from expert_index.repositories import (
    MySQLSourceSyncTaskRepository,
    SourceSyncTaskLeaseConflictError,
    SourceSyncTaskStageConflictError,
)


class RecordingCursor:
    """记录 SQL，并返回测试指定的查询结果。"""

    def __init__(
        self,
        *,
        one: dict[str, Any] | None = None,
        ones: list[dict[str, Any] | None] | None = None,
        alls: list[list[dict[str, Any]]] | None = None,
        rowcounts: list[int] | None = None,
        lastrowid: int = 801,
    ) -> None:
        self.one = one
        self.ones = list(ones or [])
        self.alls = list(alls or [])
        self.rowcounts = list(rowcounts or [])
        self.lastrowid = lastrowid
        self.rowcount = 0
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append((" ".join(sql.split()), params))
        self.rowcount = self.rowcounts.pop(0) if self.rowcounts else 0

    def fetchone(self) -> dict[str, Any] | None:
        if self.ones:
            return self.ones.pop(0)
        return self.one

    def fetchall(self) -> list[dict[str, Any]]:
        if self.alls:
            return self.alls.pop(0)
        return []


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


def _repository_with_tokens(*tokens: str) -> MySQLSourceSyncTaskRepository:
    """构造按顺序返回固定 attempt token 的 Repository。"""

    remaining = list(tokens)
    config = DatabaseConfig(
        "127.0.0.1",
        3306,
        "test",
        "test",
        "index_test",
        role=DatabaseRole.INDEX,
    )
    return MySQLSourceSyncTaskRepository(
        config,
        attempt_token_factory=lambda: remaining.pop(0),
    )


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


def _task_row(**changes: object) -> dict[str, Any]:
    """构造任务表完整查询行，供状态机 Repository 测试复用。"""

    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    values: dict[str, Any] = {
        "id": 801,
        "dedupe_key": "b" * 64,
        "batch_id": 701,
        "event_source": "poll",
        "source_table": "experts_work_experience",
        "source_record_id": 901,
        "source_updated_at": now,
        "expert_id": 101,
        "operation_type": "update",
        "target_type": "employment",
        "processing_stage": "candidate",
        "sync_status": "pending",
        "retry_count": 0,
        "attempt_token": None,
        "lease_until": None,
        "next_retry_at": None,
        "policy_version": "1.3",
        "error_stage": None,
        "error_code": None,
        "error_type": None,
        "error_message": None,
        "result_code": None,
        "started_at": None,
        "completed_at": None,
        "created_at": now,
        "updated_at": now,
    }
    values.update(changes)
    return values


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


def test_claim_source_sync_tasks_uses_skip_locked_and_fences_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pending 和过期 processing 应获得不同新 token，且仅后者增加重试计数。"""

    lease_until = datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc)
    claimed_rows = [
        _task_row(
            id=801,
            sync_status="processing",
            attempt_token="attempt-801-new",
            lease_until=lease_until,
        ),
        _task_row(
            id=802,
            source_record_id=902,
            sync_status="processing",
            retry_count=2,
            attempt_token="attempt-802-new",
            lease_until=lease_until,
        ),
    ]
    cursor = RecordingCursor(
        alls=[
            [
                {"id": 801, "sync_status": "pending"},
                {"id": 802, "sync_status": "processing"},
            ],
            claimed_rows,
        ],
        rowcounts=[0, 1, 1, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens(
        "attempt-801-new",
        "attempt-802-new",
    ).claim_source_sync_tasks(
        701,
        "experts_work_experience",
        limit=10,
        lease_seconds=120,
    )

    claim_sql, claim_params = cursor.executed[0]
    first_update_sql, first_update_params = cursor.executed[1]
    second_update_params = cursor.executed[2][1]
    assert "FOR UPDATE SKIP LOCKED" in claim_sql
    assert "batch_id=%s AND source_table=%s" in claim_sql
    assert "lease_until <= CURRENT_TIMESTAMP(6)" in claim_sql
    assert claim_params == (701, "experts_work_experience", 10)
    assert "retry_count=retry_count+%s" in first_update_sql
    assert first_update_params == ("attempt-801-new", 120, 0, 801, "pending")
    assert second_update_params == (
        "attempt-802-new",
        120,
        1,
        802,
        "processing",
    )
    assert [task.task_id for task in tasks] == [801, 802]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


def test_claim_source_sync_tasks_returns_empty_without_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有可领取任务时应提交短事务并返回空结果。"""

    cursor = RecordingCursor(alls=[[]])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository().claim_source_sync_tasks(
        701,
        "experts",
        limit=1,
        lease_seconds=60,
    )

    assert tasks == ()
    assert len(cursor.executed) == 1
    assert connection.commit_count == 1


def test_claim_source_sync_tasks_rejects_duplicate_generated_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一领取事务内 token 重复时必须在更新任务前回滚。"""

    cursor = RecordingCursor(
        alls=[
            [
                {"id": 801, "sync_status": "pending"},
                {"id": 802, "sync_status": "pending"},
            ]
        ]
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(RuntimeError, match="重复的 attempt_token"):
        _repository_with_tokens("same-token", "same-token").claim_source_sync_tasks(
            701,
            "experts",
            limit=2,
            lease_seconds=60,
        )

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


def test_renew_source_sync_task_lease_requires_current_active_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """续租 SQL 必须同时守卫 processing、token 和未过期 lease。"""

    locked = _task_row(
        sync_status="processing",
        attempt_token="attempt-801",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        lease_is_active=1,
    )
    renewed = dict(locked)
    renewed.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, renewed], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().renew_source_sync_task_lease(
        801,
        "attempt-801",
        lease_seconds=120,
    )

    update_sql, update_params = cursor.executed[1]
    assert "sync_status='processing'" in update_sql
    assert "attempt_token=%s" in update_sql
    assert "lease_until > CURRENT_TIMESTAMP(6)" in update_sql
    assert update_params == (120, 801, "attempt-801")
    assert task.attempt_token == "attempt-801"
    assert connection.commit_count == 1


@pytest.mark.parametrize(
    ("persisted_token", "lease_is_active"),
    [("attempt-old", 1), ("attempt-current", 0)],
)
def test_renew_rejects_stale_or_expired_attempt_before_update(
    monkeypatch: pytest.MonkeyPatch,
    persisted_token: str,
    lease_is_active: int,
) -> None:
    """旧 token 和过期 token 都不能续租。"""

    locked = _task_row(
        sync_status="processing",
        attempt_token=persisted_token,
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        lease_is_active=lease_is_active,
    )
    cursor = RecordingCursor(ones=[locked])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskLeaseConflictError, match="有效 attempt lease"):
        _repository().renew_source_sync_task_lease(
            801,
            "attempt-current",
            lease_seconds=120,
        )

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


def test_advance_source_sync_task_stage_allows_next_and_idempotent_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阶段只前进一格，重复提交已完成阶段可幂等返回。"""

    lease_until = datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc)
    locked = _task_row(
        sync_status="processing",
        attempt_token="attempt-801",
        lease_until=lease_until,
        lease_is_active=1,
    )
    advanced = dict(locked, processing_stage="source_read")
    advanced.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, advanced], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().advance_source_sync_task_stage(
        801,
        "attempt-801",
        next_stage="source_read",
    )

    assert task.processing_stage == "source_read"
    assert cursor.executed[1][1] == (
        "source_read",
        801,
        "candidate",
        "attempt-801",
    )

    idempotent_locked = dict(locked, processing_stage="source_read")
    idempotent_cursor = RecordingCursor(ones=[idempotent_locked])
    idempotent_connection = RecordingConnection(idempotent_cursor)
    _patch_connection(monkeypatch, idempotent_connection)

    repeated = _repository().advance_source_sync_task_stage(
        801,
        "attempt-801",
        next_stage="source_read",
    )

    assert repeated.processing_stage == "source_read"
    assert len(idempotent_cursor.executed) == 1


def test_advance_source_sync_task_stage_rejects_skipped_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不能跳过 source_read 直接声称 normalized 已完成。"""

    locked = _task_row(
        sync_status="processing",
        attempt_token="attempt-801",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        lease_is_active=1,
    )
    cursor = RecordingCursor(ones=[locked])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskStageConflictError, match="不能从 candidate"):
        _repository().advance_source_sync_task_stage(
            801,
            "attempt-801",
            next_stage="normalized",
        )

    assert connection.rollback_count == 1


def test_mark_source_sync_task_failed_clears_attempt_and_writes_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失败写回应清除租约并仅保存受限机器字段和脱敏摘要。"""

    locked = _task_row(
        processing_stage="source_read",
        sync_status="processing",
        attempt_token="attempt-801",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        lease_is_active=1,
    )
    failed = dict(
        locked,
        sync_status="failed",
        attempt_token=None,
        lease_until=None,
        error_stage="source",
        error_code="source_read_failed",
        error_type="SourceReadError",
        error_message="sanitized source read failure",
    )
    failed.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, failed], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().mark_source_sync_task_failed(
        801,
        "attempt-801",
        error_stage="source",
        error_code="source_read_failed",
        error_type="SourceReadError",
        safe_error_message="sanitized source read failure",
    )

    update_sql, update_params = cursor.executed[1]
    assert "attempt_token=NULL" in update_sql
    assert "lease_until=NULL" in update_sql
    assert "lease_until > CURRENT_TIMESTAMP(6)" in update_sql
    assert update_params[-2:] == (801, "attempt-801")
    assert task.sync_status == "failed"
    assert task.attempt_token is None


def test_mark_source_sync_task_success_requires_handed_off_receipt_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成功写回只能把有效 handed_off attempt 收口为 completed。"""

    completed_at = datetime(2026, 8, 21, 10, 4, tzinfo=timezone.utc)
    locked = _task_row(
        processing_stage="handed_off",
        sync_status="processing",
        attempt_token="attempt-801",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        lease_is_active=1,
    )
    succeeded = dict(
        locked,
        processing_stage="completed",
        sync_status="success",
        attempt_token=None,
        lease_until=None,
        result_code="downstream_applied",
        completed_at=completed_at,
    )
    succeeded.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, succeeded], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().mark_source_sync_task_success(
        801,
        "attempt-801",
        result_code="downstream_applied",
    )

    update_sql = cursor.executed[1][0]
    assert "processing_stage='completed'" in update_sql
    assert "processing_stage='handed_off'" in update_sql
    assert "attempt_token=%s" in update_sql
    assert "lease_until > CURRENT_TIMESTAMP(6)" in update_sql
    assert task.sync_status == "success"
    assert task.completed_at == completed_at


def test_mark_source_sync_task_success_rejects_pre_handoff_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有 handed_off 证据的 processing 任务不能直接成功。"""

    locked = _task_row(
        processing_stage="contract_ready",
        sync_status="processing",
        attempt_token="attempt-801",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        lease_is_active=1,
    )
    cursor = RecordingCursor(ones=[locked])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskStageConflictError, match="handed_off"):
        _repository().mark_source_sync_task_success(
            801,
            "attempt-801",
            result_code="downstream_applied",
        )

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        (
            "mark_source_sync_task_failed",
            {
                "error_stage": "source",
                "error_code": "contains spaces",
            },
        ),
        (
            "mark_source_sync_task_success",
            {"result_code": "contains spaces"},
        ),
    ],
)
def test_terminal_writes_reject_unsafe_machine_fields_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    kwargs: dict[str, object],
) -> None:
    """错误码和结果码包含业务文本时不能建立数据库连接。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("安全字段校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.sync_task.pymysql.connect",
        unexpected_connect,
    )
    method = getattr(_repository(), method_name)

    with pytest.raises(ValueError, match="安全机器标识"):
        method(801, "attempt-801", **kwargs)
