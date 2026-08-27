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


def test_list_cdc_expert_delete_ids_reads_only_matching_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P7.6 应从 Index Task 获取同批专家删除集合并校验双 ID 一致。"""

    cursor = RecordingCursor(
        alls=[
            [
                {"source_record_id": 101, "expert_id": 101},
                {"source_record_id": 102, "expert_id": 102},
            ]
        ]
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    expert_ids = _repository().list_cdc_expert_delete_ids(701)

    assert expert_ids == frozenset({101, 102})
    sql, params = cursor.executed[0]
    assert "event_source='cdc'" in sql
    assert "source_table='experts'" in sql
    assert "operation_type='delete'" in sql
    assert params == (701,)
    assert connection.commit_count == 0
    assert connection.closed is True


def test_list_cdc_expert_delete_ids_rejects_inconsistent_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """专家 DELETE 的 source_record_id/expert_id 不一致时必须停止。"""

    cursor = RecordingCursor(alls=[[{"source_record_id": 101, "expert_id": 999}]])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(RuntimeError, match="不一致"):
        _repository().list_cdc_expert_delete_ids(701)

    assert connection.closed is True


def test_claim_cdc_source_sync_task_claims_exact_pending_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """按 Event 回执领取时只更新指定 CDC Task 并生成新 token。"""

    locked = _task_row(
        event_source="cdc",
        operation_type="delete",
        source_updated_at=None,
        lease_is_active=0,
    )
    claimed = dict(locked)
    claimed.update(
        sync_status="processing",
        attempt_token="cdc-token-801",
        lease_until=datetime(2026, 8, 21, 10, 3, tzinfo=timezone.utc),
    )
    claimed.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, claimed], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository_with_tokens("cdc-token-801").claim_cdc_source_sync_task(
        801,
        lease_seconds=180,
    )

    assert task is not None and task.attempt_token == "cdc-token-801"
    update_sql, update_params = cursor.executed[1]
    assert "WHERE id=%s AND sync_status=%s" in update_sql
    assert update_params is not None and update_params[-2:] == (801, "pending")
    assert connection.commit_count == 1 and connection.rollback_count == 0


def test_claim_cdc_source_sync_task_accepts_update_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一 CDC 编排允许领取 UPDATE Task，仍使用同一 lease/fencing 逻辑。"""

    locked = _task_row(
        event_source="cdc",
        operation_type="update",
        source_updated_at=None,
        lease_is_active=0,
    )
    claimed = dict(locked)
    claimed.update(
        sync_status="processing",
        attempt_token="cdc-token-update",
        lease_until=datetime(2026, 8, 21, 10, 3, tzinfo=timezone.utc),
    )
    claimed.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, claimed], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository_with_tokens("cdc-token-update").claim_cdc_source_sync_task(
        801,
        lease_seconds=180,
    )

    assert task is not None and task.operation_type == "update"


def test_claim_cdc_source_sync_task_does_not_steal_active_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """其他 Worker 持有有效 lease 时返回 None，不覆盖 attempt_token。"""

    locked = _task_row(
        event_source="cdc",
        operation_type="delete",
        source_updated_at=None,
        sync_status="processing",
        attempt_token="other-worker",
        lease_until=datetime(2026, 8, 21, 10, 3, tzinfo=timezone.utc),
        lease_is_active=1,
    )
    cursor = RecordingCursor(one=locked)
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().claim_cdc_source_sync_task(801, lease_seconds=180)

    assert task is None
    assert len(cursor.executed) == 1
    assert connection.commit_count == 1 and connection.rollback_count == 0


def test_claim_source_sync_tasks_uses_skip_locked_and_fences_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pending 和过期 processing 应获得不同新 token，且仅后者增加重试计数。

    limit=10：第一条 801 为 pending（首次 attempt，retry_count 不增加），
    第二条 802 为过期 processing（逐条严格 FIFO 第二轮，retry_count += 1），
    第三轮 find 返回空 → 停止。
    """

    lease_until = datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc)
    past_lease = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
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
        ones=[
            # 第一轮 find 801。
            {"id": 801},
            # 第一轮 lock 801（pending，next_retry_at 为空，不走退避）。
            {
                "id": 801,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 find 802。
            {"id": 802},
            # 第二轮 lock 802（processing，lease 已过期，需 CURRENT_TIMESTAMP 比较）。
            {
                "id": 802,
                "sync_status": "processing",
                "next_retry_at": None,
                "lease_until": past_lease,
                "attempt_token": "attempt-802-old",
            },
            # 第二轮 CURRENT_TIMESTAMP。
            {"now": now},
            # 第三轮 find 返回空 → 停止。
            None,
        ],
        alls=[claimed_rows],
        # 执行序：find1 lock1 UPDATE1 find2 lock2 CURRENT_TIMESTAMP UPDATE2 find3 re-read
        rowcounts=[0, 0, 1, 0, 0, 0, 1, 0, 0],
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

    # 第一轮 find：定位 id > 0 之后最早非终态队头。
    frontier_sql = cursor.executed[0][0]
    assert "sync_status NOT IN ('success', 'dead')" in frontier_sql
    assert "ORDER BY id ASC" in frontier_sql
    assert "LIMIT 1" in frontier_sql
    assert cursor.executed[0][1] == (701, "experts_work_experience", 0)
    # 第一轮 lock：按 id 锁定。
    lock_sql = cursor.executed[1][0]
    assert "FOR UPDATE SKIP LOCKED" in lock_sql
    assert cursor.executed[1][1] == (801,)
    # 第一轮 UPDATE：801 pending，retry_count 不增加。
    first_update_sql, first_update_params = cursor.executed[2]
    assert "retry_count=retry_count+%s" in first_update_sql
    assert first_update_params == ("attempt-801-new", 120, 0, 801, "pending")
    # 第二轮 UPDATE：802 过期 processing，retry_count += 1。
    second_update_params = cursor.executed[6][1]
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
    """没有非终态任务（队头查询返回空）时应提交短事务并返回空结果。"""

    cursor = RecordingCursor(one=None)
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
    """同一领取事务内 token 重复时必须在回读前回滚。

    limit=2：两条均 pending，均生成同一 token；逐条严格 FIFO 第二轮也需 token。
    """

    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 801。
            {"id": 801},
            # 第一轮 lock 801（pending，next_retry_at 为空，不走退避）。
            {
                "id": 801,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 find 802。
            {"id": 802},
            # 第二轮 lock 802（pending，next_retry_at 为空）。
            {
                "id": 802,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
        ],
        alls=[],
        # 执行序：find1 lock1 UPDATE1 find2 lock2 UPDATE2（回读前触发重复检测）。
        rowcounts=[0, 0, 1, 0, 0, 1],
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

    # 逐条严格 FIFO 两轮：每轮 find + lock + UPDATE，重复检测发生在回读之前。
    assert len(cursor.executed) == 6
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


# ---------------------------------------------------------------------------
# P5.3A: 确定性 rejected / 可重试 retryable / claim next_retry_at 退避
# ---------------------------------------------------------------------------


def _locked_processing_row(**changes: object) -> dict[str, Any]:
    """构造一个 processing + 有效 lease 的锁定行，供 rejected/retryable 测试复用。"""

    base = _task_row(
        sync_status="processing",
        attempt_token="attempt-801",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        lease_is_active=1,
    )
    base.update(changes)
    return base


def test_mark_source_sync_task_rejected_writes_dead_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rejected 应写入 dead 终态，清理租约并记录 completed_at，不可被 claim 领取。"""

    locked = _locked_processing_row()
    dead = dict(
        locked,
        sync_status="dead",
        attempt_token=None,
        lease_until=None,
        next_retry_at=None,
        error_stage="handoff",
        error_code="receipt_rejected",
        error_type="ReceiptRejected",
        error_message="deterministic rejection",
        completed_at=datetime(2026, 8, 21, 10, 6, tzinfo=timezone.utc),
    )
    dead.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, dead], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().mark_source_sync_task_rejected(
        801,
        "attempt-801",
        error_stage="handoff",
        error_code="receipt_rejected",
        error_type="ReceiptRejected",
        safe_error_message="deterministic rejection",
    )

    update_sql, update_params = cursor.executed[1]
    assert "sync_status='dead'" in update_sql
    assert "attempt_token=NULL" in update_sql
    assert "next_retry_at=NULL" in update_sql
    assert "completed_at=CURRENT_TIMESTAMP(6)" in update_sql
    assert "lease_until > CURRENT_TIMESTAMP(6)" in update_sql
    assert update_params == (
        "handoff",
        "receipt_rejected",
        "ReceiptRejected",
        "deterministic rejection",
        801,
        "attempt-801",
    )
    assert task.sync_status == "dead"
    assert task.attempt_token is None
    assert task.next_retry_at is None
    assert task.completed_at is not None
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


def test_mark_source_sync_task_rejected_requires_active_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """错误 token 或过期 lease 必须在更新前回滚，不得写 dead。"""

    locked = _locked_processing_row(
        attempt_token="attempt-old",
        lease_is_active=1,
    )
    cursor = RecordingCursor(ones=[locked])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskLeaseConflictError, match="有效 attempt lease"):
        _repository().mark_source_sync_task_rejected(
            801,
            "attempt-current",
            error_stage="scope_consumer",
            error_code="downstream_failed",
        )

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


def test_mark_source_sync_task_rejected_rejects_unsafe_error_code_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """error_code 含业务文本时不应建立数据库连接。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("安全字段校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.sync_task.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError, match="安全机器标识"):
        _repository().mark_source_sync_task_rejected(
            801,
            "attempt-801",
            error_stage="handoff",
            error_code="contains spaces",
        )


def test_mark_source_sync_task_retryable_returns_pending_with_next_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """retry_count 低于上限应退回 pending 并设置 next_retry_at，不增加 retry_count。"""

    locked = _locked_processing_row(retry_count=1)
    pending = dict(
        locked,
        sync_status="pending",
        attempt_token=None,
        lease_until=None,
        next_retry_at=datetime(2026, 8, 21, 10, 7, tzinfo=timezone.utc),
        error_stage="scope_consumer",
        error_code="embedding_timeout",
        error_type="EmbeddingTimeout",
        error_message="transient infra failure",
        completed_at=None,
    )
    pending.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, pending], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().mark_source_sync_task_retryable(
        801,
        "attempt-801",
        error_stage="scope_consumer",
        error_code="embedding_timeout",
        max_attempts=3,
        retry_delay_seconds=60,
        error_type="EmbeddingTimeout",
        safe_error_message="transient infra failure",
    )

    update_sql, update_params = cursor.executed[1]
    assert "sync_status='pending'" in update_sql
    assert "next_retry_at=TIMESTAMPADD( SECOND, %s, CURRENT_TIMESTAMP(6) )" in update_sql
    assert "completed_at=NULL" in update_sql
    assert update_params[0] == 60  # retry_delay_seconds
    assert task.sync_status == "pending"
    assert task.attempt_token is None
    assert task.next_retry_at is not None
    assert task.completed_at is None
    assert connection.commit_count == 1


def test_mark_source_sync_task_retryable_at_max_attempts_writes_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """达到 max_attempts 应进入确定性 dead 终态，不再退回 pending。"""

    locked = _locked_processing_row(retry_count=3)
    dead = dict(
        locked,
        sync_status="dead",
        attempt_token=None,
        lease_until=None,
        next_retry_at=None,
        error_stage="scope_consumer",
        error_code="embedding_timeout",
        error_type="EmbeddingTimeout",
        error_message="exhausted retries",
        completed_at=datetime(2026, 8, 21, 10, 8, tzinfo=timezone.utc),
    )
    dead.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, dead], rowcounts=[0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().mark_source_sync_task_retryable(
        801,
        "attempt-801",
        error_stage="scope_consumer",
        error_code="embedding_timeout",
        max_attempts=3,
        retry_delay_seconds=60,
        error_type="EmbeddingTimeout",
        safe_error_message="exhausted retries",
    )

    update_sql, _ = cursor.executed[1]
    assert "sync_status='dead'" in update_sql
    assert "completed_at=CURRENT_TIMESTAMP(6)" in update_sql
    assert task.sync_status == "dead"
    assert task.next_retry_at is None
    assert task.completed_at is not None


def test_mark_source_sync_task_retryable_rejects_invalid_max_attempts_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法 max_attempts/retry_delay 不应建立数据库连接。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.sync_task.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError, match="max_attempts"):
        _repository().mark_source_sync_task_retryable(
            801,
            "attempt-801",
            error_stage="scope_consumer",
            error_code="x",
            max_attempts=-1,
            retry_delay_seconds=60,
        )


def test_mark_source_sync_task_retryable_rejects_stale_token_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """过期 token 在锁定校验阶段必须回滚，不写状态。"""

    locked = _locked_processing_row(
        attempt_token="attempt-old",
        lease_is_active=1,
    )
    cursor = RecordingCursor(ones=[locked])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskLeaseConflictError, match="有效 attempt lease"):
        _repository().mark_source_sync_task_retryable(
            801,
            "attempt-current",
            error_stage="scope_consumer",
            error_code="x",
            max_attempts=3,
            retry_delay_seconds=60,
        )

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


def test_claim_source_sync_tasks_skips_pending_with_future_next_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 pending 但 next_retry_at 未到：返回空，不领取更晚任务。"""

    future = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    cursor = RecordingCursor(
        ones=[
            {"id": 801},
            {
                "id": 801,
                "sync_status": "pending",
                "next_retry_at": future,
                "lease_until": None,
                "attempt_token": None,
            },
            {"now": now},
        ],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository().claim_source_sync_tasks(
        701,
        "experts",
        limit=5,
        lease_seconds=60,
    )

    assert tasks == ()
    # 队头定位 + 锁定 + SELECT CURRENT_TIMESTAMP(6) 比较退避。
    assert len(cursor.executed) == 3
    assert connection.commit_count == 1


def test_claim_source_sync_tasks_pending_without_next_retry_still_claimable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """普通 pending（next_retry_at IS NULL）仍可领取，退避守卫不影响默认路径。"""

    lease_until = datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc)
    claimed = _task_row(
        id=801,
        sync_status="processing",
        attempt_token="attempt-801-new",
        lease_until=lease_until,
    )
    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 801。
            {"id": 801},
            # 第一轮 lock 801（pending，next_retry_at 为空，不走退避）。
            {
                "id": 801,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 find 返回空 → 停止。
            None,
        ],
        alls=[[claimed]],
        # 执行序：find1 lock1 UPDATE1 find2 re-read。
        rowcounts=[0, 0, 1, 0, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens("attempt-801-new").claim_source_sync_tasks(
        701,
        "experts",
        limit=5,
        lease_seconds=60,
    )

    assert [task.task_id for task in tasks] == [801]
    assert connection.commit_count == 1


# P5.7A：严格 head-of-line 语义专项测试。


def test_claim_frontier_pending_not_due_returns_empty_skips_later_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 pending 但 next_retry_at 未到：返回空，不领取更晚 pending 任务。"""

    future = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    cursor = RecordingCursor(
        ones=[
            {"id": 5},
            {
                "id": 5,
                "sync_status": "pending",
                "next_retry_at": future,
                "lease_until": None,
                "attempt_token": None,
            },
            {"now": now},
        ],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository().claim_source_sync_tasks(
        701,
        "experts",
        limit=5,
        lease_seconds=60,
    )

    assert tasks == ()
    # 队头定位 + 锁定 + CURRENT_TIMESTAMP 比较退避，未发出 UPDATE。
    assert len(cursor.executed) == 3
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


def test_claim_frontier_processing_lease_valid_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 processing 且 lease 未过期：返回空，不领取更晚任务。"""

    future_lease = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    cursor = RecordingCursor(
        ones=[
            {"id": 5},
            {
                "id": 5,
                "sync_status": "processing",
                "next_retry_at": None,
                "lease_until": future_lease,
                "attempt_token": "attempt-5",
            },
            {"now": now},
        ],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository().claim_source_sync_tasks(
        701,
        "experts",
        limit=3,
        lease_seconds=60,
    )

    assert tasks == ()
    assert len(cursor.executed) == 3
    assert connection.commit_count == 1


def test_claim_frontier_pending_due_claims_only_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 pending 退避到期后作为一次重试领取并增加 retry_count。"""

    past_retry = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    claimed = _task_row(
        id=5,
        sync_status="processing",
        attempt_token="attempt-5-new",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
    )
    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 5。
            {"id": 5},
            # 第一轮 lock 5（pending，next_retry_at 已到）。
            {
                "id": 5,
                "sync_status": "pending",
                "next_retry_at": past_retry,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第一轮 CURRENT_TIMESTAMP。
            {"now": now},
            # 第二轮 find 返回空 → 停止。
            None,
        ],
        alls=[[claimed]],
        # 执行序：find1 lock1 CURRENT_TIMESTAMP UPDATE1 find2 re-read。
        rowcounts=[0, 0, 0, 1, 0, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens("attempt-5-new").claim_source_sync_tasks(
        701,
        "experts",
        limit=3,
        lease_seconds=120,
    )

    assert [task.task_id for task in tasks] == [5]
    update_sql, update_params = cursor.executed[3]
    assert "retry_count=retry_count+%s" in update_sql
    assert update_params[2] == 1
    assert connection.commit_count == 1


def test_claim_frontier_processing_expired_reclaims_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 processing 且 lease 已过期：回收该队头，retry_count += 1。"""

    past_lease = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    claimed = _task_row(
        id=5,
        sync_status="processing",
        attempt_token="attempt-5-new",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        retry_count=3,
    )
    cursor = RecordingCursor(
        ones=[
            {"id": 5},
            {
                "id": 5,
                "sync_status": "processing",
                "next_retry_at": None,
                "lease_until": past_lease,
                "attempt_token": "attempt-5-old",
            },
            {"now": now},
        ],
        alls=[[claimed]],
        rowcounts=[0, 0, 0, 1, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens("attempt-5-new").claim_source_sync_tasks(
        701,
        "experts",
        limit=1,
        lease_seconds=120,
    )

    assert [task.task_id for task in tasks] == [5]
    # 回收路径：retry_count += 1。
    _frontier_update_sql, frontier_update_params = cursor.executed[3]
    assert frontier_update_params == ("attempt-5-new", 120, 1, 5, "processing")
    assert connection.commit_count == 1


def test_claim_frontier_success_allows_next_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 success 不作为待领取队头：锁定后发现 success，越过继续扫描。

    limit=1：迭代一轮越过 success 后循环结束，返回空（需重新调用 claim 继续
    扫描更晚 task）。success 不被改变。
    """

    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 5。
            {"id": 5},
            # 第一轮 lock 5 发现 success → 越过。
            {
                "id": 5,
                "sync_status": "success",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
        ],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens("attempt-5-new").claim_source_sync_tasks(
        701,
        "experts",
        limit=1,
        lease_seconds=60,
    )

    # 队头 5 是 success：find + lock 发现 success → 越过，循环结束返回空。
    assert tasks == ()
    assert len(cursor.executed) == 2
    assert connection.commit_count == 1


def test_claim_frontier_dead_allows_next_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 dead 不作为待领取队头：锁定后发现 dead，返回空，不改变 dead。"""

    cursor = RecordingCursor(
        ones=[
            {"id": 5},
            {
                "id": 5,
                "sync_status": "dead",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
        ],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository().claim_source_sync_tasks(
        701,
        "experts",
        limit=1,
        lease_seconds=60,
    )

    assert tasks == ()
    # 未发出 UPDATE，dead 未被改变。
    assert len(cursor.executed) == 2
    assert connection.commit_count == 1


def test_claim_frontier_failed_raises_stage_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 failed/未知状态：fail-closed 抛 SourceSyncTaskStageConflictError。"""

    cursor = RecordingCursor(
        ones=[
            {"id": 5},
            {
                "id": 5,
                "sync_status": "failed",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
        ],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskStageConflictError, match="不能跳过该队头任务"):
        _repository().claim_source_sync_tasks(
            701,
            "experts",
            limit=1,
            lease_seconds=60,
        )

    assert connection.rollback_count == 1


def test_claim_frontier_retrying_raises_stage_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头 retrying 状态：fail-closed，不越过队头。"""

    cursor = RecordingCursor(
        ones=[
            {"id": 5},
            {
                "id": 5,
                "sync_status": "retrying",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
        ],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskStageConflictError, match="不能跳过该队头任务"):
        _repository().claim_source_sync_tasks(
            701,
            "experts",
            limit=1,
            lease_seconds=60,
        )

    assert connection.rollback_count == 1


def test_claim_frontier_locked_by_another_worker_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """队头被其他 worker 锁定（FOR UPDATE SKIP LOCKED 返回空）：不跳到更晚任务。"""

    cursor = RecordingCursor(
        ones=[{"id": 5}, None],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository().claim_source_sync_tasks(
        701,
        "experts",
        limit=1,
        lease_seconds=60,
    )

    assert tasks == ()
    # 队头定位 + 锁定（SKIP LOCKED 返回 None），未越过队头。
    assert len(cursor.executed) == 2
    assert connection.commit_count == 1


def test_claim_two_concurrent_do_not_claim_same_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个并发 claim 不能领取同一 Task：SKIP LOCKED 让第二个看到队头被锁。

    本测验证并发语义：第二个 claim 的锁查询返回 None，不领取，也不领取更晚任务。
    """

    cursor = RecordingCursor(
        ones=[{"id": 5}, None],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository().claim_source_sync_tasks(
        701,
        "experts",
        limit=1,
        lease_seconds=60,
    )

    assert tasks == ()
    assert len(cursor.executed) == 2
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


# P5.7A 补充：limit>1 严格 FIFO 专项测试。


def test_claim_fifo_first_claimable_second_backoff_third_claimable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """limit=3：第一条可领、第二条退避未到、第三条可领：只领取第一条。

    逐条严格 FIFO 在第二条遇到退避未到立即停止，不跳到第三条。
    """

    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    future = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    claimed_1 = _task_row(
        id=1,
        sync_status="processing",
        attempt_token="attempt-1-new",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
    )
    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 1。
            {"id": 1},
            # 第一轮 lock 1（pending，next_retry_at 为空）。
            {
                "id": 1,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 find 2。
            {"id": 2},
            # 第二轮 lock 2（pending，next_retry_at 未到）。
            {
                "id": 2,
                "sync_status": "pending",
                "next_retry_at": future,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 CURRENT_TIMESTAMP（退避未到 → 停止）。
            {"now": now},
        ],
        alls=[[claimed_1]],
        # 执行序：find1 lock1 UPDATE1 find2 lock2 CURRENT_TIMESTAMP re-read。
        rowcounts=[0, 0, 1, 0, 0, 0, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens("attempt-1-new").claim_source_sync_tasks(
        701,
        "experts",
        limit=3,
        lease_seconds=60,
    )

    assert [task.task_id for task in tasks] == [1]
    assert connection.commit_count == 1


def test_claim_fifo_first_two_claimable_third_processing_lease_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """limit=3：前两条可领、第三条 processing lease 未过期：只领取前两条。

    逐条严格 FIFO 在第三条遇到 processing lease 未过期立即停止，不跳到第四条。
    """

    now = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    future_lease = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    claimed_rows = [
        _task_row(
            id=1,
            sync_status="processing",
            attempt_token="attempt-1-new",
            lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        ),
        _task_row(
            id=2,
            source_record_id=2,
            sync_status="processing",
            attempt_token="attempt-2-new",
            lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        ),
    ]
    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 1。
            {"id": 1},
            # 第一轮 lock 1（pending）。
            {
                "id": 1,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 find 2。
            {"id": 2},
            # 第二轮 lock 2（pending）。
            {
                "id": 2,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第三轮 find 3。
            {"id": 3},
            # 第三轮 lock 3（processing，lease 未过期）。
            {
                "id": 3,
                "sync_status": "processing",
                "next_retry_at": None,
                "lease_until": future_lease,
                "attempt_token": "attempt-3-old",
            },
            # 第三轮 CURRENT_TIMESTAMP（lease 未过期 → 停止）。
            {"now": now},
        ],
        alls=[claimed_rows],
        # 执行序：find1 lock1 UPDATE1 find2 lock2 UPDATE2 find3 lock3 CURRENT_TIMESTAMP re-read。
        rowcounts=[0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens(
        "attempt-1-new",
        "attempt-2-new",
    ).claim_source_sync_tasks(
        701,
        "experts",
        limit=3,
        lease_seconds=60,
    )

    assert [task.task_id for task in tasks] == [1, 2]
    assert connection.commit_count == 1


def test_claim_fifo_three_consecutive_claimable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """limit=3：连续三条均可领：按 id 顺序领取三条。"""

    claimed_rows = [
        _task_row(
            id=1,
            sync_status="processing",
            attempt_token="attempt-1-new",
            lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        ),
        _task_row(
            id=2,
            source_record_id=2,
            sync_status="processing",
            attempt_token="attempt-2-new",
            lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        ),
        _task_row(
            id=3,
            source_record_id=3,
            sync_status="processing",
            attempt_token="attempt-3-new",
            lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        ),
    ]
    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 1 + lock 1（pending）。
            {"id": 1},
            {
                "id": 1,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 find 2 + lock 2（pending）。
            {"id": 2},
            {
                "id": 2,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第三轮 find 3 + lock 3（pending）。
            {"id": 3},
            {
                "id": 3,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第四轮 find 返回空 → 停止。
            None,
        ],
        alls=[claimed_rows],
        # 执行序：find1 lock1 UPDATE1 find2 lock2 UPDATE2 find3 lock3 UPDATE3 find4 re-read。
        rowcounts=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens(
        "attempt-1-new",
        "attempt-2-new",
        "attempt-3-new",
    ).claim_source_sync_tasks(
        701,
        "experts",
        limit=3,
        lease_seconds=60,
    )

    assert [task.task_id for task in tasks] == [1, 2, 3]
    assert connection.commit_count == 1


def test_claim_fifo_stops_when_later_task_locked_by_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """后续某条被其他事务锁定：停止，不跳过。

    limit=3：第一条可领，第二条 find 后 FOR UPDATE SKIP LOCKED 返回 None → 停止。
    """

    claimed_1 = _task_row(
        id=1,
        sync_status="processing",
        attempt_token="attempt-1-new",
        lease_until=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
    )
    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 1 + lock 1（pending）。
            {"id": 1},
            {
                "id": 1,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 find 2，但 lock 2 返回 None（被其他事务锁定）。
            {"id": 2},
            None,
        ],
        alls=[[claimed_1]],
        # 执行序：find1 lock1 UPDATE1 find2 lock2(None) re-read。
        rowcounts=[0, 0, 1, 0, 0, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    tasks = _repository_with_tokens("attempt-1-new").claim_source_sync_tasks(
        701,
        "experts",
        limit=3,
        lease_seconds=60,
    )

    assert [task.task_id for task in tasks] == [1]
    assert connection.commit_count == 1


def test_claim_fifo_non_frontier_failed_raises_stage_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非队头位置的 failed/retrying/未知状态仍 fail-closed。

    limit=3：第一条可领，第二条 find + lock 后发现 failed → 抛 StageConflictError。
    """

    cursor = RecordingCursor(
        ones=[
            # 第一轮 find 1 + lock 1（pending）。
            {"id": 1},
            {
                "id": 1,
                "sync_status": "pending",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
            # 第二轮 find 2 + lock 2 发现 failed。
            {"id": 2},
            {
                "id": 2,
                "sync_status": "failed",
                "next_retry_at": None,
                "lease_until": None,
                "attempt_token": None,
            },
        ],
        # 执行序：find1 lock1 UPDATE1 find2 lock2（拋异常前）。
        rowcounts=[0, 0, 1, 0, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskStageConflictError, match="不能跳过该队头任务"):
        _repository_with_tokens("attempt-1-new").claim_source_sync_tasks(
            701,
            "experts",
            limit=3,
            lease_seconds=60,
        )

    assert connection.rollback_count == 1
