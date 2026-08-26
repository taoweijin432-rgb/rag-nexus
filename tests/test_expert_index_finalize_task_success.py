"""P5.6 ``finalize_source_sync_task_success`` 原子落定测试。

覆盖单事务内锁定 Task + 校验 Artifact identity/payload_hash + 校验阶段属于
contract_ready/handed_off + 校验 task_result_code 命中冻结枚举 + 原子写 completed/success
并清除 lease/error；以及并发 fencing、artifact 不符、非法阶段、非法枚举的 fail-closed。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.repositories.sync_task import (
    MySQLSourceSyncTaskRepository,
    SourceSyncTaskLeaseConflictError,
    SourceSyncTaskStageConflictError,
)

# ---------------------------------------------------------------------------
# 测试桩
# ---------------------------------------------------------------------------

class RecordingCursor:
    """记录 SQL 并按序返回测试指定的查询结果。"""

    def __init__(
        self,
        *,
        ones: list[dict[str, Any] | None] | None = None,
        rowcounts: list[int] | None = None,
    ) -> None:
        self.ones = list(ones or [])
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

    def fetchone(self) -> dict[str, Any] | None:
        if self.ones:
            return self.ones.pop(0)
        return None


class RecordingConnection:
    """记录事务提交/回滚和连接关闭。"""

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
    monkeypatch.setattr(
        "expert_index.repositories.sync_task.pymysql.connect",
        lambda **_kwargs: connection,
    )


def _task_row(**changes: object) -> dict[str, Any]:
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
        "processing_stage": "contract_ready",
        "sync_status": "processing",
        "retry_count": 0,
        "attempt_token": "attempt-801",
        "lease_until": datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc),
        "next_retry_at": None,
        "policy_version": "1.3",
        "error_stage": None,
        "error_code": None,
        "error_type": None,
        "error_message": None,
        "result_code": None,
        "started_at": now,
        "completed_at": None,
        "created_at": now,
        "updated_at": now,
        "lease_is_active": 1,
    }
    values.update(changes)
    return values


def _artifact_row(**changes: object) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": 5001,
        "task_id": 801,
        "artifact_type": "contract_payload",
        "artifact_version": "v1",
        "payload_json": "{}",
        "payload_reference": None,
        "payload_hash": "a" * 64,
        "expires_at": None,
        "created_at": datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc),
    }
    values.update(changes)
    return values


# ---------------------------------------------------------------------------
# 成功路径
# ---------------------------------------------------------------------------

def test_finalize_from_contract_ready_atomically_closes_to_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """contract_ready + 有效 Artifact → 单事务原子写 completed/success。"""

    completed_at = datetime(2026, 8, 21, 10, 4, tzinfo=timezone.utc)
    locked = _task_row(processing_stage="contract_ready")
    artifact = _artifact_row()
    succeeded = dict(
        locked,
        processing_stage="completed",
        sync_status="success",
        attempt_token=None,
        lease_until=None,
        result_code="embedded",
        completed_at=completed_at,
    )
    succeeded.pop("lease_is_active")
    cursor = RecordingCursor(
        ones=[locked, artifact, succeeded],
        rowcounts=[0, 0, 1, 0],
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().finalize_source_sync_task_success(
        801,
        "attempt-801",
        task_result_code="embedded",
        artifact_id=5001,
        artifact_payload_hash="a" * 64,
    )

    update_sql = cursor.executed[2][0]
    assert "processing_stage='completed'" in update_sql
    assert "sync_status='success'" in update_sql
    assert "processing_stage IN ('contract_ready', 'handed_off')" in update_sql
    assert "attempt_token=%s" in update_sql
    assert "lease_until > CURRENT_TIMESTAMP(6)" in update_sql
    assert cursor.executed[2][1] == ("embedded", 801, "attempt-801")
    assert task.sync_status == "success"
    assert task.result_code == "embedded"
    assert task.completed_at == completed_at
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True


def test_finalize_from_handed_off_recovers_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """handed_off（崩溃恢复入口）+ 有效 Artifact → 仍可 finalize 为 success。"""

    locked = _task_row(processing_stage="handed_off")
    artifact = _artifact_row()
    succeeded = dict(locked, processing_stage="completed", sync_status="success",
                     attempt_token=None, lease_until=None, result_code="no_action_unchanged",
                     completed_at=datetime(2026, 8, 21, 10, 4, tzinfo=timezone.utc))
    succeeded.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, artifact, succeeded], rowcounts=[0, 0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().finalize_source_sync_task_success(
        801, "attempt-801",
        task_result_code="no_action_unchanged",
        artifact_id=5001, artifact_payload_hash="a" * 64,
    )

    assert task.sync_status == "success"
    assert task.result_code == "no_action_unchanged"


# ---------------------------------------------------------------------------
# 阶段守卫（frozen #7：只接受 contract_ready/handed_off）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", ["validated", "candidate", "source_read", "normalized", "chunk_built"])
def test_finalize_rejects_non_finalizable_stage(
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    """validated 等前置阶段不允许直接 finalize（Artifact 可能尚未持久化）。"""

    locked = _task_row(processing_stage=stage)
    artifact = _artifact_row()
    cursor = RecordingCursor(ones=[locked, artifact], rowcounts=[0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskStageConflictError, match="contract_ready/handed_off"):
        _repository().finalize_source_sync_task_success(
            801, "attempt-801",
            task_result_code="embedded",
            artifact_id=5001, artifact_payload_hash="a" * 64,
        )

    assert connection.rollback_count == 1
    assert connection.closed is True


# ---------------------------------------------------------------------------
# Artifact 校验（frozen #8：同事务内锁定校验）
# ---------------------------------------------------------------------------

def test_finalize_rejects_missing_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """同 task_id 无匹配 contract_payload Artifact → 确定性 rejected。"""

    locked = _task_row(processing_stage="contract_ready")
    cursor = RecordingCursor(ones=[locked, None], rowcounts=[0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskStageConflictError, match="缺少匹配的 contract_payload Artifact"):
        _repository().finalize_source_sync_task_success(
            801, "attempt-801",
            task_result_code="embedded",
            artifact_id=5001, artifact_payload_hash="a" * 64,
        )

    assert connection.rollback_count == 1


def test_finalize_rejects_artifact_payload_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact 存在但 payload_hash 与传入值不一致 → fail-closed。"""

    locked = _task_row(processing_stage="contract_ready")
    artifact = _artifact_row(payload_hash="b" * 64)
    cursor = RecordingCursor(ones=[locked, artifact], rowcounts=[0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskStageConflictError, match="payload_hash 与传入值不一致"):
        _repository().finalize_source_sync_task_success(
            801, "attempt-801",
            task_result_code="embedded",
            artifact_id=5001, artifact_payload_hash="a" * 64,
        )

    assert connection.rollback_count == 1


# ---------------------------------------------------------------------------
# 并发 fencing（frozen #9：旧 token 晚到不能修改）
# ---------------------------------------------------------------------------

def test_finalize_rejects_stale_attempt_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧 attempt token 或过期 lease → LeaseConflictError，不覆盖任务。"""

    locked = _task_row(processing_stage="contract_ready", attempt_token="new-token")
    cursor = RecordingCursor(ones=[locked], rowcounts=[0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskLeaseConflictError):
        _repository().finalize_source_sync_task_success(
            801, "old-token",
            task_result_code="embedded",
            artifact_id=5001, artifact_payload_hash="a" * 64,
        )

    assert connection.rollback_count == 1


def test_finalize_rejects_when_update_rowcount_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """UPDATE rowcount=0（阶段或 lease 在锁后被并发改）→ LeaseConflictError。"""

    locked = _task_row(processing_stage="contract_ready")
    artifact = _artifact_row()
    cursor = RecordingCursor(ones=[locked, artifact], rowcounts=[0, 0, 0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(SourceSyncTaskLeaseConflictError, match="阶段或 attempt lease 已变化"):
        _repository().finalize_source_sync_task_success(
            801, "attempt-801",
            task_result_code="embedded",
            artifact_id=5001, artifact_payload_hash="a" * 64,
        )

    assert connection.rollback_count == 1


# ---------------------------------------------------------------------------
# 枚举校验（frozen #2/#3：task_result_code 命中冻结枚举）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "code",
    ["embedded", "metadata_updated", "disabled", "mixed_mutation",
     "no_action_unchanged", "recovered_unchanged", "expert_deleted", "no_chunkable_content"],
)
def test_finalize_accepts_all_frozen_result_codes(
    monkeypatch: pytest.MonkeyPatch, code: str,
) -> None:
    """8 个冻结枚举成员全部可落定。"""

    locked = _task_row(processing_stage="contract_ready")
    artifact = _artifact_row()
    succeeded = dict(locked, processing_stage="completed", sync_status="success",
                     attempt_token=None, lease_until=None, result_code=code,
                     completed_at=datetime(2026, 8, 21, 10, 4, tzinfo=timezone.utc))
    succeeded.pop("lease_is_active")
    cursor = RecordingCursor(ones=[locked, artifact, succeeded], rowcounts=[0, 0, 1, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    task = _repository().finalize_source_sync_task_success(
        801, "attempt-801",
        task_result_code=code,
        artifact_id=5001, artifact_payload_hash="a" * 64,
    )

    assert task.result_code == code


def test_finalize_rejects_unknown_result_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """非冻结枚举的 result_code → ValueError，不触碰数据库。"""

    connection = RecordingConnection(RecordingCursor())
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ValueError, match="冻结枚举"):
        _repository().finalize_source_sync_task_success(
            801, "attempt-801",
            task_result_code="downstream_applied",
            artifact_id=5001, artifact_payload_hash="a" * 64,
        )


def test_finalize_rejects_receipt_scope_result_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """scope 级 result_code（如 scope_committed）不能直接作为 task_result_code。"""

    connection = RecordingConnection(RecordingCursor())
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ValueError, match="冻结枚举"):
        _repository().finalize_source_sync_task_success(
            801, "attempt-801",
            task_result_code="scope_committed",
            artifact_id=5001, artifact_payload_hash="a" * 64,
        )


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def test_finalize_rejects_invalid_artifact_id(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = RecordingConnection(RecordingCursor())
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ValueError, match="artifact_id"):
        _repository().finalize_source_sync_task_success(
            801, "attempt-801",
            task_result_code="embedded",
            artifact_id=0, artifact_payload_hash="a" * 64,
        )


def test_finalize_rejects_invalid_payload_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = RecordingConnection(RecordingCursor())
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ValueError, match="payload_hash"):
        _repository().finalize_source_sync_task_success(
            801, "attempt-801",
            task_result_code="embedded",
            artifact_id=5001, artifact_payload_hash="short",
        )
