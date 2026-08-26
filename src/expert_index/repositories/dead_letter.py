"""Dead Letter 查询与受控处置 Repository。

所有写操作只针对本地 Index MySQL。requeue 和处置写回使用行锁、状态守卫和
``resolution_version`` 乐观锁，不删除失败历史，也不直接推进 Checkpoint。
"""

from __future__ import annotations

from dataclasses import dataclass

import pymysql

from ..config import DatabaseConfig, require_index_database
from ..models import DEAD_LETTER_RESOLUTION_STATUSES, DeadLetterRecord
from .sync_task import _validate_positive_integer, _validate_safe_error_message

__all__ = [
    "DeadLetterConflictError",
    "DeadLetterRequeueResult",
    "MySQLDeadLetterRepository",
]

_MAX_PAGE_SIZE = 200
_RESOLUTION_TARGETS = frozenset({"resolved", "ignored"})
_SELECT_COLUMNS = """
    id, task_id, batch_id, source_table, source_record_id, expert_id,
    failure_stage, reason_code, reason_message, retry_count, artifact_id,
    resolution_status, resolution_note, resolution_operator,
    resolution_request_id, resolution_version, resolved_at, created_at, updated_at
"""


class DeadLetterConflictError(RuntimeError):
    """表示 DLQ 状态、乐观锁或关联 Task 不满足处置条件。"""


@dataclass(frozen=True)
class DeadLetterRequeueResult:
    """一次原子 requeue 的结果，包含更新后的 DLQ 记录和任务 ID。"""

    dead_letter: DeadLetterRecord
    task_id: int


class MySQLDeadLetterRepository:
    """提供 DLQ 只读查询、requeue 和终态审计写回。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存 Index 连接配置，不接受 Source 数据库配置。"""

        require_index_database(config)
        self.config = config

    def get_dead_letter(self, dead_letter_id: int) -> DeadLetterRecord | None:
        """按 DLQ ID 读取一条记录，不锁定、不修改状态。"""

        _validate_positive_integer("dead_letter_id", dead_letter_id)
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM expert_index_sync_dead_letter WHERE id=%s",
                    (dead_letter_id,),
                )
                row = cursor.fetchone()
            return _dead_letter_from_row(row) if row is not None else None
        finally:
            connection.close()

    def list_dead_letters(
        self,
        *,
        resolution_status: str | None = None,
        task_id: int | None = None,
        after_id: int = 0,
        limit: int = 100,
    ) -> tuple[DeadLetterRecord, ...]:
        """按 ID 稳定分页读取 DLQ，不返回 Payload、向量或来源正文。"""

        if resolution_status is not None and resolution_status not in DEAD_LETTER_RESOLUTION_STATUSES:
            raise ValueError(f"不支持的 DLQ 处置状态: {resolution_status!r}")
        if task_id is not None:
            _validate_positive_integer("task_id", task_id)
        if isinstance(after_id, bool) or not isinstance(after_id, int) or after_id < 0:
            raise ValueError("after_id 必须是非负整数")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit 必须位于 1 到 {_MAX_PAGE_SIZE} 之间")
        clauses = ["id>%s"]
        params: list[object] = [after_id]
        if resolution_status is not None:
            clauses.append("resolution_status=%s")
            params.append(resolution_status)
        if task_id is not None:
            clauses.append("task_id=%s")
            params.append(task_id)
        params.append(limit)
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM expert_index_sync_dead_letter "
                    f"WHERE {' AND '.join(clauses)} ORDER BY id ASC LIMIT %s",
                    tuple(params),
                )
                return tuple(_dead_letter_from_row(row) for row in cursor.fetchall())
        finally:
            connection.close()

    def requeue_dead_letter(
        self,
        dead_letter_id: int,
        *,
        operator: str,
        request_id: str,
        resolution_note: str,
        expected_resolution_version: int,
    ) -> DeadLetterRequeueResult:
        """原子地把 ``dead + DLQ pending`` 恢复为可重试 Task。

        DLQ 记录标记为 ``requeued`` 并保留审计字段；Task 保留 dedupe_key、处理阶段和
        Artifact 关联，只清理终态/租约字段。``next_retry_at`` 设为数据库当前时间，
        使下一次 claim 生成新 attempt token 并正确增加 retry_count。
        """

        self._validate_resolution_input(
            dead_letter_id,
            operator=operator,
            request_id=request_id,
            resolution_note=resolution_note,
            expected_resolution_version=expected_resolution_version,
        )
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM expert_index_sync_dead_letter "
                    "WHERE id=%s FOR UPDATE",
                    (dead_letter_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise DeadLetterConflictError("DLQ 记录不存在")
                dead_letter = _dead_letter_from_row(row)
                if dead_letter.resolution_status != "pending":
                    raise DeadLetterConflictError("DLQ 记录不是 pending，禁止重复 requeue")
                if dead_letter.resolution_version != expected_resolution_version:
                    raise DeadLetterConflictError("DLQ resolution_version 已变化")
                cursor.execute(
                    "SELECT id, sync_status FROM expert_index_sync_task "
                    "WHERE id=%s FOR UPDATE",
                    (dead_letter.task_id,),
                )
                task_row = cursor.fetchone()
                if task_row is None or task_row["sync_status"] != "dead":
                    raise DeadLetterConflictError("关联 Task 不是 dead，禁止 requeue")
                cursor.execute(
                    """
                    UPDATE expert_index_sync_task
                    SET sync_status='pending', attempt_token=NULL, lease_until=NULL,
                        next_retry_at=CURRENT_TIMESTAMP(6), completed_at=NULL
                    WHERE id=%s AND sync_status='dead'
                    """,
                    (dead_letter.task_id,),
                )
                if cursor.rowcount != 1:
                    raise DeadLetterConflictError("关联 Task 状态已变化")
                cursor.execute(
                    """
                    UPDATE expert_index_sync_dead_letter
                    SET resolution_status='requeued', resolution_note=%s,
                        resolution_operator=%s, resolution_request_id=%s,
                        resolution_version=resolution_version+1,
                        resolved_at=CURRENT_TIMESTAMP(6)
                    WHERE id=%s AND resolution_status='pending'
                      AND resolution_version=%s
                    """,
                    (
                        resolution_note,
                        operator,
                        request_id,
                        dead_letter_id,
                        expected_resolution_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise DeadLetterConflictError("DLQ 乐观锁更新失败")
                cursor.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM expert_index_sync_dead_letter WHERE id=%s",
                    (dead_letter_id,),
                )
                updated = _dead_letter_from_row(cursor.fetchone())
            connection.commit()
            return DeadLetterRequeueResult(updated, dead_letter.task_id)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def resolve_dead_letter(
        self,
        dead_letter_id: int,
        *,
        resolution_status: str,
        operator: str,
        request_id: str,
        resolution_note: str,
        expected_resolution_version: int,
    ) -> DeadLetterRecord:
        """把 pending DLQ 标记为 resolved 或 ignored，并保留完整审计信息。"""

        self._validate_resolution_input(
            dead_letter_id,
            operator=operator,
            request_id=request_id,
            resolution_note=resolution_note,
            expected_resolution_version=expected_resolution_version,
        )
        if resolution_status not in _RESOLUTION_TARGETS:
            raise ValueError("resolution_status 只能是 resolved 或 ignored")
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE expert_index_sync_dead_letter
                    SET resolution_status=%s, resolution_note=%s,
                        resolution_operator=%s, resolution_request_id=%s,
                        resolution_version=resolution_version+1,
                        resolved_at=CURRENT_TIMESTAMP(6)
                    WHERE id=%s AND resolution_status='pending'
                      AND resolution_version=%s
                    """,
                    (
                        resolution_status,
                        resolution_note,
                        operator,
                        request_id,
                        dead_letter_id,
                        expected_resolution_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise DeadLetterConflictError("DLQ 状态或 resolution_version 已变化")
                cursor.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM expert_index_sync_dead_letter WHERE id=%s",
                    (dead_letter_id,),
                )
                result = _dead_letter_from_row(cursor.fetchone())
            connection.commit()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_resolution_input(
        dead_letter_id: int,
        *,
        operator: str,
        request_id: str,
        resolution_note: str,
        expected_resolution_version: int,
    ) -> None:
        """校验人工处置的身份、说明和乐观锁版本。"""

        _validate_positive_integer("dead_letter_id", dead_letter_id)
        for field_name, value, max_length in (
            ("operator", operator, 128),
            ("request_id", request_id, 128),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                or len(value) > max_length
                or any(char in value for char in "\r\n\x00")
            ):
                raise ValueError(f"{field_name} 必须是安全的单行标识")
        _validate_safe_error_message(resolution_note)
        if resolution_note is None:
            raise ValueError("resolution_note 必须提供")
        if (
            isinstance(expected_resolution_version, bool)
            or not isinstance(expected_resolution_version, int)
            or expected_resolution_version < 0
        ):
            raise ValueError("expected_resolution_version 必须是非负整数")


def _dead_letter_from_row(row: dict[str, object] | None) -> DeadLetterRecord:
    """把 DLQ 查询行映射为不含正文的领域记录。"""

    if row is None:
        raise DeadLetterConflictError("DLQ 更新后无法读取记录")
    return DeadLetterRecord(
        dead_letter_id=int(row["id"]),
        task_id=int(row["task_id"]),
        batch_id=int(row["batch_id"]),
        source_table=str(row["source_table"]),
        source_record_id=int(row["source_record_id"]),
        expert_id=int(row["expert_id"]) if row.get("expert_id") is not None else None,
        failure_stage=str(row["failure_stage"]),
        reason_code=str(row["reason_code"]),
        reason_message=row.get("reason_message"),
        retry_count=int(row["retry_count"]),
        artifact_id=int(row["artifact_id"]) if row.get("artifact_id") is not None else None,
        resolution_status=str(row["resolution_status"]),
        resolution_note=row.get("resolution_note"),
        resolution_operator=row.get("resolution_operator"),
        resolution_request_id=row.get("resolution_request_id"),
        resolution_version=int(row["resolution_version"]),
        resolved_at=row.get("resolved_at"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
