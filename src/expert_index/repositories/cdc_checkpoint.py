"""P7.7 CDC Event success 与业务 Checkpoint 的 Index MySQL 原子提交。"""

from __future__ import annotations

from typing import Any

import pymysql

from ..cdc.finalize import (
    CdcBatchCommitRequest,
    CdcBatchNotReadyError,
    CdcBinlogPosition,
    CdcCheckpointCommitResult,
    CdcCheckpointConflictError,
)
from ..config import DatabaseConfig, require_index_database


class MySQLCdcCheckpointRepository:
    """在同一 Index 事务内落定 Event success 并 CAS 推进业务位点。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存并校验 Index 数据库配置，拒绝 Source 角色连接。"""

        require_index_database(config)
        self.config = config

    def commit_applied_batch(
        self,
        request: CdcBatchCommitRequest,
    ) -> CdcCheckpointCommitResult:
        """核验当前及更早事件，再原子提交 Event 和 Checkpoint。"""

        if not isinstance(request, CdcBatchCommitRequest):
            raise TypeError("request 必须是 CdcBatchCommitRequest")
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                checkpoint_row = _lock_checkpoint(cursor, request.destination)
                event_rows = _lock_requested_events(cursor, request)
                _validate_requested_events(event_rows, request)
                current_position = _checkpoint_position(checkpoint_row)
                relation = (
                    current_position.compare(request.applied_position)
                    if current_position is not None
                    else -1
                )
                if relation >= 0:
                    _require_events_already_success(event_rows)
                    _lock_and_reject_earlier_blockers(cursor, request)
                    result = CdcCheckpointCommitResult(
                        position=current_position or request.applied_position,
                        lock_version=int(checkpoint_row["lock_version"]),
                        already_applied=True,
                    )
                    connection.commit()
                    return result

                _lock_and_reject_earlier_blockers(cursor, request)
                _mark_requested_events_success(cursor, event_rows)
                result = _advance_checkpoint(cursor, checkpoint_row, request)
            connection.commit()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _lock_checkpoint(cursor: Any, destination: str) -> dict[str, Any] | None:
    """锁定 destination 当前业务位点；首次提交返回 None。"""

    cursor.execute(
        """
        SELECT destination, binlog_server_id, binlog_file, binlog_position,
               last_local_batch_id, lock_version
        FROM expert_index_cdc_checkpoint
        WHERE destination=%s
        FOR UPDATE
        """,
        (destination,),
    )
    return cursor.fetchone()


def _lock_requested_events(
    cursor: Any,
    request: CdcBatchCommitRequest,
) -> tuple[dict[str, Any], ...]:
    """按请求身份锁定当前 batch Event 及其 Task 终态证据。"""

    event_ids = tuple(event.cdc_event_id for event in request.events)
    if not event_ids:
        return ()
    placeholders = ", ".join(["%s"] * len(event_ids))
    cursor.execute(
        f"""
        SELECT e.id, e.cdc_event_id, e.task_id, e.status,
               e.destination, e.binlog_server_id, e.binlog_file,
               e.binlog_position, e.row_index,
               t.event_source AS task_event_source,
               t.operation_type AS task_operation_type,
               t.processing_stage AS task_processing_stage,
               t.sync_status AS task_sync_status
        FROM expert_index_cdc_event e
        LEFT JOIN expert_index_sync_task t ON t.id=e.task_id
        WHERE e.cdc_event_id IN ({placeholders})
        ORDER BY e.binlog_file, e.binlog_position, e.row_index, e.id
        FOR UPDATE
        """,
        event_ids,
    )
    return tuple(cursor.fetchall())


def _validate_requested_events(
    rows: tuple[dict[str, Any], ...],
    request: CdcBatchCommitRequest,
) -> None:
    """核验 Event 定位、Task 关联和 completed/success 终态。"""

    expected = {event.cdc_event_id: event for event in request.events}
    actual = {str(row["cdc_event_id"]): row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise CdcCheckpointConflictError("CDC 提交 Event 数量或身份不一致")
    for event_id, identity in expected.items():
        row = actual[event_id]
        if row.get("task_id") is None or row.get("binlog_server_id") is None:
            raise CdcCheckpointConflictError("CDC Event 缺少 Task 或 server_id 身份")
        actual_position = CdcBinlogPosition(
            int(row["binlog_server_id"]),
            str(row["binlog_file"]),
            int(row["binlog_position"]),
        )
        if (
            row.get("destination") != request.destination
            or int(row["task_id"]) != identity.task_id
            or actual_position != identity.position
            or int(row["row_index"]) != identity.row_index
        ):
            raise CdcCheckpointConflictError("CDC Event 不可变身份与提交请求不一致")
        if (
            row.get("task_event_source") != "cdc"
            or row.get("task_operation_type") not in {"insert", "update", "delete"}
            or row.get("task_processing_stage") != "completed"
            or row.get("task_sync_status") != "success"
        ):
            raise CdcBatchNotReadyError("CDC Task 尚未完成可提交终态")
        if row.get("status") not in {"pending", "running", "retry", "success"}:
            raise CdcBatchNotReadyError("CDC Event 处于不可提交终态")


def _require_events_already_success(rows: tuple[dict[str, Any], ...]) -> None:
    """已提交位点只接受已同步落定的 Event，拒绝自动修复矛盾。"""

    if any(row.get("status") != "success" for row in rows):
        raise CdcCheckpointConflictError("CDC Checkpoint 已越过但 Event 未标记 success")


def _lock_and_reject_earlier_blockers(cursor: Any, request: CdcBatchCommitRequest) -> None:
    """锁定所有未成功 Event，拒绝跳过当前位点之前的任务。"""

    cursor.execute(
        """
        SELECT cdc_event_id, binlog_server_id, binlog_file, binlog_position
        FROM expert_index_cdc_event
        WHERE destination=%s AND status <> 'success'
        ORDER BY binlog_file, binlog_position, row_index, id
        FOR UPDATE
        """,
        (request.destination,),
    )
    requested_ids = {event.cdc_event_id for event in request.events}
    for row in cursor.fetchall():
        event_id = row.get("cdc_event_id")
        if event_id in requested_ids:
            continue
        if row.get("binlog_server_id") is None:
            raise CdcCheckpointConflictError("CDC 更早 Event 缺少 server_id 定位")
        position = CdcBinlogPosition(
            int(row["binlog_server_id"]),
            str(row["binlog_file"]),
            int(row["binlog_position"]),
        )
        if position.compare(request.applied_position) <= 0:
            raise CdcBatchNotReadyError("CDC 更早位点仍有未成功 Event")


def _mark_requested_events_success(
    cursor: Any,
    rows: tuple[dict[str, Any], ...],
) -> None:
    """将当前 batch Event 落定为 success，不覆盖其他事件。"""

    for row in rows:
        if row.get("status") == "success":
            continue
        cursor.execute(
            """
            UPDATE expert_index_cdc_event
            SET status='success', processed_at=CURRENT_TIMESTAMP(6),
                error_type=NULL, error_message=NULL
            WHERE id=%s AND status=%s
            """,
            (int(row["id"]), row["status"]),
        )
        if cursor.rowcount != 1:
            raise CdcCheckpointConflictError("CDC Event success 写入发生并发冲突")


def _advance_checkpoint(
    cursor: Any,
    checkpoint_row: dict[str, Any] | None,
    request: CdcBatchCommitRequest,
) -> CdcCheckpointCommitResult:
    """首次创建或按 lock_version CAS 推进 CDC 业务位点。"""

    position = request.applied_position
    if checkpoint_row is None:
        try:
            cursor.execute(
                """
                INSERT INTO expert_index_cdc_checkpoint(
                    destination, binlog_server_id, binlog_file, binlog_position,
                    last_local_batch_id, lock_version, last_success_at
                ) VALUES (%s, %s, %s, %s, %s, 1, CURRENT_TIMESTAMP(6))
                """,
                (
                    request.destination,
                    position.server_id,
                    position.logfile_name,
                    position.logfile_offset,
                    request.local_batch_id,
                ),
            )
        except pymysql.IntegrityError as error:
            raise CdcCheckpointConflictError("CDC Checkpoint 首次写入发生并发冲突") from error
        return CdcCheckpointCommitResult(position, 1, False)

    old_version = int(checkpoint_row["lock_version"])
    cursor.execute(
        """
        UPDATE expert_index_cdc_checkpoint
        SET binlog_server_id=%s, binlog_file=%s, binlog_position=%s,
            last_local_batch_id=%s, lock_version=lock_version+1,
            last_success_at=CURRENT_TIMESTAMP(6)
        WHERE destination=%s AND lock_version=%s
        """,
        (
            position.server_id,
            position.logfile_name,
            position.logfile_offset,
            request.local_batch_id,
            request.destination,
            old_version,
        ),
    )
    if cursor.rowcount != 1:
        raise CdcCheckpointConflictError("CDC Checkpoint CAS 推进发生并发冲突")
    return CdcCheckpointCommitResult(position, old_version + 1, False)


def _checkpoint_position(row: dict[str, Any] | None) -> CdcBinlogPosition | None:
    """将数据库 Checkpoint 行转为领域位点。"""

    if row is None:
        return None
    return CdcBinlogPosition(
        int(row["binlog_server_id"]),
        str(row["binlog_file"]),
        int(row["binlog_position"]),
    )


__all__ = ["MySQLCdcCheckpointRepository"]
