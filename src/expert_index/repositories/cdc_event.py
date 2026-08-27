"""CDC Event 与标准 Sync Task 的 Index MySQL 原子持久化。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import pymysql

from ..cdc.ingest import (
    CdcBatchPersistencePort,
    CdcPersistenceConflictError,
    CdcTaskPlan,
    CdcTaskReceipt,
)
from ..config import DatabaseConfig, require_index_database
from ..models import NewSourceSyncTask


class MySQLCdcBatchPersistence(CdcBatchPersistencePort):
    """在一个 Index MySQL 事务中幂等写入 CDC Event 并创建/复用 Sync Task。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存并校验 Index 数据库配置；绝不接受 Source 角色连接。"""

        require_index_database(config)
        self.config = config

    def persist_batch(self, plans: Sequence[CdcTaskPlan]) -> tuple[CdcTaskReceipt, ...]:
        """原子提交一批 Event/Task，任何身份冲突都回滚整批。"""

        if not isinstance(plans, tuple):
            plans = tuple(plans)
        if not plans:
            return ()

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                receipts: list[CdcTaskReceipt] = []
                for plan in plans:
                    event_id, event_status, existing_task_id, is_new = _insert_or_verify_event(
                        cursor, plan
                    )
                    if event_status == "success":
                        if existing_task_id is None:
                            raise CdcPersistenceConflictError(
                                "success CDC Event 缺少关联 Sync Task"
                            )
                        receipts.append(
                            CdcTaskReceipt(plan.cdc_event_id, existing_task_id, "reused_success")
                        )
                        continue

                    task_id = existing_task_id or _insert_or_verify_task(cursor, plan)
                    cursor.execute(
                        """
                        UPDATE expert_index_cdc_event
                        SET task_id=%s
                        WHERE id=%s AND (task_id IS NULL OR task_id=%s)
                        """,
                        (task_id, event_id, task_id),
                    )
                    # MySQL 对“已是相同值”的 UPDATE 可能返回 0；该情况仍是安全幂等。
                    if cursor.rowcount not in (0, 1):
                        raise CdcPersistenceConflictError("CDC Event 关联 Sync Task 发生冲突")
                    if cursor.rowcount == 0:
                        cursor.execute(
                            "SELECT task_id FROM expert_index_cdc_event WHERE id=%s FOR UPDATE",
                            (event_id,),
                        )
                        linked = cursor.fetchone()
                        if linked is None or linked.get("task_id") != task_id:
                            raise CdcPersistenceConflictError(
                                "CDC Event 已关联其他 Sync Task"
                            )
                    status = "created" if is_new else "reused_pending"
                    receipts.append(CdcTaskReceipt(plan.cdc_event_id, task_id, status))
            connection.commit()
            return tuple(receipts)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _insert_or_verify_event(cursor: Any, plan: CdcTaskPlan) -> tuple[int, str, int | None, bool]:
    """按新身份或旧 position 唯一键锁定 Event，并核验不可变定位字段。"""

    event = plan.event
    cursor.execute(
        """
        SELECT id, cdc_event_id, local_batch_id, task_id, status,
               destination, binlog_file, binlog_position, row_index,
               source_schema, source_table, operation_type,
               source_record_id, expert_id, binlog_server_id
        FROM expert_index_cdc_event
        WHERE destination=%s AND binlog_file=%s
          AND binlog_position=%s AND row_index=%s
        FOR UPDATE
        """,
        ("company_expert", event.logfile_name, event.logfile_offset, event.row_index),
    )
    existing = cursor.fetchone()
    if existing is None:
        # position 不是生产身份的替代物；同一 cdc_event_id 若位置变化，必须报冲突。
        cursor.execute(
            """
            SELECT id, cdc_event_id, local_batch_id, task_id, status,
                   destination, binlog_file, binlog_position, row_index,
                   source_schema, source_table, operation_type,
                   source_record_id, expert_id, binlog_server_id
            FROM expert_index_cdc_event
            WHERE cdc_event_id=%s
            FOR UPDATE
            """,
            (plan.cdc_event_id,),
        )
        existing = cursor.fetchone()
    if existing is not None:
        _verify_event_identity(existing, plan)
        if existing.get("cdc_event_id") is None:
            cursor.execute(
                "UPDATE expert_index_cdc_event SET cdc_event_id=%s WHERE id=%s",
                (plan.cdc_event_id, existing["id"]),
            )
        return (
            int(existing["id"]),
            str(existing["status"]),
            int(existing["task_id"]) if existing.get("task_id") is not None else None,
            False,
        )

    event_time = _event_time(plan)
    try:
        cursor.execute(
            """
            INSERT INTO expert_index_cdc_event(
                cdc_event_id, local_batch_id, task_id, policy_version,
                destination, binlog_file, binlog_position, row_index, binlog_server_id,
                event_time, source_schema, source_table, operation_type,
                source_record_id, expert_id, status
            ) VALUES (
                %s, %s, NULL, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, 'pending'
            )
            """,
            (
                plan.cdc_event_id,
                plan.task.batch_id,
                plan.task.policy_version,
                "company_expert",
                event.logfile_name,
                event.logfile_offset,
                event.row_index,
                event.server_id,
                event_time,
                event.source_schema,
                event.source_table,
                event.operation_type,
                event.source_record_id,
                event.expert_id,
            ),
        )
    except pymysql.IntegrityError as exc:
        # 并发消费者可能在查询后先插入同一身份；锁定已提交行并复用它。
        cursor.execute(
            """
            SELECT id, cdc_event_id, local_batch_id, task_id, status,
                   destination, binlog_file, binlog_position, row_index,
                   source_schema, source_table, operation_type,
                   source_record_id, expert_id, binlog_server_id
            FROM expert_index_cdc_event
            WHERE cdc_event_id=%s
            FOR UPDATE
            """,
            (plan.cdc_event_id,),
        )
        existing = cursor.fetchone()
        if existing is None:
            raise CdcPersistenceConflictError("CDC Event 唯一身份写入冲突") from exc
        _verify_event_identity(existing, plan)
        return (
            int(existing["id"]),
            str(existing["status"]),
            int(existing["task_id"]) if existing.get("task_id") is not None else None,
            False,
        )
    return int(cursor.lastrowid), "pending", None, True


def _verify_event_identity(existing: dict[str, Any], plan: CdcTaskPlan) -> None:
    """拒绝同一 position/cdc_event_id 对应不同不可变事件内容。"""

    event = plan.event
    expected = {
        "cdc_event_id": plan.cdc_event_id,
        "destination": "company_expert",
        "binlog_file": event.logfile_name,
        "binlog_position": event.logfile_offset,
        "row_index": event.row_index,
        "source_schema": event.source_schema,
        "source_table": event.source_table,
        "operation_type": event.operation_type,
        "source_record_id": event.source_record_id,
        "expert_id": event.expert_id,
        "binlog_server_id": event.server_id,
    }
    for field, value in expected.items():
        current = existing.get(field)
        if current is not None and current != value:
            raise CdcPersistenceConflictError(f"CDC Event 不可变字段冲突: {field}")


def _insert_or_verify_task(cursor: Any, plan: CdcTaskPlan) -> int:
    """在同一事务中按 dedupe_key 幂等创建 CDC Sync Task。"""

    task = plan.task
    cursor.execute(
        """
        SELECT id, dedupe_key, batch_id, event_source,
               source_table, source_record_id, source_updated_at,
               expert_id, operation_type, target_type, policy_version
        FROM expert_index_sync_task
        WHERE dedupe_key=%s
        FOR UPDATE
        """,
        (task.dedupe_key,),
    )
    existing = cursor.fetchone()
    if existing is not None:
        _verify_task_identity(existing, task)
        return int(existing["id"])

    cursor.execute(
        """
        INSERT INTO expert_index_sync_task(
            dedupe_key, batch_id, event_source,
            source_table, source_record_id, source_updated_at,
            expert_id, operation_type, target_type,
            processing_stage, sync_status, retry_count, policy_version
        ) VALUES (
            %s, %s, 'cdc', %s, %s, NULL, %s, %s, %s,
            'candidate', 'pending', 0, %s
        )
        ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)
        """,
        (
            task.dedupe_key,
            task.batch_id,
            task.source_table,
            task.source_record_id,
            task.expert_id,
            task.operation_type,
            task.target_type,
            task.policy_version,
        ),
    )
    cursor.execute(
        """
        SELECT id, dedupe_key, batch_id, event_source,
               source_table, source_record_id, source_updated_at,
               expert_id, operation_type, target_type, policy_version
        FROM expert_index_sync_task
        WHERE dedupe_key=%s
        FOR UPDATE
        """,
        (task.dedupe_key,),
    )
    row = cursor.fetchone()
    if row is None:
        raise CdcPersistenceConflictError("CDC Task 幂等写入后无法读取任务")
    _verify_task_identity(row, task)
    return int(row["id"])


def _verify_task_identity(row: dict[str, Any], task: NewSourceSyncTask) -> None:
    """核验 CDC Task 的不可变业务字段，防止唯一键误复用。"""

    expected = {
        "dedupe_key": task.dedupe_key,
        "batch_id": task.batch_id,
        "event_source": "cdc",
        "source_table": task.source_table,
        "source_record_id": task.source_record_id,
        "source_updated_at": None,
        "expert_id": task.expert_id,
        "operation_type": task.operation_type,
        "target_type": task.target_type,
        "policy_version": task.policy_version,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise CdcPersistenceConflictError(f"CDC Task 不可变字段冲突: {field}")


def _event_time(plan: CdcTaskPlan) -> datetime:
    """把 Canal epoch milliseconds 转为 UTC datetime，供旧 009 event_time 使用。"""

    execute_time = plan.event.execute_time
    if execute_time is None or execute_time < 0:
        raise CdcPersistenceConflictError("CDC Event 缺少合法 execute_time")
    try:
        return datetime.fromtimestamp(execute_time / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise CdcPersistenceConflictError("CDC Event execute_time 无法归一") from exc


__all__ = ["MySQLCdcBatchPersistence"]
