"""来源同步任务的 MySQL 数据访问层。

本模块创建来源记录同步任务，并按任务 ID 读取完整持久化状态。
"""

from __future__ import annotations

import pymysql

from ..config import DatabaseConfig
from ..models import (
    CHECKPOINT_SOURCE_TABLES,
    SOURCE_SYNC_OPERATION_TYPES,
    SourceSyncTask,
)


class MySQLSourceSyncTaskRepository:
    """持久化来源记录的增量同步任务。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存索引库连接配置，数据库连接由各方法按需创建。"""

        self.config = config

    def create_source_sync_task(
        self,
        *,
        batch_id: int,
        source_table: str,
        source_record_id: int,
        expert_id: int | None,
        operation_type: str,
    ) -> int:
        """幂等创建 pending 任务并返回任务 ID。

        同一批次、来源表和来源记录已存在任务时返回原任务 ID，不覆盖其专家映射、
        操作类型、处理状态或结果字段。
        """

        _validate_positive_integer("batch_id", batch_id)
        _validate_positive_integer("source_record_id", source_record_id)
        if expert_id is not None:
            _validate_positive_integer("expert_id", expert_id)
        if source_table not in CHECKPOINT_SOURCE_TABLES:
            raise ValueError(f"不支持的同步任务来源表: {source_table}")
        if operation_type not in SOURCE_SYNC_OPERATION_TYPES:
            raise ValueError(f"不支持的同步任务操作类型: {operation_type}")

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO expert_index_sync_task(
                        batch_id, source_table, source_record_id,
                        expert_id, operation_type, sync_status, retry_count
                    ) VALUES (%s, %s, %s, %s, %s, 'pending', 0)
                    ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)
                    """,
                    (
                        batch_id,
                        source_table,
                        source_record_id,
                        expert_id,
                        operation_type,
                    ),
                )
                # INSERT 返回新 ID；唯一键冲突时 LAST_INSERT_ID 返回原任务 ID。
                task_id = int(cursor.lastrowid)
            connection.commit()
            return task_id
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_source_sync_task(self, task_id: int) -> SourceSyncTask | None:
        """按任务 ID 读取同步任务。

        返回 ``None`` 表示任务不存在。方法只读取已提交状态，不锁定或修改任务记录。
        """

        _validate_positive_integer("task_id", task_id)
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, batch_id, source_table, source_record_id,
                           expert_id, operation_type, sync_status, retry_count,
                           error_stage, error_type, error_message, result_code,
                           started_at, completed_at, created_at
                    FROM expert_index_sync_task
                    WHERE id=%s
                    """,
                    (task_id,),
                )
                row = cursor.fetchone()
            if row is None:
                return None
            if int(row["id"]) != task_id:
                raise RuntimeError("同步任务查询结果与请求 ID 不一致")
            return SourceSyncTask(
                task_id=task_id,
                batch_id=int(row["batch_id"]),
                source_table=str(row["source_table"]),
                source_record_id=int(row["source_record_id"]),
                expert_id=(int(row["expert_id"]) if row.get("expert_id") is not None else None),
                operation_type=str(row["operation_type"]),
                sync_status=str(row["sync_status"]),
                retry_count=int(row["retry_count"]),
                error_stage=row.get("error_stage"),
                error_type=row.get("error_type"),
                error_message=row.get("error_message"),
                result_code=row.get("result_code"),
                started_at=row.get("started_at"),
                completed_at=row.get("completed_at"),
                created_at=row["created_at"],
            )
        finally:
            connection.close()


def _validate_positive_integer(field_name: str, value: int) -> None:
    """校验数据库标识字段为正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} 必须是正整数")
