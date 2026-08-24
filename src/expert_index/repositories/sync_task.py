"""来源同步任务的 MySQL 数据访问层。

本模块创建来源记录同步任务，并按任务 ID 读取完整持久化状态。
"""

from __future__ import annotations

import pymysql

from ..config import DatabaseConfig, require_index_database
from ..models import NewSourceSyncTask, SourceSyncTask


class MySQLSourceSyncTaskRepository:
    """持久化来源记录的增量同步任务。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存索引库连接配置，数据库连接由各方法按需创建。"""

        require_index_database(config)
        self.config = config

    def create_source_sync_task(
        self,
        task: NewSourceSyncTask,
    ) -> int:
        """按跨扫描 ``dedupe_key`` 幂等创建 pending 任务并返回 ID。

        新任务固定从 ``candidate/pending`` 开始。唯一键冲突后会再按
        ``dedupe_key`` 及其公式字段查询；只有确认是同一业务身份时才返回
        原任务 ID。旧批次内唯一键如果误命中不同任务，方法会回滚并报错。
        """

        if not isinstance(task, NewSourceSyncTask):
            raise TypeError("task 必须是 NewSourceSyncTask")

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO expert_index_sync_task(
                        dedupe_key, batch_id, event_source,
                        source_table, source_record_id, source_updated_at,
                        expert_id, operation_type, target_type,
                        processing_stage, sync_status, retry_count,
                        policy_version
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'candidate', 'pending', 0, %s
                    )
                    ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)
                    """,
                    (
                        task.dedupe_key,
                        task.batch_id,
                        task.event_source,
                        task.source_table,
                        task.source_record_id,
                        task.source_updated_at,
                        task.expert_id,
                        task.operation_type,
                        task.target_type,
                        task.policy_version,
                    ),
                )
                # 不依赖命中的具体唯一索引；只认可完整幂等身份对应的任务。
                cursor.execute(
                    """
                    SELECT id
                    FROM expert_index_sync_task
                    WHERE dedupe_key=%s
                      AND event_source=%s
                      AND source_table=%s
                      AND source_record_id=%s
                      AND source_updated_at <=> %s
                      AND operation_type=%s
                      AND target_type=%s
                      AND policy_version=%s
                    """,
                    (
                        task.dedupe_key,
                        task.event_source,
                        task.source_table,
                        task.source_record_id,
                        task.source_updated_at,
                        task.operation_type,
                        task.target_type,
                        task.policy_version,
                    ),
                )
                identity_row = cursor.fetchone()
                if identity_row is None:
                    raise RuntimeError("同步任务唯一键冲突，但未找到匹配的 dedupe_key 身份")
                task_id = int(identity_row["id"])
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
                    SELECT id, dedupe_key, batch_id, event_source,
                           source_table, source_record_id, source_updated_at,
                           expert_id, operation_type, target_type,
                           processing_stage, sync_status, retry_count,
                           attempt_token, lease_until, next_retry_at,
                           policy_version, error_stage, error_code,
                           error_type, error_message, result_code,
                           started_at, completed_at, created_at, updated_at
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
                dedupe_key=str(row["dedupe_key"]),
                batch_id=int(row["batch_id"]),
                event_source=str(row["event_source"]),
                source_table=str(row["source_table"]),
                source_record_id=int(row["source_record_id"]),
                source_updated_at=row.get("source_updated_at"),
                expert_id=(int(row["expert_id"]) if row.get("expert_id") is not None else None),
                operation_type=str(row["operation_type"]),
                target_type=str(row["target_type"]),
                processing_stage=str(row["processing_stage"]),
                sync_status=str(row["sync_status"]),
                retry_count=int(row["retry_count"]),
                attempt_token=row.get("attempt_token"),
                lease_until=row.get("lease_until"),
                next_retry_at=row.get("next_retry_at"),
                policy_version=row.get("policy_version"),
                error_stage=row.get("error_stage"),
                error_code=row.get("error_code"),
                error_type=row.get("error_type"),
                error_message=row.get("error_message"),
                result_code=row.get("result_code"),
                started_at=row.get("started_at"),
                completed_at=row.get("completed_at"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        finally:
            connection.close()


def _validate_positive_integer(field_name: str, value: int) -> None:
    """校验数据库标识字段为正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} 必须是正整数")
