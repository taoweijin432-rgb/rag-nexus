"""来源表 Checkpoint 与批次扫描状态的 MySQL 数据访问层。

本模块提供安全位置读取与初始化，以及批次固定扫描范围的创建与查询。每个公开方法
自行建立短连接，写操作在成功后提交，异常时回滚，最后关闭连接。
"""

from __future__ import annotations

import pymysql

from ..config import DatabaseConfig, require_index_database
from ..models import (
    CHECKPOINT_SOURCE_TABLES,
    BatchSourceScan,
    SourceCheckpoint,
    SourceScanCursor,
)


class MySQLSourceStateRepository:
    """持久化来源表安全位置和批次临时扫描范围。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存索引库连接配置，数据库连接由各方法按需创建。"""

        require_index_database(config)
        self.config = config

    def get_source_checkpoint(
        self,
        source_table: str,
    ) -> SourceCheckpoint | None:
        """读取指定来源表最后安全提交的 Checkpoint。

        返回 ``None`` 表示该来源表尚未初始化。该方法只读取已有状态，不创建游标、
        不锁定记录，也不推进 Checkpoint。
        """

        if not isinstance(source_table, str) or source_table not in CHECKPOINT_SOURCE_TABLES:
            raise ValueError(f"不支持的 Checkpoint 来源表: {source_table}")

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT source_table, last_updated_at, last_id,
                           last_batch_id, last_success_time, lock_version,
                           lease_owner, lease_until
                    FROM expert_index_source_checkpoint
                    WHERE source_table=%s
                    """,
                    (source_table,),
                )
                row = cursor.fetchone()
            if row is None:
                return None
            if row.get("source_table") != source_table:
                raise RuntimeError("Checkpoint 查询结果与请求来源表不一致")
            return SourceCheckpoint(
                source_table=source_table,
                last_updated_at=row["last_updated_at"],
                last_id=int(row["last_id"]),
                last_batch_id=(
                    int(row["last_batch_id"]) if row.get("last_batch_id") is not None else None
                ),
                last_success_time=row.get("last_success_time"),
                lock_version=int(row["lock_version"]),
                lease_owner=row.get("lease_owner"),
                lease_until=row.get("lease_until"),
            )
        finally:
            connection.close()

    def initialize_source_checkpoint(self, checkpoint: SourceCheckpoint) -> None:
        """插入调用方确认的来源表同步起点。

        初始 ``lock_version`` 必须为 0。方法使用普通 ``INSERT``；来源表已有
        Checkpoint 时由主键约束拒绝，不覆盖现有安全位置。
        """

        if not isinstance(checkpoint, SourceCheckpoint):
            raise TypeError("checkpoint 必须是 SourceCheckpoint")
        if checkpoint.lock_version != 0:
            raise ValueError("初始化 Checkpoint 的 lock_version 必须为 0")
        if checkpoint.lease_owner is not None or checkpoint.lease_until is not None:
            raise ValueError("初始化 Checkpoint 不能携带扫描租约")

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO expert_index_source_checkpoint(
                        source_table, last_updated_at, last_id,
                        last_batch_id, last_success_time, lock_version,
                        lease_owner, lease_until
                    ) VALUES (%s, %s, %s, %s, %s, 0, NULL, NULL)
                    """,
                    (
                        checkpoint.source_table,
                        checkpoint.last_updated_at,
                        checkpoint.last_id,
                        checkpoint.last_batch_id,
                        checkpoint.last_success_time,
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_batch_source_scan(self, scan: BatchSourceScan) -> None:
        """保存一个批次对指定来源表的固定扫描范围。

        新记录必须处于 ``pending``，且临时游标必须等于起点。方法使用普通
        ``INSERT``；同一批次和来源表已有记录时由唯一键拒绝，不覆盖已有进度。
        """

        if not isinstance(scan, BatchSourceScan):
            raise TypeError("scan 必须是 BatchSourceScan")
        if scan.current_cursor != scan.start_cursor:
            raise ValueError("新建批次来源记录的 current_cursor 必须等于 start_cursor")
        if scan.status != "pending" or scan.started_at is not None or scan.finished_at is not None:
            raise ValueError("新建批次来源记录必须处于未开始的 pending 状态")
        if any(
            count != 0
            for count in (
                scan.candidate_count,
                scan.success_count,
                scan.failed_count,
                scan.skipped_count,
            )
        ):
            raise ValueError("新建批次来源记录的处理计数必须全部为 0")

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO expert_index_batch_source(
                        batch_id, source_table,
                        start_updated_at, start_id,
                        cursor_updated_at, cursor_id,
                        upper_bound_updated_at, upper_bound_id,
                        checkpoint_lock_version, status,
                        candidate_count, success_count,
                        failed_count, skipped_count,
                        started_at, finished_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        scan.batch_id,
                        scan.source_table,
                        scan.start_cursor.updated_at,
                        scan.start_cursor.record_id,
                        scan.current_cursor.updated_at,
                        scan.current_cursor.record_id,
                        scan.upper_bound.updated_at,
                        scan.upper_bound.record_id,
                        scan.checkpoint_lock_version,
                        scan.status,
                        scan.candidate_count,
                        scan.success_count,
                        scan.failed_count,
                        scan.skipped_count,
                        scan.started_at,
                        scan.finished_at,
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_batch_source_scan(
        self,
        batch_id: int,
        source_table: str,
    ) -> BatchSourceScan | None:
        """读取一个批次对指定来源表保存的扫描范围和临时进度。

        返回 ``None`` 表示批次尚未创建该来源表扫描记录。方法只读取状态，不创建
        记录、不锁定行，也不修改临时游标或正式 Checkpoint。
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if not isinstance(source_table, str) or source_table not in CHECKPOINT_SOURCE_TABLES:
            raise ValueError(f"不支持的 Checkpoint 来源表: {source_table}")

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT batch_id, source_table,
                           start_updated_at, start_id,
                           cursor_updated_at, cursor_id,
                           upper_bound_updated_at, upper_bound_id,
                           checkpoint_lock_version, status,
                           candidate_count, success_count,
                           failed_count, skipped_count,
                           started_at, finished_at
                    FROM expert_index_batch_source
                    WHERE batch_id=%s AND source_table=%s
                    """,
                    (batch_id, source_table),
                )
                row = cursor.fetchone()
            if row is None:
                return None
            if int(row["batch_id"]) != batch_id or row.get("source_table") != source_table:
                raise RuntimeError("批次来源扫描查询结果与请求条件不一致")
            return BatchSourceScan(
                batch_id=batch_id,
                source_table=source_table,
                start_cursor=SourceScanCursor(
                    updated_at=row["start_updated_at"],
                    record_id=int(row["start_id"]),
                ),
                current_cursor=SourceScanCursor(
                    updated_at=row["cursor_updated_at"],
                    record_id=int(row["cursor_id"]),
                ),
                upper_bound=SourceScanCursor(
                    updated_at=row["upper_bound_updated_at"],
                    record_id=int(row["upper_bound_id"]),
                ),
                checkpoint_lock_version=int(row["checkpoint_lock_version"]),
                status=row["status"],
                candidate_count=int(row["candidate_count"]),
                success_count=int(row["success_count"]),
                failed_count=int(row["failed_count"]),
                skipped_count=int(row["skipped_count"]),
                started_at=row.get("started_at"),
                finished_at=row.get("finished_at"),
            )
        finally:
            connection.close()
