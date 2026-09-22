"""来源表 Checkpoint 与批次扫描状态的 MySQL 数据访问层。

本模块提供安全位置读写、扫描 lease 管理，以及批次固定扫描范围的创建、查询与恢复。
每个公开方法自行建立短连接，写操作在成功后提交，异常时回滚，最后关闭连接。
"""

from __future__ import annotations

from typing import Any

import pymysql

from ..config import DatabaseConfig, require_index_database
from ..models import (
    PERSISTED_SOURCE_TABLES,
    BatchSourceScan,
    SourceCheckpoint,
    SourceScanCursor,
)
from .batch_id import allocate_batch_id


class SourceCheckpointNotInitializedError(RuntimeError):
    """表示指定来源表尚未建立显式初始 Checkpoint。"""


class SourceLeaseConflictError(RuntimeError):
    """表示来源扫描 lease 已被占用或调用方不再拥有有效 lease。"""


class BatchSourceRecoveryConflictError(RuntimeError):
    """表示同一来源表存在多个未完成批次，无法安全选择恢复目标。"""


class BatchSourceCursorConflictError(RuntimeError):
    """表示批次临时游标、任务证据或 Checkpoint 版本不满足 CAS 条件。"""


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

        _validate_source_table(source_table)

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
            return _source_checkpoint_from_row(row, expected_source_table=source_table)
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

    def acquire_source_lease(
        self,
        source_table: str,
        lease_owner: str,
        *,
        lease_seconds: int,
    ) -> SourceCheckpoint:
        """原子认领一张来源表的扫描 lease，并返回数据库生成的到期时间。

        方法锁定 Checkpoint 行，只允许认领空闲或已过期 lease。仍然有效的 lease 即使
        owner 文本相同也会被拒绝，调用方应使用 ``renew_source_lease`` 显式续租。
        """

        _validate_source_table(source_table)
        _validate_lease_owner(lease_owner)
        _validate_lease_seconds(lease_seconds)

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                locked_row = _lock_source_checkpoint(cursor, source_table)
                if bool(locked_row["lease_is_active"]):
                    raise SourceLeaseConflictError(
                        f"来源表 {source_table} 的扫描 lease 仍在有效期内"
                    )
                cursor.execute(
                    """
                    UPDATE expert_index_source_checkpoint
                    SET lease_owner=%s,
                        lease_until=TIMESTAMPADD(
                            SECOND, %s, CURRENT_TIMESTAMP(6)
                        )
                    WHERE source_table=%s
                      AND (
                          lease_until IS NULL
                          OR lease_until <= CURRENT_TIMESTAMP(6)
                      )
                    """,
                    (lease_owner, lease_seconds, source_table),
                )
                if cursor.rowcount != 1:
                    raise SourceLeaseConflictError(
                        f"来源表 {source_table} 的扫描 lease 认领失败"
                    )
                checkpoint = _read_source_checkpoint(cursor, source_table)
            connection.commit()
            return checkpoint
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def renew_source_lease(
        self,
        source_table: str,
        lease_owner: str,
        *,
        lease_seconds: int,
    ) -> SourceCheckpoint:
        """续租调用方当前持有且尚未过期的来源扫描 lease。"""

        _validate_source_table(source_table)
        _validate_lease_owner(lease_owner)
        _validate_lease_seconds(lease_seconds)

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                locked_row = _lock_source_checkpoint(cursor, source_table)
                if (
                    locked_row.get("lease_owner") != lease_owner
                    or not bool(locked_row["lease_is_active"])
                ):
                    raise SourceLeaseConflictError(
                        f"调用方不再拥有来源表 {source_table} 的有效扫描 lease"
                    )
                cursor.execute(
                    """
                    UPDATE expert_index_source_checkpoint
                    SET lease_until=TIMESTAMPADD(
                        SECOND, %s, CURRENT_TIMESTAMP(6)
                    )
                    WHERE source_table=%s
                      AND lease_owner=%s
                      AND lease_until > CURRENT_TIMESTAMP(6)
                    """,
                    (lease_seconds, source_table, lease_owner),
                )
                if cursor.rowcount != 1:
                    raise SourceLeaseConflictError(
                        f"来源表 {source_table} 的扫描 lease 续租失败"
                    )
                checkpoint = _read_source_checkpoint(cursor, source_table)
            connection.commit()
            return checkpoint
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def ensure_source_lease(
        self,
        source_table: str,
        lease_owner: str,
        *,
        lease_seconds: int,
    ) -> SourceCheckpoint:
        """在来源批次开始执行前确保调用方持有有效扫描租约。

        扫描四张表与串行执行之间可能跨越一个短租约周期。若租约仍由当前
        owner 持有则直接续租；若租约已过期则在同一行锁内重新认领。有效期内
        属于其他 owner 的租约不会被覆盖。
        """

        _validate_source_table(source_table)
        _validate_lease_owner(lease_owner)
        _validate_lease_seconds(lease_seconds)

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                locked_row = _lock_source_checkpoint(cursor, source_table)
                current_owner = locked_row.get("lease_owner")
                if bool(locked_row["lease_is_active"]) and current_owner != lease_owner:
                    raise SourceLeaseConflictError(
                        f"来源表 {source_table} 的扫描 lease 已被其他 owner 占用"
                    )
                cursor.execute(
                    """
                    UPDATE expert_index_source_checkpoint
                    SET lease_owner=%s,
                        lease_until=TIMESTAMPADD(
                            SECOND, %s, CURRENT_TIMESTAMP(6)
                        )
                    WHERE source_table=%s
                      AND (
                          lease_until IS NULL
                          OR lease_until <= CURRENT_TIMESTAMP(6)
                          OR lease_owner=%s
                      )
                    """,
                    (lease_owner, lease_seconds, source_table, lease_owner),
                )
                if cursor.rowcount != 1:
                    raise SourceLeaseConflictError(
                        f"来源表 {source_table} 的扫描 lease 确认失败"
                    )
                checkpoint = _read_source_checkpoint(cursor, source_table)
            connection.commit()
            return checkpoint
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def release_source_lease(
        self,
        source_table: str,
        lease_owner: str,
    ) -> SourceCheckpoint:
        """释放 owner 匹配且尚未过期的来源扫描 lease，并返回未推进的 Checkpoint。"""

        _validate_source_table(source_table)
        _validate_lease_owner(lease_owner)

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                locked_row = _lock_source_checkpoint(cursor, source_table)
                if (
                    locked_row.get("lease_owner") != lease_owner
                    or not bool(locked_row["lease_is_active"])
                ):
                    raise SourceLeaseConflictError(
                        f"调用方不再拥有来源表 {source_table} 的有效扫描 lease"
                    )
                cursor.execute(
                    """
                    UPDATE expert_index_source_checkpoint
                    SET lease_owner=NULL, lease_until=NULL
                    WHERE source_table=%s
                      AND lease_owner=%s
                      AND lease_until > CURRENT_TIMESTAMP(6)
                    """,
                    (source_table, lease_owner),
                )
                if cursor.rowcount != 1:
                    raise SourceLeaseConflictError(
                        f"来源表 {source_table} 的扫描 lease 释放失败"
                    )
                checkpoint = _read_source_checkpoint(cursor, source_table)
            connection.commit()
            return checkpoint
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_recoverable_batch_source_scan(
        self,
        source_table: str,
        lease_owner: str,
    ) -> BatchSourceScan | None:
        """在有效来源 lease 下读取唯一未完成的 Batch Source。

        ``pending/running/failed`` 都属于尚未提交安全水位的恢复范围。不存在时返回
        ``None``；存在多条时 fail-closed，避免任意选择一个重叠范围继续处理。
        """

        _validate_source_table(source_table)
        _validate_lease_owner(lease_owner)

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                locked_row = _lock_source_checkpoint(cursor, source_table)
                if (
                    locked_row.get("lease_owner") != lease_owner
                    or not bool(locked_row["lease_is_active"])
                ):
                    raise SourceLeaseConflictError(
                        f"调用方不再拥有来源表 {source_table} 的有效扫描 lease"
                    )
                cursor.execute(
                    """
                    SELECT id, batch_id, source_table,
                           start_updated_at, start_id,
                           cursor_updated_at, cursor_id,
                           upper_bound_updated_at, upper_bound_id,
                           checkpoint_lock_version, status,
                           candidate_count, success_count,
                           failed_count, skipped_count,
                           started_at, finished_at
                    FROM expert_index_batch_source
                    WHERE source_table=%s
                      AND status IN ('pending', 'running', 'failed')
                    ORDER BY id ASC
                    LIMIT 2
                    FOR UPDATE
                    """,
                    (source_table,),
                )
                rows = list(cursor.fetchall())
                if len(rows) > 1:
                    raise BatchSourceRecoveryConflictError(
                        f"来源表 {source_table} 存在多个未完成 Batch Source"
                    )
                scan = _batch_source_scan_from_row(rows[0]) if rows else None
                if scan is not None:
                    checkpoint_cursor = SourceScanCursor(
                        updated_at=locked_row["last_updated_at"],
                        record_id=int(locked_row["last_id"]),
                    )
                    if (
                        scan.start_cursor != checkpoint_cursor
                        or scan.checkpoint_lock_version
                        != int(locked_row["lock_version"])
                    ):
                        raise BatchSourceRecoveryConflictError(
                            f"来源表 {source_table} 的未完成 Batch Source "
                            "与当前安全 Checkpoint 不一致"
                        )
            connection.commit()
            return scan
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

    def create_incremental_batch_source_scan(
        self,
        source_table: str,
        lease_owner: str,
        *,
        batch_no: str,
        upper_bound: SourceScanCursor,
    ) -> BatchSourceScan:
        """原子创建一张来源表的 incremental 批次和固定扫描范围。

        方法在同一 Index MySQL 事务中锁定 Checkpoint、确认当前 lease 和不存在未完成
        范围，再插入批次主记录与 Batch Source。安全 Checkpoint 只作为起点读取，不推进。
        """

        _validate_source_table(source_table)
        _validate_lease_owner(lease_owner)
        _validate_batch_no(batch_no)
        _validate_scan_cursor("upper_bound", upper_bound)

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                checkpoint_row = _lock_source_checkpoint(cursor, source_table)
                _require_active_source_lease(
                    checkpoint_row,
                    source_table=source_table,
                    lease_owner=lease_owner,
                )
                start_cursor = SourceScanCursor(
                    updated_at=checkpoint_row["last_updated_at"],
                    record_id=int(checkpoint_row["last_id"]),
                )
                try:
                    has_new_range = upper_bound > start_cursor
                except TypeError as exc:
                    raise ValueError(
                        "upper_bound 与 Checkpoint 必须使用一致的时间类型"
                    ) from exc
                if not has_new_range:
                    raise ValueError("upper_bound 必须严格大于安全 Checkpoint")

                cursor.execute(
                    """
                    SELECT id
                    FROM expert_index_batch_source
                    WHERE source_table=%s
                      AND status IN ('pending', 'running', 'failed')
                    ORDER BY id ASC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (source_table,),
                )
                if cursor.fetchone() is not None:
                    raise BatchSourceRecoveryConflictError(
                        f"来源表 {source_table} 已存在未完成 Batch Source"
                    )

                batch_id = allocate_batch_id(cursor)
                cursor.execute(
                    """
                    INSERT INTO expert_index_batch(id, batch_no, run_mode, status)
                    VALUES (%s, %s, 'incremental', 'running')
                    """,
                    (batch_id, batch_no),
                )
                checkpoint_lock_version = int(checkpoint_row["lock_version"])
                cursor.execute(
                    """
                    INSERT INTO expert_index_batch_source(
                        batch_id, source_table,
                        start_updated_at, start_id,
                        cursor_updated_at, cursor_id,
                        upper_bound_updated_at, upper_bound_id,
                        checkpoint_lock_version, status,
                        candidate_count, success_count,
                        failed_count, skipped_count
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'pending', 0, 0, 0, 0
                    )
                    """,
                    (
                        batch_id,
                        source_table,
                        start_cursor.updated_at,
                        start_cursor.record_id,
                        start_cursor.updated_at,
                        start_cursor.record_id,
                        upper_bound.updated_at,
                        upper_bound.record_id,
                        checkpoint_lock_version,
                    ),
                )
                scan = BatchSourceScan(
                    batch_id=batch_id,
                    source_table=source_table,
                    start_cursor=start_cursor,
                    current_cursor=start_cursor,
                    upper_bound=upper_bound,
                    checkpoint_lock_version=checkpoint_lock_version,
                )
            connection.commit()
            return scan
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def advance_batch_source_scan_cursor(
        self,
        source_table: str,
        batch_id: int,
        lease_owner: str,
        *,
        checkpoint_lock_version: int,
        expected_cursor: SourceScanCursor,
        next_cursor: SourceScanCursor,
        candidate_increment: int,
    ) -> BatchSourceScan:
        """在任务证据完整后 CAS 推进 Batch Source 临时游标。

        正数 ``candidate_increment`` 表示一页候选已全部落成任务；零表示空页封口，且
        ``next_cursor`` 必须等于固定上界。已经封口的 failed 范围允许保持同一游标，仅把
        状态恢复为 running。该方法不修改正式 Checkpoint。
        """

        _validate_source_table(source_table)
        _validate_positive_integer("batch_id", batch_id)
        _validate_lease_owner(lease_owner)
        _validate_non_negative_integer(
            "checkpoint_lock_version",
            checkpoint_lock_version,
        )
        _validate_scan_cursor("expected_cursor", expected_cursor)
        _validate_scan_cursor("next_cursor", next_cursor)
        _validate_non_negative_integer("candidate_increment", candidate_increment)

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                checkpoint_row = _lock_source_checkpoint(cursor, source_table)
                _require_active_source_lease(
                    checkpoint_row,
                    source_table=source_table,
                    lease_owner=lease_owner,
                )
                if int(checkpoint_row["lock_version"]) != checkpoint_lock_version:
                    raise BatchSourceCursorConflictError(
                        "Checkpoint lock_version 已变化，拒绝推进临时游标"
                    )

                cursor.execute(
                    """
                    SELECT id, batch_id, source_table,
                           start_updated_at, start_id,
                           cursor_updated_at, cursor_id,
                           upper_bound_updated_at, upper_bound_id,
                           checkpoint_lock_version, status,
                           candidate_count, success_count,
                           failed_count, skipped_count,
                           started_at, finished_at
                    FROM expert_index_batch_source
                    WHERE batch_id=%s AND source_table=%s
                    FOR UPDATE
                    """,
                    (batch_id, source_table),
                )
                row = cursor.fetchone()
                if row is None:
                    raise BatchSourceCursorConflictError("Batch Source 不存在")
                scan = _batch_source_scan_from_row(row)
                checkpoint_cursor = SourceScanCursor(
                    updated_at=checkpoint_row["last_updated_at"],
                    record_id=int(checkpoint_row["last_id"]),
                )
                if (
                    scan.start_cursor != checkpoint_cursor
                    or scan.checkpoint_lock_version != checkpoint_lock_version
                    or scan.current_cursor != expected_cursor
                    or scan.status not in {"pending", "running", "failed"}
                ):
                    raise BatchSourceCursorConflictError(
                        "Batch Source 起点、版本、状态或期望游标已变化"
                    )
                try:
                    cursor_advances = next_cursor > expected_cursor
                    within_upper_bound = next_cursor <= scan.upper_bound
                except TypeError as exc:
                    raise ValueError("扫描游标必须使用一致的时间类型") from exc
                reactivates_sealed_range = (
                    candidate_increment == 0
                    and scan.status == "failed"
                    and expected_cursor == scan.upper_bound
                    and next_cursor == scan.upper_bound
                )
                if (
                    (not cursor_advances and not reactivates_sealed_range)
                    or not within_upper_bound
                ):
                    raise BatchSourceCursorConflictError(
                        "临时游标必须前进且不能越过固定上界"
                    )
                if candidate_increment == 0 and next_cursor != scan.upper_bound:
                    raise BatchSourceCursorConflictError(
                        "空页只能把临时游标封口到固定上界"
                    )

                expected_candidate_count = (
                    scan.candidate_count + candidate_increment
                )
                cursor.execute(
                    """
                    SELECT COUNT(*) AS task_count
                    FROM expert_index_sync_task
                    WHERE batch_id=%s AND source_table=%s
                    """,
                    (batch_id, source_table),
                )
                task_count_row = cursor.fetchone()
                task_count = int(task_count_row["task_count"] if task_count_row else -1)
                if task_count != expected_candidate_count:
                    raise BatchSourceCursorConflictError(
                        "已落库任务总数与待提交 candidate_count 不一致"
                    )

                cursor.execute(
                    """
                    UPDATE expert_index_batch_source
                    SET cursor_updated_at=%s, cursor_id=%s,
                        candidate_count=%s, status='running',
                        started_at=COALESCE(started_at, CURRENT_TIMESTAMP(6))
                    WHERE id=%s
                      AND cursor_updated_at=%s AND cursor_id=%s
                      AND checkpoint_lock_version=%s
                      AND status IN ('pending', 'running', 'failed')
                    """,
                    (
                        next_cursor.updated_at,
                        next_cursor.record_id,
                        expected_candidate_count,
                        int(row["id"]),
                        expected_cursor.updated_at,
                        expected_cursor.record_id,
                        checkpoint_lock_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise BatchSourceCursorConflictError(
                        "Batch Source 临时游标 CAS 更新失败"
                    )
                updated_scan = _read_batch_source_scan_by_id(
                    cursor,
                    int(row["id"]),
                )
            connection.commit()
            return updated_scan
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
        _validate_source_table(source_table)

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
            return _batch_source_scan_from_row(row)
        finally:
            connection.close()


def _validate_source_table(source_table: str) -> None:
    """校验来源表属于四表 Checkpoint 白名单。"""

    if not isinstance(source_table, str) or source_table not in PERSISTED_SOURCE_TABLES:
        raise ValueError(f"不支持的 Checkpoint 来源表: {source_table}")


def _validate_lease_owner(lease_owner: str) -> None:
    """校验 lease owner 能安全写入 Checkpoint 字段。"""

    if (
        not isinstance(lease_owner, str)
        or not lease_owner.strip()
        or len(lease_owner) > 128
    ):
        raise ValueError("lease_owner 必须是长度不超过 128 的非空字符串")


def _validate_lease_seconds(lease_seconds: int) -> None:
    """校验 lease 时长为正整数秒。"""

    if (
        isinstance(lease_seconds, bool)
        or not isinstance(lease_seconds, int)
        or lease_seconds < 1
    ):
        raise ValueError("lease_seconds 必须是正整数")


def _validate_batch_no(batch_no: str) -> None:
    """校验 incremental 批次号可安全写入批次主表。"""

    if (
        not isinstance(batch_no, str)
        or not batch_no.strip()
        or batch_no != batch_no.strip()
        or len(batch_no) > 64
    ):
        raise ValueError("batch_no 必须是长度不超过 64 的非空字符串")


def _validate_positive_integer(field_name: str, value: int) -> None:
    """校验数据库无符号主键参数为正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} 必须是正整数")


def _validate_non_negative_integer(field_name: str, value: int) -> None:
    """校验计数或版本参数为非负整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} 必须是非负整数")


def _validate_scan_cursor(field_name: str, value: SourceScanCursor) -> None:
    """校验临时扫描位置使用显式复合游标模型。"""

    if not isinstance(value, SourceScanCursor):
        raise TypeError(f"{field_name} 必须是 SourceScanCursor")


def _require_active_source_lease(
    checkpoint_row: dict[str, Any],
    *,
    source_table: str,
    lease_owner: str,
) -> None:
    """要求 Checkpoint 行仍由调用方持有有效来源 lease。"""

    if (
        checkpoint_row.get("lease_owner") != lease_owner
        or not bool(checkpoint_row["lease_is_active"])
    ):
        raise SourceLeaseConflictError(
            f"调用方不再拥有来源表 {source_table} 的有效扫描 lease"
        )


def _lock_source_checkpoint(cursor: Any, source_table: str) -> dict[str, Any]:
    """锁定 Checkpoint，并由 Index MySQL 判断 lease 当前是否有效。"""

    cursor.execute(
        """
        SELECT source_table, last_updated_at, last_id,
               last_batch_id, last_success_time, lock_version,
               lease_owner, lease_until,
               (
                   lease_owner IS NOT NULL
                   AND lease_until > CURRENT_TIMESTAMP(6)
               ) AS lease_is_active
        FROM expert_index_source_checkpoint
        WHERE source_table=%s
        FOR UPDATE
        """,
        (source_table,),
    )
    row = cursor.fetchone()
    if row is None:
        raise SourceCheckpointNotInitializedError(
            f"来源表 {source_table} 尚未初始化 Checkpoint"
        )
    if row.get("source_table") != source_table:
        raise RuntimeError("锁定的 Checkpoint 与请求来源表不一致")
    return row


def _read_source_checkpoint(cursor: Any, source_table: str) -> SourceCheckpoint:
    """在当前事务中读取刚写入的完整 Checkpoint 状态。"""

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
        raise RuntimeError("Checkpoint 在 lease 状态写入后不存在")
    return _source_checkpoint_from_row(row, expected_source_table=source_table)


def _read_batch_source_scan_by_id(cursor: Any, row_id: int) -> BatchSourceScan:
    """在当前事务中读取刚更新的完整 Batch Source 状态。"""

    cursor.execute(
        """
        SELECT id, batch_id, source_table,
               start_updated_at, start_id,
               cursor_updated_at, cursor_id,
               upper_bound_updated_at, upper_bound_id,
               checkpoint_lock_version, status,
               candidate_count, success_count,
               failed_count, skipped_count,
               started_at, finished_at
        FROM expert_index_batch_source
        WHERE id=%s
        """,
        (row_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Batch Source 在临时游标更新后不存在")
    return _batch_source_scan_from_row(row)


def _source_checkpoint_from_row(
    row: dict[str, Any],
    *,
    expected_source_table: str,
) -> SourceCheckpoint:
    """把数据库 Checkpoint 行转换为显式领域模型。"""

    if row.get("source_table") != expected_source_table:
        raise RuntimeError("Checkpoint 查询结果与请求来源表不一致")
    return SourceCheckpoint(
        source_table=expected_source_table,
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


def _batch_source_scan_from_row(row: dict[str, Any]) -> BatchSourceScan:
    """把数据库 Batch Source 行转换为固定范围扫描模型。"""

    return BatchSourceScan(
        batch_id=int(row["batch_id"]),
        source_table=str(row["source_table"]),
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
        status=str(row["status"]),
        candidate_count=int(row["candidate_count"]),
        success_count=int(row["success_count"]),
        failed_count=int(row["failed_count"]),
        skipped_count=int(row["skipped_count"]),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )
