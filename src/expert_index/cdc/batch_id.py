"""CDC 本地批次号分配器。

Canal ``batch_id`` 只在当前连接内有效，不能写入业务 Checkpoint。该模块使用 Index
MySQL 命名锁保证同一 destination 只有一个 CDC Worker，并从已落库 Event/Checkpoint
恢复本地批次号，空 batch 不产生数据库行。
"""

from __future__ import annotations

import hashlib
import threading

import pymysql

from ..config import DatabaseConfig, require_index_database


class CdcWorkerAlreadyRunningError(RuntimeError):
    """表示同一 Canal destination 已有 CDC Worker 持有运行锁。"""


class MySQLCdcLocalBatchIdProvider:
    """为单一 destination 分配可恢复的 Index 本地批次号。"""

    def __init__(self, config: DatabaseConfig, *, destination: str) -> None:
        """建立持有命名锁的 Index 连接，并从已有状态恢复起点。"""

        require_index_database(config)
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError("destination 必须是非空字符串")
        self.config = config
        self.destination = destination
        self._mutex = threading.Lock()
        self._connection = pymysql.connect(**config.as_kwargs())
        self._lock_name = _lock_name(destination)
        try:
            self._acquire_worker_lock()
            self._next_id = self._load_max_id() + 1
        except Exception:
            self.close()
            raise

    def __call__(self) -> int:
        """返回下一个本地批次号；仅非空 Canal batch 会实际使用该号码。"""

        with self._mutex:
            if self._connection is None:
                raise RuntimeError("CDC 本地批次号分配器已关闭")
            value = self._next_id
            self._next_id += 1
            if value < 1:
                raise RuntimeError("CDC 本地批次号溢出")
            return value

    def close(self) -> None:
        """释放 destination 运行锁和 Index 连接。"""

        connection = self._connection
        self._connection = None
        if connection is None:
            return
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT RELEASE_LOCK(%s)", (self._lock_name,))
        except pymysql.Error:
            # 连接关闭本身也会释放 MySQL 命名锁；清理阶段不覆盖原始异常。
            return
        finally:
            connection.close()

    def _acquire_worker_lock(self) -> None:
        """以非阻塞方式取得同一 destination 的单 Worker 锁。"""

        with self._connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK(%s, 0) AS acquired", (self._lock_name,))
            row = cursor.fetchone()
        if not row or int(row.get("acquired", 0)) != 1:
            raise CdcWorkerAlreadyRunningError(
                f"destination={self.destination!r} 已有 CDC Worker 运行"
            )

    def _load_max_id(self) -> int:
        """读取所有 Index 批次来源中的最大编号，避免与 Poll 批次键冲突。"""

        max_ids: list[int] = []
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COALESCE(MAX(local_batch_id), 0) AS max_id
                FROM expert_index_cdc_event
                WHERE destination=%s
                """,
                (self.destination,),
            )
            event_row = cursor.fetchone() or {}
            max_ids.append(int(event_row.get("max_id", 0)))
            cursor.execute(
                """
                SELECT COALESCE(last_local_batch_id, 0) AS max_id
                FROM expert_index_cdc_checkpoint
                WHERE destination=%s
                """,
                (self.destination,),
            )
            checkpoint_row = cursor.fetchone() or {}
            max_ids.append(int(checkpoint_row.get("max_id", 0)))
            # Poll 和 CDC 共用 Sync Task 的 batch_id/source 唯一键；同时纳入
            # 已创建但尚未产生 Task 的全量/增量批次，保证新编号不会碰撞。
            for table, column in (
                ("expert_index_batch", "id"),
                ("expert_index_batch_source", "batch_id"),
                ("expert_index_sync_task", "batch_id"),
            ):
                cursor.execute(f"SELECT COALESCE(MAX({column}), 0) AS max_id FROM {table}")
                row = cursor.fetchone() or {}
                max_ids.append(int(row.get("max_id", 0)))
        return max(max_ids, default=0)


def _lock_name(destination: str) -> str:
    """将 destination 转为长度稳定且不含敏感信息的 MySQL 锁名。"""

    digest = hashlib.sha256(destination.encode("utf-8")).hexdigest()[:32]
    return f"expert_index:cdc:{digest}"


__all__ = ["CdcWorkerAlreadyRunningError", "MySQLCdcLocalBatchIdProvider"]
