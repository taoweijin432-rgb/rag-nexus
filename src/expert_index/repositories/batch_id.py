"""全量、单条、Poll 与 CDC 共用的 MySQL Batch ID 分配器。"""

from __future__ import annotations

import logging
from typing import Any

import pymysql

from ..config import DatabaseConfig, require_index_database
from ..constants import (
    BATCH_ID_SEQUENCE_NAME_COLUMN,
    BATCH_ID_SEQUENCE_TABLE,
    BATCH_ID_SEQUENCE_VALUE_COLUMN,
    GLOBAL_BATCH_ID_SEQUENCE_NAME,
    MAX_UNSIGNED_BIGINT,
)

log = logging.getLogger(__name__)


class BatchIdAllocationError(RuntimeError):
    """表示统一 Batch ID 序列缺失、耗尽或返回非法结果。"""


def allocate_batch_id(cursor: Any) -> int:
    """使用调用方当前事务原子分配一个全局唯一 Batch ID。

    :param cursor: 已连接 Index MySQL 的事务 Cursor。
    :return: 本事务独占的新 Batch ID。
    :raises BatchIdAllocationError: 序列未初始化、已耗尽或驱动未返回有效 ID。

    ``LAST_INSERT_ID(expr)`` 的值按 MySQL Connection 隔离，因此并发连接不会读取
    到其他事务分配的编号。调用方负责提交或回滚当前事务。
    """

    if not callable(getattr(cursor, "execute", None)):
        raise TypeError("cursor 缺少 execute")

    cursor.execute(
        f"""
        UPDATE `{BATCH_ID_SEQUENCE_TABLE}`
        SET `{BATCH_ID_SEQUENCE_VALUE_COLUMN}` = LAST_INSERT_ID(
            `{BATCH_ID_SEQUENCE_VALUE_COLUMN}` + 1
        )
        WHERE `{BATCH_ID_SEQUENCE_NAME_COLUMN}`=%s
          AND `{BATCH_ID_SEQUENCE_VALUE_COLUMN}` < %s
        """,
        (GLOBAL_BATCH_ID_SEQUENCE_NAME, MAX_UNSIGNED_BIGINT),
    )
    if int(getattr(cursor, "rowcount", 0)) != 1:
        raise BatchIdAllocationError("统一 Batch ID 序列未初始化或已耗尽")

    batch_id = int(getattr(cursor, "lastrowid", 0) or 0)
    if not 1 <= batch_id <= MAX_UNSIGNED_BIGINT:
        raise BatchIdAllocationError("MySQL 未返回有效的统一 Batch ID")
    return batch_id


class MySQLBatchIdAllocator:
    """通过短事务从 Index MySQL 统一序列分配 Batch ID。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存已校验的 Index 数据库配置。"""

        require_index_database(config)
        self.config = config

    def allocate(self) -> int:
        """在独立事务中分配并提交一个全局唯一 Batch ID。"""

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                batch_id = allocate_batch_id(cursor)
            connection.commit()
            return batch_id
        except Exception:
            connection.rollback()
            log.exception(
                "Batch ID allocation failed for index database %s",
                self.config.database,
            )
            raise
        finally:
            connection.close()


__all__ = [
    "BatchIdAllocationError",
    "MySQLBatchIdAllocator",
    "allocate_batch_id",
]
