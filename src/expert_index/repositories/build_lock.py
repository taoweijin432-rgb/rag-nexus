"""构建任务互斥锁的 MySQL 数据访问层。

本模块为 single 和 full 构建提供共享命名锁。锁连接在上下文完整执行期间保持打开，
退出上下文时释放锁并关闭连接。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress

import pymysql

from ..config import DatabaseConfig, require_index_database

# single/full 使用同一锁名前缀，避免两个构建任务同时改写索引派生数据。
BUILD_LOCK_PREFIX = "expert_index_build_v1"


class ConcurrentBuildError(RuntimeError):
    """表示另一个专家索引构建任务正在持有全局互斥锁。"""


class MySQLBuildLockRepository:
    """使用 MySQL 命名锁控制 single/full 构建任务互斥。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存索引库连接配置，获取锁时再建立数据库连接。"""

        require_index_database(config)
        self.config = config

    @contextmanager
    def build_lock(self, run_mode: str) -> Iterator[None]:
        """在 single 或 full 构建的完整调用期间持有 MySQL 命名锁。

        ``GET_LOCK(..., 0)`` 会立即尝试获取锁。锁被占用时抛出
        ``ConcurrentBuildError``，不会等待或进入任务主体。上下文退出时释放锁并关闭
        连接；释放失败不会覆盖任务主体已经产生的异常。
        """

        if run_mode not in {"single", "full"}:
            raise ValueError("build_lock 第一版只支持 single/full")
        # 数据库名用于隔离同一 MySQL 实例中的不同索引库，锁名最长保留 64 个字符。
        lock_name = f"{BUILD_LOCK_PREFIX}:{self.config.database}"[:64]
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        acquired = False
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT GET_LOCK(%s, 0) AS acquired", (lock_name,))
                row = cursor.fetchone()
                acquired = bool(row and row.get("acquired") == 1)
            if not acquired:
                raise ConcurrentBuildError(f"已有专家索引加工任务运行，当前 {run_mode} 任务未启动")
            yield
        finally:
            if acquired:
                with suppress(Exception), connection.cursor() as cursor:
                    cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
            connection.close()
