"""专家索引数据库结构初始化器。

本模块读取并执行调用方指定的 SQL 文件，用于创建或迁移专家索引所需的数据库结构。
"""

from __future__ import annotations

from pathlib import Path

import pymysql

from .config import DatabaseConfig


class MySQLSchemaInitializer:
    """在 MySQL 索引库中执行 Schema SQL 文件。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存索引库连接配置，执行 SQL 时再建立数据库连接。"""

        self.config = config

    def initialize_schema(self, sql_path: Path) -> None:
        """按顺序执行指定 SQL 文件中的全部非空语句。

        文件使用 UTF-8 读取并按分号拆分。全部语句成功后提交；任意语句失败时回滚
        当前连接并继续抛出异常。无论执行结果如何，方法最终都会关闭连接。

        MySQL DDL 可能隐式提交，因此回滚只影响数据库仍允许回滚的语句。
        """

        script = sql_path.read_text(encoding="utf-8")
        statements = [statement.strip() for statement in script.split(";") if statement.strip()]
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
