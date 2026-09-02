"""Resume 独立 Index Schema 初始化器。

初始化器只连接 ``RESUME_INDEX_DB_*``，执行版本库中的 Resume 控制表 DDL。它不连接
Source，不创建数据库，不读取 Resume 正文，也不会调用 Expert Schema 初始化入口。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymysql

from .config import ResumeSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESUME_INDEX_DDL = PROJECT_ROOT / "sql" / "020_create_resume_index_tables.sql"


class _SchemaCursor(Protocol):
    """执行控制表 DDL 所需的最小 Cursor 协议。"""

    def execute(self, sql: str) -> object: ...

    def close(self) -> None: ...


class _SchemaConnection(Protocol):
    """初始化事务所需的最小 Connection 协议。"""

    def cursor(self) -> _SchemaCursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ResumeSchemaInitializationResult:
    """不含凭据的 Resume Index 初始化摘要。"""

    schema_name: str
    statements_executed: int


class MySQLResumeSchemaInitializer:
    """在已隔离的 Resume Index Schema 中执行控制表 DDL。"""

    def __init__(
        self,
        settings: ResumeSettings,
        *,
        connection_factory: Callable[..., _SchemaConnection] = pymysql.connect,
    ) -> None:
        """保存已校验配置和连接工厂，构造阶段不建立数据库连接。

        :param settings: Resume 专用配置；构造时再次执行隔离门禁。
        :param connection_factory: 生产默认使用 ``pymysql.connect``，测试注入 Fake。
        """

        if not isinstance(settings, ResumeSettings):
            raise TypeError("settings 必须是 ResumeSettings")
        settings.validate_isolation_boundaries()
        self._settings = settings
        self._connection_factory = connection_factory

    def initialize_schema(
        self,
        sql_path: Path = DEFAULT_RESUME_INDEX_DDL,
    ) -> ResumeSchemaInitializationResult:
        """按顺序执行 Resume Index DDL，并返回安全摘要。

        :param sql_path: 待执行的 UTF-8 SQL 文件。正式 CLI 固定使用
            :data:`DEFAULT_RESUME_INDEX_DDL`；参数仅保留给隔离测试。
        :return: 目标 Schema 和成功执行的语句数量。
        :raises ValueError: SQL 文件不包含任何可执行语句。

        全部语句成功后提交；任一语句失败时回滚当前连接并继续抛出异常。MySQL DDL
        可能隐式提交，因此 rollback 不能替代部署前备份和独立 Schema 边界。
        """

        statements = _read_sql_statements(sql_path)
        connection = self._connection_factory(**self._settings.index_db.as_kwargs(autocommit=False))
        try:
            cursor = connection.cursor()
            try:
                for statement in statements:
                    cursor.execute(statement)
            finally:
                cursor.close()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return ResumeSchemaInitializationResult(
            schema_name=self._settings.index_db.database,
            statements_executed=len(statements),
        )


def _read_sql_statements(sql_path: Path) -> tuple[str, ...]:
    """读取分号分隔的初始化 SQL，并拒绝空脚本。"""

    if not isinstance(sql_path, Path):
        raise TypeError("sql_path 必须是 pathlib.Path")
    script = sql_path.read_text(encoding="utf-8")
    statements = tuple(statement.strip() for statement in script.split(";") if statement.strip())
    if not statements:
        raise ValueError("Resume Index DDL 不包含可执行语句")
    return statements
