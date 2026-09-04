"""仅供测试辅助 Reader 使用的 MySQL 连接生命周期。"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from expert_index.config import DatabaseConfig


class TestMySQLConnectionError(RuntimeError):
    """测试环境 MySQL 连接或关闭失败。"""


class TestMySQLConnection:
    """延迟创建测试连接，不进入 ``rag_vector_store`` 生产包。"""

    __test__ = False

    def __init__(
        self,
        settings: DatabaseConfig,
        *,
        connector_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._connector_factory = connector_factory
        self._connection: Any | None = None

    @property
    def connection(self) -> Any:
        if self._connection is None:
            self.connect()
        return self._connection

    def connect(self) -> Any:
        if self._connection is not None:
            return self._connection
        factory = self._connector_factory
        if factory is None:
            try:
                from pymysql import connect
            except ImportError as exc:
                raise TestMySQLConnectionError(
                    "PyMySQL 未安装，无法建立测试 MySQL 连接"
                ) from exc
            factory = connect
        try:
            self._connection = factory(**self.settings.as_kwargs(autocommit=True))
        except Exception as exc:
            raise TestMySQLConnectionError(
                "测试 MySQL 连接失败: "
                f"host={self.settings.host!r}, port={self.settings.port}, "
                f"database={self.settings.database!r}, "
                f"driver_error={_safe_driver_diagnostic(exc)}"
            ) from exc
        return self._connection

    def close(self) -> None:
        if self._connection is None:
            return
        connection = self._connection
        self._connection = None
        try:
            connection.close()
        except Exception as exc:
            raise TestMySQLConnectionError(
                "测试 MySQL 连接关闭失败: "
                f"driver_error={_safe_driver_diagnostic(exc)}"
            ) from exc

    def __enter__(self) -> Any:
        return self.connection

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def database_config_from_test_env(
    environ: Mapping[str, str] | None = None,
) -> DatabaseConfig:
    """从测试环境变量构造现有 ``DatabaseConfig``，不引入生产配置类型。"""

    env = os.environ if environ is None else environ
    port_text = _first(env, "MYSQL_PORT", "SOURCE_DB_PORT", "DB_PORT") or "3306"
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("MYSQL_PORT 必须是整数") from exc
    return DatabaseConfig(
        host=_first(env, "MYSQL_HOST", "SOURCE_DB_HOST", "DB_HOST") or "127.0.0.1",
        port=port,
        user=_first(env, "MYSQL_USER", "SOURCE_DB_USER", "DB_USER") or "",
        password=_first(
            env,
            "MYSQL_PASSWORD",
            "SOURCE_DB_PASSWORD",
            "DB_PASSWORD",
        )
        or "",
        database=_first(
            env,
            "MYSQL_DATABASE",
            "SOURCE_DB_NAME",
            "DB_NAME",
        )
        or "",
    )


def _first(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value is not None and value.strip():
            return value
    return None


def _safe_driver_diagnostic(error: Exception) -> str:
    details = [type(error).__name__]
    errno = getattr(error, "errno", None)
    sqlstate = getattr(error, "sqlstate", None)
    if errno is not None:
        details.append(f"errno={errno}")
    if sqlstate is not None:
        details.append(f"sqlstate={sqlstate}")
    return " ".join(details)


__all__ = [
    "TestMySQLConnection",
    "TestMySQLConnectionError",
    "database_config_from_test_env",
]
