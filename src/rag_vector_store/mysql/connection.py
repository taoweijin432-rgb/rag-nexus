"""MySQL 同步连接生命周期与安全错误封装。

本模块是 MySQL Adapter 的基础设施层，只负责按配置创建/关闭连接和隐藏驱动
敏感错误；它不执行业务 SQL、不读取 ``expert_chunk``，也不管理 Task State 或
checkpoint。上层 Reader/Repository 通过它复用一致的连接和异常边界。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import MySQLSettings


class MySQLConnectionError(RuntimeError):
    """MySQL driver 缺失、连接失败或关闭失败。"""


class MySQLConnection:
    """延迟创建并可安全关闭的同步 MySQL 连接。

    本模块只负责连接生命周期，不执行任何业务 SQL，也不实现 ``expert_chunk``
    Reader。连接采用延迟创建，便于只构造配置或执行本地单元测试时不触碰数据库。
    """

    def __init__(
        self,
        settings: MySQLSettings,
        *,
        connector_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._connector_factory = connector_factory
        self._connection: Any | None = None

    @property
    def connection(self) -> Any:
        """返回已建立的连接；首次访问时建立连接。"""

        if self._connection is None:
            self.connect()
        return self._connection

    def connect(self) -> Any:
        """创建同步连接，不在异常中暴露密码或完整连接参数。

        驱动异常只保留类型、errno 和 SQL state 等安全诊断；调用方可据此把环境
        不可用分类为 BLOCKED，但不会把密码、连接参数或完整驱动文本写入日志。
        """

        if self._connection is not None:
            return self._connection

        factory = self._connector_factory
        if factory is None:
            try:
                from pymysql import connect
            except ImportError as exc:
                raise MySQLConnectionError(
                    "PyMySQL 未安装，无法建立 MySQL 连接"
                ) from exc
            factory = connect

        try:
            # 同事的 DatabaseConfig 暴露 as_kwargs()，Task2 的兼容
            # MySQLSettings 暴露 connector_kwargs；两者都最终落到 PyMySQL，
            # 避免在两个包中维护不同的驱动和连接协议。
            self._connection = factory(**_connection_kwargs(self.settings))
        except Exception as exc:
            raise MySQLConnectionError(
                "MySQL 连接失败: "
                f"host={self.settings.host!r}, port={self.settings.port}, "
                f"database={self.settings.database or '<unset>'!r}, "
                f"driver_error={_safe_driver_diagnostic(exc)}"
            ) from exc
        return self._connection

    def close(self) -> None:
        """关闭当前连接；重复关闭是安全的。"""

        if self._connection is None:
            return
        connection = self._connection
        self._connection = None
        try:
            connection.close()
        except Exception as exc:
            raise MySQLConnectionError(
                "MySQL 连接关闭失败: "
                f"driver_error={_safe_driver_diagnostic(exc)}"
            ) from exc

    def __enter__(self) -> Any:
        return self.connection

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def _safe_driver_diagnostic(error: Exception) -> str:
    """只保留驱动类型和可用于排障的编号，不暴露原始错误文本。"""

    details = [type(error).__name__]
    errno = getattr(error, "errno", None)
    sqlstate = getattr(error, "sqlstate", None)
    if errno is not None:
        details.append(f"errno={errno}")
    if sqlstate is not None:
        details.append(f"sqlstate={sqlstate}")
    return " ".join(details)


def _connection_kwargs(settings: Any) -> dict[str, Any]:
    """兼容同事 DatabaseConfig 与 Task2 旧配置入口的公共连接参数。"""

    as_kwargs = getattr(settings, "as_kwargs", None)
    if callable(as_kwargs):
        return dict(as_kwargs(autocommit=True))
    connector_kwargs = getattr(settings, "connector_kwargs", None)
    if isinstance(connector_kwargs, dict):
        return dict(connector_kwargs)
    raise TypeError("MySQL 配置必须提供 as_kwargs() 或 connector_kwargs")
