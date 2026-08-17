"""Task State 独立数据库配置。

Task State 不允许复用 expert_chunk 读取配置（MYSQL_*/SOURCE_DB_*/DB_*），
必须使用独立的 ``TASK_STATE_MYSQL_*`` 环境变量，从源头隔离业务库与状态库。

执行 migration（sql/006_create_vectorization_task_state.sql）前必须确认目标
数据库为本地开发库（如 ``expert_dev``），禁止在公司业务数据库执行。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from ..config import MySQLSettings


class TaskStateMySQLSettings(MySQLSettings):
    """Task State 专用 MySQL 配置（``TASK_STATE_MYSQL_*``）。

    继承 ``MySQLSettings`` 的连接参数与 ``connector_kwargs`` 行为，但
    ``from_env`` 只读取 ``TASK_STATE_MYSQL_*`` 变量，不读取、不回退到
    ``MYSQL_*`` / ``SOURCE_DB_*`` / ``DB_*``，保证与业务库配置完全隔离。

    数据库名、用户名、密码不提供业务默认值；缺少时由调用方明确报告
    BLOCKED，避免误连其他环境。
    """

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> TaskStateMySQLSettings:
        env = os.environ if environ is None else environ

        values: dict[str, Any] = {
            "host": _env_get(env, "TASK_STATE_MYSQL_HOST", "127.0.0.1"),
            "database": _env_optional(env, "TASK_STATE_MYSQL_DATABASE"),
            "user": _env_optional(env, "TASK_STATE_MYSQL_USER"),
        }

        password = _env_optional(env, "TASK_STATE_MYSQL_PASSWORD")
        if password is not None:
            values["password"] = SecretStr(password)

        port = _env_optional(env, "TASK_STATE_MYSQL_PORT")
        if port is not None:
            try:
                values["port"] = int(port)
            except ValueError as exc:
                raise ValueError("TASK_STATE_MYSQL_PORT 必须是整数") from exc

        return cls(**values)

    @property
    def connector_kwargs(self) -> dict[str, Any]:
        """覆写父类，显式设置 ``autocommit=True``。

        Task State Store 每条 SQL 独立提交，不依赖跨语句事务；
        显式 autocommit 避免隐式事务在多线程连接池中残留状态。
        """

        kwargs = super().connector_kwargs
        kwargs["autocommit"] = True
        return kwargs


def _env_get(
    env: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    value = env.get(name)
    return default if value is None or not value.strip() else value


def _env_optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return None if value is None or not value.strip() else value