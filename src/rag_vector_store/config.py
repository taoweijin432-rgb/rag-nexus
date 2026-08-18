"""从环境变量加载 Milvus 和 MySQL 配置。

配置模块只负责把进程环境转换为强类型设置，不创建连接、不读取数据库，也不
改变现有 ``.env``。凭据通过 ``SecretStr``/连接参数传递，错误诊断不能回显密钥。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

# 运行时只允许正式 Contract v2；旧兼容模式已退役。
SchemaMode = Literal["contract"]


class MilvusSettings(BaseModel):
    """供正式 Contract Milvus Repository 使用的连接和集合配置。

    ``from_env`` 显式读取环境变量，不依赖 dotenv 加载器。这样可以让密钥只存在
    于进程环境中，也不会把模块绑定到某一种部署方式。Collection 名称和索引参数
    也属于配置的一部分；Repository 会在写入前进行兼容性检查。
    """

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(default="http://localhost:19530", min_length=1)
    token: SecretStr | None = None
    user: str | None = None
    password: SecretStr | None = None
    db_name: str | None = None
    timeout: float | None = Field(default=None, gt=0)

    collection_name: str = Field(
        default="expert_candidates_v14_contract_dev",
        min_length=1,
    )
    schema_mode: SchemaMode = "contract"
    index_type: str = Field(default="HNSW", min_length=1)
    metric_type: str = Field(default="COSINE", min_length=1)
    index_params: dict[str, Any] = Field(
        default_factory=lambda: {"M": 16, "efConstruction": 200}
    )
    destructive_maintenance_collections: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_authentication(self) -> MilvusSettings:
        has_user = self.user is not None
        has_password = self.password is not None
        if has_user != has_password:
            raise ValueError("MILVUS_USER and MILVUS_PASSWORD must be provided together")
        if self.token is not None and (has_user or has_password):
            raise ValueError("configure either token or user/password, not both")
        return self

    @property
    def client_kwargs(self) -> dict[str, Any]:
        """只返回 ``pymilvus.MilvusClient`` 支持的构造参数。"""

        kwargs: dict[str, Any] = {"uri": self.uri}
        if self.token is not None:
            kwargs["token"] = self.token.get_secret_value()
        elif self.user is not None and self.password is not None:
            kwargs["user"] = self.user
            kwargs["password"] = self.password.get_secret_value()
        if self.db_name is not None:
            kwargs["db_name"] = self.db_name
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return kwargs

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MilvusSettings:
        """从 ``MILVUS_*`` 环境变量创建配置。

        URI 提供本地开发默认值；需要认证时，应始终通过环境变量传入凭据，
        不要把凭据提交到代码中。
        """

        env = os.environ if environ is None else environ
        values: dict[str, Any] = {
            "uri": _get(env, "MILVUS_URI", "http://localhost:19530"),
            "token": _optional(env, "MILVUS_TOKEN"),
            "user": _optional(env, "MILVUS_USER"),
            "password": _optional(env, "MILVUS_PASSWORD"),
            "db_name": _optional(env, "MILVUS_DB_NAME"),
            "collection_name": _get(
                env,
                "MILVUS_COLLECTION_NAME",
                "expert_candidates_v14_contract_dev",
            ),
            "schema_mode": _get(env, "MILVUS_SCHEMA_MODE", "contract"),
            "index_type": _get(env, "MILVUS_INDEX_TYPE", "HNSW"),
            "metric_type": _get(env, "MILVUS_METRIC_TYPE", "COSINE"),
        }

        timeout = _optional(env, "MILVUS_TIMEOUT")
        if timeout is not None:
            values["timeout"] = float(timeout)

        index_params = _optional(env, "MILVUS_INDEX_PARAMS")
        if index_params is not None:
            try:
                parsed_index_params = json.loads(index_params)
            except json.JSONDecodeError as exc:
                raise ValueError("MILVUS_INDEX_PARAMS 必须是合法 JSON") from exc
            if not isinstance(parsed_index_params, dict):
                raise ValueError("MILVUS_INDEX_PARAMS 必须是 JSON 对象")
            values["index_params"] = parsed_index_params

        destructive_raw = _optional(env, "MILVUS_ALLOWED_DESTRUCTIVE_COLLECTIONS")
        values["destructive_maintenance_collections"] = (
            tuple(
                name.strip()
                for name in destructive_raw.split(",")
                if name.strip()
            )
            if destructive_raw is not None
            else ()
        )

        return cls(**values)


class MySQLSettings(BaseModel):
    """供只读 MySQL 连接基础设施使用的配置。

    数据库名称、用户名和密码不提供业务默认值。缺少这些环境变量时，验证脚本应
    报告 BLOCKED，而不是猜测本地数据库身份或连接目标；本配置本身不负责选择
    只读事务或执行任何 SQL。
    """

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=3306, ge=1, le=65535)
    database: str | None = Field(default=None, min_length=1)
    user: str | None = Field(default=None, min_length=1)
    password: SecretStr | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MySQLSettings:
        """从 MYSQL_* 环境变量读取配置，不加载或修改 .env 文件。

        同事的 expert_index 仍以 DB_*/SOURCE_DB_* 命名；这里保留
        MYSQL_* 作为 Task2 的清晰边界，同时接受这些已有变量作为回退，
        使 Reader 可以直接复用同事的 DatabaseConfig 配置而不复制一套环境协议。
        """

        env = os.environ if environ is None else environ
        port = _first_optional(env, "MYSQL_PORT", "SOURCE_DB_PORT", "DB_PORT")
        values: dict[str, Any] = {
            "host": _first(
                env,
                "MYSQL_HOST",
                "SOURCE_DB_HOST",
                "DB_HOST",
                default="127.0.0.1",
            ),
            "database": _first_optional(
                env,
                "MYSQL_DATABASE",
                "SOURCE_DB_NAME",
                "DB_NAME",
            ),
            "user": _first_optional(
                env,
                "MYSQL_USER",
                "SOURCE_DB_USER",
                "DB_USER",
            ),
            "password": (
                SecretStr(password)
                if (
                    password := _first_optional(
                        env,
                        "MYSQL_PASSWORD",
                        "SOURCE_DB_PASSWORD",
                        "DB_PASSWORD",
                    )
                ) is not None
                else None
            ),
        }
        if port is not None:
            try:
                values["port"] = int(port)
            except ValueError as exc:
                raise ValueError("MYSQL_PORT 必须是整数") from exc
        return cls(**values)

    @property
    def connector_kwargs(self) -> dict[str, Any]:
        """返回 PyMySQL 与同事 DatabaseConfig 共用的基础连接参数。"""

        kwargs: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
        }
        if self.database is not None:
            kwargs["database"] = self.database
        if self.user is not None:
            kwargs["user"] = self.user
        if self.password is not None:
            kwargs["password"] = self.password.get_secret_value()
        return kwargs


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or not value.strip() else value


def _optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return None if value is None or not value.strip() else value


def _first(
    env: Mapping[str, str],
    *names: str,
    default: str,
) -> str:
    for name in names:
        value = env.get(name)
        if value is not None and value.strip():
            return value
    return default


def _first_optional(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value is not None and value.strip():
            return value
    return None
