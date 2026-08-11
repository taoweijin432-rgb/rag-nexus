"""从环境变量加载 Milvus 配置。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SchemaMode = Literal["legacy", "expert"]


class MilvusSettings(BaseModel):
    """供 :class:`MilvusVectorRepository` 使用的连接和集合配置。

    ``from_env`` 显式读取环境变量，不依赖 dotenv 加载器。这样可以让密钥只存在
    于进程环境中，也不会把模块绑定到某一种部署方式。
    """

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(default="http://localhost:19530", min_length=1)
    token: str | None = None
    user: str | None = None
    password: str | None = None
    db_name: str | None = None
    timeout: float | None = Field(default=None, gt=0)

    collection_name: str = Field(default="rag_chunks", min_length=1)
    schema_mode: SchemaMode = "legacy"
    chunk_id_field: str = Field(default="chunk_id", min_length=1)
    text_field: str = Field(default="text", min_length=1)
    embedding_field: str = Field(default="embedding", min_length=1)
    metadata_field: str = Field(default="metadata", min_length=1)

    chunk_id_max_length: int = Field(default=512, ge=1, le=65535)
    text_max_length: int = Field(default=65535, ge=1, le=65535)
    expert_vector_doc_id_max_length: int = Field(default=256, ge=1, le=65535)
    expert_chunk_id_max_length: int = Field(default=256, ge=1, le=65535)
    expert_chunk_text_max_length: int = Field(default=16384, ge=1, le=65535)
    index_type: str = Field(default="AUTOINDEX", min_length=1)
    metric_type: str = Field(default="COSINE", min_length=1)
    index_params: dict[str, Any] = Field(default_factory=dict)

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
            kwargs["token"] = self.token
        elif self.user is not None and self.password is not None:
            kwargs["user"] = self.user
            kwargs["password"] = self.password
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
            "collection_name": _get(env, "MILVUS_COLLECTION_NAME", "rag_chunks"),
            "schema_mode": _get(env, "MILVUS_SCHEMA_MODE", "legacy"),
            "index_type": _get(env, "MILVUS_INDEX_TYPE", "AUTOINDEX"),
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

        return cls(**values)


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or not value.strip() else value


def _optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return None if value is None or not value.strip() else value
