"""从环境变量加载 Milvus 配置。

配置模块只负责把进程环境转换为强类型设置，不创建连接、不读取数据库，也不
改变现有 ``.env``。凭据通过 ``SecretStr``/连接参数传递，错误诊断不能回显密钥。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

# 运行时只允许正式 Contract 4.1.0；旧兼容模式已退役。
SchemaMode = Literal["contract"]
ExpertStorageProfile = Literal["v15", "v16", "v16_external_formal"]
ExpertV16Analyzer = Literal["chinese", "jieba_targeted"]

# 默认生产 V15 Collection；历史 dev/v14/e2e 名称只作数据保留，不再是运行目标。
DEFAULT_MILVUS_COLLECTION_NAME = "expert_candidates_v15_contract"
EXPERT_V16_COLLECTION_NAME = "expert_candidates_v16_contract"
EXPERT_V16_EXTERNAL_FORMAL_COLLECTION_NAME = "v16_external_jieba_accurate_v1"



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
        default=DEFAULT_MILVUS_COLLECTION_NAME,
        min_length=1,
    )
    expert_storage_profile: ExpertStorageProfile = "v15"
    # 默认保持旧 V16 兼容；新建 Jieba 大词典集合必须显式选择，V15/Resume 不读取词典。
    expert_v16_analyzer: ExpertV16Analyzer = "chinese"
    expert_v16_dictionary_path: str | None = Field(default=None, min_length=1)
    expert_v16_dictionary_version: str | None = Field(default=None, min_length=1)
    expert_v16_dictionary_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    expert_v16_external_hmm: bool = False
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
        if (
            self.expert_storage_profile == "v16"
            and self.collection_name != EXPERT_V16_COLLECTION_NAME
        ):
            raise ValueError(
                "EXPERT_MILVUS_STORAGE_PROFILE=v16 时 "
                f"MILVUS_COLLECTION_NAME 必须是 {EXPERT_V16_COLLECTION_NAME!r}"
            )
        if (
            self.expert_storage_profile == "v15"
            and self.collection_name == EXPERT_V16_COLLECTION_NAME
        ):
            raise ValueError(
                f"MILVUS_COLLECTION_NAME={EXPERT_V16_COLLECTION_NAME!r} 时 "
                "EXPERT_MILVUS_STORAGE_PROFILE 必须是 v16"
            )
        if (self.collection_name == EXPERT_V16_EXTERNAL_FORMAL_COLLECTION_NAME
                and self.expert_storage_profile != "v16_external_formal"):
            raise ValueError("外部 Jieba 正式集合必须使用 v16_external_formal Profile")
        if self.expert_v16_analyzer == "jieba_targeted" and not all((
            self.expert_v16_dictionary_path,
            self.expert_v16_dictionary_version,
            self.expert_v16_dictionary_sha256,
        )):
            raise ValueError("Jieba 大词典必须配置本地路径、版本及 SHA256")
        if self.expert_storage_profile == "v16_external_formal":
            if not all((self.expert_v16_dictionary_path, self.expert_v16_dictionary_version,
                        self.expert_v16_dictionary_sha256)):
                raise ValueError("外部 Jieba 必须配置词典路径、版本及 SHA256")
            suffix = "_contract"
            named_formal = (
                self.expert_storage_profile == "v16_external_formal"
                and self.expert_v16_external_hmm
                and self.collection_name == EXPERT_V16_EXTERNAL_FORMAL_COLLECTION_NAME
            )
            if not (self.collection_name.startswith("expert_candidates_v16_extjieba_")
                    and self.collection_name.endswith(suffix)) and not named_formal:
                raise ValueError("外部 Jieba 必须显式绑定匹配测试/正式 Profile 的集合")
        elif self.collection_name.startswith("expert_candidates_v16_extjieba_"):
            raise ValueError("外部 Jieba 集合必须绑定对应的外部分词 Profile")
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
                DEFAULT_MILVUS_COLLECTION_NAME,
            ),
            "expert_storage_profile": _get(
                env,
                "EXPERT_MILVUS_STORAGE_PROFILE",
                "v15",
            ),
            "expert_v16_analyzer": _get(env, "EXPERT_V16_ANALYZER", "chinese"),
            "expert_v16_dictionary_path": _optional(env, "EXPERT_V16_DICTIONARY_PATH"),
            "expert_v16_dictionary_version": _optional(env, "EXPERT_V16_DICTIONARY_VERSION"),
            "expert_v16_dictionary_sha256": _optional(env, "EXPERT_V16_DICTIONARY_SHA256"),
            "expert_v16_external_hmm": _get(env, "EXPERT_V16_EXTERNAL_HMM", "false"),
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


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or not value.strip() else value


def _optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return None if value is None or not value.strip() else value
