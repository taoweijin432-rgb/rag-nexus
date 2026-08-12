"""配置读取与版本常量。

这里支持“源业务库”和“派生索引库”两套连接变量。当前演示环境只有一套 DB_*，
所以 INDEX_DB_* 会安全地回退到 DB_*；
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 四种版本各自管理不同兼容边界，不能合并成一个含义模糊的 version。
CHUNK_POLICY_VERSION = "1.2"
# Metadata 新增 Milvus 主键 ``vector_doc_id``，属于必填字段变更，因此升级主版本。
METADATA_SCHEMA_VERSION = "2.0"
DICTIONARY_VERSION = "2026.08"
# v2 将 ``vector_doc_id`` 设为 Metadata 必填字段，并采用新版来源追溯字段。
CONTRACT_VERSION = "2.0.0"

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(path: Path | None = None) -> None:
    """加载简单 ``KEY=value`` 文件，但不覆盖部署环境已经注入的变量。"""

    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return
    for number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"{env_path} 第 {number} 行不是 KEY=value")
        name, value = (part.strip() for part in line.split("=", 1))
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"{env_path} 第 {number} 行变量名无效")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


@dataclass(frozen=True)
class DatabaseConfig:
    """
        DatabaseConfig 是连接参数的"标准化容器"，把分散的参数打包成不可变对象，
        并提供统一的 as_kwargs() 方法生成 PyMySQL 连接参数，
        统一 PyMySQL 的安全默认值。让数据库连接代码干净、统一、不易出错。
    """

    host: str
    port: int
    user: str
    password: str
    database: str

    def as_kwargs(self, *, autocommit: bool = False) -> dict[str, Any]:
        # DictCursor 能让 SQL 字段与领域字段的映射清晰可读；较短超时避免任务挂死。
        import pymysql

        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": "utf8mb4",           # 支持 emoji 等 4 字节字符
            "cursorclass": pymysql.cursors.DictCursor,  # 返回字典而非元组，字段名清晰
            "connect_timeout": 8,           # 连接超时 8 秒
            "read_timeout": 60,             # 读超时 60 秒
            "write_timeout": 60,            # 写超时 60 秒
            "autocommit": autocommit,       # 是否自动提交事务
        }


@dataclass(frozen=True)
class Settings:
    """流水线运行配置，包含连接边界、版本和默认时区。"""

    source_db: DatabaseConfig
    index_db: DatabaseConfig
    timezone: str = "Asia/Shanghai"
    chunk_policy_version: str = CHUNK_POLICY_VERSION
    metadata_schema_version: str = METADATA_SCHEMA_VERSION
    dictionary_version: str = DICTIONARY_VERSION

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量构造配置；未设置密码变量时才拒绝，允许本地空密码。"""

        load_env_file()
        base_password = os.getenv("DB_PASSWORD")
        if base_password is None:
            raise RuntimeError("缺少 DB_PASSWORD，无法连接 MySQL")

        source = DatabaseConfig(
            host=os.getenv("SOURCE_DB_HOST", os.getenv("DB_HOST", "127.0.0.1")),
            port=int(os.getenv("SOURCE_DB_PORT", os.getenv("DB_PORT", "3306"))),
            user=os.getenv("SOURCE_DB_USER", os.getenv("DB_USER", "root")),
            password=os.getenv("SOURCE_DB_PASSWORD", base_password),
            database=os.getenv("SOURCE_DB_NAME", os.getenv("DB_NAME", "expert_demo")),
        )
        index = DatabaseConfig(
            host=os.getenv("INDEX_DB_HOST", os.getenv("DB_HOST", "127.0.0.1")),
            port=int(os.getenv("INDEX_DB_PORT", os.getenv("DB_PORT", "3306"))),
            user=os.getenv("INDEX_DB_USER", os.getenv("DB_USER", "root")),
            password=os.getenv("INDEX_DB_PASSWORD", base_password),
            database=os.getenv("INDEX_DB_NAME", os.getenv("DB_NAME", "expert_demo")),
        )
        return cls(source_db=source, index_db=index)
