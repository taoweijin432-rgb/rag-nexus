"""读取专家来源库与本地索引状态库配置，并维护版本常量。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# 四种版本各自管理不同兼容边界，不能合并成一个含义模糊的 version。
CHUNK_POLICY_VERSION = "1.3"
# 4.0：Metadata 从三类 superset 改为按 chunk_type 分类组装（per-type），
# 不适用于当前类型的字段不再输出占位，属于主版本变化。
METADATA_SCHEMA_VERSION = "4.0"
DICTIONARY_VERSION = "2026.08"
# v4 删除最终表/导出中的 chunk_id、source_item_key、dictionary_version、data_version；
# 4.1 将 ExpertChunkEmbeddingInput.metadata 从统一 superset 改为 per-type 模型。
CONTRACT_VERSION = "4.1.0"

DEFAULT_INCREMENTAL_BATCH_SIZE = 500
DEFAULT_INCREMENTAL_LEASE_SECONDS = 120
MAX_INCREMENTAL_BATCH_SIZE = 1000

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatabaseBoundaryError(RuntimeError):
    """表示数据库角色或 Source/Index 物理目标不满足写入边界。"""


class DatabaseRole(str, Enum):
    """标识连接配置允许承担的数据库职责。"""

    UNSPECIFIED = "unspecified"
    SOURCE = "source"
    INDEX = "index"
    ISOLATED_TEST = "isolated_test"


@dataclass(frozen=True)
class IncrementalScanConfig:
    """保存候选扫描开关、单页数量和来源表租约时长。

    该配置只控制 P3 候选扫描，不表示任务已完成，也不允许自动推进安全 Checkpoint。
    """

    enabled: bool = False
    batch_size: int = DEFAULT_INCREMENTAL_BATCH_SIZE
    lease_seconds: int = DEFAULT_INCREMENTAL_LEASE_SECONDS

    def __post_init__(self) -> None:
        """在建立数据库连接前校验候选分页和租约参数。"""

        if not isinstance(self.enabled, bool):
            raise TypeError("incremental enabled 必须是布尔值")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= MAX_INCREMENTAL_BATCH_SIZE
        ):
            raise ValueError(
                f"incremental batch_size 必须位于 1 到 {MAX_INCREMENTAL_BATCH_SIZE} 之间"
            )
        if (
            isinstance(self.lease_seconds, bool)
            or not isinstance(self.lease_seconds, int)
            or self.lease_seconds < 1
        ):
            raise ValueError("incremental lease_seconds 必须是正整数")


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
    """保存一个 MySQL Schema 的连接参数、用途角色和安全连接默认值。"""

    host: str
    port: int
    user: str
    password: str
    database: str
    role: DatabaseRole = DatabaseRole.UNSPECIFIED

    def targets_same_schema(self, other: DatabaseConfig) -> bool:
        """判断两个配置是否指向同一 MySQL 主机、端口和 Schema。

        用户名和密码不参与目标判断；即使写账号不同，只要物理 Schema 相同就不能把
        Index 状态写入 Source。常见本机地址会统一为 ``127.0.0.1``。
        """

        if not isinstance(other, DatabaseConfig):
            return False
        return self._target_identity() == other._target_identity()

    def _target_identity(self) -> tuple[str, int, str]:
        """返回不包含凭据的规范化数据库目标标识。"""

        host = self.host.strip().casefold()
        if host in {"localhost", "::1", "[::1]"}:
            host = "127.0.0.1"
        return host, self.port, self.database.strip().casefold()

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
    incremental_scan: IncrementalScanConfig = field(
        default_factory=IncrementalScanConfig
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """从环境变量构造 Source/Index 配置，并要求 Index 使用独立显式变量。"""

        if env is None:
            load_env_file()
            values: Mapping[str, str] = os.environ
        else:
            values = env

        source_password = _first_present(values, "SOURCE_DB_PASSWORD", "DB_PASSWORD")
        if source_password is None:
            raise RuntimeError("缺少 SOURCE_DB_PASSWORD 或 DB_PASSWORD，无法连接来源 MySQL")

        source = DatabaseConfig(
            host=values.get("SOURCE_DB_HOST", values.get("DB_HOST", "127.0.0.1")),
            port=_parse_port(
                "SOURCE_DB_PORT",
                values.get("SOURCE_DB_PORT", values.get("DB_PORT", "3306")),
            ),
            user=values.get("SOURCE_DB_USER", values.get("DB_USER", "root")),
            password=source_password,
            database=values.get("SOURCE_DB_NAME", values.get("DB_NAME", "expert_demo")),
            role=DatabaseRole.SOURCE,
        )
        index = DatabaseConfig(
            host=_required_value(values, "INDEX_DB_HOST"),
            port=_parse_port("INDEX_DB_PORT", _required_value(values, "INDEX_DB_PORT")),
            user=_required_value(values, "INDEX_DB_USER"),
            password=_required_value(values, "INDEX_DB_PASSWORD", allow_empty=True),
            database=_required_value(values, "INDEX_DB_NAME"),
            role=DatabaseRole.INDEX,
        )
        incremental_scan = IncrementalScanConfig(
            enabled=_parse_boolean(
                "INCREMENTAL_ENABLED",
                values.get("INCREMENTAL_ENABLED", "0"),
            ),
            batch_size=_parse_integer(
                "INCREMENTAL_BATCH_SIZE",
                values.get(
                    "INCREMENTAL_BATCH_SIZE",
                    str(DEFAULT_INCREMENTAL_BATCH_SIZE),
                ),
                minimum=1,
                maximum=MAX_INCREMENTAL_BATCH_SIZE,
            ),
            lease_seconds=_parse_integer(
                "INCREMENTAL_LEASE_SECONDS",
                values.get(
                    "INCREMENTAL_LEASE_SECONDS",
                    str(DEFAULT_INCREMENTAL_LEASE_SECONDS),
                ),
                minimum=1,
            ),
        )
        return cls(
            source_db=source,
            index_db=index,
            incremental_scan=incremental_scan,
        )

    def validate_index_write_boundary(self) -> None:
        """拒绝使用非 Index 配置或把状态写入 Source 所在物理 Schema。"""

        require_index_database(self.index_db)
        if self.source_db.targets_same_schema(self.index_db):
            raise DatabaseBoundaryError(
                "INDEX_DB_* 与来源库指向同一 MySQL Schema，已拒绝状态写入"
            )


def require_index_database(config: DatabaseConfig) -> None:
    """确保状态 Repository 收到由 ``INDEX_DB_*`` 构造的连接配置。"""

    if not isinstance(config, DatabaseConfig) or config.role is not DatabaseRole.INDEX:
        raise DatabaseBoundaryError("状态 Repository 只能使用 INDEX_DB_* 连接配置")


def _first_present(values: Mapping[str, str], *names: str) -> str | None:
    """返回第一个已声明的环境变量值，允许密码显式为空字符串。"""

    for name in names:
        if name in values:
            return values[name]
    return None


def _required_value(
    values: Mapping[str, str],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    """读取正式入口必需的环境变量，并对非密码字段拒绝空值。"""

    if name not in values or (not allow_empty and not values[name].strip()):
        raise RuntimeError(f"缺少必需配置 {name}")
    return values[name]


def _parse_port(name: str, value: str) -> int:
    """把端口配置转换为有效的 TCP 端口号。"""

    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} 必须位于 1 到 65535 之间")
    return port


def _parse_boolean(name: str, value: str) -> bool:
    """把显式环境变量解析为布尔值，并拒绝含义不明确的文本。"""

    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false 或 1/0")


def _parse_integer(
    name: str,
    value: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """把整数环境变量转换后校验闭区间范围。"""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        if maximum is None:
            raise ValueError(f"{name} 必须大于等于 {minimum}")
        raise ValueError(f"{name} 必须位于 {minimum} 到 {maximum} 之间")
    return parsed
