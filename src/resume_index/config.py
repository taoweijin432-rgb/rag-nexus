"""Resume 独立运行配置与 Expert 隔离门禁。

本模块只把环境变量转换为强类型配置，不建立 MySQL/Milvus 连接，也不读取业务数据。
Resume Source 可以与 Expert Source 位于同一个只读 Schema；Resume Index 和 Milvus
Collection 必须与 Expert 物理隔离。所有 Resume 变量均要求显式声明，禁止回退到
``SOURCE_DB_*``、``INDEX_DB_*`` 或默认 Expert Collection。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from expert_index.config import (
    DEFAULT_INCREMENTAL_BATCH_SIZE,
    DEFAULT_INCREMENTAL_LEASE_SECONDS,
    MAX_INCREMENTAL_BATCH_SIZE,
    CdcRuntimeConfig,
    DatabaseBoundaryError,
    DatabaseConfig,
    DatabaseRole,
    IncrementalScanConfig,
    Settings,
    load_env_file,
)
from rag_vector_store.config import DEFAULT_MILVUS_COLLECTION_NAME

# Resume CDC 使用独立 Canal destination 与表订阅，禁止复用 Expert 的
# ``company_expert`` 和 ``experts_*`` 契约。默认保持关闭，只有部署完成真实
# Probe、ACK/重投验证后才允许由环境变量显式开启。
RESUME_CANAL_DESTINATION = "company_resume"
RESUME_CANAL_SUBSCRIPTION = (
    r"expert\.(resume|resume_work_experience|resume_project_experience|"
    r"resume_education_experience)"
)


def _default_resume_cdc_runtime() -> CdcRuntimeConfig:
    """返回隔离且关闭的 Resume CDC 默认配置。"""

    return CdcRuntimeConfig(
        destination=RESUME_CANAL_DESTINATION,
        subscription=RESUME_CANAL_SUBSCRIPTION,
    )


@dataclass(frozen=True)
class PhysicalDatabaseTarget:
    """不含凭据的 MySQL 物理目标，用于跨链路隔离比较。

    ``host + port + database`` 唯一标识一个待访问 Schema。用户名和密码不参与比较，
    避免不同账号掩盖两条链路实际写入同一 Schema 的风险。
    """

    host: str
    port: int
    database: str

    @classmethod
    def from_database_config(cls, config: DatabaseConfig) -> PhysicalDatabaseTarget:
        """从带角色的连接配置提取不含凭据的物理目标。"""

        if not isinstance(config, DatabaseConfig):
            raise TypeError("config 必须是 DatabaseConfig")
        return cls(host=config.host, port=config.port, database=config.database)

    def matches_database_config(self, config: DatabaseConfig) -> bool:
        """判断连接配置是否指向同一物理 Schema。"""

        if not isinstance(config, DatabaseConfig):
            return False
        return self._identity() == PhysicalDatabaseTarget.from_database_config(config)._identity()

    def _identity(self) -> tuple[str, int, str]:
        """统一本机地址和大小写，避免字符串差异绕过隔离门禁。"""

        host = self.host.strip().casefold()
        if host in {"localhost", "::1", "[::1]"}:
            host = "127.0.0.1"
        return host, self.port, self.database.strip().casefold()


@dataclass(frozen=True)
class ResumeSettings:
    """Resume 预检及后续处理链使用的完整隔离配置。

    :param source_db: Resume 四张来源表的只读连接。它可以与 Expert Source 相同，
        但只能由 Resume Source Adapter 使用固定 ``resume_*`` 表白名单。
    :param index_db: Resume 批次、任务、Checkpoint、artifact 和 DLQ 的写入目标。
        该目标必须同时区别于 Resume Source、Expert Source 和 Expert Index。
    :param milvus_collection_name: Resume 独立 Collection 名称；不得使用 Expert Collection。
    :param expert_source_target: 当前 Expert Source 的不含凭据目标，仅用于避免 Resume
        状态误写 Expert 来源库；允许与 ``source_db`` 相同。
    :param expert_index_target: 当前 Expert Index 的不含凭据目标，仅用于隔离比较。
    :param expert_milvus_collection_name: 当前 Expert Collection 名称，仅用于隔离比较。
    """

    source_db: DatabaseConfig
    index_db: DatabaseConfig
    milvus_collection_name: str
    expert_source_target: PhysicalDatabaseTarget
    expert_index_target: PhysicalDatabaseTarget
    expert_milvus_collection_name: str
    incremental_scan: IncrementalScanConfig = field(default_factory=IncrementalScanConfig)
    cdc_runtime: CdcRuntimeConfig = field(default_factory=_default_resume_cdc_runtime)

    def __post_init__(self) -> None:
        """任何构造方式都必须经过角色和物理目标隔离校验。"""

        self.validate_isolation_boundaries()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ResumeSettings:
        """从显式环境变量构造 Resume 配置。

        :param environ: 可选环境变量映射。测试传入独立字典；不传时先加载项目 `.env`，
            再读取当前进程环境。方法不会修改已有环境变量。
        :return: 已完成隔离校验的 :class:`ResumeSettings`。
        :raises RuntimeError: 任一 Resume 必需变量或 Expert 隔离参照缺失。
        :raises ValueError: 端口格式非法或物理隔离条件不满足。

        Resume Source/Index 的主机、端口、用户、密码和 Schema 全部必须以
        ``RESUME_*`` 显式提供。现有 Expert 变量只作为隔离参照，不作为回退值。
        """

        if environ is None:
            load_env_file()
            values: Mapping[str, str] = os.environ
        else:
            values = environ

        source_db = DatabaseConfig(
            host=_require_environment_value(values, "RESUME_SOURCE_DB_HOST"),
            port=_parse_tcp_port(
                "RESUME_SOURCE_DB_PORT",
                _require_environment_value(values, "RESUME_SOURCE_DB_PORT"),
            ),
            user=_require_environment_value(values, "RESUME_SOURCE_DB_USER"),
            password=_require_environment_value(
                values,
                "RESUME_SOURCE_DB_PASSWORD",
                allow_empty=True,
            ),
            database=_require_environment_value(values, "RESUME_SOURCE_DB_NAME"),
            role=DatabaseRole.SOURCE,
        )
        index_db = DatabaseConfig(
            host=_require_environment_value(values, "RESUME_INDEX_DB_HOST"),
            port=_parse_tcp_port(
                "RESUME_INDEX_DB_PORT",
                _require_environment_value(values, "RESUME_INDEX_DB_PORT"),
            ),
            user=_require_environment_value(values, "RESUME_INDEX_DB_USER"),
            password=_require_environment_value(
                values,
                "RESUME_INDEX_DB_PASSWORD",
                allow_empty=True,
            ),
            database=_require_environment_value(values, "RESUME_INDEX_DB_NAME"),
            role=DatabaseRole.INDEX,
        )

        # Expert 账号和密码不需要进入 Resume 进程；只读取 Source/Index 的物理
        # 定位字段作为 fail-closed 参照。缺少参照时无法证明不会串写，因此拒绝启动。
        expert_source_target = PhysicalDatabaseTarget(
            host=_require_environment_value(values, "SOURCE_DB_HOST"),
            port=_parse_tcp_port(
                "SOURCE_DB_PORT",
                _require_environment_value(values, "SOURCE_DB_PORT"),
            ),
            database=_require_environment_value(values, "SOURCE_DB_NAME"),
        )
        expert_index_target = PhysicalDatabaseTarget(
            host=_require_environment_value(values, "INDEX_DB_HOST"),
            port=_parse_tcp_port(
                "INDEX_DB_PORT",
                _require_environment_value(values, "INDEX_DB_PORT"),
            ),
            database=_require_environment_value(values, "INDEX_DB_NAME"),
        )
        expert_collection = values.get(
            "MILVUS_COLLECTION_NAME",
            DEFAULT_MILVUS_COLLECTION_NAME,
        ).strip()
        if not expert_collection:
            expert_collection = DEFAULT_MILVUS_COLLECTION_NAME
        incremental_scan = IncrementalScanConfig(
            enabled=_parse_boolean(
                "RESUME_INCREMENTAL_ENABLED",
                values.get("RESUME_INCREMENTAL_ENABLED", "0"),
            ),
            batch_size=_parse_integer(
                "RESUME_INCREMENTAL_BATCH_SIZE",
                values.get(
                    "RESUME_INCREMENTAL_BATCH_SIZE",
                    str(DEFAULT_INCREMENTAL_BATCH_SIZE),
                ),
                minimum=1,
                maximum=MAX_INCREMENTAL_BATCH_SIZE,
            ),
            lease_seconds=_parse_integer(
                "RESUME_INCREMENTAL_LEASE_SECONDS",
                values.get(
                    "RESUME_INCREMENTAL_LEASE_SECONDS",
                    str(DEFAULT_INCREMENTAL_LEASE_SECONDS),
                ),
                minimum=1,
            ),
        )
        cdc_runtime = CdcRuntimeConfig(
            enabled=_parse_boolean(
                "RESUME_CDC_ENABLED",
                values.get("RESUME_CDC_ENABLED", "0"),
            ),
            host=values.get("RESUME_CANAL_HOST", "127.0.0.1"),
            port=_parse_tcp_port(
                "RESUME_CANAL_PORT",
                values.get("RESUME_CANAL_PORT", "11111"),
            ),
            destination=values.get(
                "RESUME_CANAL_DESTINATION",
                RESUME_CANAL_DESTINATION,
            ),
            client_id=values.get("RESUME_CANAL_CLIENT_ID", ""),
            subscription=values.get(
                "RESUME_CANAL_SUBSCRIPTION",
                RESUME_CANAL_SUBSCRIPTION,
            ),
            username=values.get("RESUME_CANAL_USERNAME", ""),
            password=values.get("RESUME_CANAL_PASSWORD", ""),
            server_version=values.get("RESUME_CANAL_SERVER_VERSION", "1.1.8"),
            server_mode=values.get("RESUME_CANAL_SERVER_MODE", "tcp"),
            tcp_port=_parse_tcp_port(
                "RESUME_CANAL_TCP_PORT",
                values.get("RESUME_CANAL_TCP_PORT", "11111"),
            ),
            probe_batch_id=_parse_integer(
                "RESUME_CDC_PROBE_BATCH_ID",
                values.get("RESUME_CDC_PROBE_BATCH_ID", "0"),
                minimum=0,
            ),
            probe_entry_count=_parse_integer(
                "RESUME_CDC_PROBE_ENTRY_COUNT",
                values.get("RESUME_CDC_PROBE_ENTRY_COUNT", "0"),
                minimum=0,
            ),
            source_connected=_parse_boolean(
                "RESUME_CDC_SOURCE_CONNECTED",
                values.get("RESUME_CDC_SOURCE_CONNECTED", "0"),
            ),
            ack_confirmed=_parse_boolean(
                "RESUME_CDC_ACK_CONFIRMED",
                values.get("RESUME_CDC_ACK_CONFIRMED", "0"),
            ),
            protocol_ready=_parse_boolean(
                "RESUME_CDC_PROTOCOL_READY",
                values.get("RESUME_CDC_PROTOCOL_READY", "0"),
            ),
        )

        return cls(
            source_db=source_db,
            index_db=index_db,
            milvus_collection_name=_require_environment_value(
                values,
                "RESUME_MILVUS_COLLECTION_NAME",
            ),
            expert_source_target=expert_source_target,
            expert_index_target=expert_index_target,
            expert_milvus_collection_name=expert_collection,
            incremental_scan=incremental_scan,
            cdc_runtime=cdc_runtime,
        )

    def to_processing_settings(self) -> Settings:
        """构造仅绑定 Resume Source/Index 的公共加工配置。

        公共配置中的增量开关和分页参数只来自 ``RESUME_INCREMENTAL_*``，CDC 参数
        只来自 ``RESUME_CDC_*``/``RESUME_CANAL_*``，均不会读取 Expert 同名配置。
        """

        self.validate_isolation_boundaries()
        return Settings(
            source_db=self.source_db,
            index_db=self.index_db,
            incremental_scan=self.incremental_scan,
            cdc_runtime=self.cdc_runtime,
        )

    def validate_isolation_boundaries(self) -> None:
        """在建立任何连接前验证 Resume 与 Expert 的写入隔离。

        允许 Resume/Expert 共用只读 Source Schema；禁止 Resume Index 指向任一 Source、
        Expert Index，或 Resume Collection 与 Expert Collection 同名。
        """

        if (
            not isinstance(self.source_db, DatabaseConfig)
            or self.source_db.role is not DatabaseRole.SOURCE
        ):
            raise DatabaseBoundaryError("Resume Source 必须使用 RESUME_SOURCE_DB_* 构造")
        if (
            not isinstance(self.index_db, DatabaseConfig)
            or self.index_db.role is not DatabaseRole.INDEX
        ):
            raise DatabaseBoundaryError("Resume Index 必须使用 RESUME_INDEX_DB_* 构造")
        if self.source_db.targets_same_schema(self.index_db):
            raise DatabaseBoundaryError(
                "RESUME_INDEX_DB_* 与 Resume Source 指向同一 MySQL Schema，已拒绝状态写入"
            )
        if self.expert_source_target.matches_database_config(self.index_db):
            raise DatabaseBoundaryError(
                "RESUME_INDEX_DB_* 与 Expert Source 指向同一 MySQL Schema，已拒绝状态写入"
            )
        if self.expert_index_target.matches_database_config(self.index_db):
            raise DatabaseBoundaryError(
                "RESUME_INDEX_DB_* 与 Expert Index 指向同一 MySQL Schema，已拒绝串写"
            )

        resume_collection = _normalize_collection_name(
            "RESUME_MILVUS_COLLECTION_NAME",
            self.milvus_collection_name,
        )
        expert_collection = _normalize_collection_name(
            "MILVUS_COLLECTION_NAME",
            self.expert_milvus_collection_name,
        )
        if resume_collection.casefold() == expert_collection.casefold():
            raise DatabaseBoundaryError(
                "RESUME_MILVUS_COLLECTION_NAME 与 Expert Collection 相同，已拒绝串写"
            )
        if self.cdc_runtime.destination.strip() != RESUME_CANAL_DESTINATION:
            raise DatabaseBoundaryError(
                "RESUME_CANAL_DESTINATION 必须固定为 company_resume，已拒绝串接其他链路"
            )
        if self.cdc_runtime.subscription.strip() != RESUME_CANAL_SUBSCRIPTION:
            raise DatabaseBoundaryError(
                "RESUME_CANAL_SUBSCRIPTION 与冻结的四张 Resume 来源表不一致，已拒绝启动"
            )


def _require_environment_value(
    values: Mapping[str, str],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    """读取必需变量；密码允许显式空值，但不允许变量缺失。"""

    if name not in values or (not allow_empty and not values[name].strip()):
        raise RuntimeError(f"缺少必需配置 {name}")
    return values[name]


def _parse_tcp_port(name: str, value: str) -> int:
    """解析并限制 MySQL TCP 端口到 1~65535。"""

    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} 必须位于 1 到 65535 之间")
    return port


def _normalize_collection_name(name: str, value: str) -> str:
    """清洗 Collection 名称并拒绝空字符串。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _parse_boolean(name: str, value: str) -> bool:
    """解析显式布尔环境变量，拒绝可能产生歧义的文本。"""

    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 0/1、true/false、yes/no 或 on/off")


def _parse_integer(
    name: str,
    value: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """解析带上下界的 Resume 增量整数配置。"""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        upper = f" 到 {maximum}" if maximum is not None else "以上"
        raise ValueError(f"{name} 必须位于 {minimum}{upper}")
    return parsed
