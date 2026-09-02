"""Resume 启动前只读预检编排。

预检先验证 Index/Collection 隔离，再连接 Resume Source 查询 ``information_schema``。
它不连接 Resume Index 或 Milvus，不读取简历正文，也不执行任何 DDL/DML。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pymysql

from .config import ResumeSettings
from .source_schema import RESUME_SOURCE_TABLES, require_resume_source_schema


class _ClosableConnection(Protocol):
    """预检所需的最小连接协议，便于生产和测试使用同一编排。"""

    def cursor(self) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ResumeSourcePreflightResult:
    """不含凭据和业务数据的启动预检摘要。"""

    source_schema: str
    source_tables: tuple[str, ...]
    index_schema: str
    milvus_collection_name: str


def run_resume_source_preflight(
    settings: ResumeSettings,
    *,
    connection_factory: Callable[..., _ClosableConnection] = pymysql.connect,
) -> ResumeSourcePreflightResult:
    """验证隔离边界并只读检查四张 Resume 来源表。

    :param settings: 已由 :meth:`ResumeSettings.from_env` 构造的专用配置。
    :param connection_factory: MySQL 连接工厂。生产默认使用 ``pymysql.connect``；
        测试注入 Fake，避免访问真实数据库。
    :return: 可安全输出的 Schema、表名和独立 Collection 摘要。
    :raises DatabaseBoundaryError: Index 或 Collection 与 Expert 冲突。
    :raises ResumeSourceSchemaError: 四张来源表结构不满足已冻结契约。

    该函数不会创建 Index 表或 Milvus Collection。即使来源预检成功，后续处理链仍需
    在自己的阶段显式初始化 Resume Index，并单独校验 Milvus Schema。
    """

    if not isinstance(settings, ResumeSettings):
        raise TypeError("settings 必须是 ResumeSettings")
    settings.validate_isolation_boundaries()

    connection = connection_factory(**settings.source_db.as_kwargs(autocommit=True))
    try:
        report = require_resume_source_schema(
            connection,
            schema_name=settings.source_db.database,
        )
    finally:
        connection.close()

    return ResumeSourcePreflightResult(
        source_schema=report.schema_name,
        source_tables=RESUME_SOURCE_TABLES,
        index_schema=settings.index_db.database,
        milvus_collection_name=settings.milvus_collection_name,
    )
