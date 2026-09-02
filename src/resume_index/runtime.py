"""Resume 全量处理运行时组装。

运行时在创建 Embedding/Milvus 组件前先执行只读 Source 预检，并把公共向量服务显式
绑定到 ``RESUME_MILVUS_COLLECTION_NAME``。任何依赖都不从 Expert Pipeline 或默认
Collection 隐式复用。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from expert_index.config import DatabaseBoundaryError
from expert_index.ports import ExpertVectorSyncPort
from expert_index.repositories import (
    MySQLBatchItemRepository,
    MySQLBatchRepository,
    MySQLBuildLockRepository,
)
from expert_index.vector_sync import ExpertVectorSynchronizer
from rag_vector_store import (
    MilvusSettings,
    VectorIngestionService,
    create_vector_ingestion_service,
)

from .config import ResumeSettings
from .pipeline import ResumeIndexPipeline
from .preflight import ResumeSourcePreflightResult, run_resume_source_preflight
from .source import MySQLResumeSourceAdapter


@dataclass(frozen=True)
class ResumeRuntime:
    """已通过来源预检并完成依赖组装的 Resume 全量运行时。"""

    pipeline: ResumeIndexPipeline
    preflight: ResumeSourcePreflightResult


def build_resume_runtime(
    settings: ResumeSettings,
    *,
    source: MySQLResumeSourceAdapter | None = None,
    build_lock_repository: MySQLBuildLockRepository | None = None,
    batch_repository: MySQLBatchRepository | None = None,
    batch_item_repository: MySQLBatchItemRepository | None = None,
    vector_sync: ExpertVectorSyncPort | None = None,
    vector_service: VectorIngestionService | None = None,
    base_milvus_settings: MilvusSettings | None = None,
    preflight_runner: Callable[[ResumeSettings], ResumeSourcePreflightResult] = (
        run_resume_source_preflight
    ),
) -> ResumeRuntime:
    """按隔离顺序构造 Resume Pipeline。

    Parameters
    ----------
    settings:
        已完成 Expert/Resume 数据库和 Collection 隔离校验的专用配置。
    source, build_lock_repository, batch_repository, batch_item_repository:
        可选依赖注入；生产默认分别绑定 Resume Source 和 Resume Index Schema。
    vector_sync:
        可选完整快照同步端口。测试可注入 Fake；生产未注入时由本函数组装。
    vector_service:
        可选已绑定 Resume Collection 的公共向量服务。与 ``vector_sync`` 同时传入时
        拒绝启动，避免调用方误以为两个对象都会生效。
    base_milvus_settings:
        Milvus 连接、认证和索引参数。Collection 名称无论原值为何都会被 Resume 配置
        覆盖；未传时从通用 ``MILVUS_*`` 读取连接信息。
    preflight_runner:
        只读来源结构预检。生产使用默认实现，测试注入不连接数据库的 Fake。

    Returns
    -------
    ResumeRuntime
        包含预检摘要和仅能处理 Resume 的 Pipeline。
    """

    if not isinstance(settings, ResumeSettings):
        raise TypeError("settings 必须是 ResumeSettings")
    settings.validate_isolation_boundaries()
    if vector_sync is not None and vector_service is not None:
        raise ValueError("vector_sync 与 vector_service 只能显式传入一个")
    if vector_service is not None:
        _require_resume_vector_service(settings, vector_service)

    # Source 预检必须先于状态写入、Embedding 或 Milvus 客户端构造。
    preflight = preflight_runner(settings)

    resolved_vector_sync = vector_sync
    if resolved_vector_sync is None:
        resolved_vector_service = vector_service
        if resolved_vector_service is None:
            milvus_settings = _resume_milvus_settings(
                settings,
                base=base_milvus_settings,
            )
            resolved_vector_service = create_vector_ingestion_service(
                milvus_settings=milvus_settings,
            )
        resolved_vector_sync = ExpertVectorSynchronizer(resolved_vector_service)

    pipeline = ResumeIndexPipeline(
        settings,
        source=source,
        build_lock_repository=build_lock_repository,
        batch_repository=batch_repository,
        batch_item_repository=batch_item_repository,
        vector_sync=resolved_vector_sync,
    )
    return ResumeRuntime(pipeline=pipeline, preflight=preflight)


def _resume_milvus_settings(
    settings: ResumeSettings,
    *,
    base: MilvusSettings | None,
) -> MilvusSettings:
    """复制连接参数并强制覆盖为 Resume 独立 Collection。"""

    resolved_base = base or MilvusSettings.from_env()
    return resolved_base.model_copy(update={"collection_name": settings.milvus_collection_name})


def _require_resume_vector_service(
    settings: ResumeSettings,
    vector_service: VectorIngestionService,
) -> None:
    """拒绝无法证明已绑定 Resume Collection 的外部向量服务。"""

    repository = getattr(vector_service, "repository", None)
    repository_settings = getattr(repository, "settings", None)
    collection_name = getattr(repository_settings, "collection_name", None)
    if (
        not isinstance(collection_name, str)
        or collection_name.strip() != settings.milvus_collection_name.strip()
    ):
        raise DatabaseBoundaryError(
            "显式 vector_service 未绑定 RESUME_MILVUS_COLLECTION_NAME，已拒绝启动"
        )


__all__ = ["ResumeRuntime", "build_resume_runtime"]
