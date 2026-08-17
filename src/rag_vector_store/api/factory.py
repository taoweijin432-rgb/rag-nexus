"""生产环境 FastAPI 应用工厂。

提供 ``create_production_app()``，从环境变量组装全部真实依赖：

    env (TASK_STATE_MYSQL_* / MYSQL_* / MILVUS_* / EMBEDDING_*)
      ↓
    TaskStateMySQLSettings      → TaskStateStore
    MySQLSettings               → MySqlExpertChunkReader
    MilvusSettings              → ContractMilvusVectorRepository
    CompanyCloudEmbeddingModel  → ContractEmbeddingWorker
                                   → ContractBatchIngestionService
                                   → VectorizationPipelineRunner
                                   → AsyncTaskExecutor
                                   → ServiceContainer
                                   → create_app(services)

启动::

    uv run uvicorn rag_vector_store.api.factory:create_production_app --factory

不修改 Runner / Executor / TaskStateStore / API 路由逻辑——本模块只做组装。
"""

from __future__ import annotations

import logging
from datetime import timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI

from ..chunking.contract_ingestion import ContractBatchIngestionService
from ..chunking.contract_pipeline import ContractEmbeddingWorker
from ..config import MilvusSettings, MySQLSettings
from ..embedding_adapters.company_cloud import CompanyCloudEmbeddingModel
from ..milvus.repository import ContractMilvusVectorRepository
from ..mysql.expert_chunk_reader import MySqlExpertChunkReader
from ..runner.async_executor import AsyncTaskExecutor
from ..runner.pipeline_runner import VectorizationPipelineRunner
from ..task_state.config import TaskStateMySQLSettings
from ..task_state.store import TaskStateStore
from .app import ServiceContainer, create_app

__all__ = ["create_production_app"]

_logger = logging.getLogger(__name__)

# 数据库时区：MySQL 中的 DATETIME(6) 按 Asia/Shanghai 解释（与 Reader 一致）。
_DB_TIMEZONE = ZoneInfo("Asia/Shanghai")


def create_production_app(
    *,
    max_workers: int = 4,
    timezone_override: timezone | None = None,
) -> FastAPI:
    """从环境变量组装全部依赖，构建可运行的 FastAPI 应用。

    Args:
        max_workers: AsyncTaskExecutor 线程池大小，默认 4。
        timezone_override: 覆盖默认的 Asia/Shanghai 数据库时区。

    Returns:
        已注入 ServiceContainer 的 FastAPI 实例。
    """

    db_tz = timezone_override or _DB_TIMEZONE

    # ── 1. Task State Store（独立 TASK_STATE_MYSQL_* 配置） ──
    task_state_settings = TaskStateMySQLSettings.from_env()
    task_state_store = TaskStateStore(
        task_state_settings,
        database_timezone=db_tz,
    )

    # ── 2. MySQL Reader（MYSQL_* / SOURCE_DB_* / DB_* 配置） ──
    mysql_settings = MySQLSettings.from_env()
    reader = MySqlExpertChunkReader(
        mysql_settings,
        database_timezone=db_tz,
    )

    # ── 3. Embedding Model（EMBEDDING_* 配置） ──
    embedding_model = CompanyCloudEmbeddingModel.from_env()

    # ── 4. Contract Embedding Worker（校验/Eligibility/向量化） ──
    worker = ContractEmbeddingWorker(embedding_model)

    # ── 5. Milvus Repository（MILVUS_* 配置） ──
    milvus_settings = MilvusSettings.from_env()
    repository = ContractMilvusVectorRepository(milvus_settings)

    # ── 6. Contract Batch Ingestion Service ──
    ingestion_service = ContractBatchIngestionService(worker, repository)

    # ── 7. Pipeline Runner（编排层） ──
    runner = VectorizationPipelineRunner(
        task_state_store=task_state_store,
        reader=reader,
        ingestion_service=ingestion_service,
    )

    # ── 8. Async Executor ──
    executor = AsyncTaskExecutor(runner, max_workers=max_workers)

    # ── 9. 组装 ServiceContainer + FastAPI ──
    services = ServiceContainer(
        task_state_store=task_state_store,
        executor=executor,
    )

    _logger.info(
        "生产应用已组装: collection=%s, max_workers=%d",
        milvus_settings.collection_name,
        max_workers,
    )

    return create_app(services)
