"""Resume 普通 Poll 增量运行时组装。

本模块只把 Resume 专用配置和 Source Adapter 注入现有增量编排，不复制任务状态机、
Scope Handler、Hash 差分、Embedding、Milvus 收敛或 Checkpoint 规则。Resume 首版仅
支持 Poll，不构造或暴露 Expert CDC Delete Processor。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from expert_index.incremental_runtime import IncrementalOnceResult, IncrementalRuntime
from expert_index.scanner import IncrementalScanDisabledError
from rag_vector_store import MilvusSettings, VectorIngestionService

from .config import ResumeSettings
from .dataset import RESUME_SPEC
from .preflight import ResumeSourcePreflightResult, run_resume_source_preflight
from .runtime import resolve_resume_vector_service
from .source import MySQLResumeSourceAdapter


@dataclass(frozen=True)
class ResumeIncrementalRuntime:
    """已通过 Resume 来源预检和写入隔离门禁的 Poll 运行时。"""

    runtime: IncrementalRuntime
    preflight: ResumeSourcePreflightResult

    def run_once(self, worker_id: str, *, cycle_id: str) -> IncrementalOnceResult:
        """执行一轮四表 Resume Poll，并返回不含正文的安全计数结果。"""

        return self.runtime.run_once(worker_id, cycle_id=cycle_id)


def build_resume_incremental_runtime(
    settings: ResumeSettings,
    *,
    max_attempts: int,
    retry_delay_seconds: int,
    task_lease_seconds: int,
    source: MySQLResumeSourceAdapter | None = None,
    vector_service: VectorIngestionService | None = None,
    base_milvus_settings: MilvusSettings | None = None,
    preflight_runner: Callable[[ResumeSettings], ResumeSourcePreflightResult] = (
        run_resume_source_preflight
    ),
) -> ResumeIncrementalRuntime:
    """按 fail-closed 顺序组装 Resume Poll 全链路。

    Parameters
    ----------
    settings:
        Resume 专用 Source、Index、Collection 以及 ``RESUME_INCREMENTAL_*`` 设置。
    max_attempts, retry_delay_seconds, task_lease_seconds:
        任务自动重试上限、重试冷却秒数和任务租约秒数；均由 CLI 显式提供。
    source:
        可选 Resume Source Adapter。生产默认创建只读 Adapter，测试可注入 Fake。
    vector_service, base_milvus_settings:
        可选公共向量服务或 Milvus 连接参数。Collection 始终强制绑定 Resume 配置。
    preflight_runner:
        只读四表结构预检；必须先于 Embedding/Milvus 服务构造完成。
    """

    if not isinstance(settings, ResumeSettings):
        raise TypeError("settings 必须是 ResumeSettings")
    settings.validate_isolation_boundaries()
    if not settings.incremental_scan.enabled:
        raise IncrementalScanDisabledError(
            "RESUME_INCREMENTAL_ENABLED 未开启，Resume Poll 入口保持禁用"
        )

    # 先确认来源结构，再组装可能持有外部客户端的向量服务；两步均不执行写入。
    preflight = preflight_runner(settings)
    resolved_vector_service = resolve_resume_vector_service(
        settings,
        vector_service=vector_service,
        base_milvus_settings=base_milvus_settings,
    )
    resolved_source = source or MySQLResumeSourceAdapter(settings.source_db)

    # 公共增量入口默认仍是 Expert；Resume 必须显式提供规格、Source 和关闭 CDC。
    from expert_index.incremental_runtime import build_incremental_runtime

    runtime = build_incremental_runtime(
        settings.to_processing_settings(),
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        task_lease_seconds=task_lease_seconds,
        vector_service=resolved_vector_service,
        dataset_spec=RESUME_SPEC,
        source=resolved_source,
        include_cdc_delete_processor=False,
    )
    return ResumeIncrementalRuntime(runtime=runtime, preflight=preflight)


__all__ = ["ResumeIncrementalRuntime", "build_resume_incremental_runtime"]
