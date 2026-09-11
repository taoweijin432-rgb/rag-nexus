"""Resume 单人物与全量批次处理入口。

本模块复用 :class:`expert_index.pipeline.ExpertIndexPipeline` 的批次、重试、Contract 和
向量同步编排，只替换四个明确的数据集边界：Resume Source、人物身份、物理扫描游标和
``RESUME_SPEC``。公共下游首版仍以 ``expert_id`` 承载 ``resume_id``，但本模块公开返回
值只使用 Resume 语义名称。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from expert_index.models import ExpertAggregate
from expert_index.pipeline import (
    BatchBuildResult,
    BuildResult,
    ExpertIndexPipeline,
    VectorSyncSummary,
)
from expert_index.ports import ExpertVectorSyncPort
from expert_index.repositories import (
    MySQLBatchItemRepository,
    MySQLBatchRepository,
    MySQLBuildLockRepository,
)

from .config import ResumeSettings
from .dataset import RESUME_SPEC
from .source import MySQLResumeSourceAdapter, ResumeCandidate


@dataclass(frozen=True)
class ResumeVectorSyncSummary:
    """当前命令完成的 Resume 人物向量同步安全计数。"""

    resume_count: int
    current_count: int
    eligible_count: int
    embedded_count: int
    metadata_updated_count: int
    unchanged_count: int
    deleted_count: int
    delete_not_found_count: int

    @classmethod
    def from_internal(cls, summary: VectorSyncSummary) -> ResumeVectorSyncSummary:
        """把公共 Pipeline 的内部 Expert 兼容计数转换为 Resume 语义。"""

        return cls(
            resume_count=summary.expert_count,
            current_count=summary.current_count,
            eligible_count=summary.eligible_count,
            embedded_count=summary.embedded_count,
            metadata_updated_count=summary.metadata_updated_count,
            unchanged_count=summary.unchanged_count,
            deleted_count=summary.deleted_count,
            delete_not_found_count=summary.delete_not_found_count,
        )


@dataclass(frozen=True)
class ResumeBuildResult:
    """单人物构建结果；不包含来源正文或 Chunk Text。"""

    batch_id: int
    batch_no: str
    resume_id: int
    extracted_count: int
    stage_count: int
    chunk_count: int
    partial_chunk_count: int
    vector_sync: ResumeVectorSyncSummary

    @classmethod
    def from_internal(cls, result: BuildResult) -> ResumeBuildResult:
        """把内部兼容结果中的 ``expert_id`` 恢复为公开 ``resume_id``。"""

        return cls(
            batch_id=result.batch_id,
            batch_no=result.batch_no,
            resume_id=result.expert_id,
            extracted_count=result.extracted_count,
            stage_count=result.stage_count,
            chunk_count=result.chunk_count,
            partial_chunk_count=result.partial_chunk_count,
            vector_sync=ResumeVectorSyncSummary.from_internal(result.vector_sync),
        )


@dataclass(frozen=True)
class ResumeBatchBuildResult:
    """Resume 全量、续跑或失败重试的安全计数结果。

    ``upper_bound_source_record_id`` 和 ``cursor_source_record_id`` 均表示
    ``resume.id`` 物理游标，不是 ``resume_id``。
    """

    batch_id: int
    batch_no: str
    status: str
    upper_bound_source_record_id: int
    cursor_source_record_id: int
    candidate_count: int
    success_count: int
    failed_count: int
    rejected_count: int
    deactivated_count: int
    extracted_count: int
    stage_count: int
    chunk_count: int
    vector_sync: ResumeVectorSyncSummary

    @classmethod
    def from_internal(cls, result: BatchBuildResult) -> ResumeBatchBuildResult:
        """把公共批次结果中的物理 ``*_id`` 字段转换为明确 Resume 名称。"""

        return cls(
            batch_id=result.batch_id,
            batch_no=result.batch_no,
            status=result.status,
            upper_bound_source_record_id=result.upper_bound_id,
            cursor_source_record_id=result.cursor_id,
            candidate_count=result.candidate_count,
            success_count=result.success_count,
            failed_count=result.failed_count,
            rejected_count=result.rejected_count,
            deactivated_count=result.deactivated_count,
            extracted_count=result.extracted_count,
            stage_count=result.stage_count,
            chunk_count=result.chunk_count,
            vector_sync=ResumeVectorSyncSummary.from_internal(result.vector_sync),
        )


class _ResumePipelineCore(ExpertIndexPipeline):
    """把公共全量编排的 Source 钩子绑定到 Resume 语义。"""

    def _resume_source(self) -> MySQLResumeSourceAdapter:
        return cast(MySQLResumeSourceAdapter, self.source)

    def _get_full_scan_upper_bound(self, max_people: int | None) -> int:
        return self._resume_source().get_full_scan_upper_bound(max_resumes=max_people)

    def _load_full_scan_candidates(
        self,
        *,
        after_cursor_id: int,
        upper_bound_cursor_id: int,
        limit: int,
    ) -> list[object]:
        return list(
            self._resume_source().load_full_scan_candidates(
                after_source_record_id=after_cursor_id,
                upper_bound_source_record_id=upper_bound_cursor_id,
                limit=limit,
            )
        )

    def _load_person_candidate(self, person_id: int) -> ResumeCandidate:
        return self._resume_source().load_resume_candidate(person_id)

    def _single_person_is_deleted(self, person_id: int) -> bool:
        return self._resume_source().load_resume_candidate(person_id).is_deleted

    def _load_person_aggregate(self, person_id: int) -> ExpertAggregate:
        return self._resume_source().load_resume(person_id)

    @staticmethod
    def _candidate_person_id(candidate: object) -> int:
        return cast(ResumeCandidate, candidate).resume_id

    @staticmethod
    def _candidate_cursor_id(candidate: object) -> int:
        return cast(ResumeCandidate, candidate).source_record_id

    def _validate_candidate_page(
        self,
        candidates: list[object],
        *,
        after_id: int,
        upper_bound_id: int,
    ) -> None:
        """按 ``resume.id`` 校验候选页物理游标。"""

        cursor_ids = [self._candidate_cursor_id(candidate) for candidate in candidates]
        if cursor_ids != sorted(set(cursor_ids)):
            raise ValueError("Resume 候选页必须按唯一物理游标严格升序排列")
        if cursor_ids[0] <= after_id or cursor_ids[-1] > upper_bound_id:
            raise ValueError("Resume 候选页超出当前全量物理游标范围")


class ResumeIndexPipeline:
    """公开的 Resume 单人物、全量、续跑和失败重试入口。

    构造时再次执行 Resume 隔离门禁，并把状态 Repository 固定到
    ``RESUME_INDEX_DB_*``。调用方不能从本类进入 Expert CDC 或 Expert CLI。
    """

    def __init__(
        self,
        settings: ResumeSettings,
        *,
        source: MySQLResumeSourceAdapter | None = None,
        build_lock_repository: MySQLBuildLockRepository | None = None,
        batch_repository: MySQLBatchRepository | None = None,
        batch_item_repository: MySQLBatchItemRepository | None = None,
        vector_sync: ExpertVectorSyncPort,
    ) -> None:
        if not isinstance(settings, ResumeSettings):
            raise TypeError("settings 必须是 ResumeSettings")
        settings.validate_isolation_boundaries()

        # 公共加工规则只需要 Source/Index、时区和版本常量；这里显式从 Resume 配置
        # 构造，不读取或回退到任何 Expert 数据库环境变量。
        processing_settings = settings.to_processing_settings()
        self.settings = settings
        self._core = _ResumePipelineCore(
            processing_settings,
            source=source or MySQLResumeSourceAdapter(settings.source_db),
            build_lock_repository=build_lock_repository,
            batch_repository=batch_repository,
            batch_item_repository=batch_item_repository,
            vector_sync=vector_sync,
            dataset_spec=RESUME_SPEC,
        )

    def build_single_resume(
        self,
        resume_id: int,
        *,
        data_version: str,
        batch_no: str | None = None,
    ) -> ResumeBuildResult:
        """完整重建一位 Resume 人物，并同步其当前向量快照。"""

        result = self._core.build_single_expert(
            resume_id,
            data_version=data_version,
            batch_no=batch_no,
        )
        return ResumeBuildResult.from_internal(result)

    def build_resume_batch(
        self,
        *,
        data_version: str,
        page_size: int = 200,
        max_resumes: int | None = None,
        batch_no: str | None = None,
        worker_count: int = 1,
    ) -> ResumeBatchBuildResult:
        """按 ``resume.id`` 固定上界运行全量或限定数量批次。

        ``worker_count`` 只并发执行来源读取、标准化和 Chunk/Contract 准备；向量批量
        写入、明细终态和页级 Checkpoint 仍顺序提交，保证原批次可安全断点续跑。
        """

        result = self._core.build_expert_batch(
            data_version=data_version,
            page_size=page_size,
            max_experts=max_resumes,
            batch_no=batch_no,
            worker_count=worker_count,
        )
        return ResumeBatchBuildResult.from_internal(result)

    def resume_batch(
        self,
        batch_id: int,
        *,
        page_size: int = 200,
        worker_count: int = 1,
    ) -> ResumeBatchBuildResult:
        """从原批次最后可靠的 ``resume.id`` 页边界继续执行。

        并发度只影响本次续跑进程，不改变原批次的数据版本、固定上界、累计统计或已经
        提交的物理游标，因此旧单线程批次可直接切换到多线程续跑。
        """

        return ResumeBatchBuildResult.from_internal(
            self._core.resume_expert_batch(
                batch_id,
                page_size=page_size,
                worker_count=worker_count,
            )
        )

    def retry_failed_resumes(
        self,
        batch_id: int,
        *,
        page_size: int = 200,
    ) -> ResumeBatchBuildResult:
        """只重试一个 partial 全量批次中的失败 Resume 人物。"""

        return ResumeBatchBuildResult.from_internal(
            self._core.retry_failed_batch_experts(batch_id, page_size=page_size)
        )


__all__ = [
    "ResumeBatchBuildResult",
    "ResumeBuildResult",
    "ResumeIndexPipeline",
    "ResumeVectorSyncSummary",
]
