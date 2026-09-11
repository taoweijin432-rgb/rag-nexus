"""批次失败人物的分页重试和重新汇总。"""

from __future__ import annotations

from ..repositories import FullBatchCheckpoint
from ..source import MAX_FULL_SCAN_PAGE_SIZE
from .helpers import (
    _build_batch_error_summary,
    _checkpoint_vector_sync,
    _vector_sync_persistence_fields,
)
from .results import BatchBuildResult, VectorSyncSummary
from .state import MAX_BATCH_ERROR_SUMMARIES, _ExpertProgress


class _RetryBuildMixin:
    """实现 partial 全量批次中的失败人物重试。

    使用方需要提供批次及明细 Repository、人物读取、人物处理和向量同步方法。
    重试完成后以全部批次明细为准重新计算批次终态。
    """

    def retry_failed_batch_experts(
        self,
        batch_id: int,
        *,
        page_size: int = 200,
    ) -> BatchBuildResult:
        """只重试一个 partial full 批次中的失败专家。

        方法复用原批次、数据版本和固定扫描范围，不重新扫描成功、拒绝或已停用专家。
        failed 明细以及上次重试异常退出遗留的 running 明细各尝试一次；结束后从全部
        批次明细重新聚合数量，全部成功则把批次改为 success，否则仍保持 partial。

        Args:
            batch_id: 需要重试失败明细的 full 批次主键。
            page_size: 每次加载的失败或遗留 running 明细数量上限。

        Returns:
            重新汇总后的原批次结果；向量同步摘要只统计本次重试。

        Raises:
            ValueError: 批次 ID 或分页大小不合法。
            Exception: 批次不是可重试状态，或读取、持久化、同步失败。
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= MAX_FULL_SCAN_PAGE_SIZE
        ):
            raise ValueError(f"page_size 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间")
        with self.build_lock_repository.build_lock("full"):
            checkpoint = self.batch_repository.load_partial_batch_for_retry(batch_id)
            return self._retry_failed_batch_experts_locked(
                checkpoint,
                page_size=page_size,
            )

    def _retry_failed_batch_experts_locked(
        self,
        checkpoint: FullBatchCheckpoint,
        *,
        page_size: int,
    ) -> BatchBuildResult:
        """在已取得全量锁后分页重试失败明细，并用明细汇总收口原批次。

        Args:
            checkpoint: 原 partial 批次的版本、扫描范围和累计统计快照。
            page_size: 单次读取可重试人物 ID 的数量上限。

        Returns:
            根据全部批次明细重新计算的批次结果。
        """

        after_expert_id = 0
        error_summaries: list[dict[str, object]] = []
        # 返回值只描述本次重试；持久化值从原批次累计统计继续增加。
        vector_sync = VectorSyncSummary()
        persisted_vector_sync = _checkpoint_vector_sync(checkpoint)
        while True:
            expert_ids = self.batch_item_repository.load_retryable_batch_item_ids(
                checkpoint.batch_id,
                after_expert_id=after_expert_id,
                limit=page_size,
            )
            if not expert_ids:
                break

            for expert_id in expert_ids:
                progress = _ExpertProgress(expert_id=expert_id)
                self.batch_item_repository.mark_batch_item_running(
                    checkpoint.batch_id,
                    expert_id,
                )
                terminal_status: str
                result_code: str | None = None
                error_type: str | None = None
                try:
                    candidate = self._load_person_candidate(expert_id)
                    if candidate.is_deleted:
                        progress.vector_sync = self._sync_expert_vector_snapshot(
                            expert_id,
                            [],
                        )
                        terminal_status = "deactivated"
                        result_code = "expert_deleted"
                    else:
                        terminal_status = self._process_active_expert(
                            expert_id,
                            data_version=checkpoint.data_version,
                            batch_id=checkpoint.batch_id,
                            progress=progress,
                        )
                        if terminal_status == "rejected":
                            result_code = "no_chunkable_content"
                except Exception as exc:  # noqa: BLE001
                    terminal_status = "failed"
                    result_code = "expert_processing_failed"
                    error_type = type(exc).__name__

                self.batch_item_repository.finish_batch_item(
                    checkpoint.batch_id,
                    expert_id,
                    status=terminal_status,
                    extracted_count=progress.extracted_count,
                    stage_count=0,
                    chunk_count=progress.chunk_count,
                    result_code=result_code,
                    error_type=error_type,
                )
                if progress.vector_sync is not None:
                    vector_sync = vector_sync.add(progress.vector_sync)
                    persisted_vector_sync = persisted_vector_sync.add(progress.vector_sync)
                if terminal_status == "failed" and (
                    len(error_summaries) < MAX_BATCH_ERROR_SUMMARIES
                ):
                    error_summaries.append(
                        {
                            f"{self.dataset_spec.name}_id": expert_id,
                            "error_type": error_type,
                        }
                    )

            after_expert_id = expert_ids[-1]

        summary = self.batch_item_repository.summarize_batch_items(checkpoint.batch_id)
        unfinished_count = summary.pending_count + summary.running_count
        final_status = "partial" if summary.failed_count or unfinished_count else "success"
        error_summary = _build_batch_error_summary(
            summary.failed_count,
            error_summaries,
        )
        if error_summary is None and unfinished_count:
            error_summary = {"unfinished_item_count": unfinished_count}

        self.batch_repository.finish_batch(
            checkpoint.batch_id,
            status=final_status,
            extracted_count=summary.extracted_count,
            stage_count=summary.stage_count,
            chunk_count=summary.chunk_count,
            rejected_count=summary.rejected_count,
            cursor_id=checkpoint.upper_bound_id,
            candidate_count=summary.candidate_count,
            success_count=summary.success_count,
            failed_count=summary.failed_count,
            deactivated_count=summary.deactivated_count,
            error_summary=error_summary,
            **_vector_sync_persistence_fields(persisted_vector_sync),
        )
        return BatchBuildResult(
            batch_id=checkpoint.batch_id,
            batch_no=checkpoint.batch_no,
            status=final_status,
            upper_bound_id=checkpoint.upper_bound_id,
            cursor_id=checkpoint.upper_bound_id,
            candidate_count=summary.candidate_count,
            success_count=summary.success_count,
            failed_count=summary.failed_count,
            rejected_count=summary.rejected_count,
            deactivated_count=summary.deactivated_count,
            extracted_count=summary.extracted_count,
            stage_count=summary.stage_count,
            chunk_count=summary.chunk_count,
            vector_sync=vector_sync,
        )
