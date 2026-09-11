"""全量批次的新建、断点续跑、分页执行和安全收口。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial

from ..ports import ExpertBatchSyncInput
from ..repositories import FullBatchCheckpoint
from ..source import MAX_FULL_SCAN_PAGE_SIZE
from .helpers import (
    _build_batch_error_summary,
    _checkpoint_vector_sync,
    _generate_full_batch_number,
    _validate_data_version,
    _vector_sync_persistence_fields,
)
from .results import BatchBuildResult, VectorSyncSummary
from .state import MAX_BATCH_ERROR_SUMMARIES, _ExpertProgress

# 全量并发只用于来源读取、标准化和 Chunk/Contract 准备。数据库终态、向量批量同步和
# 页级 Checkpoint 仍由协调线程顺序提交，避免并发写游标破坏既有断点恢复语义。
DEFAULT_FULL_BUILD_WORKER_COUNT = 1
MAX_FULL_BUILD_WORKER_COUNT = 16


@dataclass(frozen=True)
class _PreparedFullCandidate:
    """一位候选在线程内完成领域准备后的安全结果。

    ``sync_input`` 只在人物可以进入向量同步时存在；确定性或临时领域异常会被隔离为
    ``failed``，异常正文不会跨线程保存或写入 Index 数据库。
    """

    candidate: object
    progress: _ExpertProgress
    terminal_status: str
    result_code: str | None
    error_type: str | None
    sync_input: ExpertBatchSyncInput | None


class _BatchBuildMixin:
    """实现固定扫描上界的全量构建与断点续跑。

    使用方需要提供 Source 分页方法、批次及明细 Repository、批量 Contract 构建和
    向量同步方法。只有整页明细全部落到终态后才推进持久化游标。
    """

    def build_expert_batch(
        self,
        *,
        data_version: str,
        page_size: int = 200,
        max_experts: int | None = None,
        batch_no: str | None = None,
        worker_count: int = DEFAULT_FULL_BUILD_WORKER_COUNT,
    ) -> BatchBuildResult:
        """在全局加工锁内运行全量或限定专家数量的批量任务。

        ``max_experts`` 会在任务开始时把固定扫描上界收缩到前 N 位候选的最后一个 ID，
        因此限制作用于整个任务范围，而不是在分页循环中途停止。全局锁覆盖上界计算、
        分页处理和批次收口，防止其他构建任务在此期间并发执行。

        Args:
            data_version: 整个批次共用的数据版本，必须是非空字符串。
            page_size: 每次从来源表读取的最大候选数，不能超过来源适配器上限。
            max_experts: 可选的任务总人数上限，用于计算固定扫描上界。
            batch_no: 可选业务批次号；省略时根据当前时间生成。
            worker_count: 页内领域准备线程数。默认 1 保持现有 Expert 行为；大于 1 时
                只并行来源读取、标准化和 Chunk/Contract 构建。

        Returns:
            固定扫描范围内的批次状态、可靠游标和累计计数。

        Raises:
            ValueError: 数据版本、分页大小或人数上限不合法。
            Exception: 构建锁、来源读取、持久化或向量同步失败。
        """

        _validate_data_version(data_version)
        if not 1 <= page_size <= MAX_FULL_SCAN_PAGE_SIZE:
            raise ValueError(f"page_size 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间")
        if max_experts is not None and (
            isinstance(max_experts, bool) or not isinstance(max_experts, int) or max_experts < 1
        ):
            raise ValueError("max_experts 必须是正整数或 None")
        _validate_full_build_worker_count(worker_count)
        with self.build_lock_repository.build_lock("full"):
            return self._build_expert_batch_locked(
                data_version=data_version,
                page_size=page_size,
                max_experts=max_experts,
                batch_no=batch_no,
                worker_count=worker_count,
            )

    def resume_expert_batch(
        self,
        batch_id: int,
        *,
        page_size: int = 200,
        worker_count: int = DEFAULT_FULL_BUILD_WORKER_COUNT,
    ) -> BatchBuildResult:
        """从指定 full 批次最后完整提交页之后继续执行。

        方法在全局锁内恢复原批次的版本、固定上界、可靠游标和累计计数，再进入与新
        批次相同的分页流程。已提交完整页不会重复处理；未推进游标的中断页会整体重跑。

        Args:
            batch_id: 待续跑的 full 批次主键，必须是正整数。
            page_size: 本次续跑每页读取的最大候选数。
            worker_count: 页内领域准备线程数；不改变原批次号、固定上界或可靠游标。

        Returns:
            原批次续跑后的状态、游标和累计计数。
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if not 1 <= page_size <= MAX_FULL_SCAN_PAGE_SIZE:
            raise ValueError(f"page_size 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间")
        _validate_full_build_worker_count(worker_count)
        with self.build_lock_repository.build_lock("full"):
            checkpoint = self.batch_repository.resume_batch(batch_id)
            return self._build_expert_batch_locked(
                data_version=checkpoint.data_version,
                page_size=page_size,
                max_experts=None,
                batch_no=checkpoint.batch_no,
                checkpoint=checkpoint,
                worker_count=worker_count,
            )

    def _build_expert_batch_locked(
        self,
        *,
        data_version: str,
        page_size: int,
        max_experts: int | None,
        batch_no: str | None,
        checkpoint: FullBatchCheckpoint | None = None,
        worker_count: int = DEFAULT_FULL_BUILD_WORKER_COUNT,
    ) -> BatchBuildResult:
        """在已取得互斥锁的前提下执行固定范围分页构建。

        执行顺序：

        1. 新任务从 Source 固定 ``upper_bound_id`` 并创建批次；续跑任务恢复原检查点；
        2. 从新任务的 0 或续跑批次的可靠 ``cursor_id`` 开始 Keyset Pagination；
        3. 每页先登记 pending，再按 running→终态记录每位候选；
        4. active 专家复用 ``_process_active_expert``，逻辑删除和无内容分别进入
           deactivated/rejected，意外异常才计入 failed；
        5. 每页全部明细可靠落库后，才持久化游标和累计计数；
        6. 逻辑删除、无内容或陈旧来源导致 Chunk 停用时同步重算 metadata_hash；
        7. 只要存在专家级 failed，最终批次为 partial，否则为 success；
        8. 分页或批次记录错误会终止任务并将批次记为 failed。

        返回值反映固定范围内的最终游标和累计结果。续跑不创建新批次，也不会重新计算
        或扩大原批次扫描上界。

        Args:
            data_version: 标准化和 Contract 构建使用的数据版本。
            page_size: 单页候选数量上限。
            max_experts: 新批次的可选人数上限；续跑时必须为 None。
            batch_no: 新批次使用或续跑批次恢复的业务批次号。
            checkpoint: 续跑检查点；为 None 时创建新批次并固定扫描上界。
            worker_count: 当前命令允许的页内领域准备线程数。

        Returns:
            本次新建或续跑后的批量构建结果。
        """

        if checkpoint is None:
            # 创建批次前固定扫描上界，使任务运行期间新增的专家不进入本次范围。
            upper_bound_id = self._get_full_scan_upper_bound(max_experts)
            actual_batch_no = batch_no or _generate_full_batch_number()
            batch_id = self.batch_repository.start_batch(
                actual_batch_no,
                "full",
                data_version=data_version,
                upper_bound_id=upper_bound_id,
            )
            cursor_id = 0
            candidate_count = success_count = failed_count = 0
            rejected_count = deactivated_count = 0
            extracted_count = stage_count = chunk_count = 0
        else:
            # 从持久化检查点恢复原批次范围和完整页计数，保持续跑前后统计连续。
            batch_id = checkpoint.batch_id
            actual_batch_no = checkpoint.batch_no
            upper_bound_id = checkpoint.upper_bound_id
            cursor_id = checkpoint.cursor_id
            candidate_count = checkpoint.candidate_count
            success_count = checkpoint.success_count
            failed_count = checkpoint.failed_count
            rejected_count = checkpoint.rejected_count
            deactivated_count = checkpoint.deactivated_count
            extracted_count = checkpoint.extracted_count
            stage_count = checkpoint.stage_count
            chunk_count = checkpoint.chunk_count
        error_summaries: list[dict[str, object]] = []
        # 续跑也只汇总当前命令新完成的同步，不混入检查点之前的历史处理。
        vector_sync = VectorSyncSummary()
        persisted_vector_sync = (
            VectorSyncSummary() if checkpoint is None else _checkpoint_vector_sync(checkpoint)
        )
        reliable_persisted_vector_sync = persisted_vector_sync
        # 该快照只在完整页进度成功落库后更新。批次级错误使用它收口，确保游标与累计
        # 数量属于同一个页边界，续跑时不会重复累计中断页。
        reliable_progress = {
            "cursor_id": cursor_id,
            "candidate_count": candidate_count,
            "success_count": success_count,
            "failed_count": failed_count,
            "rejected_count": rejected_count,
            "deactivated_count": deactivated_count,
            "extracted_count": extracted_count,
            "stage_count": stage_count,
            "chunk_count": chunk_count,
        }

        try:
            # 线程池在整个命令期间复用，避免每一页重复创建线程。worker_count=1 时不
            # 创建线程，确保 Expert 现有默认执行顺序、异常边界和资源占用完全不变。
            with ExitStack() as stack:
                executor = (
                    None
                    if worker_count == 1
                    else stack.enter_context(
                        ThreadPoolExecutor(
                            max_workers=worker_count,
                            thread_name_prefix=f"{self.dataset_spec.name}-full",
                        )
                    )
                )
                while cursor_id < upper_bound_id:
                    candidates = self._load_full_scan_candidates(
                        after_cursor_id=cursor_id,
                        upper_bound_cursor_id=upper_bound_id,
                        limit=page_size,
                    )
                    if not candidates:
                        # 固定范围内没有后续候选时，把游标推进到上界并落库，表示扫描完成。
                        cursor_id = upper_bound_id
                        self.batch_repository.update_batch_progress(
                            batch_id,
                            cursor_id=cursor_id,
                            candidate_count=candidate_count,
                            success_count=success_count,
                            failed_count=failed_count,
                            rejected_count=rejected_count,
                            deactivated_count=deactivated_count,
                            extracted_count=extracted_count,
                            stage_count=stage_count,
                            chunk_count=chunk_count,
                            **_vector_sync_persistence_fields(persisted_vector_sync),
                        )
                        reliable_progress = {
                            "cursor_id": cursor_id,
                            "candidate_count": candidate_count,
                            "success_count": success_count,
                            "failed_count": failed_count,
                            "rejected_count": rejected_count,
                            "deactivated_count": deactivated_count,
                            "extracted_count": extracted_count,
                            "stage_count": stage_count,
                            "chunk_count": chunk_count,
                        }
                        reliable_persisted_vector_sync = persisted_vector_sync
                        break
                    self._validate_candidate_page(
                        candidates,
                        after_id=cursor_id,
                        upper_bound_id=upper_bound_id,
                    )
                    person_ids = [self._candidate_person_id(candidate) for candidate in candidates]
                    # 先幂等登记整页明细。登记失败时不开始处理人物，游标保持在上一完整页。
                    self.batch_item_repository.prepare_batch_items(
                        batch_id,
                        person_ids,
                    )

                    # running 状态仍由协调线程按候选顺序写入；线程只访问各自独立的来源
                    # 连接和内存对象，不并发修改 Batch/Item/Checkpoint Repository。
                    for person_id in person_ids:
                        self.batch_item_repository.mark_batch_item_running(
                            batch_id,
                            person_id,
                        )
                    candidate_count += len(candidates)

                    prepared = self._prepare_full_candidate_page(
                        candidates,
                        data_version=data_version,
                        batch_id=batch_id,
                        executor=executor,
                    )
                    batch_inputs = [
                        item.sync_input for item in prepared if item.sync_input is not None
                    ]

                    # 向量同步保持单一批量提交边界。其内部已有独立的 Embedding 微批
                    # 并发；这里不让多个线程共享 Milvus Client 或写同一页状态。
                    sync_results = self._sync_experts_batch(batch_inputs)

                    # executor.map 保持输入顺序，因而终态、累计计数和可靠游标继续按照
                    # 来源物理游标顺序提交，与旧单线程批次兼容。
                    for item in prepared:
                        person_id = self._candidate_person_id(item.candidate)
                        if item.terminal_status != "failed":
                            item.progress.vector_sync = sync_results.get(person_id)
                        self.batch_item_repository.finish_batch_item(
                            batch_id,
                            person_id,
                            status=item.terminal_status,
                            extracted_count=item.progress.extracted_count,
                            stage_count=0,
                            chunk_count=item.progress.chunk_count,
                            result_code=item.result_code,
                            error_type=item.error_type,
                        )
                        if item.progress.vector_sync is not None:
                            vector_sync = vector_sync.add(item.progress.vector_sync)
                            persisted_vector_sync = persisted_vector_sync.add(
                                item.progress.vector_sync
                            )

                        if item.terminal_status == "success":
                            success_count += 1
                        elif item.terminal_status == "rejected":
                            rejected_count += 1
                        elif item.terminal_status == "deactivated":
                            deactivated_count += 1
                        else:
                            failed_count += 1
                            if len(error_summaries) < MAX_BATCH_ERROR_SUMMARIES:
                                error_summaries.append(
                                    {
                                        f"{self.dataset_spec.name}_id": person_id,
                                        "error_type": item.error_type,
                                    }
                                )

                        extracted_count += item.progress.extracted_count
                        chunk_count += item.progress.chunk_count

                    # 只有整页领域准备、向量同步和明细终态全部成功落库后才推进游标。
                    # 任一线程或基础设施异常都会保留上一页快照，续跑时安全重做当前页。
                    cursor_id = self._candidate_cursor_id(candidates[-1])
                    self.batch_repository.update_batch_progress(
                        batch_id,
                        cursor_id=cursor_id,
                        candidate_count=candidate_count,
                        success_count=success_count,
                        failed_count=failed_count,
                        rejected_count=rejected_count,
                        deactivated_count=deactivated_count,
                        extracted_count=extracted_count,
                        stage_count=stage_count,
                        chunk_count=chunk_count,
                        **_vector_sync_persistence_fields(persisted_vector_sync),
                    )
                    reliable_progress = {
                        "cursor_id": cursor_id,
                        "candidate_count": candidate_count,
                        "success_count": success_count,
                        "failed_count": failed_count,
                        "rejected_count": rejected_count,
                        "deactivated_count": deactivated_count,
                        "extracted_count": extracted_count,
                        "stage_count": stage_count,
                        "chunk_count": chunk_count,
                    }
                    reliable_persisted_vector_sync = persisted_vector_sync

        except Exception as exc:
            # 批次级错误无法安全跳过。使用最后完整页快照结束批次，使中断页可以重跑。
            self.batch_repository.finish_batch(
                batch_id,
                status="failed",
                extracted_count=reliable_progress["extracted_count"],
                stage_count=reliable_progress["stage_count"],
                chunk_count=reliable_progress["chunk_count"],
                rejected_count=reliable_progress["rejected_count"],
                cursor_id=reliable_progress["cursor_id"],
                candidate_count=reliable_progress["candidate_count"],
                success_count=reliable_progress["success_count"],
                failed_count=reliable_progress["failed_count"],
                deactivated_count=reliable_progress["deactivated_count"],
                error_summary={"error_type": type(exc).__name__},
                **_vector_sync_persistence_fields(reliable_persisted_vector_sync),
            )
            raise

        # 最终状态写回失败时直接抛出，不重复调用同一方法，以免掩盖数据库异常。
        final_status = "partial" if failed_count else "success"
        self.batch_repository.finish_batch(
            batch_id,
            status=final_status,
            extracted_count=extracted_count,
            stage_count=stage_count,
            chunk_count=chunk_count,
            rejected_count=rejected_count,
            cursor_id=cursor_id,
            candidate_count=candidate_count,
            success_count=success_count,
            failed_count=failed_count,
            deactivated_count=deactivated_count,
            error_summary=_build_batch_error_summary(
                failed_count,
                error_summaries,
            ),
            **_vector_sync_persistence_fields(persisted_vector_sync),
        )

        return BatchBuildResult(
            batch_id=batch_id,
            batch_no=actual_batch_no,
            status=final_status,
            upper_bound_id=upper_bound_id,
            cursor_id=cursor_id,
            candidate_count=candidate_count,
            success_count=success_count,
            failed_count=failed_count,
            rejected_count=rejected_count,
            deactivated_count=deactivated_count,
            extracted_count=extracted_count,
            stage_count=stage_count,
            chunk_count=chunk_count,
            vector_sync=vector_sync,
        )

    def _prepare_full_candidate_page(
        self,
        candidates: list[object],
        *,
        data_version: str,
        batch_id: int,
        executor: ThreadPoolExecutor | None,
    ) -> list[_PreparedFullCandidate]:
        """顺序或并发准备一页人物，并始终按候选物理游标顺序返回结果。"""

        prepare = partial(
            self._prepare_full_candidate,
            data_version=data_version,
            batch_id=batch_id,
        )
        if executor is None:
            return [prepare(candidate) for candidate in candidates]
        # map 会并发执行但按输入顺序产出；有利于保持状态审计和既有测试的稳定顺序。
        return list(executor.map(prepare, candidates))

    def _prepare_full_candidate(
        self,
        candidate: object,
        *,
        data_version: str,
        batch_id: int,
    ) -> _PreparedFullCandidate:
        """在线程内读取并构建一位人物的 Contract，不执行任何状态库或向量写入。"""

        person_id = self._candidate_person_id(candidate)
        progress = _ExpertProgress(expert_id=person_id)
        try:
            if candidate.is_deleted:
                terminal_status = "deactivated"
                result_code = "expert_deleted"
                contracts = self._contracts_for_expert([])
            else:
                terminal_status, contracts = self._build_expert_contracts_for_batch(
                    person_id,
                    data_version=data_version,
                    batch_id=batch_id,
                    progress=progress,
                )
                result_code = "no_chunkable_content" if terminal_status == "rejected" else None
            sync_input = ExpertBatchSyncInput(
                expert_id=person_id,
                records=contracts,
            )
            error_type = None
        # 人物领域异常仍按既有语义隔离，不让一个坏 Resume 取消同页其他线程。
        except Exception as exc:  # noqa: BLE001
            terminal_status = "failed"
            result_code = "expert_processing_failed"
            error_type = type(exc).__name__
            sync_input = None
        return _PreparedFullCandidate(
            candidate=candidate,
            progress=progress,
            terminal_status=terminal_status,
            result_code=result_code,
            error_type=error_type,
            sync_input=sync_input,
        )


def _validate_full_build_worker_count(worker_count: int) -> None:
    """限制页内线程数量，避免误配置耗尽 Source MySQL 连接。"""

    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or not 1 <= worker_count <= MAX_FULL_BUILD_WORKER_COUNT
    ):
        raise ValueError(f"worker_count 必须位于 1 到 {MAX_FULL_BUILD_WORKER_COUNT} 之间")
