"""全量批次的新建、断点续跑、分页执行和安全收口。"""

from __future__ import annotations

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
        with self.build_lock_repository.build_lock("full"):
            return self._build_expert_batch_locked(
                data_version=data_version,
                page_size=page_size,
                max_experts=max_experts,
                batch_no=batch_no,
            )

    def resume_expert_batch(
        self,
        batch_id: int,
        *,
        page_size: int = 200,
    ) -> BatchBuildResult:
        """从指定 full 批次最后完整提交页之后继续执行。

        方法在全局锁内恢复原批次的版本、固定上界、可靠游标和累计计数，再进入与新
        批次相同的分页流程。已提交完整页不会重复处理；未推进游标的中断页会整体重跑。

        Args:
            batch_id: 待续跑的 full 批次主键，必须是正整数。
            page_size: 本次续跑每页读取的最大候选数。

        Returns:
            原批次续跑后的状态、游标和累计计数。
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if not 1 <= page_size <= MAX_FULL_SCAN_PAGE_SIZE:
            raise ValueError(f"page_size 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间")
        with self.build_lock_repository.build_lock("full"):
            checkpoint = self.batch_repository.resume_batch(batch_id)
            return self._build_expert_batch_locked(
                data_version=checkpoint.data_version,
                page_size=page_size,
                max_experts=None,
                batch_no=checkpoint.batch_no,
                checkpoint=checkpoint,
            )

    def _build_expert_batch_locked(
        self,
        *,
        data_version: str,
        page_size: int,
        max_experts: int | None,
        batch_no: str | None,
        checkpoint: FullBatchCheckpoint | None = None,
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
                # 先幂等登记整页明细。登记失败时不开始处理专家，游标保持在上一完整页。
                self.batch_item_repository.prepare_batch_items(
                    batch_id,
                    person_ids,
                )

                # 阶段 1：逐专家读、生成、校验（CPU，不触发 Embedding），
                # 收集需要同步的专家及其 Contract 快照，领域异常隔离为 failed。
                # failed 专家不进入批量同步；deleted/rejected 用空快照进入，
                # 以删除该专家已存在但本次不再产出的向量。
                batch_inputs: list[ExpertBatchSyncInput] = []
                pending: list[
                    tuple[
                        object,
                        _ExpertProgress,
                        str,
                        str | None,
                        str | None,
                    ]
                ] = []
                for candidate in candidates:
                    person_id = self._candidate_person_id(candidate)
                    candidate_count += 1
                    progress = _ExpertProgress(expert_id=person_id)
                    self.batch_item_repository.mark_batch_item_running(
                        batch_id,
                        person_id,
                    )
                    terminal_status: str
                    result_code: str | None = None
                    error_type: str | None = None
                    try:
                        if candidate.is_deleted:
                            # 删除候选不读取正文，用空快照删除下游历史向量。
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
                            if terminal_status == "rejected":
                                result_code = "no_chunkable_content"
                        batch_inputs.append(
                            ExpertBatchSyncInput(
                                expert_id=person_id,
                                records=contracts,
                            )
                        )
                    # 只隔离单专家领域处理异常。批次明细或页游标写入失败会继续向外抛出，
                    # 终止整个批次，避免在审计状态未落库时跳过候选。
                    except Exception as exc:  # noqa: BLE001
                        terminal_status = "failed"
                        result_code = "expert_processing_failed"
                        error_type = type(exc).__name__
                    pending.append((candidate, progress, terminal_status, result_code, error_type))

                # 阶段 2：跨专家批量同步，合并 Embedding 与 Milvus 写入。
                # 任一专家同步失败会 fail-closed 中断整页，由外层 catch 收口为 failed。
                sync_results = self._sync_experts_batch(batch_inputs)

                # 落库阶段：逐专家回填同步结果、持久化终态、累计批次数量。
                for candidate, progress, terminal_status, result_code, error_type in pending:
                    person_id = self._candidate_person_id(candidate)
                    if terminal_status != "failed":
                        progress.vector_sync = sync_results.get(person_id)
                    # 先持久化专家终态，再把该专家计入批次累计数量。
                    self.batch_item_repository.finish_batch_item(
                        batch_id,
                        person_id,
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

                    if terminal_status == "success":
                        success_count += 1
                    elif terminal_status == "rejected":
                        rejected_count += 1
                    elif terminal_status == "deactivated":
                        deactivated_count += 1
                    else:
                        failed_count += 1
                        if len(error_summaries) < MAX_BATCH_ERROR_SUMMARIES:
                            error_summaries.append(
                                {
                                    f"{self.dataset_spec.name}_id": person_id,
                                    "error_type": error_type,
                                }
                            )

                    # 按已实际完成的阶段累计数量；failed 专家也可能已读取来源或生成 Chunk。
                    extracted_count += progress.extracted_count
                    chunk_count += progress.chunk_count

                # 整页明细全部到达终态后，才把游标和累计数量作为同一检查点持久化。
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
