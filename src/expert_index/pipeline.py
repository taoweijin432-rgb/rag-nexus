"""专家索引加工流程的编排模块。

本文件负责把源数据读取、内存标准化、Chunk 构建、契约校验和向量同步
组织成可执行的单专家及批量任务。固定处理顺序为：

``Source → Normalize/Stage（内存） → Chunk → Validator → Vector Sync``

本文件只控制执行顺序和任务状态，不实现各阶段的领域规则：

* ``source.py``：扫描全量候选，并读取一位专家的四表一致性快照；
* ``normalize.py``：清洗和标准化来源字段，生成 Stage 对象；
* ``chunk_builder.py``：生成三类 Chunk Text 与扁平 Metadata；
* ``validators.py``：在向量同步前验证文本、ID、哈希、枚举和 PII；
* ``repository.py``：持久化批次、专家明细和分页检查点；
* ``vector_sync.py``：把内存中的 Chunk Contract 交给下游向量服务。

批量任务按固定 ID 上界分页，逐专家隔离领域处理失败，并只在完整一页处理结束后推进
持久化游标。这样可以从可靠页边界续跑；批次级错误则停止任务并保留最后完整页进度。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .chunk_builder import ChunkFactory
from .config import Settings
from .contract_bridge import chunk_records_to_contracts
from .models import ChunkRecord, ExpertCandidate
from .normalize import normalize_aggregate
from .ports import ExpertVectorSyncPort
from .repository import FullBatchCheckpoint, MySQLIndexRepository
from .source import MAX_FULL_SCAN_PAGE_SIZE, MySQLSourceAdapter
from .validators import validate_chunks

# 批次主表只保存有限数量的失败样本；每位专家的完整机器终态由批次明细表记录。
MAX_BATCH_ERROR_SUMMARIES = 20


@dataclass(frozen=True)
class BuildResult:
    """单专家任务完成后返回的批次标识和处理数量。

    该结果不包含源数据或 Chunk 正文；相同统计同时写入批次主表，便于按 ``batch_id``
    回查执行结果。
    """

    batch_id: int  # expert_index_batch.id，用于回查本次运行。
    batch_no: str  # 单专家业务批次号；调用方不传时自动生成。
    expert_id: int  # 本次重建的 experts.id。
    extracted_count: int  # 1 条主档加工作、项目、教育来源记录数。
    stage_count: int  # 不再持久化 Stage，因此新任务恒为 0。
    chunk_count: int  # 成功同步到下游的最终 Chunk 行数。
    partial_chunk_count: int  # 当前流程不生成 partial Chunk，因此恒为 0。


@dataclass(frozen=True)
class BatchBuildResult:
    """批量任务完成后返回的扫描范围、最终游标和累计数量。

    该对象由新批次和续跑入口共同返回，只包含任务状态和计数，不包含专家原始字段。

    ``rejected_count`` 表示存在但无可生成内容的专家数；``deactivated_count`` 表示
    主档已逻辑删除且历史 Chunk 已完成停用的专家数。两者都不是程序失败。
    """

    batch_id: int  # expert_index_batch.id。
    batch_no: str  # 全量业务批次号。
    status: str  # success/partial；批次级异常会抛出而不返回该对象。
    upper_bound_id: int  # 启动时固定的最大 experts.id。
    cursor_id: int  # 本轮内存中最后完成调度的专家 ID。
    candidate_count: int  # 实际遍历的专家候选数，包括逻辑删除候选。
    success_count: int  # 正常完成读取、生成、校验和向量同步的专家数。
    failed_count: int  # 本轮发生异常但未阻止后续处理的专家数。
    rejected_count: int  # 存在但没有可生成内容，并已停用旧 Chunk 的专家数。
    deactivated_count: int  # 主档逻辑删除，并已停用历史 Chunk 的专家数。
    extracted_count: int  # 所有已推进专家读取到的四表来源记录总数。
    stage_count: int  # 纯内存构建新增量为 0；续跑旧批次时保留其历史值。
    chunk_count: int  # 已生成的 Chunk 总数；同步失败专家可能已产生 Chunk。


@dataclass
class _ExpertProgress:
    """记录一位专家在当前调用中已完成阶段的数量。

    ``_process_active_expert`` 在来源读取和 Chunk 构建完成后更新对应字段，使外层在
    成功或异常时都能记录已经实际发生的数量。该对象只存在于内存。
    """

    expert_id: int
    extracted_count: int = 0
    chunk_count: int = 0


class ExpertIndexPipeline:
    """提供单专家构建、新建批量任务、断点续跑和失败重试入口。

    默认从配置创建 MySQL Source 和 Repository，也支持注入同接口实现。所有入口共享
    ``_process_active_expert``，确保单专家与批量任务采用相同的标准化、校验和同步规则。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        source: MySQLSourceAdapter | None = None,
        repository: MySQLIndexRepository | None = None,
        vector_sync: ExpertVectorSyncPort,
    ) -> None:
        self.settings = settings
        self.source = source or MySQLSourceAdapter(settings.source_db)
        self.repository = repository or MySQLIndexRepository(settings.index_db)
        self.vector_sync = vector_sync
        self.factory = ChunkFactory(settings)

    def build_single_expert(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_no: str | None = None,
    ) -> BuildResult:
        """在全局加工锁内创建 ``single`` 批次并完整重建一位专家。

        方法先校验专家 ID 和数据版本，再取得 single/full 共用的互斥锁。取得锁后创建
        批次并执行完整专家处理；已有任务持锁时立即失败，不会并发改写批次或向量。
        返回成功批次的数量摘要，处理异常由内部批次收口后继续向调用方抛出。
        """

        _validate_data_version(data_version)
        if expert_id <= 0:
            raise ValueError("expert_id 必须大于 0")
        with self.repository.build_lock("single"):
            return self._build_single_expert_locked(
                expert_id,
                data_version=data_version,
                batch_no=batch_no,
            )

    def _build_single_expert_locked(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_no: str | None,
    ) -> BuildResult:
        """创建并收口一个已经取得互斥锁的单专家批次。

        成功时把处理数量写入批次主表并返回 ``BuildResult``；异常时把已完成数量和安全
        错误摘要写入 failed 批次，再原样抛出异常。领域处理统一委托给
        ``_process_active_expert``。
        """

        actual_batch_no = batch_no or _generate_single_batch_number(expert_id)
        batch_id = self.repository.start_batch(actual_batch_no, "single")
        progress = _ExpertProgress(expert_id=expert_id)
        try:
            outcome = self._process_active_expert(
                expert_id,
                data_version=data_version,
                batch_id=batch_id,
                progress=progress,
            )
            # 当前只生成完整 Chunk；返回结构中的 partial 数量固定为 0。
            partial_count = 0
            self.repository.finish_batch(
                batch_id,
                status="success",
                extracted_count=progress.extracted_count,
                stage_count=0,
                chunk_count=progress.chunk_count,
                rejected_count=1 if outcome == "rejected" else 0,
            )
            return BuildResult(
                batch_id=batch_id,
                batch_no=actual_batch_no,
                expert_id=expert_id,
                extracted_count=progress.extracted_count,
                stage_count=0,
                chunk_count=progress.chunk_count,
                partial_chunk_count=partial_count,
            )
        except Exception as exc:
            # 批次错误摘要只记录异常类型和截断消息，不保存专家正文或 Chunk Text。
            self.repository.finish_batch(
                batch_id,
                status="failed",
                extracted_count=progress.extracted_count,
                stage_count=0,
                chunk_count=progress.chunk_count,
                rejected_count=1,
                error_summary={"error_type": type(exc).__name__, "message": str(exc)[:400]},
            )
            raise

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
        """

        _validate_data_version(data_version)
        if not 1 <= page_size <= MAX_FULL_SCAN_PAGE_SIZE:
            raise ValueError(
                f"page_size 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间"
            )
        if max_experts is not None and (
            isinstance(max_experts, bool)
            or not isinstance(max_experts, int)
            or max_experts < 1
        ):
            raise ValueError("max_experts 必须是正整数或 None")
        with self.repository.build_lock("full"):
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
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if not 1 <= page_size <= MAX_FULL_SCAN_PAGE_SIZE:
            raise ValueError(
                f"page_size 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间"
            )
        with self.repository.build_lock("full"):
            checkpoint = self.repository.resume_batch(batch_id)
            return self._build_expert_batch_locked(
                data_version=checkpoint.data_version,
                page_size=page_size,
                max_experts=None,
                batch_no=checkpoint.batch_no,
                checkpoint=checkpoint,
            )

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
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= MAX_FULL_SCAN_PAGE_SIZE
        ):
            raise ValueError(
                f"page_size 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间"
            )
        with self.repository.build_lock("full"):
            checkpoint = self.repository.load_partial_batch_for_retry(batch_id)
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
        """在已取得全量锁后分页重试失败明细，并用明细汇总收口原批次。"""

        after_expert_id = 0
        error_summaries: list[dict[str, object]] = []
        while True:
            expert_ids = self.repository.load_retryable_batch_item_ids(
                checkpoint.batch_id,
                after_expert_id=after_expert_id,
                limit=page_size,
            )
            if not expert_ids:
                break

            for expert_id in expert_ids:
                progress = _ExpertProgress(expert_id=expert_id)
                self.repository.mark_batch_item_running(checkpoint.batch_id, expert_id)
                terminal_status: str
                result_code: str | None = None
                error_type: str | None = None
                try:
                    candidate = self.source.load_expert_candidate(expert_id)
                    if candidate.is_deleted:
                        self._sync_expert_vector_snapshot(expert_id, [])
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

                self.repository.finish_batch_item(
                    checkpoint.batch_id,
                    expert_id,
                    status=terminal_status,
                    extracted_count=progress.extracted_count,
                    stage_count=0,
                    chunk_count=progress.chunk_count,
                    result_code=result_code,
                    error_type=error_type,
                )
                if terminal_status == "failed" and (
                    len(error_summaries) < MAX_BATCH_ERROR_SUMMARIES
                ):
                    error_summaries.append(
                        {"expert_id": expert_id, "error_type": error_type}
                    )

            after_expert_id = expert_ids[-1]

        summary = self.repository.summarize_batch_items(checkpoint.batch_id)
        unfinished_count = summary.pending_count + summary.running_count
        final_status = (
            "partial"
            if summary.failed_count or unfinished_count
            else "success"
        )
        error_summary = _build_batch_error_summary(
            summary.failed_count,
            error_summaries,
        )
        if error_summary is None and unfinished_count:
            error_summary = {"unfinished_item_count": unfinished_count}

        self.repository.finish_batch(
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
        """

        if checkpoint is None:
            # 创建批次前固定扫描上界，使任务运行期间新增的专家不进入本次范围。
            upper_bound_id = self.source.get_full_scan_upper_bound(
                max_experts=max_experts,
            )
            actual_batch_no = batch_no or _generate_full_batch_number()
            batch_id = self.repository.start_batch(
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
                candidates = self.source.load_full_scan_candidates(
                    after_id=cursor_id,
                    upper_bound_id=upper_bound_id,
                    limit=page_size,
                )
                if not candidates:
                    # 固定范围内没有后续候选时，把游标推进到上界并落库，表示扫描完成。
                    cursor_id = upper_bound_id
                    self.repository.update_batch_progress(
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
                    break
                _validate_expert_candidate_page(
                    candidates,
                    after_id=cursor_id,
                    upper_bound_id=upper_bound_id,
                )
                # 先幂等登记整页明细。登记失败时不开始处理专家，游标保持在上一完整页。
                self.repository.prepare_batch_items(
                    batch_id,
                    [candidate.id for candidate in candidates],
                )

                for candidate in candidates:
                    candidate_count += 1
                    progress = _ExpertProgress(expert_id=candidate.id)
                    self.repository.mark_batch_item_running(batch_id, candidate.id)
                    terminal_status: str
                    result_code: str | None = None
                    error_type: str | None = None
                    try:
                        if candidate.is_deleted:
                            # 删除候选不读取正文，直接用空快照删除下游历史向量。
                            self._sync_expert_vector_snapshot(candidate.id, [])
                            terminal_status = "deactivated"
                            result_code = "expert_deleted"
                        else:
                            terminal_status = self._process_active_expert(
                                candidate.id,
                                data_version=data_version,
                                batch_id=batch_id,
                                progress=progress,
                            )
                            if terminal_status == "rejected":
                                result_code = "no_chunkable_content"
                    # 只隔离单专家领域处理异常。批次明细或页游标写入失败会继续向外抛出，
                    # 终止整个批次，避免在审计状态未落库时跳过候选。
                    except Exception as exc:  # noqa: BLE001
                        terminal_status = "failed"
                        result_code = "expert_processing_failed"
                        error_type = type(exc).__name__

                    # 先持久化专家终态，再把该专家计入批次累计数量。
                    self.repository.finish_batch_item(
                        batch_id,
                        candidate.id,
                        status=terminal_status,
                        extracted_count=progress.extracted_count,
                        stage_count=0,
                        chunk_count=progress.chunk_count,
                        result_code=result_code,
                        error_type=error_type,
                    )

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
                                    "expert_id": candidate.id,
                                    "error_type": error_type,
                                }
                            )

                    # 按已实际完成的阶段累计数量；failed 专家也可能已读取来源或生成 Chunk。
                    extracted_count += progress.extracted_count
                    chunk_count += progress.chunk_count

                # 整页明细全部到达终态后，才把游标和累计数量作为同一检查点持久化。
                cursor_id = candidates[-1].id
                self.repository.update_batch_progress(
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

        except Exception as exc:
            # 批次级错误无法安全跳过。使用最后完整页快照结束批次，使中断页可以重跑。
            self.repository.finish_batch(
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
            )
            raise

        # 最终状态写回失败时直接抛出，不重复调用同一方法，以免掩盖数据库异常。
        final_status = "partial" if failed_count else "success"
        self.repository.finish_batch(
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
        )

    def _process_active_expert(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_id: int,
        progress: _ExpertProgress,
    ) -> str:
        """执行一位 active 专家的完整领域链路并返回业务终态。

        * Source 读取主档、工作、项目、教育四表一致性快照；
        * Normalize 在内存中生成 Profile/Employment/Project Stage；
        * ChunkFactory 生成三类 Chunk；
        * Validator 在向量同步前执行完整契约检查；
        * 有 Chunk 时，通过内存 Contract 同步完整专家快照；
        * 无 Chunk 时，通过空快照删除该专家的旧向量。

        返回 ``success`` 或 ``rejected``。异常原样交给入口处理：单专家入口结束整个
        批次，批量入口只结束当前专家。``progress`` 保存异常发生前已完成阶段的数量。
        """

        aggregate = self.source.load_expert(expert_id)
        progress.extracted_count = (
            1
            + len(aggregate.employments)
            + len(aggregate.projects)
            + len(aggregate.educations)
        )

        profile, employments, projects = normalize_aggregate(
            aggregate,
            data_version=data_version,
            build_batch_id=batch_id,
        )
        chunks = self.factory.build_all(profile, employments, projects)
        if not chunks:
            # 无可生成内容属于 rejected 业务结果，用空快照清除旧向量。
            self._sync_expert_vector_snapshot(expert_id, [])
            return "rejected"

        validate_chunks(chunks)
        # 校验完成后记录 Chunk 数量；向量同步失败时外层会把本专家标记为 failed。
        progress.chunk_count = len(chunks)
        self._sync_expert_vector_snapshot(expert_id, chunks)
        return "success"

    def _sync_expert_vector_snapshot(
        self,
        expert_id: int,
        chunks: list[ChunkRecord],
    ) -> None:
        """把 Chunk 转为公共 Contract，并同步一位专家的完整向量快照。

        ``chunks`` 为空时也会调用下游，用于删除该专家已存在但本次不再产出的向量。
        """

        updated_at = datetime.now(ZoneInfo(self.settings.timezone))
        contracts = chunk_records_to_contracts(
            chunks,
            updated_at=updated_at,
            timezone_name=self.settings.timezone,
        )
        self.vector_sync.sync_expert(expert_id, contracts)


def _validate_data_version(data_version: str) -> None:
    """拒绝空数据版本，避免创建无法追踪来源版本的构建批次。"""

    if not data_version.strip():
        raise ValueError("data_version 不能为空")


def _validate_expert_candidate_page(
    candidates: list[ExpertCandidate],
    *,
    after_id: int,
    upper_bound_id: int,
) -> None:
    """检查候选页 ID 严格递增，并且位于当前游标和固定上界之间。

    校验失败会终止批次且不推进游标，防止重复、倒序或越界 ID 导致漏处理、重复处理
    或分页死循环。
    """

    ids = [candidate.id for candidate in candidates]
    if ids != sorted(set(ids)):
        raise ValueError("Source 候选页必须按唯一 expert id 严格升序排列")
    if ids[0] <= after_id or ids[-1] > upper_bound_id:
        raise ValueError("Source 候选页超出当前全量游标范围")


def _build_batch_error_summary(
    failed_count: int,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """生成不含正文的受限批次错误摘要，供 ``finish_batch`` JSON 字段使用。"""

    if failed_count == 0:
        return None
    return {
        "failed_expert_count": failed_count,
        "sample_errors": errors,
        "sample_truncated": failed_count > len(errors),
    }


def _generate_single_batch_number(expert_id: int) -> str:
    """生成单专家业务批次号，不创建或修改任何专家数据。"""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
    return f"single_{expert_id}_{timestamp}"


def _generate_full_batch_number() -> str:
    """为未指定批次号的全量任务生成带时间戳的唯一业务编号。"""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
    return f"full_{timestamp}"
