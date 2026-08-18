"""单专家与第一版全量 Chunk 加工的总编排模块。

本模块按照固定顺序组织领域链路：

``Source → Normalize/Stage → Chunk → Validator → Stage MySQL → Vector Sync``

各模块职责保持清晰边界：

* ``source.py``：扫描全量候选，并读取一位专家的四表一致性快照；
* ``normalize.py``：清洗和标准化来源字段，生成 Stage 对象；
* ``chunk_builder.py``：生成三类 Chunk Text 与扁平 Metadata；
* ``validators.py``：在写库前验证文本、ID、哈希、枚举和 PII；
* ``repository.py``：管理批次，并以专家为事务单位保存 Stage；
* 本文件：决定调用顺序、分页循环、失败隔离、计数累加和批次状态。

第三阶段把原 ``build_one`` 中的核心加工抽为 ``_process_expert``；第四阶段新增
``build_many``。当前版本在此基础上补齐批次明细、页游标、逻辑删除/无内容生命周期、
Metadata Hash 重算和第一版全局互斥，使全量任务可以从 CLI 安全启动。

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
from .repository import MySQLIndexRepository
from .source import MAX_FULL_SCAN_PAGE_SIZE, MySQLSourceAdapter
from .validators import validate_chunks

# 批次错误摘要只保留有限数量的专家 ID 和异常类型。每位专家的完整机器终态写入
# expert_index_batch_item；这里限制样本长度，避免大批次产生过大的汇总 JSON。
MAX_BATCH_ERROR_SUMMARIES = 20


@dataclass(frozen=True)
class BuildResult:
    """一次单专家加工任务返回给 CLI/API 的安全数量摘要。

    ``build_one`` 创建并结束一个 ``single`` 批次后返回该对象。它不包含 Chunk 正文
    或经历信息；Chunk 仅在本次调用内存中存在，批次统计由 Repository 写入
    ``expert_index_batch``。
    """

    batch_id: int  # expert_index_batch.id，用于回查本次运行。
    batch_no: str  # 单专家业务批次号；调用方不传时由 _single_batch_no 自动生成。
    expert_id: int  # 本次重建的 experts.id。
    extracted_count: int  # 1 条主档加工作、项目、教育来源记录数。
    stage_count: int  # 成功事务提交的 Profile、Employment、Project Stage 行数。
    chunk_count: int  # 成功事务提交的最终 Chunk 行数。
    partial_chunk_count: int  # 旧 CLI 兼容字段；V1.3 当前恒为 0。


@dataclass(frozen=True)
class BatchBuildResult:
    """第一版全量加工完成后返回的安全批次摘要。

    该对象由 :meth:`ExpertIndexPipeline.build_many` 返回，供未来 CLI、调度平台或 API
    输出运行结果。它只包含固定上界、最终游标和数量，不包含任何专家原始字段。

    ``rejected_count`` 表示存在但无可生成内容的专家数；``deactivated_count`` 表示
    主档已逻辑删除且历史 Chunk 已完成停用的专家数。两者都不是程序失败。
    """

    batch_id: int  # expert_index_batch.id。
    batch_no: str  # 全量业务批次号。
    status: str  # success/partial；批次级异常会抛出而不返回该对象。
    upper_bound_id: int  # 启动时固定的最大 experts.id。
    cursor_id: int  # 本轮内存中最后完成调度的专家 ID。
    candidate_count: int  # 实际遍历的专家候选数，包括逻辑删除候选。
    success_count: int  # 正常完成读取、生成、校验和事务落库的专家数。
    failed_count: int  # 本轮发生异常但未阻止后续处理的专家数。
    rejected_count: int  # 存在但没有可生成内容，并已停用旧 Chunk 的专家数。
    deactivated_count: int  # 主档逻辑删除，并已停用历史 Chunk 的专家数。
    extracted_count: int  # 所有已推进专家读取到的四表来源记录总数。
    stage_count: int  # 所有成功专家提交的 Stage 总数。
    chunk_count: int  # 所有成功专家提交的 Chunk 总数。


@dataclass
class _ExpertProgress:
    """单专家内部可变进度，供成功返回和异常计数共同使用。

    ``_process_expert`` 每完成一个阶段就更新相应数量。即使 Normalize、Validator 或
    Repository 中途抛出异常，外层 ``build_one``/``build_many`` 仍能知道任务已经读取
    了多少来源记录。该对象只存在于内存，不对应数据库模型，也不向 CLI 直接暴露。
    """

    expert_id: int
    extracted_count: int = 0
    stage_count: int = 0
    chunk_count: int = 0


class ExpertIndexPipeline:
    """单专家与第一版全量加工流程的总调度器。

    默认使用真实 MySQL Source/Repository；测试可以注入 Fake 实现。所有清洗、文本
    构建和 SQL 映射仍由专门模块负责，本类只编排它们，不复制业务规则。
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

    def build_one(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_no: str | None = None,
    ) -> BuildResult:
        """在全局加工锁内创建 ``single`` 批次并完整重建一位专家。

        第一版 single/full 共用 Repository 的 MySQL 命名锁，因此全量运行期间本命令会
        立即拒绝启动，不会与全量任务同时改写同一专家。锁释放由上下文管理器保证。
        实际单专家批次编排由 :meth:`_build_one_locked` 完成。
        """

        _validate_data_version(data_version)
        if expert_id <= 0:
            raise ValueError("expert_id 必须大于 0")
        with self.repository.build_lock("single"):
            return self._build_one_locked(
                expert_id,
                data_version=data_version,
                batch_no=batch_no,
            )

    def _build_one_locked(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_no: str | None,
    ) -> BuildResult:
        """执行已经取得互斥锁的单专家批次。

        本函数现在只负责单专家批次的开始、成功/失败收口和返回值；真正的
        Source→Normalize→Chunk→Validator→Repository 顺序由 ``_process_expert``
        统一实现，因此以后修正规则不会让单专家和全量路径发生漂移。

        现有 CLI 仍调用本方法，参数和返回结构保持兼容。发生异常时先将批次标记为
        failed，再继续抛出原异常。
        """

        actual_batch_no = batch_no or _single_batch_no(expert_id)
        batch_id = self.repository.start_batch(actual_batch_no, "single")
        progress = _ExpertProgress(expert_id=expert_id)
        try:
            outcome = self._process_expert(
                expert_id,
                data_version=data_version,
                batch_id=batch_id,
                progress=progress,
            )
            # V1.3 不再生成 partial Chunk，保留旧返回字段但固定为 0。
            partial_count = 0
            self.repository.finish_batch(
                batch_id,
                status="success",
                extracted_count=progress.extracted_count,
                stage_count=progress.stage_count,
                chunk_count=progress.chunk_count,
                rejected_count=1 if outcome == "rejected" else 0,
            )
            return BuildResult(
                batch_id=batch_id,
                batch_no=actual_batch_no,
                expert_id=expert_id,
                extracted_count=progress.extracted_count,
                stage_count=progress.stage_count,
                chunk_count=progress.chunk_count,
                partial_chunk_count=partial_count,
            )
        except Exception as exc:
            # 只记录异常类型与截断后的消息，不记录背景、经历或 Chunk Text。
            self.repository.finish_batch(
                batch_id,
                status="failed",
                extracted_count=progress.extracted_count,
                stage_count=progress.stage_count,
                chunk_count=progress.chunk_count,
                rejected_count=1,
                error_summary={"error_type": type(exc).__name__, "message": str(exc)[:400]},
            )
            raise

    def build_many(
        self,
        *,
        data_version: str,
        page_size: int = 200,
        max_experts: int | None = None,
        batch_no: str | None = None,
    ) -> BatchBuildResult:
        """在全局加工锁内运行全量或限定专家数量的批量任务。

        锁从固定来源上界之前一直持有到批次终态写回之后，保证同一时间只有一个全量
        或单专家加工任务。``max_experts`` 限制本批次最多覆盖的候选专家数；Source 会
        在启动时把扫描上界收缩到前 N 位候选的最后一个 ID，而不是在循环中途截断。
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
            return self._build_many_locked(
                data_version=data_version,
                page_size=page_size,
                max_experts=max_experts,
                batch_no=batch_no,
            )

    def _build_many_locked(
        self,
        *,
        data_version: str,
        page_size: int,
        max_experts: int | None,
        batch_no: str | None,
    ) -> BatchBuildResult:
        """按固定主键上界全量加工专家，并可靠记录每页进度。

        构成
        ----
        1. 校验参数并从 Source 固定 ``upper_bound_id``；
        2. 创建一个 ``full`` 批次；
        3. 从 ``after_id=0`` 开始按 Keyset Pagination 读取候选页；
        4. 每页先登记 pending，再按 running→终态记录每位候选；
        5. active 专家复用 ``_process_expert``，逻辑删除和无内容分别进入
           deactivated/rejected，意外异常才计入 failed；
        6. 每页全部明细可靠落库后，才持久化游标和累计计数；
        7. 逻辑删除、无内容或陈旧来源导致 Chunk 停用时同步重算 metadata_hash；
        8. 只要存在专家级 failed，最终批次为 partial，否则为 success；
        9. 分页或批次记录错误会终止任务并将批次记为 failed。

        使用位置
        --------
        本方法由 ``cli.py build-batch`` 调用，也可供未来调度平台复用。第一版实现全量
        新批次，不自动恢复中断批次；数据库中的页游标与专家明细已为后续续跑留下基础。
        """

        # 先固定来源上界，再创建批次；max_experts 会把上界收缩为前 N 位候选的最后
        # 一个 ID。即使任务运行期间新增专家，本轮范围也不会改变。
        upper_bound_id = self.source.get_full_scan_upper_bound(
            max_experts=max_experts,
        )
        actual_batch_no = batch_no or _full_batch_no()
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
        error_summaries: list[dict[str, object]] = []

        try:
            while cursor_id < upper_bound_id:
                candidates = self.source.load_full_scan_candidates(
                    after_id=cursor_id,
                    upper_bound_id=upper_bound_id,
                    limit=page_size,
                )
                if not candidates:
                    # Keyset 查询在固定范围内已经没有行。把内存游标推进到固定上界，
                    # 并同步落库，明确表示固定范围扫描完成。这不会跳过实际返回的记录。
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
                    break
                _validate_candidate_page(
                    candidates,
                    after_id=cursor_id,
                    upper_bound_id=upper_bound_id,
                )
                # 整页先登记 pending。若这里失败，任何候选都不会开始，批次级异常分支
                # 会结束任务，游标也仍停留在上一页。
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
                            # 删除候选不读取四表正文；Repository 只清理 Stage，空快照
                            # 通过内存同步交给下游删除该专家的历史向量。
                            self.repository.mark_deleted_stage(
                                candidate.id,
                                build_batch_id=batch_id,
                            )
                            self._sync_expert_vectors(candidate.id, [])
                            terminal_status = "deactivated"
                            result_code = "expert_deleted"
                        else:
                            terminal_status = self._process_expert(
                                candidate.id,
                                data_version=data_version,
                                batch_id=batch_id,
                                progress=progress,
                            )
                            if terminal_status == "rejected":
                                result_code = "no_chunkable_content"
                    # 这里只隔离专家领域处理和专家事务异常。下面的批次明细、页游标写入
                    # 位于 except 之外；它们失败时属于批次级错误，必须终止而不能跳过。
                    except Exception as exc:  # noqa: BLE001
                        terminal_status = "failed"
                        result_code = "expert_processing_failed"
                        error_type = type(exc).__name__

                    # 专家终态必须先可靠写入，再累计批次计数和推进页游标。
                    self.repository.finish_batch_item(
                        batch_id,
                        candidate.id,
                        status=terminal_status,
                        extracted_count=progress.extracted_count,
                        stage_count=progress.stage_count,
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

                    # rejected 已提交当前 Stage，failed 可能只完成部分 Source 读取；这些
                    # 都是已经真实发生的安全数量，因此统一累计。
                    extracted_count += progress.extracted_count
                    stage_count += progress.stage_count
                    chunk_count += progress.chunk_count

                # 只有整页批次明细均到达终态后才推进并持久化游标。
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

        except Exception as exc:
            # 页查询、候选顺序或其他编排级异常无法安全跳过，必须终止整个批次。
            self.repository.finish_batch(
                batch_id,
                status="failed",
                extracted_count=extracted_count,
                stage_count=stage_count,
                chunk_count=chunk_count,
                rejected_count=rejected_count,
                cursor_id=cursor_id,
                candidate_count=candidate_count,
                success_count=success_count,
                failed_count=failed_count,
                deactivated_count=deactivated_count,
                error_summary={"error_type": type(exc).__name__},
            )
            raise

        # 最终 finish 放在循环 try 之外：若批次状态写回自身失败，直接向上抛出，避免
        # 再次调用同一个失败方法并掩盖第一次数据库异常。
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
            error_summary=_batch_error_summary(failed_count, error_summaries),
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

    def _process_expert(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_id: int,
        progress: _ExpertProgress,
    ) -> str:
        """执行一位 active 专家的完整领域链路并返回业务终态。

        该函数是第三阶段抽出的核心复用单元，只被本类的 ``build_one`` 和
        ``build_many`` 调用：

        * Source 读取主档、工作、项目、教育四表一致性快照；
        * Normalize 生成 Profile/Employment/Project Stage；
        * ChunkFactory 生成三类 Chunk；
        * Validator 在入库前执行完整契约检查；
        * 有 Chunk 时，Repository 原子保存 Stage，随后以内存 Contract 同步下游；
        * 无 Chunk 时，Repository 保存 rejected Stage，并以下游空快照删除旧向量。

        返回 ``success`` 或 ``rejected``。任何意外异常仍原样抛给外层：单专家入口会把
        批次记为 failed；全量入口会把当前专家记为 failed 并继续下一位。``progress``
        在每一步成功后更新，供两类入口正确统计。
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
            # “专家存在但无可生成内容”是业务拒绝而非程序错误。保存当前 Stage、标记
            # rejected，并通过下游空快照删除旧向量，避免旧数据继续参与检索。
            progress.stage_count = self.repository.persist_rejected_stage_snapshot(
                profile,
                employments,
                projects,
                reason="no_chunkable_content",
            )
            self._sync_expert_vectors(expert_id, [])
            return "rejected"

        validate_chunks(chunks)
        progress.stage_count = self.repository.persist_stage_snapshot(
            profile,
            employments,
            projects,
        )
        # Stage 已提交后记录本轮内存产出数；向量同步若失败，外层会把本专家标记为
        # failed，重跑时由下游当前状态和哈希完成幂等修复。
        progress.chunk_count = len(chunks)
        self._sync_expert_vectors(expert_id, chunks)
        return "success"

    def _sync_expert_vectors(
        self,
        expert_id: int,
        chunks: list[ChunkRecord],
    ) -> None:
        """把一位专家的当前 Chunk 快照同步到下游公共 Python 服务。"""

        updated_at = datetime.now(ZoneInfo(self.settings.timezone))
        contracts = chunk_records_to_contracts(
            chunks,
            updated_at=updated_at,
            timezone_name=self.settings.timezone,
        )
        self.vector_sync.sync_expert(expert_id, contracts)


def _validate_data_version(data_version: str) -> None:
    """在创建批次前验证业务版本，供单专家和全量入口共同调用。"""

    if not data_version.strip():
        raise ValueError("data_version 不能为空")


def _validate_candidate_page(
    candidates: list[ExpertCandidate],
    *,
    after_id: int,
    upper_bound_id: int,
) -> None:
    """防御性检查 Source 返回页的顺序和固定范围。

    正常 MySQL Source 已通过 ``ORDER BY id ASC`` 保证该契约；此处仍在 Pipeline 边界
    校验，防止替代 Source、测试 Fake 或未来实现返回重复/倒序 ID，造成游标回退、死循环
    或专家重复加工。失败属于批次级错误，因此不会继续推进游标。
    """

    ids = [candidate.id for candidate in candidates]
    if ids != sorted(set(ids)):
        raise ValueError("Source 候选页必须按唯一 expert id 严格升序排列")
    if ids[0] <= after_id or ids[-1] > upper_bound_id:
        raise ValueError("Source 候选页超出当前全量游标范围")


def _batch_error_summary(
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


def _single_batch_no(expert_id: int) -> str:
    """生成单专家业务批次号，不创建或修改任何专家数据。"""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
    return f"single_{expert_id}_{timestamp}"


def _full_batch_no() -> str:
    """生成全量任务业务批次号，供 ``build_many`` 未显式传编号时使用。"""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
    return f"full_{timestamp}"


# 保留旧私有函数名，避免项目外仍有调试脚本直接引用时立即中断；正式调用应通过
# build_one，由它使用语义更明确的 _single_batch_no。该别名不产生额外行为。
def _batch_no(expert_id: int) -> str:
    """兼容旧名称，委托给 :func:`_single_batch_no`。"""

    return _single_batch_no(expert_id)
