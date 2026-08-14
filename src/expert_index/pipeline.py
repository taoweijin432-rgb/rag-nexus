"""单专家与第一版全量 Chunk 加工的总编排模块。

本模块按照固定顺序组织领域链路：

``Source → Normalize/Stage → Chunk → Validator → MySQL``

各模块职责保持清晰边界：

* ``source.py``：扫描全量候选，并读取一位专家的四表一致性快照；
* ``normalize.py``：清洗和标准化来源字段，生成 Stage 对象；
* ``chunk_builder.py``：生成三类 Chunk Text 与扁平 Metadata；
* ``validators.py``：在写库前验证文本、ID、哈希、枚举和 PII；
* ``repository.py``：管理批次，并以专家为事务单位保存 Stage 与 Chunk；
* 本文件：决定调用顺序、分页循环、失败隔离、计数累加和批次状态。

第三阶段把原 ``build_one`` 中的核心加工抽为 ``_process_expert``；第四阶段新增
``build_many``，按固定 ID 上界和 Keyset Pagination 处理全量专家。一位专家失败只
计入当前批次，不会中断后续专家；分页读取等批次级错误仍会终止任务。

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .chunk_builder import ChunkFactory
from .config import Settings
from .models import ExpertCandidate
from .normalize import normalize_aggregate
from .repository import MySQLIndexRepository
from .source import MAX_FULL_SCAN_PAGE_SIZE, MySQLSourceAdapter
from .validators import validate_chunks

# 批次错误摘要只保留有限数量的专家 ID 和异常类型。完整失败明细将在下一阶段写入
# expert_index_batch_item；限制长度也能避免一个大批次生成过大的 error_summary JSON。
MAX_BATCH_ERROR_SUMMARIES = 20


@dataclass(frozen=True)
class BuildResult:
    """一次单专家加工任务返回给 CLI/API 的安全数量摘要。

    ``build_one`` 创建并结束一个 ``single`` 批次后返回该对象。它不包含 Chunk 正文
    或经历信息；真正的加工结果保存在 ``expert_chunk``，批次统计由 Repository 写入
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

    ``rejected_count`` 与 ``deactivated_count`` 已提前进入返回契约，但第三、第四阶段
    暂不改变旧 Chunk，因此当前保持 0；下一阶段实现生命周期事务后再正式累加。
    """

    batch_id: int  # expert_index_batch.id。
    batch_no: str  # 全量业务批次号。
    status: str  # success/partial；批次级异常会抛出而不返回该对象。
    upper_bound_id: int  # 启动时固定的最大 experts.id。
    cursor_id: int  # 本轮内存中最后完成调度的专家 ID。
    candidate_count: int  # 实际遍历的专家候选数，包括逻辑删除候选。
    success_count: int  # 正常完成读取、生成、校验和事务落库的专家数。
    failed_count: int  # 本轮发生异常但未阻止后续处理的专家数。
    rejected_count: int  # 无可生成内容专家数；下一阶段接入，当前为 0。
    deactivated_count: int  # 逻辑删除并完成历史停用的专家数；下一阶段接入，当前为 0。
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


class NoChunkableContentError(ValueError):
    """专家存在但无法生成任何 Chunk。

    单专家命令仍把它当作失败并抛出；全量编排当前把它隔离为专家级失败。下一阶段会
    捕获该类型，将专家明细记为 rejected，并在同一事务中停用历史 Chunk。
    """


class DeletedExpertLifecyclePendingError(RuntimeError):
    """逻辑删除专家等待下一阶段执行历史 Chunk 停用。

    Source 候选必须保留已删除专家，但真正停用需要 Repository 同步重算
    ``metadata_hash``。第三、第四阶段不提前实现该事务，因此先把这类候选安全隔离为
    专家级失败，保证不会错误调用 ``load_expert`` 或把旧 Chunk 当作已处理。
    """


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
    ) -> None:
        self.settings = settings
        self.source = source or MySQLSourceAdapter(settings.source_db)
        self.repository = repository or MySQLIndexRepository(settings.index_db)
        self.factory = ChunkFactory(settings)

    def build_one(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_no: str | None = None,
    ) -> BuildResult:
        """创建 ``single`` 批次，完整重建一位专家并返回数量摘要。

        本函数现在只负责单专家批次的开始、成功/失败收口和返回值；真正的
        Source→Normalize→Chunk→Validator→Repository 顺序由 ``_process_expert``
        统一实现，因此以后修正规则不会让单专家和全量路径发生漂移。

        现有 CLI 仍调用本方法，参数和返回结构保持兼容。发生异常时先将批次标记为
        failed，再继续抛出原异常。
        """

        _validate_data_version(data_version)
        if expert_id <= 0:
            raise ValueError("expert_id 必须大于 0")

        actual_batch_no = batch_no or _single_batch_no(expert_id)
        batch_id = self.repository.start_batch(actual_batch_no, "single")
        progress = _ExpertProgress(expert_id=expert_id)
        try:
            self._process_expert(
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
                rejected_count=partial_count,
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
        batch_no: str | None = None,
    ) -> BatchBuildResult:
        """按固定主键上界全量加工专家，并隔离单专家失败。

        构成
        ----
        1. 校验参数并从 Source 固定 ``upper_bound_id``；
        2. 创建一个 ``full`` 批次；
        3. 从 ``after_id=0`` 开始按 Keyset Pagination 读取候选页；
        4. 每位 active 专家复用 ``_process_expert``，失败时只累计失败数；
        5. 每页全部候选到达终态后，把内存游标推进到该页最后一个 ID；
        6. 全部成功时批次为 success，存在专家级失败时为 partial；
        7. 分页读取等批次级异常会将批次记为 failed，并继续向调用方抛出。

        使用位置
        --------
        本方法是未来 ``cli.py build-batch`` 和调度平台的编排入口。当前阶段尚未增加 CLI，
        便于先单独审查 Pipeline。批次明细、游标持久化、删除/拒绝停用与并发锁将在
        Repository 后续阶段接入；所以现在不应在生产环境直接调用本方法。
        """

        _validate_data_version(data_version)
        if not 1 <= page_size <= MAX_FULL_SCAN_PAGE_SIZE:
            raise ValueError(
                f"page_size 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间"
            )

        # 先固定来源上界，再创建批次；即使任务运行期间新增专家，本轮范围也不会改变。
        upper_bound_id = self.source.get_full_scan_upper_bound()
        actual_batch_no = batch_no or _full_batch_no()
        batch_id = self.repository.start_batch(actual_batch_no, "full")

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
                    # 明确表示本轮范围扫描完成；这不会跳过任何实际返回的专家记录。
                    cursor_id = upper_bound_id
                    break
                _validate_candidate_page(
                    candidates,
                    after_id=cursor_id,
                    upper_bound_id=upper_bound_id,
                )

                for candidate in candidates:
                    candidate_count += 1
                    progress = _ExpertProgress(expert_id=candidate.id)
                    try:
                        if candidate.is_deleted:
                            # 后续生命周期阶段会在这里替换为 Repository 原子停用操作。
                            raise DeletedExpertLifecyclePendingError(
                                f"专家 {candidate.id} 已逻辑删除，等待历史 Chunk 停用阶段"
                            )
                        self._process_expert(
                            candidate.id,
                            data_version=data_version,
                            batch_id=batch_id,
                            progress=progress,
                        )
                    # 全量隔离边界必须捕获任意专家级异常，包括数据库驱动异常；批次级
                    # 页读取异常位于外层，不会被这里吞掉。
                    except Exception as exc:  # noqa: BLE001
                        # 单专家失败只影响自己的事务。错误摘要不保存异常消息，避免第三方
                        # 驱动或未来规则把源文本带入批次 JSON。
                        failed_count += 1
                        if len(error_summaries) < MAX_BATCH_ERROR_SUMMARIES:
                            error_summaries.append(
                                {
                                    "expert_id": candidate.id,
                                    "error_type": type(exc).__name__,
                                }
                            )
                    else:
                        success_count += 1
                    finally:
                        # 成功和失败都累计已实际推进的安全数量；失败事务的 stage/chunk
                        # 保持 0，因此不会被误算为已经写库。
                        extracted_count += progress.extracted_count
                        stage_count += progress.stage_count
                        chunk_count += progress.chunk_count

                # 只有整页候选均已成功或被隔离为专家级失败后才推进游标。
                cursor_id = candidates[-1].id

        except Exception as exc:
            # 页查询、候选顺序或其他编排级异常无法安全跳过，必须终止整个批次。
            self.repository.finish_batch(
                batch_id,
                status="failed",
                extracted_count=extracted_count,
                stage_count=stage_count,
                chunk_count=chunk_count,
                rejected_count=rejected_count,
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
    ) -> None:
        """执行一位 active 专家的完整领域链路，不管理批次开始和结束。

        该函数是第三阶段抽出的核心复用单元，只被本类的 ``build_one`` 和
        ``build_many`` 调用：

        * Source 读取主档、工作、项目、教育四表一致性快照；
        * Normalize 生成 Profile/Employment/Project Stage；
        * ChunkFactory 生成三类 Chunk；
        * Validator 在入库前执行完整契约检查；
        * Repository 以该专家为一个事务保存 Stage、Chunk 和陈旧 Chunk 状态。

        任何异常都原样抛给外层：单专家入口会把整个批次记为 failed；全量入口会隔离
        当前专家并继续下一位。``progress`` 在每一步成功后更新，供两类入口正确统计。
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
            raise NoChunkableContentError(f"专家 {expert_id} 没有可生成的 Chunk")

        validate_chunks(chunks)
        progress.stage_count = self.repository.persist_expert(
            profile,
            employments,
            projects,
            chunks,
        )
        # 只有 Repository 事务成功提交后才记录 Chunk 数，失败事务保持 0。
        progress.chunk_count = len(chunks)


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
