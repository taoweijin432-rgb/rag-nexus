"""P5.2 正式范围 Consumer（scope-based）。

复用 :class:`ContractConsumerCore`（Work-Item 无关判定与执行核心），实现
``ScopeHandoffRequest → 查询该 scope 当前向量状态 → 对 desired Contracts 执行核心
→ 计算并逻辑下线 stale → 返回 ScopeHandoffReceipt`` 的正式范围收敛链路。

架构边界（无环依赖）：

    expert_index.handoff 定义 ScopeHandoffRequest/ScopeHandoffReceipt（上游交付契约）；
    expert_index 已单向依赖 rag_vector_store。为避免 rag_vector_store 反向 import
    expert_index 形成环，本模块只依赖结构化只读 Protocol（``ScopeView`` /
    ``ScopeHandoffRequestView``）与 ``ScopeReceiptSink``（接收 Receipt 字段并返回
    上游真实 ``ScopeHandoffReceipt``）。生产组装由上游注入 Receipt 工厂，测试直接
    传真实 P5.1 Request/Receipt。本模块运行时不 import expert_index 任何类型。

职责：

    - 只读查询该 scope 的当前 active 向量状态（复用
      ``list_vectors_by_expert`` + ``get_vectors``）；
    - 把 desired Contracts 包成 :class:`ContractUpsertIntent`（不携带 stale），
      一次调用 ``ContractConsumerCore.process`` 收敛；不复制三 Hash / metadata-only /
      重新激活判定逻辑；
    - desired 全部成功后再计算 ``stale = scope 内 active IDs − desired IDs``，
      经 ``ContractConsumerCore`` 执行单个 :class:`ContractDisableInput` 逻辑下线；
    - 禁止物理删除、禁止 Work Item/Batch、禁止写公共任务状态/制品存储/检查点；
    - 不实现 retry/lease/requeue（状态表适配层职责）。

两阶段与防短路（对齐修订版计划 §P5.2 范围收敛语义）：

    - 先收敛全部 desired，再下线 stale；desired 出现 rejected/failed 时立即返回
      Receipt，**绝不执行 stale 下线**；
    - stale 下线失败返回 ``failed/retryable=True`` 并注明 desired 已完成——
      重放时主路径因三 Hash 幂等自然收敛，仅补做 stale 下线。

Receipt 计数：

    - ``desired_count = len(request.contracts)``；
    - ``embedding_count``：embedding_performed=True 的 desired 数；
    - ``metadata_only_count``：metadata_only=True 的 desired 数；
    - ``unchanged_count``：result_code==unchanged_skipped 的 desired 数；
    - ``disable_count``：本次 stale 目标数（成功重放时若 active stale 已为空则为 0，
      属等价收敛结果）；
    - identity/token/payload_hash/scope 全部从 Request 原样回填。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from expert_contracts import ExpertChunkEmbeddingInput

from ..ingestion.errors import VectorIngestionError
from ..ingestion.models import VectorExpertList, VectorQueryResult
from .consumer_core import (
    ContractConsumerCore,
    ContractDisableInput,
)
from .consumer_models import (
    ERROR_CODE_SCOPE_ACTIVE_STATE_INVALID,
    ERROR_CODE_SCOPE_QUERY_FAILED,
    ERROR_CODE_SCOPE_STALE_DISABLE_FAILED,
    ERROR_CODE_SCOPE_UNSUPPORTED_TYPE,
    RESULT_CODE_TARGETS_DISABLED,
    RESULT_CODE_UNCHANGED_SKIPPED,
    ExecutionResult,
)

# ---------------------------------------------------------------------------
# 只读 Protocol（无环依赖缝隙）
# ---------------------------------------------------------------------------
_SCOPE_TYPES = frozenset({"expert", "profile", "employment", "project"})
_PROFILE_SOURCE_TABLE = "experts+experts_education_experience"
_EMPLOYMENT_SOURCE_TABLE = "experts_work_experience"
_PROJECT_SOURCE_TABLE = "experts_project_experience"


@runtime_checkable
class ScopeView(Protocol):
    """上游 ``IncrementalImpactScope`` 的结构化只读视图。"""

    scope_type: str
    expert_id: int
    chunk_type: str | None
    source_table: str | None
    source_field_id: str | None


@runtime_checkable
class ScopeHandoffRequestView(Protocol):
    """上游 ``ScopeHandoffRequest`` 的结构化只读视图。"""

    request_id: str
    task_id: int
    batch_id: int
    attempt_token: str
    payload_hash: str
    impact_scope: ScopeView
    contracts: tuple[ExpertChunkEmbeddingInput, ...]


@runtime_checkable
class ScopeHandoffReceiptView(Protocol):
    """范围交付回执的下游输出视图。

    该协议只声明上游编排需要核验的字段，不绑定 ``expert_index`` 的具体模型；
    生产 Receipt 由注入的 sink 构造，测试可使用同结构 Fake。各计数必须能由
    下游逐项结果确定性重算，失败时不得伪造 committed。
    """

    request_id: str
    task_id: int
    batch_id: int
    attempt_token: str
    payload_hash: str
    impact_scope: ScopeView
    status: str
    result_code: str
    desired_results: tuple[tuple[str, str], ...]
    stale_vector_doc_ids: tuple[str, ...]
    desired_count: int
    embedding_count: int
    metadata_only_count: int
    disable_count: int
    unchanged_count: int
    retryable: bool
    error_type: str | None
    error_message: str | None


class ScopeReceiptSink(Protocol):
    """接收 Receipt 字段并返回上游真实 ``ScopeHandoffReceipt`` 的工厂。

    生产组装由上游注入（``ScopeHandoffReceipt`` 构造器），测试直接传真实类；
    本模块不 import expert_index.handoff，避免反向依赖成环。
    """

    def make_receipt(
        self,
        *,
        request_id: str,
        task_id: int,
        batch_id: int,
        attempt_token: str,
        payload_hash: str,
        impact_scope: ScopeView,
        status: str,
        result_code: str,
        desired_results: tuple[tuple[str, str], ...],
        stale_vector_doc_ids: tuple[str, ...],
        desired_count: int,
        embedding_count: int,
        metadata_only_count: int,
        disable_count: int,
        unchanged_count: int,
        retryable: bool,
        error_type: str | None,
        error_message: str | None,
    ) -> ScopeHandoffReceiptView:
        """按字段组装并返回上游 Receipt 实例。"""
        ...


@runtime_checkable
class ScopeVectorStatePort(Protocol):
    """范围当前向量状态的只读查询端口（生产注入 ``VectorIngestionService``）。

    复用既有公开方法，不新增向量能力：
        - ``list_vectors_by_expert`` 拿该专家全部 vector_doc_id；
        - ``get_vectors`` 拿 record（含 metadata JSON 与投影后的三 Hash）。
    """

    def list_vectors_by_expert(self, expert_id: int) -> VectorExpertList:
        """按专家列出 Milvus 中全部 vector_doc_id（仅轻量字段）。"""
        ...

    def get_vectors(self, vector_doc_ids: list[str]) -> VectorQueryResult:
        """按 vector_doc_id 集合批量查询记录（含 metadata + 投影三 Hash）。"""
        ...


# ---------------------------------------------------------------------------
# 作用域内 active 状态记录
# ---------------------------------------------------------------------------
class _ScopeActiveState:
    """scope 过滤后某条已存 active 向量的轻量状态。"""

    __slots__ = ("chunk_type", "expert_id", "source_field_id", "source_table", "vector_doc_id")

    def __init__(
        self,
        *,
        vector_doc_id: str,
        expert_id: int,
        chunk_type: str,
        source_table: str,
        source_field_id: str,
    ) -> None:
        self.vector_doc_id = vector_doc_id
        self.expert_id = expert_id
        self.chunk_type = chunk_type
        self.source_table = source_table
        self.source_field_id = source_field_id


class _ScopeStateError(ValueError):
    """scope active 状态提取失败（字段缺失/类型非法），确定性。"""


class _ScopeQueryError(Exception):
    """scope 查询基础设施失败，可重试。"""


def _extract_str(metadata: dict[str, object], field: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _ScopeStateError(f"metadata 缺少合法 {field}")
    return value


def _extract_bool(metadata: dict[str, object], field: str) -> bool:
    value = metadata.get(field)
    if not isinstance(value, bool):
        raise _ScopeStateError(f"metadata.{field} 必须是布尔值，收到 {type(value).__name__}")
    return value


def _in_scope(
    *,
    scope: ScopeView,
    expert_id: int,
    chunk_type: str,
    source_table: str,
    source_field_id: str,
) -> bool:
    """判断一条已存向量是否属于当前 scope（精确过滤）。

    - ``expert`` 范围：该专家全部 active Chunk（跨三类）；
    - ``profile`` 范围：仅 profile_summary（source_table/profile_source_field_id 校验）；
    - ``employment`` 范围：仅对应 source_field_id 的 employment family；
    - ``project`` 范围：仅对应 source_field_id 的 project family。
    """

    if expert_id != scope.expert_id:
        return False
    if scope.scope_type == "expert":
        return chunk_type in ("profile_summary", "employment", "project")
    if chunk_type != scope.chunk_type:
        return False
    if source_table != scope.source_table:
        return False
    if scope.scope_type == "profile":
        return source_field_id == scope.source_field_id
    return source_field_id == scope.source_field_id


def _expected_scope_invariants(scope: ScopeView) -> None:
    """校验 scope 字段组合与 INCREMENTAL_IMPACT_SCOPE_TYPES 一致（fail-closed）。"""

    if scope.scope_type not in _SCOPE_TYPES:
        raise _ScopeStateError(f"不支持的影响范围类型: {scope.scope_type}")
    if scope.scope_type == "expert":
        if scope.chunk_type is not None or scope.source_table is not None or scope.source_field_id is not None:
            raise _ScopeStateError("expert 范围 chunk_type/source_table/source_field_id 必须为 None")
        return
    if scope.chunk_type is None or scope.source_table is None or scope.source_field_id is None:
        raise _ScopeStateError(f"{scope.scope_type} 范围必须提供 chunk_type/source_table/source_field_id")
    expected_table = {
        "profile": _PROFILE_SOURCE_TABLE,
        "employment": _EMPLOYMENT_SOURCE_TABLE,
        "project": _PROJECT_SOURCE_TABLE,
    }[scope.scope_type]
    if scope.source_table != expected_table:
        raise _ScopeStateError(f"{scope.scope_type} 范围 source_table 必须为 {expected_table}")
    expected_chunk = {"profile": "profile_summary", "employment": "employment", "project": "project"}[scope.scope_type]
    if scope.chunk_type != expected_chunk:
        raise _ScopeStateError(f"{scope.scope_type} 范围 chunk_type 必须为 {expected_chunk}")


def _filter_active_scope(
    *,
    scope: ScopeView,
    records: list[dict[str, object]],
    missing: list[str],
) -> tuple[list[_ScopeActiveState] | None, str | None, bool]:
    """从 get_vectors 记录中提取属于当前 scope 且 active 的状态。

    返回 (active|None, error_message_if_invalid, is_retryable)。
    - ``missing`` 非空表示 listing 声明存在但 get_vectors 未命中，视为存储不一致，
      fail-closed 返回 failed/retryable（避免漏下线 stale 或误判 committed）；
    - metadata 字段缺失/类型非法 → 确定性 rejected（is_retryable=False）；
    - is_active 非 bool → 确定性 rejected。
    非 scope 内或 inactive 的记录被排除，不进入 stale。
    """

    if missing:
        return None, f"scope_query_failed：get_vectors 未命中 listing 声明存在的 {len(missing)} 条记录", True
    active: list[_ScopeActiveState] = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return None, "active_state_invalid：已存记录缺少 metadata 对象", False
        vector_doc_id = record.get("vector_doc_id")
        if not isinstance(vector_doc_id, str) or not vector_doc_id:
            return None, "active_state_invalid：已存记录缺少 vector_doc_id", False
        try:
            is_active = _extract_bool(metadata, "is_active")
            if not is_active:
                continue
            expert_id_raw = metadata.get("expert_id")
            if isinstance(expert_id_raw, bool) or not isinstance(expert_id_raw, int) or expert_id_raw < 1:
                return None, "active_state_invalid：metadata.expert_id 非法", False
            chunk_type = _extract_str(metadata, "chunk_type")
            source_table = _extract_str(metadata, "source_table")
            source_field_id = _extract_str(metadata, "source_field_id")
        except _ScopeStateError as error:
            return None, f"active_state_invalid：{error}", False
        if not _in_scope(
            scope=scope,
            expert_id=expert_id_raw,
            chunk_type=chunk_type,
            source_table=source_table,
            source_field_id=source_field_id,
        ):
            continue
        active.append(
            _ScopeActiveState(
                vector_doc_id=vector_doc_id,
                expert_id=expert_id_raw,
                chunk_type=chunk_type,
                source_table=source_table,
                source_field_id=source_field_id,
            )
        )
    return active, None, False


def _single_line_truncate(value: str, limit: int = 512) -> str:
    """归一为单行并截断到限制长度（满足 Receipt error_message 约束）。"""

    flat = " ".join(value.splitlines())
    if len(flat) > limit:
        flat = flat[:limit]
    return flat


# ---------------------------------------------------------------------------
# 正式范围 Consumer
# ---------------------------------------------------------------------------
class ScopeConsumer:
    """正式 scope-based 范围 Consumer（复用 ContractConsumerCore）。

    注入：
        ``core``：:class:`ContractConsumerCore` 实例（已绑定向量能力）；
        ``state_port``：只读 scope 向量状态查询端口（生产 VectorIngestionService）；
        ``receipt_sink``：上游 Receipt 工厂。

    入口：
        :meth:`process` 接收 ``ScopeHandoffRequestView``，返回上游 Receipt。
    """

    def __init__(
        self,
        *,
        core: ContractConsumerCore,
        state_port: ScopeVectorStatePort,
        receipt_sink: ScopeReceiptSink,
    ) -> None:
        self._core = core
        self._state_port = state_port
        self._receipt_sink = receipt_sink

    def process(self, request: ScopeHandoffRequestView) -> ScopeHandoffReceiptView:
        """执行范围收敛：查询 active → 收敛 desired → 下线 stale → 返回 Receipt。"""

        scope = request.impact_scope
        try:
            _expected_scope_invariants(scope)
        except _ScopeStateError as error:
            return self._receipt(
                request,
                status="rejected",
                result_code=f"scope_invalid:{scope.scope_type}",
                error_type=ERROR_CODE_SCOPE_UNSUPPORTED_TYPE,
                error_message=_single_line_truncate(str(error)),
            )

        # 阶段 0：查询该专家全部 vector_doc_id，再取 metadata 做 scope 精确过滤。
        try:
            listing = self._state_port.list_vectors_by_expert(scope.expert_id)
        except VectorIngestionError as error:
            return self._receipt(
                request,
                status="failed",
                result_code="scope_query_failed",
                error_type=ERROR_CODE_SCOPE_QUERY_FAILED,
                error_message=_single_line_truncate(f"list_vectors_by_expert 失败：{error}"),
                retryable=True,
            )
        active_states, query_error, retryable = self._load_active_scope(
            scope=scope,
            vector_doc_ids=list(listing.vector_doc_ids),
        )
        if query_error is not None:
            status = "failed" if retryable else "rejected"
            return self._receipt(
                request,
                status=status,
                result_code=("scope_query_failed" if retryable else "scope_active_state_invalid"),
                error_type=(ERROR_CODE_SCOPE_QUERY_FAILED if retryable else ERROR_CODE_SCOPE_ACTIVE_STATE_INVALID),
                error_message=_single_line_truncate(query_error),
                counts=_Counts(),
                disable_count=0,
                retryable=retryable,
            )
        assert active_states is not None  # query_error 已 return，此处必非 None

        # 阶段 1：把 desired Contracts 交核心做 Eligibility 预判 + 三 Hash 差分收敛。
        # 通过 core.process_contracts 复用 Eligibility 规则（不复制），不携带 stale。
        desired_ids = {contract.vector_doc_id for contract in request.contracts}
        desired_results = self._core.process_contracts(list(request.contracts))

        # 阶段 1.5：desired 出现 rejected/failed 时立即返回 Receipt，绝不执行 stale。
        block = self._classify_desired(desired_results)
        if block is not None:
            return self._receipt(
                request,
                status=block.status,
                result_code=block.result_code,
                error_type=block.error_type,
                error_message=block.error_message,
                counts=block.counts,
                desired_results=desired_results,
                retryable=block.retryable,
            )

        # 阶段 2：desired 全部收敛后计算 stale = scope 内 active IDs − desired IDs，
        # 经核心执行单个 ContractDisableInput 逻辑下线（不复制 disable 逻辑）。
        stale_ids = sorted(vid for vid in (s.vector_doc_id for s in active_states) if vid not in desired_ids)
        desired_disable_count = self._desired_disable_count(desired_results)
        stale_error: tuple[str, bool] | None = None
        stale_success_count = 0
        if stale_ids:
            disable_input = ContractDisableInput(
                vector_doc_id=stale_ids[0],
                target_vector_doc_ids=tuple(stale_ids),
            )
            stale_results = self._core.process([disable_input])
            stale_result = stale_results[0]
            if stale_result.status != "committed":
                stale_error = (
                    _single_line_truncate(
                        f"stale_disable_failed（desired 已完成）：{stale_result.reason or ''}"
                    ),
                    stale_result.retryable,
                )
            else:
                stale_success_count = len(stale_result.disabled_ids)

        counts = self._summarize(desired_results)
        if stale_error is not None:
            return self._receipt(
                request,
                status="failed",
                result_code="scope_stale_disable_failed",
                error_type=ERROR_CODE_SCOPE_STALE_DISABLE_FAILED,
                error_message=stale_error[0],
                counts=counts,
                desired_results=desired_results,
                disable_count=desired_disable_count,  # 只保留已成功的 desired disable，不计失败的 stale
                retryable=stale_error[1],
            )
        return self._receipt(
            request,
            status="committed",
            result_code="scope_committed",
            counts=counts,
            desired_results=desired_results,
            stale_vector_doc_ids=tuple(stale_ids),
            disable_count=desired_disable_count + stale_success_count,
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _load_active_scope(
        self,
        *,
        scope: ScopeView,
        vector_doc_ids: list[str],
    ) -> tuple[list[_ScopeActiveState] | None, str | None, bool]:
        """读取 scope 内 active 向量状态。返回 (active|None, error, retryable)。"""

        if not vector_doc_ids:
            return [], None, False
        try:
            query = self._state_port.get_vectors(vector_doc_ids)
        except VectorIngestionError as error:
            return None, f"get_vectors 失败：{error}", True
        active, error, retryable = _filter_active_scope(
            scope=scope, records=query.records, missing=query.missing
        )
        return active, error, retryable

    def _classify_desired(
        self,
        results: list[ExecutionResult],
    ) -> _DesiredBlock | None:
        """分类 desired 结果；任一 rejected→整体 rejected，任一 failed→整体 failed。

        正式 desired 收敛成功只接受 embedded/metadata_updated/unchanged_skipped；
        targets_disabled/upsert_skipped/duplicate_superseded/planned 不作为正式
        desired 成功，fail-closed 转 rejected。
        """

        counts = self._summarize(results)
        rejected_msg: str | None = None
        failed_msg: str | None = None
        failed_retryable = False
        for result in results:
            if result.status == "rejected":
                rejected_msg = rejected_msg or _single_line_truncate(
                    f"{result.vector_doc_id}: {result.reason or result.error_code or ''}"
                )
            elif result.status == "failed":
                failed_msg = failed_msg or _single_line_truncate(
                    f"{result.vector_doc_id}: {result.reason or result.error_code or ''}"
                )
                failed_retryable = failed_retryable or result.retryable
            elif result.status not in ("committed", "skipped"):
                # planned/未知 → fail-closed rejected
                rejected_msg = rejected_msg or _single_line_truncate(
                    f"{result.vector_doc_id}: 非 desired 收敛终态 {result.status}"
                )
            elif result.status == "skipped" and result.result_code != RESULT_CODE_UNCHANGED_SKIPPED:
                # duplicate_superseded/upsert_skipped 不作为正式 desired 成功
                rejected_msg = rejected_msg or _single_line_truncate(
                    f"{result.vector_doc_id}: desired 落选 {result.result_code}"
                )
        if rejected_msg is not None:
            return _DesiredBlock(
                status="rejected",
                result_code="scope_desired_rejected",
                error_type="scope_desired_rejected",
                error_message=rejected_msg,
                counts=counts,
                retryable=False,
            )
        if failed_msg is not None:
            return _DesiredBlock(
                status="failed",
                result_code="scope_desired_failed",
                error_type="scope_desired_failed",
                error_message=failed_msg,
                counts=counts,
                retryable=failed_retryable,
            )
        return None

    @staticmethod
    def _summarize(results: list[ExecutionResult]) -> _Counts:
        """统计 desired 计数（embedding/metadata_only/unchanged）。"""

        embedding = sum(1 for r in results if r.embedding_performed)
        metadata_only = sum(1 for r in results if r.metadata_only)
        unchanged = sum(1 for r in results if r.result_code == RESULT_CODE_UNCHANGED_SKIPPED)
        return _Counts(embedding=embedding, metadata_only=metadata_only, unchanged=unchanged)

    @staticmethod
    def _desired_disable_count(results: list[ExecutionResult]) -> int:
        """统计 desired 中 Eligibility 逻辑下线（targets_disabled）的 disabled_ids 数。"""

        return sum(
            len(r.disabled_ids)
            for r in results
            if r.result_code == RESULT_CODE_TARGETS_DISABLED and r.status == "committed"
        )

    def _receipt(
        self,
        request: ScopeHandoffRequestView,
        *,
        status: str,
        result_code: str,
        error_type: str | None = None,
        error_message: str | None = None,
        counts: _Counts | None = None,
        desired_results: list[ExecutionResult] | None = None,
        stale_vector_doc_ids: tuple[str, ...] = (),
        disable_count: int = 0,
        retryable: bool = False,
    ) -> ScopeHandoffReceiptView:
        """组装上游 Receipt（identity/token/payload_hash/scope 原样回填）。"""

        counts = counts or _Counts()
        item_results = tuple(
            (result.vector_doc_id, result.result_code)
            for result in (desired_results or [])
            if result.status in {"committed", "skipped"}
        )
        return self._receipt_sink.make_receipt(
            request_id=request.request_id,
            task_id=request.task_id,
            batch_id=request.batch_id,
            attempt_token=request.attempt_token,
            payload_hash=request.payload_hash,
            impact_scope=request.impact_scope,
            status=status,
            result_code=result_code,
            desired_results=item_results,
            stale_vector_doc_ids=stale_vector_doc_ids,
            desired_count=len(request.contracts),
            embedding_count=counts.embedding,
            metadata_only_count=counts.metadata_only,
            disable_count=disable_count,
            unchanged_count=counts.unchanged,
            retryable=retryable,
            error_type=error_type,
            error_message=error_message,
        )


# ---------------------------------------------------------------------------
# 内部聚合类型
# ---------------------------------------------------------------------------
class _Counts:
    __slots__ = ("embedding", "metadata_only", "unchanged")

    def __init__(
        self,
        *,
        embedding: int = 0,
        metadata_only: int = 0,
        unchanged: int = 0,
    ) -> None:
        self.embedding = embedding
        self.metadata_only = metadata_only
        self.unchanged = unchanged


class _DesiredBlock:
    __slots__ = (
        "counts",
        "error_message",
        "error_type",
        "result_code",
        "retryable",
        "status",
    )

    def __init__(
        self,
        *,
        status: str,
        result_code: str,
        error_type: str,
        error_message: str,
        counts: _Counts,
        retryable: bool,
    ) -> None:
        self.status = status
        self.result_code = result_code
        self.error_type = error_type
        self.error_message = error_message
        self.counts = counts
        self.retryable = retryable


__all__ = [
    "ScopeConsumer",
    "ScopeHandoffReceiptView",
    "ScopeHandoffRequestView",
    "ScopeReceiptSink",
    "ScopeVectorStatePort",
    "ScopeView",
]
