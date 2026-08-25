"""ContractConsumerCore：Work Item 无关的增量消费判定与执行核心（D2A）。

核心输入是**标准 Contract 输入 / Disable 输入**，不再依赖 V15 Work Item 行模型：

    标准 Contract 输入 / Disable 输入 → ContractConsumerCore → ExecutionResult
    （Work Item 行模型 → V15WorkItemConsumer 兼容入口 → 本核心）

职责边界：
    - 只实现「Contract 输入 → 差分判定 → 调用既有向量能力 → 结构化结果」；
    - 不实现任何任务状态持久化：不新增 StateStorePort、不连接 MySQL、
      不写 ``v15_vectorization_work_item`` 或公共状态表；
    - 不做 retry / lease / requeue（后续唯一状态表适配层的职责）；
    - 不感知 Embedding 实现（藏于 ``upsert_chunks`` 编排内）、不做物理删除、
      不生成 chunk_text。

metadata-only 安全门失败语义（D3A 修正，对齐修订版计划 §4.4）：
    ``update_metadata`` 因旧向量 model_space / embedding_dim 与正式配置不一致而
    安全门失败时，本核心 **确定性拒绝**：

    - 不调用 Embedding、不调用 upsert_chunks、不修改 Milvus 任何数据；
    - status=rejected、error_code=model_space_mismatch、retryable=False、
      fallback_used=False；
    - 绝不允许静默重新 Embedding、复用不匹配旧向量或写占位向量——该类失败
      属确定性模型空间错误，应进入重试/DLQ 由人工介入，而不是悄悄重嵌。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from expert_contracts.models import ExpertChunkEmbeddingInput

from ..ingestion.errors import (
    VectorContractError,
    VectorIngestionError,
)
from ..ingestion.models import (
    VectorDeletionResult,
    VectorIngestionResult,
    VectorQueryResult,
)
from .consumer_models import (
    CONSUMER_ACTION_ADAPT_UPSERT,
    CONSUMER_ACTION_DISABLE_STALE_TARGETS,
    CONSUMER_ACTION_DISABLE_TARGETS,
    CONSUMER_ACTION_EMBEDDING_UPSERT,
    CONSUMER_ACTION_FETCH_EXISTING,
    CONSUMER_ACTION_METADATA_ONLY_UPDATE,
    CONSUMER_ACTION_SKIP_UNCHANGED,
    CONSUMER_ACTION_SYNTHESIZE_FROM_STORED,
    ERROR_CODE_CONTRACT_ADAPTATION_FAILED,
    ERROR_CODE_DISABLE_INFRA_FAILED,
    ERROR_CODE_ELIGIBILITY_DISABLE_FAILED,
    ERROR_CODE_EXISTING_QUERY_FAILED,
    ERROR_CODE_EXISTING_STATE_INVALID,
    ERROR_CODE_METADATA_ONLY_INCONSISTENT,
    ERROR_CODE_METADATA_STORED_TEXT_MISSING,
    ERROR_CODE_METADATA_TARGET_MISSING,
    ERROR_CODE_METADATA_UPDATE_INFRA_FAILED,
    ERROR_CODE_MODEL_SPACE_MISMATCH,
    ERROR_CODE_STALE_DISABLE_FAILED,
    ERROR_CODE_STALE_SOURCE_EVENT_ID,
    ERROR_CODE_UPSERT_CONTRACT_FAILED,
    ERROR_CODE_UPSERT_INFRA_FAILED,
    RESULT_CODE_DUPLICATE_SUPERSEDED,
    RESULT_CODE_EMBEDDED,
    RESULT_CODE_METADATA_UPDATED,
    RESULT_CODE_TARGETS_DISABLED,
    RESULT_CODE_UNCHANGED_SKIPPED,
    RESULT_CODE_UPSERT_SKIPPED,
    ConsumerOperation,
    ExecutionResult,
)


class SourceEventOrderView(Protocol):
    """只读的事件顺序查询缝隙（非 StateStorePort，零写入）。

    生产实现由最终确认的唯一状态表适配层提供；测试使用 Fake。核心仅在输入
    携带 ``source_event_id`` 且本视图有记录时做旧事件拦截。
    """

    def last_applied_source_event_id(self, vector_doc_id: str) -> int | None:
        """返回该 vector_doc_id 最近一次成功应用的 source_event_id；未知为 None。"""
        ...


class VectorMutationPort(Protocol):
    """Consumer 依赖的最小向量能力面（生产注入 ``VectorIngestionService``）。

    刻意不含物理删除方法；Embedding Worker 藏于 ``upsert_chunks`` 编排之内，
    核心不直接感知 Embedding 实现。
    """

    def upsert_chunks(
        self, chunks: Sequence[ExpertChunkEmbeddingInput]
    ) -> VectorIngestionResult:
        """正式 Chunk 向量化并写入 Milvus。"""
        ...

    def update_metadata(
        self, chunks: Sequence[ExpertChunkEmbeddingInput]
    ) -> VectorIngestionResult:
        """metadata-only 更新（复用旧向量，零 Embedding）。"""
        ...

    def disable_vectors(
        self, vector_doc_ids: Sequence[str]
    ) -> VectorDeletionResult:
        """逻辑下线（保留实体与向量，仅置 is_active=false）。"""
        ...

    def get_vectors(self, vector_doc_ids: Sequence[str]) -> VectorQueryResult:
        """批量查询已存记录（顶层投影三个 hash；不含 embedding）。"""
        ...


# ---------------------------------------------------------------------------
# 标准 Contract 输入 / Disable 输入（核心的唯一入口形态）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdaptedContract:
    """适配产物：完整正式契约 + Eligibility 预判（None 表示允许 Embedding）。"""

    contract: ExpertChunkEmbeddingInput
    eligibility_skip_reason: str | None = None


@dataclass(frozen=True)
class ContractUpsertIntent:
    """标准 upsert 意图：调用方已完成 Contract 适配（含 Eligibility 预判）。

    ``contract.vector_doc_id`` 即目标主键；chunk_text 已由上游保证原样透传，
    本核心不改写、不重算哈希。
    """

    contract: ExpertChunkEmbeddingInput
    eligibility_skip_reason: str | None = None
    stale_vector_doc_ids: tuple[str, ...] = ()
    source_event_id: int | None = None

    @property
    def vector_doc_id(self) -> str:
        return self.contract.vector_doc_id


@dataclass(frozen=True)
class ContractMetadataUpdateIntent:
    """标准 metadata_update 意图：目标 id + 从已存实体构造 Contract 的工厂。

    Work Item 契约禁止携带 chunk_text，metadata-only 更新必须复用已存正文：
    核心先读取已存记录，再以 ``(stored_chunk_text, stored_content_hash)`` 调用
    :attr:`adapt_from_stored` 构造完整 Contract。工厂失败抛出
    :class:`ContractSynthesisError`（确定性拒绝），绝不伪造哈希或文本。
    """

    vector_doc_id: str
    adapt_from_stored: Callable[[str, str], AdaptedContract]
    stale_vector_doc_ids: tuple[str, ...] = ()
    source_event_id: int | None = None


@dataclass(frozen=True)
class ContractDisableInput:
    """标准逻辑下线意图：保留实体与向量，仅置 is_active=false。

    ``target_vector_doc_ids`` 为完整待下线集合（含主 id，去重且保持顺序）；
    留空时退化为仅主 id。物理删除被端口形状排除，永不发生。
    """

    vector_doc_id: str
    target_vector_doc_ids: tuple[str, ...] = ()
    source_event_id: int | None = None


@dataclass(frozen=True)
class PreRejectedInput:
    """调用方已确定性失败的输入透传载体。

    核心先执行旧事件拦截（保持既有判定顺序），未命中拦截时按携带的稳定
    error_code 原样产出 rejected 结果。
    """

    vector_doc_id: str
    operation: ConsumerOperation
    detail: str
    error_code: str
    source_event_id: int | None = None


#: 核心入口接受的输入联合类型。
ContractConsumerInput = (
    ContractUpsertIntent
    | ContractMetadataUpdateIntent
    | ContractDisableInput
    | PreRejectedInput
)


class ContractSynthesisError(ValueError):
    """metadata_update 工厂从已存实体构造 Contract 失败（确定性拒绝）。"""


def resolve_winner_indices(
    keys: Sequence[tuple[str, int | None]],
) -> dict[int, int]:
    """批内同 vector_doc_id 幂等裁决，返回 ``下标 -> 胜出下标`` 映射。

    规则：source_event_id 最高者胜（None 视为最低）；并列时取输入顺序靠后者
    ——同一事件重复投递时后到的覆盖先到的，与"最后写入获胜"一致。非胜出
    下标映射到其所属的最终胜者下标。兼容入口与核心共用本实现。
    """

    def _rank(index: int) -> tuple[int, int]:
        event = keys[index][1]
        return (event if event is not None else -1, index)

    best: dict[str, int] = {}
    for index, (vector_doc_id, _event) in enumerate(keys):
        current = best.get(vector_doc_id)
        if current is None or _rank(index) >= _rank(current):
            best[vector_doc_id] = index
    winner_indices = set(best.values())
    mapping: dict[int, int] = {}
    for index, (vector_doc_id, _event) in enumerate(keys):
        if index in winner_indices:
            mapping[index] = index
        else:
            mapping[index] = best[vector_doc_id]
    return mapping


def duplicate_superseded_result(
    *,
    vector_doc_id: str,
    operation: ConsumerOperation,
    winner_index: int | None,
) -> ExecutionResult:
    """构造批内落选者的幂等跳过结果（不执行任何动作）。"""

    detail = f"（输入序号 {winner_index}）" if winner_index is not None else ""
    return ExecutionResult(
        vector_doc_id=vector_doc_id,
        operation=operation,
        status="skipped",
        result_code=RESULT_CODE_DUPLICATE_SUPERSEDED,
        reason=(
            "duplicate_superseded：同 vector_doc_id 存在更新事件"
            f"{detail}，本条按幂等规则落选，不执行"
        ),
    )


@dataclass(frozen=True)
class _ExistingVectorState:
    """增量判定所需的已存实体轻量状态（严格 fail-closed 提取）。

    ``source_hash`` 允许为 ``None``：旧版行可能未写 source_hash，缺失时不
    fail-closed，而是交由 ``_decide_text_mode`` 路由到 metadata-only
    （content_hash 相同且 source_hash 缺失 → 收敛 previous state 的非向量更新，
    不误判为三 Hash 全同、不调用 Embedding）。``content_hash`` / ``metadata_hash``
    缺失仍 fail-closed。
    """

    vector_doc_id: str
    content_hash: str
    metadata_hash: str
    source_hash: str | None
    is_active: bool


class _ExistingStateError(ValueError):
    """已存实体轻量状态提取失败（缺 hash / is_active 非 bool 等，确定性）。"""


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    """按出现顺序去重。"""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def validation_error_summary(error: Exception) -> str:
    """提取 pydantic 校验错误的简明摘要（供拒绝原因透出）。"""

    parts = [
        f"{'.'.join(str(p) for p in item.get('loc', ()))}: {item.get('msg', '')}"
        for item in getattr(error, "errors", list)()
    ]
    return "；".join(part for part in parts if part) or str(error)


def _extract_existing_state(record: dict[str, object]) -> _ExistingVectorState:
    """从 get_vectors 返回记录中提取轻量状态（content/metadata 严格 fail-closed）。

    - ``content_hash`` / ``metadata_hash``：优先顶层投影字段，其次 metadata JSON
      内键；缺失或类型非法即抛错，交由调用方转 rejected；
    - ``source_hash``：优先顶层投影字段，其次 metadata JSON 内键；缺失时返回
      ``None``（历史行可能未写），但 **存在且类型非法** 仍 fail-closed
      （与 V15 仓储侧的存储语义一致）；
    - ``is_active``：V15 将其平铺在 metadata JSON 中；缺失或非布尔值一律抛错
      （与 V15 仓储侧 strict-bool 提取语义一致）。
    """

    vector_doc_id = record.get("vector_doc_id")
    if not isinstance(vector_doc_id, str) or not vector_doc_id:
        raise _ExistingStateError("记录缺少合法 vector_doc_id")
    metadata = record.get("metadata")
    flat: dict[str, object] = dict(metadata) if isinstance(metadata, dict) else {}

    def _hash_field(field: str) -> str:
        for source in (record, flat):
            value = source.get(field)
            if isinstance(value, str) and value:
                return value
        raise _ExistingStateError(f"已存记录缺少 {field}")

    def _optional_hash_field(field: str) -> str | None:
        for source in (record, flat):
            value = source.get(field)
            if value is None:
                continue
            if isinstance(value, str) and value:
                return value
            # 存在但类型非法（空字符串 / 非字符串）→ fail-closed，不静默当缺失。
            raise _ExistingStateError(
                f"已存记录 {field} 类型非法：{type(value).__name__}"
            )
        return None

    is_active = flat.get("is_active")
    if not isinstance(is_active, bool):
        raise _ExistingStateError(
            f"is_active 必须是布尔值，收到 {type(is_active).__name__}"
        )
    return _ExistingVectorState(
        vector_doc_id=vector_doc_id,
        content_hash=_hash_field("content_hash"),
        metadata_hash=_hash_field("metadata_hash"),
        source_hash=_optional_hash_field("source_hash"),
        is_active=is_active,
    )


class ContractConsumerCore:
    """通用增量消费核心（纯业务、无状态持久化、Work Item 无关）。

    参数：
        ``vector_store``：满足 :class:`VectorMutationPort` 的向量能力入口。
        ``event_order``：可选只读事件顺序视图，用于 source_event_id 旧事件拦截；
          未提供时不拦截（生产由后续唯一状态表适配层接入）。
        ``eligibility_skip_of``：可选的 Eligibility 预判函数，把一条正式 Contract 映射为
          skip reason（``None`` 表示允许 Embedding）。正式范围 Consumer 经
          :meth:`process_contracts` 调用时内部复用该函数，避免调用方复制 Eligibility 规则；
          生产注入 ``EmbeddingEligibilityValidator().get_skip_reason``。

    入口：
        :meth:`process` 接收标准 Contract/Disable 输入序列。批内先做幂等裁决：
        同 ``vector_doc_id`` 取 source_event_id 最高者（并列取输入靠后者），落选者
        记 skipped(duplicate_superseded)，胜者按输入顺序执行并逐条返回结果。

        :meth:`process_contracts` 接收正式 ``ExpertChunkEmbeddingInput`` 序列：内部对每条
        调用 ``eligibility_skip_of`` 得到 skip reason，包成 :class:`ContractUpsertIntent`
        后委托 :meth:`process`。不携带 stale（范围 Consumer 在收敛后单独计算并下线 stale）。

        ``plan_only=True`` 时只生成执行计划（status=planned），除 ``get_vectors``
        只读查询外不触发任何写入动作。
    """

    def __init__(
        self,
        *,
        vector_store: VectorMutationPort,
        event_order: SourceEventOrderView | None = None,
        eligibility_skip_of: Callable[[ExpertChunkEmbeddingInput], str | None] | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._event_order = event_order
        self._eligibility_skip_of = eligibility_skip_of

    # ------------------------------------------------------------------
    # 批量入口
    # ------------------------------------------------------------------
    def process(
        self,
        inputs: Sequence[ContractConsumerInput],
        *,
        plan_only: bool = False,
    ) -> list[ExecutionResult]:
        """按输入顺序消费一批标准输入并返回逐条执行结果。"""

        keys = [(item.vector_doc_id, item.source_event_id) for item in inputs]
        winner_by_index = resolve_winner_indices(keys)
        results: list[ExecutionResult] = []
        for index, item in enumerate(inputs):
            winner_index = winner_by_index.get(index)
            if winner_index != index:
                results.append(
                    duplicate_superseded_result(
                        vector_doc_id=item.vector_doc_id,
                        operation=self._operation_of(item),
                        winner_index=(
                            winner_index if winner_index is not None else None
                        ),
                    )
                )
                continue
            results.append(self._process_one(item, plan_only=plan_only))
        return results

    def process_contracts(
        self,
        contracts: Sequence[ExpertChunkEmbeddingInput],
        *,
        plan_only: bool = False,
    ) -> list[ExecutionResult]:
        """面向正式 Contract 的入口：内部做 Eligibility 预判后委托 :meth:`process`。

        调用方传入正式 ``ExpertChunkEmbeddingInput`` 序列，本方法对每条调用构造时注入的
        ``eligibility_skip_of`` 得到 skip reason（``None`` 表示允许 Embedding），包成
        :class:`ContractUpsertIntent`（``eligibility_skip_reason`` 透传）后再走
        :meth:`process`。这样 Eligibility 规则只在构造时注入的单一函数中实现，
        调用方（如正式范围 Consumer）不复制 Eligibility 判定逻辑，也不漏判 active/
        verification_status 导致的 skip。

        fail-closed：若未注入 ``eligibility_skip_of``，本方法拒绝执行（避免静默按
        ``None`` 放行 ineligible desired）。兼容入口 :meth:`process` 不受此约束
        （调用方自行构造 intent 并预填 skip reason）。

        不携带 ``stale_vector_doc_ids``：范围 Consumer 在 desired 全部收敛后单独计算并
        下线 stale（见 :mod:`scope_consumer`）。
        """

        if self._eligibility_skip_of is None:
            raise RuntimeError(
                "process_contracts 要求 ContractConsumerCore 注入 eligibility_skip_of，"
                "未注入时拒绝执行以避免静默放行 ineligible desired"
            )
        intents: list[ContractConsumerInput] = []
        for contract in contracts:
            skip_reason = (
                self._eligibility_skip_of(contract)
                if self._eligibility_skip_of is not None
                else None
            )
            intents.append(
                ContractUpsertIntent(contract=contract, eligibility_skip_reason=skip_reason)
            )
        return self.process(intents, plan_only=plan_only)

    @staticmethod
    def _operation_of(item: ContractConsumerInput) -> ConsumerOperation:
        """推断输入对应的操作标签（仅用于落选结果的展示）。"""

        if isinstance(item, ContractUpsertIntent):
            return "upsert"
        if isinstance(item, ContractMetadataUpdateIntent):
            return "metadata_update"
        if isinstance(item, PreRejectedInput):
            return item.operation
        return "disable"

    # ------------------------------------------------------------------
    # 单条处理
    # ------------------------------------------------------------------
    def _process_one(
        self,
        item: ContractConsumerInput,
        *,
        plan_only: bool,
    ) -> ExecutionResult:
        stale_reason = self._stale_event_reason(item)
        if stale_reason is not None:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation=self._operation_of(item),
                status="rejected",
                reason=stale_reason,
                error_code=ERROR_CODE_STALE_SOURCE_EVENT_ID,
            )
        if isinstance(item, PreRejectedInput):
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation=item.operation,
                status="rejected",
                reason=item.detail,
                error_code=item.error_code,
            )
        if isinstance(item, ContractUpsertIntent):
            return self._handle_upsert_like(
                contract=item.contract,
                op="upsert",
                eligibility_skip_reason=item.eligibility_skip_reason,
                stale_ids=item.stale_vector_doc_ids,
                actions=[],
                plan_only=plan_only,
            )
        if isinstance(item, ContractMetadataUpdateIntent):
            return self._handle_metadata_update(item, plan_only=plan_only)
        return self._handle_disable(item, plan_only=plan_only)

    def _stale_event_reason(self, item: ContractConsumerInput) -> str | None:
        """旧 source_event_id 拦截；返回拒绝原因（None 表示放行）。"""

        incoming = item.source_event_id
        if self._event_order is None or incoming is None:
            return None
        last = self._event_order.last_applied_source_event_id(item.vector_doc_id)
        if last is None:
            return None
        if incoming < last:
            return (
                f"stale_source_event_id：incoming={incoming} < "
                f"last_applied={last}，旧事件不得覆盖新事件"
            )
        return None

    def _load_record(
        self,
        vector_doc_id: str,
        operation: ConsumerOperation,
    ) -> tuple[dict[str, object] | None, ExecutionResult | None]:
        """读取单条已存记录；基础设施失败时直接产出 failed 结果。"""

        try:
            query = self._vector_store.get_vectors([vector_doc_id])
        except VectorIngestionError as error:
            return None, ExecutionResult(
                vector_doc_id=vector_doc_id,
                operation=operation,
                status="failed",
                reason=f"existing_query_failed：{error}",
                error_code=ERROR_CODE_EXISTING_QUERY_FAILED,
                retryable=True,
            )
        record = next(
            (
                row
                for row in query.records
                if row.get("vector_doc_id") == vector_doc_id
            ),
            None,
        )
        return record, None

    # ------------------------------------------------------------------
    # upsert 形态业务主体（普通 upsert 与 metadata_update 合成路径共用）
    # ------------------------------------------------------------------
    def _handle_upsert_like(
        self,
        *,
        contract: ExpertChunkEmbeddingInput,
        op: ConsumerOperation,
        eligibility_skip_reason: str | None,
        stale_ids: tuple[str, ...],
        actions: list[str],
        plan_only: bool,
        preloaded_state: _ExistingVectorState | None = None,
        state_loaded: bool = False,
    ) -> ExecutionResult:
        """差分判定与执行主体：Eligibility → 读取已存 → 差分 → 执行计划。

        Contract 适配已由调用方完成（兼容入口的适配器端口或直接构造 intent），
        此处登记 adapt_upsert 动作以保持执行轨迹完整。
        ``preloaded_state``/``state_loaded`` 由 metadata_update 合成路径传入以
        复用已读取的已存状态（None 且 loaded=True 表示确认不存在），避免重复
        查询；普通 upsert 路径两者取默认值自行查询。
        """

        vid = contract.vector_doc_id
        actions.append(CONSUMER_ACTION_ADAPT_UPSERT)

        # Eligibility 不通过 → 逻辑下线目标集合（不是 skip、不 Embedding）。
        if eligibility_skip_reason is not None:
            return self._execute_eligibility_disable(
                vector_doc_id=vid,
                stale_ids=stale_ids,
                op=op,
                skip_reason=eligibility_skip_reason,
                actions=tuple(actions),
                plan_only=plan_only,
            )

        if not state_loaded:
            record, failure = self._load_record(vid, op)
            if failure is not None:
                return failure.model_copy(update={"actions": tuple(actions)})
            if record is None:
                preloaded_state = None
                state_loaded = True
            else:
                try:
                    preloaded_state = _extract_existing_state(record)
                    state_loaded = True
                except _ExistingStateError as error:
                    return ExecutionResult(
                        vector_doc_id=vid,
                        operation=op,
                        status="rejected",
                        reason=f"existing_state_invalid：{error}",
                        error_code=ERROR_CODE_EXISTING_STATE_INVALID,
                        actions=tuple(actions),
                    )
        actions.append(CONSUMER_ACTION_FETCH_EXISTING)

        mode = self._decide_text_mode(
            preloaded_state,
            contract.content_hash,
            contract.metadata_hash,
            contract.source.source_hash,
        )
        result = self._execute_text_plan(
            contract=contract,
            mode=mode,
            actions=actions,
            plan_only=plan_only,
            op=op,
        )
        if result.status in ("committed", "skipped"):
            result = self._apply_stale_disables(
                main_vector_doc_id=vid,
                stale_ids=stale_ids,
                main_result=result,
                plan_only=plan_only,
            )
        return result

    @staticmethod
    def _decide_text_mode(
        state: _ExistingVectorState | None,
        content_hash: str,
        metadata_hash: str,
        source_hash: str,
    ) -> str:
        """差分判定：embedding_upsert / metadata_only_update / skip_unchanged。

        三 Hash 动作矩阵（C1）：

        1. ``state`` 为 None（已存实体不存在）→ embedding_upsert；
        2. ``content_hash`` 变化 → embedding_upsert；
        3. ``content_hash`` 不变但已存 ``source_hash`` 缺失 → metadata-only
           （不误判为三 Hash 全同、不调用 Embedding，收敛 previous state）；
        4. ``content_hash`` 不变但 ``source_hash`` 或 ``metadata_hash`` 变化 →
           metadata-only（Embedding 调用次数为 0）；
        5. 三 Hash 全等且已存 active → skip_unchanged；
        6. 三 Hash 全等但已存 inactive、本次输入 active → metadata-only
           （重新激活守卫；逻辑下线只翻转 is_active、不重算已存 metadata_hash，
           故 hash 全等但仍需写入以恢复 active）。
        """

        if state is None:
            return CONSUMER_ACTION_EMBEDDING_UPSERT
        content_changed = state.content_hash != content_hash
        if content_changed:
            return CONSUMER_ACTION_EMBEDDING_UPSERT
        # 已存 source_hash 缺失：不能判定为三 Hash 全同，content_hash 不变时
        # 路由到 metadata-only（零 Embedding），收敛 previous state。
        if state.source_hash is None:
            return CONSUMER_ACTION_METADATA_ONLY_UPDATE
        source_changed = state.source_hash != source_hash
        metadata_changed = state.metadata_hash != metadata_hash
        if not source_changed and not metadata_changed and state.is_active:
            return CONSUMER_ACTION_SKIP_UNCHANGED
        if not source_changed and not metadata_changed and not state.is_active:
            # 重新激活守卫：三 Hash 全等但已存 inactive → metadata-only 恢复 active。
            return CONSUMER_ACTION_METADATA_ONLY_UPDATE
        return CONSUMER_ACTION_METADATA_ONLY_UPDATE

    def _execute_text_plan(
        self,
        *,
        contract: ExpertChunkEmbeddingInput,
        mode: str,
        actions: list[str],
        plan_only: bool,
        op: ConsumerOperation,
    ) -> ExecutionResult:
        """执行差分计划；plan_only 下只登记将要执行的动作。"""

        if plan_only:
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,
                status="planned",
                reason="plan_only：仅生成执行计划，未触发任何写入",
                actions=tuple(actions) + (mode,),
            )

        if mode == CONSUMER_ACTION_SKIP_UNCHANGED:
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,
                status="skipped",
                result_code=RESULT_CODE_UNCHANGED_SKIPPED,
                reason="hash_unchanged：三 hash 与已存实体一致且仍 active",
                actions=tuple(actions) + (mode,),
            )

        if mode == CONSUMER_ACTION_METADATA_ONLY_UPDATE:
            actions.append(mode)
            try:
                outcome = self._run_metadata_only(contract)
            except VectorIngestionError as error:
                return ExecutionResult(
                    vector_doc_id=contract.vector_doc_id,
                    operation=op,
                    status="failed",
                    reason=f"metadata_update_infra_failed：{error}",
                    error_code=ERROR_CODE_METADATA_UPDATE_INFRA_FAILED,
                    retryable=True,
                    actions=tuple(actions),
                )
            if outcome == "updated":
                return ExecutionResult(
                    vector_doc_id=contract.vector_doc_id,
                    operation=op,
                    status="committed",
                    result_code=RESULT_CODE_METADATA_UPDATED,
                    metadata_only=True,
                    reason="metadata_updated：仅元数据变化，复用旧向量，零 Embedding",
                    actions=tuple(actions),
                )
            if outcome == "gate_blocked":
                # D3A 修正：安全门失败（旧向量 model_space/dim 与正式配置不一致）
                # → 确定性拒绝。绝不降级重新 Embedding、绝不复用不匹配旧向量、
                # 绝不修改 Milvus；retryable=False（非基础设施抖动）。
                return ExecutionResult(
                    vector_doc_id=contract.vector_doc_id,
                    operation=op,
                    status="rejected",
                    reason=(
                        "model_space_mismatch：metadata-only 安全门失败，"
                        "已存向量模型空间/维度与正式配置不一致，"
                        "拒绝更新且不重新 Embedding（需全量重建通道处理）"
                    ),
                    error_code=ERROR_CODE_MODEL_SPACE_MISMATCH,
                    actions=tuple(actions),
                )
            # outcome == "inconsistent"：确定性不一致 → rejected。
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,
                status="rejected",
                reason="metadata_only_inconsistent：与已存实体的主键/正文/hash 不一致",
                error_code=ERROR_CODE_METADATA_ONLY_INCONSISTENT,
                actions=tuple(actions),
            )

        actions.append(CONSUMER_ACTION_EMBEDDING_UPSERT)
        return self._run_embedding_upsert(contract=contract, actions=tuple(actions), op=op)

    def _run_metadata_only(self, contract: ExpertChunkEmbeddingInput) -> str:
        """调用 update_metadata；返回 updated / gate_blocked / inconsistent。

        两类失败严格区分（对齐 VectorIngestionService 的分层语义）：

        - 一致性门禁（主键/content_hash/chunk_text 必须与已存完全一致）以
          raise ``VectorContractError`` 表达 → 确定性 rejected；
        - 安全门（model_space/embedding_dim）以 failed item +
          recommendation=call_upsert 返回 → 确定性 rejected
          （error_code=model_space_mismatch，见模块 docstring）；
        - 其余基础设施错误继续抛出，由调用方转 failed（可重试）。
        """

        try:
            result = self._vector_store.update_metadata([contract])
        except VectorContractError:
            return "inconsistent"
        entry = result.results[0]
        if entry.status == "metadata_updated":
            return "updated"
        if entry.status == "failed" and entry.recommendation == "call_upsert":
            return "gate_blocked"
        return "inconsistent"

    def _run_embedding_upsert(
        self,
        *,
        contract: ExpertChunkEmbeddingInput,
        actions: tuple[str, ...],
        op: ConsumerOperation,
    ) -> ExecutionResult:
        """全量 upsert（新 Chunk / 正文变化）。"""

        try:
            result = self._vector_store.upsert_chunks([contract])
        except VectorContractError as error:
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,
                status="rejected",
                reason=f"upsert_contract_failed：{error}",
                error_code=ERROR_CODE_UPSERT_CONTRACT_FAILED,
                actions=actions,
            )
        except VectorIngestionError as error:
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,
                status="failed",
                reason=f"upsert_infra_failed：{error}",
                error_code=ERROR_CODE_UPSERT_INFRA_FAILED,
                retryable=True,
                actions=actions,
            )
        entry = result.results[0]
        if entry.status == "embedded":
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,
                status="committed",
                result_code=RESULT_CODE_EMBEDDED,
                embedding_performed=True,
                reason="embedded：Embedding + upsert 成功",
                actions=actions,
            )
        if entry.status == "skipped":
            # 防御分支：正常链路已在 Eligibility 预判中拦截，不应触达。
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,
                status="skipped",
                result_code=RESULT_CODE_UPSERT_SKIPPED,
                reason=f"upsert_skipped：{entry.error_message or 'eligibility 跳过'}",
                actions=actions,
            )
        return ExecutionResult(
            vector_doc_id=contract.vector_doc_id,
            operation=op,
            status="failed",
            reason=f"upsert_unexpected_status：{entry.status}",
            error_code=ERROR_CODE_UPSERT_INFRA_FAILED,
            retryable=True,
            actions=actions,
        )

    def _execute_eligibility_disable(
        self,
        *,
        vector_doc_id: str,
        stale_ids: tuple[str, ...],
        op: ConsumerOperation,
        skip_reason: str,
        actions: tuple[str, ...],
        plan_only: bool,
    ) -> ExecutionResult:
        """Eligibility 不通过 → 目标集合逻辑下线（正确性要求，非 skip）。"""

        targets = _dedupe_keep_order([vector_doc_id, *stale_ids])
        planned_actions = actions + (CONSUMER_ACTION_DISABLE_TARGETS,)
        if plan_only:
            return ExecutionResult(
                vector_doc_id=vector_doc_id,
                operation=op,
                status="planned",
                reason=f"eligibility_disable：{skip_reason} → 逻辑下线目标集合",
                actions=planned_actions,
                disabled_ids=tuple(targets),
            )
        try:
            deletion = self._vector_store.disable_vectors(targets)
        except VectorIngestionError as error:
            return ExecutionResult(
                vector_doc_id=vector_doc_id,
                operation=op,
                status="failed",
                reason=f"eligibility_disable_failed：{skip_reason}；{error}",
                error_code=ERROR_CODE_ELIGIBILITY_DISABLE_FAILED,
                retryable=True,
                actions=planned_actions,
            )
        return ExecutionResult(
            vector_doc_id=vector_doc_id,
            operation=op,
            status="committed",
            result_code=RESULT_CODE_TARGETS_DISABLED,
            reason=(
                f"eligibility_disable：{skip_reason}；已下线 "
                f"{deletion.deleted_count} 条，未命中 {deletion.not_found_count} 条"
            ),
            actions=planned_actions,
            disabled_ids=tuple(targets),
        )

    # ------------------------------------------------------------------
    # metadata_update：取回已存原文后经工厂构造 Contract
    # ------------------------------------------------------------------
    def _handle_metadata_update(
        self,
        intent: ContractMetadataUpdateIntent,
        *,
        plan_only: bool,
    ) -> ExecutionResult:
        """metadata-update 主体：读旧实体 → 工厂构造完整 Contract → 复用主体判定。

        目标不存在时没有可 Embedding 的文本、无法回退全量 upsert，只能确定性
        rejected；工厂构造失败同样确定性 rejected（绝不伪造文本或哈希）。
        只有目标存在且构造成功时才进入共享差分主体。
        """

        record, failure = self._load_record(intent.vector_doc_id, "metadata_update")
        actions = [CONSUMER_ACTION_FETCH_EXISTING]
        if failure is not None:
            return failure.model_copy(update={"actions": tuple(actions)})
        if record is None:
            return ExecutionResult(
                vector_doc_id=intent.vector_doc_id,
                operation="metadata_update",
                status="rejected",
                reason=(
                    "metadata_target_missing：Milvus 无此 vector_doc_id，"
                    "metadata-only 更新缺少可复用的 chunk_text，无法构造 Contract"
                ),
                error_code=ERROR_CODE_METADATA_TARGET_MISSING,
                actions=tuple(actions),
            )
        try:
            state = _extract_existing_state(record)
        except _ExistingStateError as error:
            return ExecutionResult(
                vector_doc_id=intent.vector_doc_id,
                operation="metadata_update",
                status="rejected",
                reason=f"existing_state_invalid：{error}",
                error_code=ERROR_CODE_EXISTING_STATE_INVALID,
                actions=tuple(actions),
            )
        stored_text = record.get("chunk_text")
        if not isinstance(stored_text, str) or not stored_text.strip():
            return ExecutionResult(
                vector_doc_id=intent.vector_doc_id,
                operation="metadata_update",
                status="rejected",
                reason="metadata_stored_text_missing：已存实体缺少非空 chunk_text",
                error_code=ERROR_CODE_METADATA_STORED_TEXT_MISSING,
                actions=tuple(actions),
            )

        try:
            adapted = intent.adapt_from_stored(stored_text, state.content_hash)
        except ContractSynthesisError as error:
            return ExecutionResult(
                vector_doc_id=intent.vector_doc_id,
                operation="metadata_update",
                status="rejected",
                reason=f"contract_adaptation_failed：{error}",
                error_code=ERROR_CODE_CONTRACT_ADAPTATION_FAILED,
                actions=tuple(actions),
            )
        actions.append(CONSUMER_ACTION_SYNTHESIZE_FROM_STORED)

        return self._handle_upsert_like(
            contract=adapted.contract,
            op="metadata_update",
            eligibility_skip_reason=adapted.eligibility_skip_reason,
            stale_ids=intent.stale_vector_doc_ids,
            actions=actions,
            plan_only=plan_only,
            preloaded_state=state,
            state_loaded=True,
        )

    # ------------------------------------------------------------------
    # disable：最小输入契约
    # ------------------------------------------------------------------
    def _handle_disable(
        self,
        item: ContractDisableInput,
        *,
        plan_only: bool,
    ) -> ExecutionResult:
        """disable 主体：按给定 id 集合调用逻辑下线（无物理删除）。"""

        ids = tuple(_dedupe_keep_order([item.vector_doc_id, *item.target_vector_doc_ids]))
        actions = [CONSUMER_ACTION_DISABLE_TARGETS]
        if plan_only:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation="disable",
                status="planned",
                reason="plan_only：仅规划逻辑下线，不触发写入",
                actions=tuple(actions),
                disabled_ids=ids,
            )
        try:
            deletion = self._vector_store.disable_vectors(list(ids))
        except VectorIngestionError as error:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation="disable",
                status="failed",
                reason=f"disable_infra_failed：{error}",
                error_code=ERROR_CODE_DISABLE_INFRA_FAILED,
                retryable=True,
                actions=tuple(actions),
            )
        return ExecutionResult(
            vector_doc_id=item.vector_doc_id,
            operation="disable",
            status="committed",
            result_code=RESULT_CODE_TARGETS_DISABLED,
            reason=(
                f"disabled：逻辑下线 {deletion.deleted_count} 条"
                f"（幂等重放计入），未命中 {deletion.not_found_count} 条"
            ),
            actions=tuple(actions),
            disabled_ids=ids,
        )

    # ------------------------------------------------------------------
    # stale 子块：与主操作解耦的独立逻辑下线
    # ------------------------------------------------------------------
    def _apply_stale_disables(
        self,
        *,
        main_vector_doc_id: str,
        stale_ids: tuple[str, ...],
        main_result: ExecutionResult,
        plan_only: bool,
    ) -> ExecutionResult:
        """主操作收敛后独立处理 stale 子块；stale 失败不回滚主操作效果。

        主操作成功后若 stale 列表非空，则单独走逻辑下线；stale 下线失败时整体
        记 failed（可重试），reason 注明主操作已完成——重放时主路径因 hash 幂等
        自然收敛，仅补做 stale 下线。主操作为 skipped 且 stale 全部下线成功时，
        结果升级为 committed（存在实际写入效果，result_code 改记 targets_disabled）。
        """

        stales = [
            vid
            for vid in _dedupe_keep_order(stale_ids)
            if vid != main_vector_doc_id
        ]
        if not stales:
            return main_result
        planned_actions = main_result.actions + (CONSUMER_ACTION_DISABLE_STALE_TARGETS,)
        base_reason = main_result.reason or ""
        if plan_only:
            return main_result.model_copy(
                update={
                    "actions": planned_actions,
                    "reason": (
                        f"{base_reason}；stale 子块待下线: {stales}"
                        if base_reason
                        else f"stale 子块待下线: {stales}"
                    ),
                },
            )
        try:
            deletion = self._vector_store.disable_vectors(stales)
        except VectorIngestionError as error:
            # 显式重建以强制执行字段不变量（failed 必须携带 error_code、
            # 不得残留 result_code / embedding / metadata 标记）。
            return ExecutionResult(
                vector_doc_id=main_result.vector_doc_id,
                operation=main_result.operation,
                status="failed",
                reason=(
                    f"{base_reason or 'main_done'}；"
                    f"stale_disable_failed（主操作已完成，重放仅补 stale）: {error}"
                ),
                actions=planned_actions,
                disabled_ids=main_result.disabled_ids,
                error_code=ERROR_CODE_STALE_DISABLE_FAILED,
                retryable=True,
            )
        merged_disabled = tuple(
            _dedupe_keep_order([*main_result.disabled_ids, *stales])
        )
        stale_note = (
            f"stale 已下线 {deletion.deleted_count} 条，"
            f"未命中 {deletion.not_found_count} 条"
        )
        if main_result.status != "skipped":
            # 非 skip 主操作保持原状态，仅合并 reason / 动作 / disabled_ids。
            return main_result.model_copy(
                update={
                    "reason": f"{base_reason}；{stale_note}" if base_reason else stale_note,
                    "actions": planned_actions,
                    "disabled_ids": merged_disabled,
                },
            )
        # skip 升级为 committed：净效果是纯逻辑下线，分类码改记 targets_disabled。
        return ExecutionResult(
            vector_doc_id=main_result.vector_doc_id,
            operation=main_result.operation,
            status="committed",
            result_code=RESULT_CODE_TARGETS_DISABLED,
            reason=f"{base_reason}；{stale_note}" if base_reason else stale_note,
            actions=planned_actions,
            disabled_ids=merged_disabled,
        )


__all__ = [
    "AdaptedContract",
    "ContractConsumerCore",
    "ContractConsumerInput",
    "ContractDisableInput",
    "ContractMetadataUpdateIntent",
    "ContractSynthesisError",
    "ContractUpsertIntent",
    "PreRejectedInput",
    "SourceEventOrderView",
    "VectorMutationPort",
    "duplicate_superseded_result",
    "resolve_winner_indices",
    "validation_error_summary",
]
