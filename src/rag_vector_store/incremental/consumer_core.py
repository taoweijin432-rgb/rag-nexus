"""ContractConsumerCore：正式增量消费判定与执行核心。

核心输入是**标准 Contract 输入 / Disable 输入**，不再依赖 V15 Work Item 行模型：

    标准 Contract 输入 / Disable 输入 → ContractConsumerCore → ExecutionResult
    （Scope Handoff → ScopeConsumer → 本核心）

职责边界：
    - 只实现「Contract 输入 → 差分判定 → 调用既有向量能力 → 结构化结果」；
    - 不实现任何任务状态持久化：不新增 StateStorePort、不连接 MySQL、
      不写公共状态表；
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
from typing import Any, Protocol

from ..core.adapters import VectorConsumerContractAdapter
from ..core.hash_state import read_hash_field
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
    CONSUMER_ACTION_DISABLE_TARGETS,
    CONSUMER_ACTION_EMBEDDING_UPSERT,
    CONSUMER_ACTION_FETCH_EXISTING,
    CONSUMER_ACTION_METADATA_ONLY_UPDATE,
    CONSUMER_ACTION_SKIP_UNCHANGED,
    ERROR_CODE_DISABLE_INFRA_FAILED,
    ERROR_CODE_DOWNSTREAM_RESULT_MISMATCH,
    ERROR_CODE_ELIGIBILITY_DISABLE_FAILED,
    ERROR_CODE_EXISTING_QUERY_FAILED,
    ERROR_CODE_EXISTING_STATE_INVALID,
    ERROR_CODE_METADATA_ONLY_INCONSISTENT,
    ERROR_CODE_METADATA_UPDATE_INFRA_FAILED,
    ERROR_CODE_MODEL_SPACE_MISMATCH,
    ERROR_CODE_UPSERT_CONTRACT_FAILED,
    ERROR_CODE_UPSERT_INFRA_FAILED,
    RESULT_CODE_EMBEDDED,
    RESULT_CODE_METADATA_UPDATED,
    RESULT_CODE_TARGETS_DISABLED,
    RESULT_CODE_UNCHANGED_SKIPPED,
    RESULT_CODE_UPSERT_SKIPPED,
    ConsumerOperation,
    ExecutionResult,
)


class VectorMutationPort(Protocol):
    """Consumer 依赖的最小向量能力面（生产注入 ``VectorIngestionService``）。

    刻意不含物理删除方法；Embedding Worker 藏于 ``upsert_chunks`` 编排之内，
    核心不直接感知 Embedding 实现。
    """

    def upsert_chunks(
        self, chunks: Sequence[Any]
    ) -> VectorIngestionResult:
        """正式 Chunk 向量化并写入 Milvus。"""
        ...

    def update_metadata(
        self, chunks: Sequence[Any]
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
class ContractUpsertIntent:
    """标准 upsert 意图：调用方已完成 Contract 适配（含 Eligibility 预判）。

    ``contract.vector_doc_id`` 即目标主键；chunk_text 已由上游保证原样透传，
    本核心不改写、不重算哈希。
    """

    contract: Any
    eligibility_skip_reason: str | None = None

    @property
    def vector_doc_id(self) -> str:
        return self.contract.vector_doc_id


@dataclass(frozen=True)
class ContractDisableInput:
    """标准逻辑下线意图：保留实体与向量，仅置 is_active=false。

    ``target_vector_doc_ids`` 为完整待下线集合（含主 id，去重且保持顺序）；
    留空时退化为仅主 id。物理删除被端口形状排除，永不发生。
    """

    vector_doc_id: str
    target_vector_doc_ids: tuple[str, ...] = ()


#: 核心入口接受的输入联合类型。
ContractConsumerInput = ContractUpsertIntent | ContractDisableInput


class _AttributeContractAdapter:
    """默认兼容适配器：只读取标准 Contract 属性，不绑定任何领域类型。"""

    def vector_doc_id(self, value: object) -> str:
        result = getattr(value, "vector_doc_id", None)
        if not isinstance(result, str) or not result:
            raise TypeError("Contract 缺少合法 vector_doc_id")
        return result

    def content_hash(self, value: object) -> str:
        return value.content_hash  # type: ignore[attr-defined]

    def metadata_hash(self, value: object) -> str:
        return value.metadata_hash  # type: ignore[attr-defined]

    def source_hash(self, value: object) -> str | None:
        source = getattr(value, "source", None)
        result = getattr(source, "source_hash", None)
        return result if isinstance(result, str) else None

    def skip_reason(self, value: object) -> str | None:
        return None


class _AdapterEligibilityBinding:
    """把可选 Eligibility 函数和领域 Adapter 绑定到原有单一回调槽。"""

    def __init__(
        self,
        adapter: VectorConsumerContractAdapter,
        callback: Callable[[Any], str | None] | None,
    ) -> None:
        self.adapter = adapter
        self.callback = callback

    def __call__(self, value: Any) -> str | None:
        if self.callback is not None:
            return self.callback(value)
        return self.adapter.skip_reason(value)


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


class _DownstreamResultMismatch(ValueError):
    """下游结果无法与本次请求一一对应。"""


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    """按出现顺序去重。"""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


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

    try:
        content_hash = read_hash_field(record, "content_hash", required=True)
        metadata_hash = read_hash_field(record, "metadata_hash", required=True)
        source_hash = read_hash_field(record, "source_hash", required=False)
    except ValueError as exc:
        raise _ExistingStateError(str(exc)) from exc

    is_active = flat.get("is_active")
    if not isinstance(is_active, bool):
        raise _ExistingStateError(
            f"is_active 必须是布尔值，收到 {type(is_active).__name__}"
        )
    return _ExistingVectorState(
        vector_doc_id=vector_doc_id,
        content_hash=content_hash,
        metadata_hash=metadata_hash,
        source_hash=source_hash,
        is_active=is_active,
    )


class ContractConsumerCore:
    """通用增量消费核心（纯业务、无状态持久化、Work Item 无关）。

    参数：
        ``vector_store``：满足 :class:`VectorMutationPort` 的向量能力入口。
        ``eligibility_skip_of``：可选的 Eligibility 预判函数，把一条正式 Contract 映射为
          skip reason（``None`` 表示允许 Embedding）。正式范围 Consumer 经
          :meth:`process_contracts` 调用时内部复用该函数，避免调用方复制 Eligibility 规则；
          生产注入 ``EmbeddingEligibilityValidator().get_skip_reason``。

    入口：
        :meth:`process` 接收标准 Contract/Disable 输入序列。批内
        ``vector_doc_id`` 必须唯一，重复输入会在任何写入前 fail-closed。

        :meth:`process_contracts` 接收正式 ``ExpertChunkEmbeddingInput`` 序列：内部对每条
        调用 ``eligibility_skip_of`` 得到 skip reason，包成 :class:`ContractUpsertIntent`
        后委托 :meth:`process`。不携带 stale（范围 Consumer 在收敛后单独计算并下线 stale）。
    """

    def __init__(
        self,
        *,
        vector_store: VectorMutationPort,
        eligibility_skip_of: Callable[[Any], str | None] | None = None,
        contract_adapter: VectorConsumerContractAdapter | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._eligibility_skip_of = (
            _AdapterEligibilityBinding(contract_adapter, eligibility_skip_of)
            if contract_adapter is not None
            else eligibility_skip_of
        )

    @property
    def _contract_adapter(self) -> VectorConsumerContractAdapter:
        """读取本实例绑定的 Adapter；默认使用兼容的属性访问适配器。"""

        callback = self._eligibility_skip_of
        adapter = getattr(callback, "adapter", None)
        return adapter if adapter is not None else _AttributeContractAdapter()

    # ------------------------------------------------------------------
    # 批量入口
    # ------------------------------------------------------------------
    def process(
        self,
        inputs: Sequence[ContractConsumerInput],
    ) -> list[ExecutionResult]:
        """按输入顺序消费一批标准输入并返回逐条执行结果。"""

        if not isinstance(inputs, Sequence):
            raise TypeError("Consumer 输入必须是 ContractConsumerInput 序列")
        items = list(inputs)
        if any(
            not isinstance(item, (ContractUpsertIntent, ContractDisableInput))
            for item in items
        ):
            raise TypeError(
                "Consumer 输入只能是 ContractUpsertIntent 或 ContractDisableInput"
            )
        vector_doc_ids = [item.vector_doc_id for item in items]
        if any(not isinstance(value, str) or not value.strip() for value in vector_doc_ids):
            raise ValueError("Consumer 输入必须包含非空 vector_doc_id")
        if len(vector_doc_ids) != len(set(vector_doc_ids)):
            raise ValueError("Consumer 批次不能包含重复 vector_doc_id")
        return [self._process_one(item) for item in items]

    def process_contracts(
        self,
        contracts: Sequence[Any],
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

        stale 由范围 Consumer 在 desired 全部收敛后单独计算并下线
        （见 :mod:`scope_consumer`）。
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
        return self.process(intents)

    # ------------------------------------------------------------------
    # 单条处理
    # ------------------------------------------------------------------
    def _process_one(
        self,
        item: ContractConsumerInput,
    ) -> ExecutionResult:
        if isinstance(item, ContractUpsertIntent):
            return self._handle_upsert_like(
                contract=item.contract,
                op="upsert",
                eligibility_skip_reason=item.eligibility_skip_reason,
                actions=[],
            )
        return self._handle_disable(item)

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
        try:
            if not isinstance(query, VectorQueryResult):
                raise _DownstreamResultMismatch(
                    "get_vectors 返回值不是 VectorQueryResult"
                )
            record_ids = [
                row.get("vector_doc_id")
                for row in query.records
            ]
            if any(not isinstance(value, str) for value in record_ids):
                raise _DownstreamResultMismatch(
                    "get_vectors records 缺少合法 vector_doc_id"
                )
            returned_ids = set(record_ids)
            missing_ids = set(query.missing)
            if returned_ids.union(missing_ids) != {vector_doc_id}:
                raise _DownstreamResultMismatch(
                    "get_vectors 返回的 ID 集合与请求不一致"
                )
            if returned_ids.intersection(missing_ids):
                raise _DownstreamResultMismatch(
                    "get_vectors records 与 missing 存在重复 ID"
                )
            record = query.records[0] if query.records else None
        except _DownstreamResultMismatch as error:
            return None, ExecutionResult(
                vector_doc_id=vector_doc_id,
                operation=operation,
                status="failed",
                reason=f"existing_query_result_mismatch：{error}",
                error_code=ERROR_CODE_DOWNSTREAM_RESULT_MISMATCH,
                retryable=True,
            )
        return record, None

    # ------------------------------------------------------------------
    # upsert 形态业务主体（普通 upsert 与 metadata_update 合成路径共用）
    # ------------------------------------------------------------------
    def _handle_upsert_like(
        self,
        *,
        contract: Any,
        op: ConsumerOperation,
        eligibility_skip_reason: str | None,
        actions: list[str],
    ) -> ExecutionResult:
        """差分判定与执行主体：Eligibility → 读取已存 → 差分 → 执行。

        Contract 适配已由调用方完成，
        此处登记 adapt_upsert 动作以保持执行轨迹完整。
        """

        adapter = self._contract_adapter
        vid = adapter.vector_doc_id(contract)
        actions.append(CONSUMER_ACTION_ADAPT_UPSERT)

        # Eligibility 不通过 → 逻辑下线目标集合（不是 skip、不 Embedding）。
        if eligibility_skip_reason is not None:
            return self._execute_eligibility_disable(
                vector_doc_id=vid,
                op=op,
                skip_reason=eligibility_skip_reason,
                actions=tuple(actions),
            )

        record, failure = self._load_record(vid, op)
        if failure is not None:
            return failure.model_copy(update={"actions": tuple(actions)})
        if record is None:
            existing_state = None
        else:
            try:
                existing_state = _extract_existing_state(record)
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
            existing_state,
            adapter.content_hash(contract),
            adapter.metadata_hash(contract),
            adapter.source_hash(contract),
        )
        result = self._execute_text_plan(
            contract=contract,
            mode=mode,
            actions=actions,
            op=op,
        )
        return result

    @staticmethod
    def _decide_text_mode(
        state: _ExistingVectorState | None,
        content_hash: str,
        metadata_hash: str,
        source_hash: str | None,
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
        contract: Any,
        mode: str,
        actions: list[str],
        op: ConsumerOperation,
    ) -> ExecutionResult:
        """执行三 Hash 差分计划。"""

        if mode == CONSUMER_ACTION_SKIP_UNCHANGED:
            return ExecutionResult(
                vector_doc_id=self._contract_adapter.vector_doc_id(contract),
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
                    vector_doc_id=self._contract_adapter.vector_doc_id(contract),
                    operation=op,
                    status="failed",
                    reason=f"metadata_update_infra_failed：{error}",
                    error_code=ERROR_CODE_METADATA_UPDATE_INFRA_FAILED,
                    retryable=True,
                    actions=tuple(actions),
                )
            if outcome == "updated":
                return ExecutionResult(
                    vector_doc_id=self._contract_adapter.vector_doc_id(contract),
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
                    vector_doc_id=self._contract_adapter.vector_doc_id(contract),
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
            if outcome == "result_mismatch":
                return ExecutionResult(
                    vector_doc_id=self._contract_adapter.vector_doc_id(contract),
                    operation=op,
                    status="failed",
                    reason="metadata_update_result_mismatch：下游返回结果无法与请求对齐",
                    error_code=ERROR_CODE_DOWNSTREAM_RESULT_MISMATCH,
                    retryable=True,
                    actions=tuple(actions),
                )
            # outcome == "inconsistent"：确定性不一致 → rejected。
            return ExecutionResult(
                vector_doc_id=self._contract_adapter.vector_doc_id(contract),
                operation=op,
                status="rejected",
                reason="metadata_only_inconsistent：与已存实体的主键/正文/hash 不一致",
                error_code=ERROR_CODE_METADATA_ONLY_INCONSISTENT,
                actions=tuple(actions),
            )

        actions.append(CONSUMER_ACTION_EMBEDDING_UPSERT)
        return self._run_embedding_upsert(contract=contract, actions=tuple(actions), op=op)

    def _run_metadata_only(self, contract: Any) -> str:
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
        try:
            entry = self._single_ingestion_result(
                result,
                self._contract_adapter.vector_doc_id(contract),
            )
        except _DownstreamResultMismatch:
            return "result_mismatch"
        if entry.status == "metadata_updated":
            return "updated"
        if entry.status == "failed" and entry.recommendation == "call_upsert":
            return "gate_blocked"
        return "inconsistent"

    def _run_embedding_upsert(
        self,
        *,
        contract: Any,
        actions: tuple[str, ...],
        op: ConsumerOperation,
    ) -> ExecutionResult:
        """全量 upsert（新 Chunk / 正文变化）。"""

        try:
            result = self._vector_store.upsert_chunks([contract])
        except VectorContractError as error:
            return ExecutionResult(
                vector_doc_id=self._contract_adapter.vector_doc_id(contract),
                operation=op,
                status="rejected",
                reason=f"upsert_contract_failed：{error}",
                error_code=ERROR_CODE_UPSERT_CONTRACT_FAILED,
                actions=actions,
            )
        except VectorIngestionError as error:
            return ExecutionResult(
                vector_doc_id=self._contract_adapter.vector_doc_id(contract),
                operation=op,
                status="failed",
                reason=f"upsert_infra_failed：{error}",
                error_code=ERROR_CODE_UPSERT_INFRA_FAILED,
                retryable=True,
                actions=actions,
            )
        try:
            entry = self._single_ingestion_result(
                result,
                self._contract_adapter.vector_doc_id(contract),
            )
        except _DownstreamResultMismatch as error:
            return ExecutionResult(
                vector_doc_id=self._contract_adapter.vector_doc_id(contract),
                operation=op,
                status="failed",
                reason=f"upsert_result_mismatch：{error}",
                error_code=ERROR_CODE_DOWNSTREAM_RESULT_MISMATCH,
                retryable=True,
                actions=actions,
            )
        if entry.status == "embedded":
            return ExecutionResult(
                vector_doc_id=self._contract_adapter.vector_doc_id(contract),
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
                vector_doc_id=self._contract_adapter.vector_doc_id(contract),
                operation=op,
                status="skipped",
                result_code=RESULT_CODE_UPSERT_SKIPPED,
                reason=f"upsert_skipped：{entry.error_message or 'eligibility 跳过'}",
                actions=actions,
            )
        return ExecutionResult(
            vector_doc_id=self._contract_adapter.vector_doc_id(contract),
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
        op: ConsumerOperation,
        skip_reason: str,
        actions: tuple[str, ...],
    ) -> ExecutionResult:
        """Eligibility 不通过 → 当前目标逻辑下线（正确性要求，非 skip）。"""

        targets = [vector_doc_id]
        disable_actions = actions + (CONSUMER_ACTION_DISABLE_TARGETS,)
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
                actions=disable_actions,
            )
        try:
            self._validate_disable_result(deletion, len(targets))
        except _DownstreamResultMismatch as error:
            return ExecutionResult(
                vector_doc_id=vector_doc_id,
                operation=op,
                status="failed",
                reason=f"eligibility_disable_result_mismatch：{error}",
                error_code=ERROR_CODE_DOWNSTREAM_RESULT_MISMATCH,
                retryable=True,
                actions=disable_actions,
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
            actions=disable_actions,
            disabled_ids=tuple(targets),
        )

    # ------------------------------------------------------------------
    # disable：最小输入契约
    # ------------------------------------------------------------------
    def _handle_disable(
        self,
        item: ContractDisableInput,
    ) -> ExecutionResult:
        """disable 主体：按给定 id 集合调用逻辑下线（无物理删除）。"""

        ids = tuple(_dedupe_keep_order([item.vector_doc_id, *item.target_vector_doc_ids]))
        actions = [CONSUMER_ACTION_DISABLE_TARGETS]
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
        try:
            self._validate_disable_result(deletion, len(ids))
        except _DownstreamResultMismatch as error:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation="disable",
                status="failed",
                reason=f"disable_result_mismatch：{error}",
                error_code=ERROR_CODE_DOWNSTREAM_RESULT_MISMATCH,
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

    @staticmethod
    def _single_ingestion_result(
        result: object,
        expected_vector_doc_id: str,
    ) -> Any:
        """校验单条 upsert/update_metadata 返回结果。"""

        if not isinstance(result, VectorIngestionResult):
            raise _DownstreamResultMismatch(
                "下游返回值不是 VectorIngestionResult"
            )
        if len(result.results) != 1:
            raise _DownstreamResultMismatch(
                f"期望 1 条结果，实际 {len(result.results)} 条"
            )
        entry = result.results[0]
        if entry.vector_doc_id != expected_vector_doc_id:
            raise _DownstreamResultMismatch(
                "返回 vector_doc_id 与请求不一致"
            )
        return entry

    @staticmethod
    def _validate_disable_result(
        result: object,
        expected_requested_count: int,
    ) -> VectorDeletionResult:
        """校验逻辑下线结果的请求数量。"""

        if not isinstance(result, VectorDeletionResult):
            raise _DownstreamResultMismatch(
                "下游返回值不是 VectorDeletionResult"
            )
        if result.requested_count != expected_requested_count:
            raise _DownstreamResultMismatch(
                "requested_count 与请求数量不一致"
            )
        return result


__all__ = [
    "ContractConsumerCore",
    "ContractConsumerInput",
    "ContractDisableInput",
    "ContractUpsertIntent",
    "VectorMutationPort",
]
