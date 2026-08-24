"""通用 Work Item Consumer 的纯业务执行核心（Phase 3A）。

职责边界：
    只实现“Work Item 业务载荷 → 执行计划 → 调用既有向量能力 → 结构化结果”
    的业务编排，**不实现任何任务状态持久化**：

    - 不新增 StateStorePort，不定义任何状态存储接口；
    - 不连接 MySQL、不调用增量状态仓储、不写 ``v15_vectorization_work_item``；
    - 不写公共状态表，不允许双写两张状态表；
    - 不做 retry / lease / requeue（那是后续唯一状态表适配层的职责）；
    - 只返回结构化 :class:`ExecutionResult`（committed / skipped / rejected /
      failed / planned），由调用方决定状态流转。

复用边界：
    - Work Item → Contract 适配复用 Phase 2 薄适配器（本模块不重新组装、
      不重新生成 chunk_text，文本始终原样透传）；
    - Embedding 经 ``VectorMutationPort.upsert_chunks`` 内部编排复用既有
      Embedding Worker，Consumer 不直接感知 Embedding 实现；
    - metadata-only 更新复用 ``update_metadata``（零 Embedding，安全门失败时
      由本核心降级为全量 upsert）；
    - 逻辑下线复用 ``disable_vectors``（保留实体与向量，仅置 is_active=false，
      禁止物理删除）。

事件顺序拦截：
    通过可选的只读 :class:`SourceEventOrderView` 查询缝隙获取某 vector_doc_id
    最近已应用的 ``source_event_id``。它不是 StateStorePort：单方法、只读、
    零写入，生产环境由最终确认的唯一状态表适配层提供实现。比较规则：

    - ``last`` 与 ``incoming`` 任一为 None → 放行（无序信息可用时不拦截）；
    - ``incoming < last`` → 旧事件，返回 rejected(stale)，绝不覆盖新事件；
    - ``incoming == last`` → 视为同一事件重放，放行（由 hash 幂等判定收敛）；
    - ``incoming > last`` → 新事件，放行。

判定矩阵（operation=upsert / 合成后的 metadata_update 共用）：
    - Eligibility 不通过（is_active=false / verification_status=rejected）
      → 目标集合逻辑下线（不是 skip、不 Embedding）：本应下线的 active 向量
      若被跳过会继续参与召回，属于正确性问题；
    - 已存实体不存在 → Embedding + upsert（新 Chunk）；
    - content_hash 变化 → 重新 Embedding + upsert；
    - 仅 metadata_hash 变化 → metadata-only update（零 Embedding）；安全门
      失败（model_space/dim 不一致，recommendation=call_upsert）→ 自动降级
      为全量 upsert；
    - 两个 Hash 均未变化且已存实体仍 active → skip；
    - 两个 Hash 均未变化但已存实体 inactive、本次输入 active → 走 metadata-only
      更新重新激活。守卫原因：逻辑下线只翻转 ``is_active``、不重算已存的
      ``metadata_hash``，因此“存储 inactive + 双 hash 全等”是真实场景，
      按 skip 处理会漏掉重新激活。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

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
    CONSUMER_ACTION_ADAPT_DISABLE,
    CONSUMER_ACTION_ADAPT_UPSERT,
    CONSUMER_ACTION_DISABLE_STALE_TARGETS,
    CONSUMER_ACTION_DISABLE_TARGETS,
    CONSUMER_ACTION_EMBEDDING_UPSERT,
    CONSUMER_ACTION_FETCH_EXISTING,
    CONSUMER_ACTION_METADATA_ONLY_UPDATE,
    CONSUMER_ACTION_SKIP_UNCHANGED,
    CONSUMER_ACTION_SYNTHESIZE_FROM_STORED,
    ExecutionResult,
)
from .contract_adapter import (
    DisableWorkItem,
    WorkItemContractAdaptation,
    WorkItemContractAdapterError,
)
from .models import V15VectorizationWorkItem


class SourceEventOrderView(Protocol):
    """只读的事件顺序查询缝隙（非 StateStorePort，零写入）。

    生产实现由最终确认的唯一状态表适配层提供；测试使用 Fake。Consumer 仅在
    Work Item 携带 ``source_event_id`` 且本视图有记录时做旧事件拦截。
    """

    def last_applied_source_event_id(self, vector_doc_id: str) -> int | None:
        """返回该 vector_doc_id 最近一次成功应用的 source_event_id；未知为 None。"""
        ...


class WorkItemAdapterPort(Protocol):
    """Work Item → Contract 适配器端口（生产注入 Phase 2 薄适配器）。"""

    def adapt_upsert(
        self, work_item: V15VectorizationWorkItem
    ) -> WorkItemContractAdaptation:
        """upsert 行 → 完整 Contract 适配产物。"""
        ...

    def adapt_disable(
        self, work_item: V15VectorizationWorkItem
    ) -> DisableWorkItem:
        """disable 行 → 待下线 id 集合。"""
        ...


class VectorMutationPort(Protocol):
    """Consumer 依赖的最小向量能力面（生产注入 ``VectorIngestionService``）。

    刻意不含物理删除方法；Embedding Worker 藏于 ``upsert_chunks`` 编排之内，
    Consumer 不直接感知 Embedding 实现。
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
        """批量查询已存记录（顶层投影两个 hash；不含 embedding）。"""
        ...


@dataclass(frozen=True)
class _ExistingVectorState:
    """增量判定所需的已存实体轻量状态（严格 fail-closed 提取）。"""

    vector_doc_id: str
    content_hash: str
    metadata_hash: str
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


def _validation_summary(error: ValidationError) -> str:
    """提取 pydantic 校验错误的简明摘要。"""

    parts = [
        f"{'.'.join(str(p) for p in item.get('loc', ()))}: {item.get('msg', '')}"
        for item in error.errors()
    ]
    return "；".join(part for part in parts if part) or str(error)


class V15WorkItemConsumer:
    """通用 Work Item 增量消费核心（纯业务、无状态持久化）。

    参数：
        ``adapter``：满足 :class:`WorkItemAdapterPort` 的薄适配器。
        ``vector_store``：满足 :class:`VectorMutationPort` 的向量能力入口。
        ``event_order``：可选只读事件顺序视图，用于 source_event_id 旧事件拦截；
          未提供时不拦截（生产由后续唯一状态表适配层接入）。

    入口：
        :meth:`process` 接收 Work Item 行序列。批内先做幂等裁决：同
        ``vector_doc_id`` 取 source_event_id 最高者（并列取输入靠后者），落选者
        记 skipped(duplicate_superseded)，胜者按输入顺序执行并逐条返回结果。

        ``plan_only=True`` 时只生成执行计划（status=planned），除 ``get_vectors``
        只读查询外不触发任何写入动作。
    """

    def __init__(
        self,
        *,
        adapter: WorkItemAdapterPort,
        vector_store: VectorMutationPort,
        event_order: SourceEventOrderView | None = None,
    ) -> None:
        self._adapter = adapter
        self._vector_store = vector_store
        self._event_order = event_order

    # ------------------------------------------------------------------
    # 批量入口
    # ------------------------------------------------------------------
    def process(
        self,
        work_items: Sequence[V15VectorizationWorkItem],
        *,
        plan_only: bool = False,
    ) -> list[ExecutionResult]:
        """按输入顺序消费一批 Work Item 并返回逐条执行结果。"""

        winner_by_index = self._resolve_winners(work_items)
        results: list[ExecutionResult] = []
        for index, item in enumerate(work_items):
            winner_index = winner_by_index.get(index)
            if winner_index is None:
                results.append(self._suppressed_result(item, winner_index=None))
                continue
            if winner_index != index:
                results.append(self._suppressed_result(item, winner_index=winner_index))
                continue
            results.append(self._process_one(item, plan_only=plan_only))
        return results

    @staticmethod
    def _resolve_winners(
        work_items: Sequence[V15VectorizationWorkItem],
    ) -> dict[int, int]:
        """批内同 vector_doc_id 幂等裁决，返回 ``下标 -> 胜出下标`` 映射。

        规则：source_event_id 最高者胜（None 视为最低）；并列时取输入顺序
        靠后者——同一事件重复投递时后到的覆盖先到的，与“最后写入获胜”一致。
        非胜出下标映射到其所属的最终胜者下标。
        """

        def _rank(index: int) -> tuple[int, int]:
            event = work_items[index].source_event_id
            return (event if event is not None else -1, index)

        best: dict[str, int] = {}
        for index, item in enumerate(work_items):
            current = best.get(item.vector_doc_id)
            if current is None or _rank(index) >= _rank(current):
                best[item.vector_doc_id] = index
        winner_indices = set(best.values())
        mapping: dict[int, int] = {}
        for index, item in enumerate(work_items):
            if index in winner_indices:
                mapping[index] = index
            else:
                mapping[index] = best[item.vector_doc_id]
        return mapping

    @staticmethod
    def _suppressed_result(
        item: V15VectorizationWorkItem,
        *,
        winner_index: int | None,
    ) -> ExecutionResult:
        """构造批内落选者的幂等跳过结果（不执行任何动作）。"""

        detail = (
            f"（输入序号 {winner_index}）" if winner_index is not None else ""
        )
        return ExecutionResult(
            vector_doc_id=item.vector_doc_id,
            operation=item.operation,
            status="skipped",
            reason=(
                "duplicate_superseded：同 vector_doc_id 存在更新事件"
                f"{detail}，本条按幂等规则落选，不执行"
            ),
        )

    # ------------------------------------------------------------------
    # 单条处理
    # ------------------------------------------------------------------
    def _process_one(
        self,
        item: V15VectorizationWorkItem,
        *,
        plan_only: bool,
    ) -> ExecutionResult:
        stale_reason = self._stale_event_reason(item)
        if stale_reason is not None:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation=item.operation,
                status="rejected",
                reason=stale_reason,
            )
        if item.operation == "upsert":
            return self._handle_upsert_like(item, plan_only=plan_only)
        if item.operation == "metadata_update":
            return self._handle_metadata_update(item, plan_only=plan_only)
        return self._handle_disable(item, plan_only=plan_only)

    def _stale_event_reason(self, item: V15VectorizationWorkItem) -> str | None:
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
        operation: str,
    ) -> tuple[dict[str, object] | None, ExecutionResult | None]:
        """读取单条已存记录；基础设施失败时直接产出 failed 结果。"""

        try:
            query = self._vector_store.get_vectors([vector_doc_id])
        except VectorIngestionError as error:
            return None, ExecutionResult(
                vector_doc_id=vector_doc_id,
                operation=operation,  # type: ignore[arg-type]
                status="failed",
                reason=f"existing_query_failed：{error}",
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
    # upsert（以及合成后的 metadata_update 共用主体）
    # ------------------------------------------------------------------
    def _handle_upsert_like(
        self,
        item: V15VectorizationWorkItem,
        *,
        plan_only: bool,
        prefix_actions: tuple[str, ...] = (),
        preloaded_state: _ExistingVectorState | None = None,
        state_loaded: bool = False,
        operation_label: str | None = None,
    ) -> ExecutionResult:
        """upsert 形态业务主体：适配 → Eligibility → 差分 → 执行。

        ``preloaded_state``/``state_loaded`` 由 metadata_update 合成路径传入以
        复用已读取的已存状态（None 且 loaded=True 表示确认不存在），避免重复
        查询；普通 upsert 路径两者取默认值自行查询。
        """

        op = operation_label or item.operation
        actions = list(prefix_actions)

        try:
            adaptation = self._adapter.adapt_upsert(item)
        except WorkItemContractAdapterError as error:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="rejected",
                reason=f"contract_adaptation_failed：{error}",
                actions=tuple(actions),
            )
        actions.append(CONSUMER_ACTION_ADAPT_UPSERT)
        contract = adaptation.contract

        # Eligibility 不通过 → 逻辑下线目标集合（不是 skip、不 Embedding）。
        if adaptation.eligibility_skip_reason is not None:
            return self._execute_eligibility_disable(
                item,
                op=op,
                skip_reason=adaptation.eligibility_skip_reason,
                actions=tuple(actions),
                plan_only=plan_only,
            )

        if not state_loaded:
            record, failure = self._load_record(contract.vector_doc_id, op)
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
                        vector_doc_id=contract.vector_doc_id,
                        operation=op,  # type: ignore[arg-type]
                        status="rejected",
                        reason=f"existing_state_invalid：{error}",
                        actions=tuple(actions),
                    )
        actions.append(CONSUMER_ACTION_FETCH_EXISTING)

        mode = self._decide_text_mode(
            preloaded_state,
            contract.content_hash,
            contract.metadata_hash,
        )
        result = self._execute_text_plan(
            item=item,
            contract=contract,
            mode=mode,
            actions=actions,
            plan_only=plan_only,
            op=op,
        )
        if result.status in ("committed", "skipped"):
            result = self._apply_stale_disables(item, result, plan_only=plan_only)
        return result

    @staticmethod
    def _decide_text_mode(
        state: _ExistingVectorState | None,
        content_hash: str,
        metadata_hash: str,
    ) -> str:
        """差分判定：embedding_upsert / metadata_only_update / skip_unchanged。

        ``state`` 为 None 表示已存实体不存在（新 Chunk）。重新激活守卫：hash
        全等但已存实体 inactive、本次输入 active 时走 metadata-only 更新
        （见模块 docstring 的判定矩阵说明）。
        """

        if state is None:
            return CONSUMER_ACTION_EMBEDDING_UPSERT
        content_changed = state.content_hash != content_hash
        metadata_changed = state.metadata_hash != metadata_hash
        if not content_changed and not metadata_changed and state.is_active:
            return CONSUMER_ACTION_SKIP_UNCHANGED
        if not content_changed:
            return CONSUMER_ACTION_METADATA_ONLY_UPDATE
        return CONSUMER_ACTION_EMBEDDING_UPSERT

    def _execute_text_plan(
        self,
        *,
        item: V15VectorizationWorkItem,
        contract: ExpertChunkEmbeddingInput,
        mode: str,
        actions: list[str],
        plan_only: bool,
        op: str,
    ) -> ExecutionResult:
        """执行差分计划；plan_only 下只登记将要执行的动作。"""

        if plan_only:
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="planned",
                reason="plan_only：仅生成执行计划，未触发任何写入",
                actions=tuple(actions) + (mode,),
            )

        if mode == CONSUMER_ACTION_SKIP_UNCHANGED:
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="skipped",
                reason="hash_unchanged：双 hash 与已存实体一致且仍 active",
                actions=tuple(actions) + (mode,),
            )

        if mode == CONSUMER_ACTION_METADATA_ONLY_UPDATE:
            actions.append(mode)
            try:
                outcome = self._run_metadata_only(contract)
            except VectorIngestionError as error:
                return ExecutionResult(
                    vector_doc_id=contract.vector_doc_id,
                    operation=op,  # type: ignore[arg-type]
                    status="failed",
                    reason=f"metadata_update_infra_failed：{error}",
                    actions=tuple(actions),
                )
            if outcome == "updated":
                return ExecutionResult(
                    vector_doc_id=contract.vector_doc_id,
                    operation=op,  # type: ignore[arg-type]
                    status="committed",
                    reason="metadata_updated：仅元数据变化，复用旧向量，零 Embedding",
                    actions=tuple(actions),
                )
            if outcome == "gate_blocked":
                # 安全门失败（model_space/dim 不一致）→ 降级全量 upsert。
                actions.append(CONSUMER_ACTION_EMBEDDING_UPSERT)
                return self._run_embedding_upsert(
                    contract=contract,
                    actions=tuple(actions),
                    fallback_used=True,
                    op=op,
                )
            # outcome == "inconsistent"：确定性不一致 → rejected。
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="rejected",
                reason="metadata_only_inconsistent：与已存实体的主键/正文/hash 不一致",
                actions=tuple(actions),
            )

        actions.append(CONSUMER_ACTION_EMBEDDING_UPSERT)
        return self._run_embedding_upsert(
            contract=contract,
            actions=tuple(actions),
            fallback_used=False,
            op=op,
        )

    def _run_metadata_only(self, contract: ExpertChunkEmbeddingInput) -> str:
        """调用 update_metadata；返回 updated / gate_blocked / inconsistent。

        两类失败严格区分（对齐 VectorIngestionService 的分层语义）：

        - 一致性门禁（主键/content_hash/chunk_text 必须与已存完全一致）以
          raise ``VectorContractError`` 表达 → 确定性 rejected；
        - 安全门（model_space/embedding_dim）以 failed item +
          recommendation=call_upsert 返回 → 自动降级全量 upsert；
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
        fallback_used: bool,
        op: str,
    ) -> ExecutionResult:
        """全量 upsert（新 Chunk / 正文变化 / 安全门降级共用）。"""

        try:
            result = self._vector_store.upsert_chunks([contract])
        except VectorContractError as error:
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="rejected",
                reason=f"upsert_contract_failed：{error}",
                actions=actions,
                fallback_used=fallback_used,
            )
        except VectorIngestionError as error:
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="failed",
                reason=f"upsert_infra_failed：{error}",
                actions=actions,
                fallback_used=fallback_used,
            )
        entry = result.results[0]
        if entry.status == "embedded":
            suffix = "（安全门降级后全量重嵌）" if fallback_used else ""
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="committed",
                reason=f"embedded：Embedding + upsert 成功{suffix}",
                actions=actions,
                fallback_used=fallback_used,
            )
        if entry.status == "skipped":
            return ExecutionResult(
                vector_doc_id=contract.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="skipped",
                reason=f"upsert_skipped：{entry.error_message or 'eligibility 跳过'}",
                actions=actions,
                fallback_used=fallback_used,
            )
        return ExecutionResult(
            vector_doc_id=contract.vector_doc_id,
            operation=op,  # type: ignore[arg-type]
            status="failed",
            reason=f"upsert_unexpected_status：{entry.status}",
            actions=actions,
            fallback_used=fallback_used,
        )

    def _execute_eligibility_disable(
        self,
        item: V15VectorizationWorkItem,
        *,
        op: str,
        skip_reason: str,
        actions: tuple[str, ...],
        plan_only: bool,
    ) -> ExecutionResult:
        """Eligibility 不通过 → 目标集合逻辑下线（正确性要求，非 skip）。"""

        targets = _dedupe_keep_order(
            [item.vector_doc_id, *item.stale_vector_doc_ids]
        )
        planned_actions = actions + (CONSUMER_ACTION_DISABLE_TARGETS,)
        if plan_only:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="planned",
                reason=f"eligibility_disable：{skip_reason} → 逻辑下线目标集合",
                actions=planned_actions,
                disabled_ids=tuple(targets),
            )
        try:
            deletion = self._vector_store.disable_vectors(targets)
        except VectorIngestionError as error:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation=op,  # type: ignore[arg-type]
                status="failed",
                reason=f"eligibility_disable_failed：{skip_reason}；{error}",
                actions=planned_actions,
            )
        return ExecutionResult(
            vector_doc_id=item.vector_doc_id,
            operation=op,  # type: ignore[arg-type]
            status="committed",
            reason=(
                f"eligibility_disable：{skip_reason}；已下线 "
                f"{deletion.deleted_count} 条，未命中 {deletion.not_found_count} 条"
            ),
            actions=planned_actions,
            disabled_ids=tuple(targets),
        )

    # ------------------------------------------------------------------
    # metadata_update：取回已存原文后合成 upsert 形态
    # ------------------------------------------------------------------
    def _handle_metadata_update(
        self,
        item: V15VectorizationWorkItem,
        *,
        plan_only: bool,
    ) -> ExecutionResult:
        """metadata-update 主体：读旧实体 → 合成完整 Contract → 复用主体判定。

        Work Item 契约禁止携带 chunk_text，因此目标不存在时没有可 Embedding
        的文本、无法回退全量 upsert，只能确定性 rejected；只有目标存在但
        metadata-only 安全门失败时才降级全量 upsert（文本取自已存实体）。
        """

        record, failure = self._load_record(item.vector_doc_id, "metadata_update")
        actions = [CONSUMER_ACTION_FETCH_EXISTING]
        if failure is not None:
            return failure.model_copy(update={"actions": tuple(actions)})
        if record is None:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation="metadata_update",
                status="rejected",
                reason=(
                    "metadata_target_missing：Milvus 无此 vector_doc_id，"
                    "metadata-only 更新缺少可复用的 chunk_text，无法构造 Contract"
                ),
                actions=tuple(actions),
            )
        try:
            state = _extract_existing_state(record)
        except _ExistingStateError as error:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation="metadata_update",
                status="rejected",
                reason=f"existing_state_invalid：{error}",
                actions=tuple(actions),
            )
        stored_text = record.get("chunk_text")
        if not isinstance(stored_text, str) or not stored_text.strip():
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation="metadata_update",
                status="rejected",
                reason="metadata_stored_text_missing：已存实体缺少非空 chunk_text",
                actions=tuple(actions),
            )

        synthesized = self._synthesize_upsert_item(item, state, stored_text)
        if isinstance(synthesized, ExecutionResult):
            return synthesized.model_copy(update={"actions": tuple(actions)})
        actions.append(CONSUMER_ACTION_SYNTHESIZE_FROM_STORED)

        return self._handle_upsert_like(
            synthesized,
            plan_only=plan_only,
            prefix_actions=tuple(actions),
            preloaded_state=state,
            state_loaded=True,
            operation_label="metadata_update",
        )

    def _synthesize_upsert_item(
        self,
        item: V15VectorizationWorkItem,
        state: _ExistingVectorState,
        stored_text: str,
    ) -> V15VectorizationWorkItem | ExecutionResult:
        """把 metadata_update 行合成为 upsert 形态行。

        chunk_text 与 content_hash 取自已存实体（绝不重算、绝不伪造哈希）；
        合成行复用原行的其余列（metadata 载荷、metadata_hash、身份列、事件列）。
        """

        payload = item.model_dump()
        payload.update(
            {
                "operation": "upsert",
                "chunk_text": stored_text,
                "content_hash": state.content_hash,
            }
        )
        try:
            return V15VectorizationWorkItem.model_validate(payload)
        except ValidationError as error:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation="metadata_update",
                status="rejected",
                reason=f"metadata_synth_invalid：{_validation_summary(error)}",
            )

    # ------------------------------------------------------------------
    # disable：最小输入契约
    # ------------------------------------------------------------------
    def _handle_disable(
        self,
        item: V15VectorizationWorkItem,
        *,
        plan_only: bool,
    ) -> ExecutionResult:
        """disable 主体：适配出待下线 id 集合后调用逻辑下线（无物理删除）。"""

        try:
            target = self._adapter.adapt_disable(item)
        except WorkItemContractAdapterError as error:
            return ExecutionResult(
                vector_doc_id=item.vector_doc_id,
                operation="disable",
                status="rejected",
                reason=f"disable_adaptation_failed：{error}",
            )
        actions = [CONSUMER_ACTION_ADAPT_DISABLE, CONSUMER_ACTION_DISABLE_TARGETS]
        ids = tuple(target.vector_doc_ids)
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
                actions=tuple(actions),
            )
        return ExecutionResult(
            vector_doc_id=item.vector_doc_id,
            operation="disable",
            status="committed",
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
        item: V15VectorizationWorkItem,
        main_result: ExecutionResult,
        *,
        plan_only: bool,
    ) -> ExecutionResult:
        """主操作收敛后独立处理 stale 子块；stale 失败不回滚主操作效果。

        主操作成功后若 stale 列表非空，则单独走逻辑下线；stale 下线失败时整体
        记 failed（可重试），reason 注明主操作已完成——重放时主路径因 hash 幂等
        自然收敛，仅补做 stale 下线。主操作为 skipped 且 stale 全部下线成功时，
        结果升级为 committed（存在实际写入效果）。
        """

        stales = [
            vid
            for vid in _dedupe_keep_order(item.stale_vector_doc_ids)
            if vid != item.vector_doc_id
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
            return main_result.model_copy(
                update={
                    "status": "failed",
                    "reason": (
                        f"{base_reason or 'main_done'}；"
                        f"stale_disable_failed（主操作已完成，重放仅补 stale）: {error}"
                    ),
                    "actions": planned_actions,
                },
            )
        merged_disabled = tuple(
            _dedupe_keep_order([*main_result.disabled_ids, *stales])
        )
        stale_note = (
            f"stale 已下线 {deletion.deleted_count} 条，"
            f"未命中 {deletion.not_found_count} 条"
        )
        upgraded_status = (
            "committed" if main_result.status == "skipped" else main_result.status
        )
        return main_result.model_copy(
            update={
                "status": upgraded_status,
                "reason": f"{base_reason}；{stale_note}" if base_reason else stale_note,
                "actions": planned_actions,
                "disabled_ids": merged_disabled,
            },
        )


def _extract_existing_state(record: dict[str, object]) -> _ExistingVectorState:
    """从 get_vectors 返回记录中提取轻量状态（严格 fail-closed）。

    - content_hash / metadata_hash：优先顶层投影字段，其次 metadata JSON 内键；
      缺失或类型非法即抛错，交由调用方转 rejected；
    - is_active：V15 将其平铺在 metadata JSON 中；缺失或非布尔值一律抛错
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

    is_active = flat.get("is_active")
    if not isinstance(is_active, bool):
        raise _ExistingStateError(
            f"is_active 必须是布尔值，收到 {type(is_active).__name__}"
        )
    return _ExistingVectorState(
        vector_doc_id=vector_doc_id,
        content_hash=_hash_field("content_hash"),
        metadata_hash=_hash_field("metadata_hash"),
        is_active=is_active,
    )


__all__ = [
    "SourceEventOrderView",
    "V15WorkItemConsumer",
    "VectorMutationPort",
    "WorkItemAdapterPort",
]
