"""V15WorkItemConsumer：Work Item 兼容入口（D2A 抽取后的薄翻译层）。

D2A 之后本模块不再承载判定与执行逻辑，只负责把 V15 Work Item 行翻译成
:mod:`consumer_core` 的标准输入并委托执行，保持既有调用与测试兼容：

    V15VectorizationWorkItem
        → V15WorkItemContractAdapter（Phase 2 薄适配器，经 WorkItemAdapterPort）
        → ContractConsumerCore（标准 Contract / Disable 输入）
        → ExecutionResult

翻译规则：
    - 批内幂等裁决与落选结果构造复用核心的共享实现；
    - upsert 胜者：先经适配器得到完整 Contract 与 Eligibility 预判；适配失败
      以 :class:`PreRejectedInput` 透传（核心仍先做旧事件拦截再拒绝，保持既有
      判定顺序）;
    - metadata_update 胜者：携带"从已存实体构造 Contract"的工厂闭包，合成行
      校验/适配失败在核心读取已存记录之后才发生，与既有行为一致；
    - disable 胜者：目标集合由适配器去重排序后透传。

本模块不实现状态持久化、不感知 Embedding 实现、不做物理删除。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import ValidationError

from .consumer_core import (
    AdaptedContract,
    ContractConsumerCore,
    ContractDisableInput,
    ContractMetadataUpdateIntent,
    ContractSynthesisError,
    ContractUpsertIntent,
    PreRejectedInput,
    SourceEventOrderView,
    VectorMutationPort,
    duplicate_superseded_result,
    resolve_winner_indices,
    validation_error_summary,
)
from .consumer_models import (
    ERROR_CODE_CONTRACT_ADAPTATION_FAILED,
    ERROR_CODE_DISABLE_ADAPTATION_FAILED,
    ExecutionResult,
)
from .contract_adapter import (
    DisableWorkItem,
    WorkItemContractAdaptation,
    WorkItemContractAdapterError,
)
from .models import V15VectorizationWorkItem


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


class V15WorkItemConsumer:
    """通用 Work Item 增量消费兼容入口（薄翻译层，无状态持久化）。

    参数：
        ``adapter``：满足 :class:`WorkItemAdapterPort` 的薄适配器。
        ``vector_store``：满足 :class:`VectorMutationPort` 的向量能力入口。
        ``event_order``：可选只读事件顺序视图，用于 source_event_id 旧事件拦截。

    行为与 D2A 抽取前一致：批内同 vector_doc_id 幂等裁决、Eligibility 路由、
    hash 差分、metadata-only 更新、安全门失败拒绝、逻辑下线与 stale 子块补偿，
    全部委托 :class:`ContractConsumerCore` 执行。
    """

    def __init__(
        self,
        *,
        adapter: WorkItemAdapterPort,
        vector_store: VectorMutationPort,
        event_order: SourceEventOrderView | None = None,
    ) -> None:
        self._adapter = adapter
        self._core = ContractConsumerCore(
            vector_store=vector_store, event_order=event_order
        )

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

        keys = [(item.vector_doc_id, item.source_event_id) for item in work_items]
        winner_by_index = resolve_winner_indices(keys)

        core_inputs: list[
            ContractUpsertIntent
            | ContractMetadataUpdateIntent
            | ContractDisableInput
            | PreRejectedInput
        ] = []
        winner_positions: list[int] = []
        results: list[ExecutionResult | None] = [None] * len(work_items)

        for index, item in enumerate(work_items):
            winner_index = winner_by_index.get(index)
            if winner_index != index:
                results[index] = duplicate_superseded_result(
                    vector_doc_id=item.vector_doc_id,
                    operation=item.operation,
                    winner_index=winner_index if winner_index is not None else None,
                )
                continue
            core_inputs.append(self._to_core_input(item))
            winner_positions.append(index)

        core_results = self._core.process(core_inputs, plan_only=plan_only)
        for position, result in zip(winner_positions, core_results, strict=True):
            results[position] = result
        return [result for result in results if result is not None]

    # ------------------------------------------------------------------
    # Work Item → 核心标准输入
    # ------------------------------------------------------------------
    def _to_core_input(
        self,
        item: V15VectorizationWorkItem,
    ) -> (
        ContractUpsertIntent
        | ContractMetadataUpdateIntent
        | ContractDisableInput
        | PreRejectedInput
    ):
        """把单条胜出 Work Item 翻译成核心标准输入。"""

        event_id = item.source_event_id
        stale_ids = tuple(item.stale_vector_doc_ids)
        if item.operation == "upsert":
            try:
                adaptation = self._adapter.adapt_upsert(item)
            except WorkItemContractAdapterError as error:
                return PreRejectedInput(
                    vector_doc_id=item.vector_doc_id,
                    operation="upsert",
                    detail=f"contract_adaptation_failed：{error}",
                    error_code=ERROR_CODE_CONTRACT_ADAPTATION_FAILED,
                    source_event_id=event_id,
                )
            return ContractUpsertIntent(
                contract=adaptation.contract,
                eligibility_skip_reason=adaptation.eligibility_skip_reason,
                stale_vector_doc_ids=stale_ids,
                source_event_id=event_id,
            )
        if item.operation == "metadata_update":
            return ContractMetadataUpdateIntent(
                vector_doc_id=item.vector_doc_id,
                adapt_from_stored=self._metadata_factory(item),
                stale_vector_doc_ids=stale_ids,
                source_event_id=event_id,
            )
        try:
            target = self._adapter.adapt_disable(item)
        except WorkItemContractAdapterError as error:
            return PreRejectedInput(
                vector_doc_id=item.vector_doc_id,
                operation="disable",
                detail=f"disable_adaptation_failed：{error}",
                error_code=ERROR_CODE_DISABLE_ADAPTATION_FAILED,
                source_event_id=event_id,
            )
        return ContractDisableInput(
            vector_doc_id=item.vector_doc_id,
            target_vector_doc_ids=tuple(target.vector_doc_ids),
            source_event_id=event_id,
        )

    def _metadata_factory(
        self,
        item: V15VectorizationWorkItem,
    ) -> Callable[[str, str], AdaptedContract]:
        """构造 metadata_update 的 Contract 工厂（延迟到核心读取已存实体后）。

        chunk_text 与 content_hash 取自已存实体（绝不重算、绝不伪造哈希）；
        合成行复用原行的其余列（metadata 载荷、metadata_hash、身份列、事件列）。
        行校验或适配失败抛出 :class:`ContractSynthesisError`，由核心确定性拒绝。
        """

        def build(stored_text: str, stored_content_hash: str) -> AdaptedContract:
            payload = item.model_dump()
            payload.update(
                {
                    "operation": "upsert",
                    "chunk_text": stored_text,
                    "content_hash": stored_content_hash,
                }
            )
            try:
                synthesized = V15VectorizationWorkItem.model_validate(payload)
            except ValidationError as error:
                raise ContractSynthesisError(
                    validation_error_summary(error)
                ) from error
            try:
                adaptation = self._adapter.adapt_upsert(synthesized)
            except WorkItemContractAdapterError as error:
                raise ContractSynthesisError(str(error)) from error
            return AdaptedContract(
                contract=adaptation.contract,
                eligibility_skip_reason=adaptation.eligibility_skip_reason,
            )

        return build

__all__ = [
    "AdaptedContract",
    "ContractConsumerCore",
    "ContractDisableInput",
    "ContractMetadataUpdateIntent",
    "ContractSynthesisError",
    "ContractUpsertIntent",
    "PreRejectedInput",
    "SourceEventOrderView",
    "V15WorkItemConsumer",
    "VectorMutationPort",
    "WorkItemAdapterPort",
]
