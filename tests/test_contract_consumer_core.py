"""ContractConsumerCore 单元测试（D2A/D3A）。

只测 Work Item 无关的核心链路：标准 Contract / Disable 输入 → 差分判定 →
Fake 向量能力面 → 结构化结果。全部依赖 Fake，不连接 MySQL、不连接 Milvus、
不调用真实 Embedding、不做任何状态表写入；并锁定 D3A 修正后的 metadata-only
安全门失败语义（确定性拒绝、零 Embedding、零写入）与 ``ExecutionResult``
新增字段的不变量。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from expert_contracts.models import ExpertChunkEmbeddingInput
from rag_vector_store.incremental import (
    CONSUMER_ACTION_DISABLE_TARGETS,
    CONSUMER_ACTION_EMBEDDING_UPSERT,
    CONSUMER_ACTION_METADATA_ONLY_UPDATE,
    CONSUMER_ACTION_SKIP_UNCHANGED,
    CONSUMER_ACTION_SYNTHESIZE_FROM_STORED,
    AdaptedContract,
    ContractConsumerCore,
    ContractDisableInput,
    ContractMetadataUpdateIntent,
    ContractUpsertIntent,
    PreRejectedInput,
    resolve_winner_indices,
)
from rag_vector_store.ingestion.errors import (
    VectorContractError,
    VectorIngestionError,
    VectorMilvusError,
)
from rag_vector_store.ingestion.models import (
    VectorDeletionResult,
    VectorIngestionResult,
    VectorItemResult,
    VectorQueryResult,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" /
    "expert_chunk_embedding_input_v4.jsonl"
)

_DEFAULT_TEXT = "上游传入的最终正文，本模块不改写"
_CHANGED_TEXT = "正文发生变化后的新文本"
_METADATA_HASH = "c" * 64
# 与 _intent() 默认 employment 契约同源的 source_hash，供 _stored_record 默认
# seed，使三 Hash 全同 skip 测试不再被路由到 metadata-only。
# 使用哨兵类延迟到 _fixture_contract 定义后计算，并保持 pyright 类型可 narrow。


class _UnsetSourceHash:
    """哨兵：_stored_record 的 source_hash 默认值，避免与 None/str 淑淆。"""


_USE_DEFAULT_SOURCE_HASH: Final = _UnsetSourceHash()


def _fixture_contract(chunk_type: str) -> ExpertChunkEmbeddingInput:
    with _FIXTURE_PATH.open(encoding="utf-8") as stream:
        payloads = [json.loads(line) for line in stream if line.strip()]
    by_type = {str(p["chunk_type"]): p for p in payloads}
    return ExpertChunkEmbeddingInput.model_validate(by_type[chunk_type])


def _intent(
    chunk_type: str = "employment",
    *,
    text: str = _DEFAULT_TEXT,
    eligibility_skip_reason: str | None = None,
    stale_ids: tuple[str, ...] = (),
    event_id: int | None = None,
) -> ContractUpsertIntent:
    """以合法契约为模板构造核心 upsert 意图（哈希与正文自洽）。"""

    contract = _fixture_contract(chunk_type).model_copy(
        update={
            "chunk_text": text,
            "char_count": len(text),
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "metadata_hash": _METADATA_HASH,
        }
    )
    return ContractUpsertIntent(
        contract=contract,
        eligibility_skip_reason=eligibility_skip_reason,
        stale_vector_doc_ids=stale_ids,
        source_event_id=event_id,
    )


def _stored_record(
    vector_doc_id: str,
    *,
    content_hash: str | None = None,
    metadata_hash: str = _METADATA_HASH,
    is_active: bool = True,
    chunk_text: str = _DEFAULT_TEXT,
    source_hash: str | None | _UnsetSourceHash = _USE_DEFAULT_SOURCE_HASH,
) -> dict[str, object]:
    """构造已存记录。

    ``source_hash`` 默认 seed 与 employment fixture 同源（三 Hash 全同 skip
    测试需要）；传 ``None`` 模拟旧版行未写 source_hash 的历史场景；传字符串
    模拟 source_hash 变化的 metadata-only 场景。
    """
    if isinstance(source_hash, _UnsetSourceHash):
        source_hash = _fixture_contract("employment").source.source_hash
    metadata: dict[str, object] = {
        "content_hash": (
            content_hash
            if content_hash is not None
            else hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        ),
        "metadata_hash": metadata_hash,
        "is_active": is_active,
    }
    if source_hash is not None:
        metadata["source_hash"] = source_hash
    return {
        "vector_doc_id": vector_doc_id,
        "chunk_text": chunk_text,
        "metadata": metadata,
    }


class FakeEmbeddingWorker:
    def __init__(self) -> None:
        self.embed_calls: list[int] = []

    def embed(self, chunks: list[ExpertChunkEmbeddingInput]) -> dict[str, None]:
        self.embed_calls.append(len(chunks))
        return {}


class FakeVectorStore:
    """最小向量能力面假实现：记录调用并可脚本化 update_metadata 分支。"""

    def __init__(
        self,
        *,
        embedding_worker: FakeEmbeddingWorker | None = None,
        update_mode: str = "ok",
    ) -> None:
        self.embedding_worker = embedding_worker
        self.update_mode = update_mode
        self.existing: dict[str, dict[str, object]] = {}
        self.upsert_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.update_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.disable_calls: list[list[str]] = []
        self.get_calls: list[list[str]] = []

    def seed(self, record: dict[str, object]) -> None:
        self.existing[str(record["vector_doc_id"])] = record

    def get_vectors(self, vector_doc_ids: Sequence[str]) -> VectorQueryResult:
        self.get_calls.append(list(vector_doc_ids))
        records = [
            self.existing[vid] for vid in vector_doc_ids if vid in self.existing
        ]
        missing = [vid for vid in vector_doc_ids if vid not in self.existing]
        return VectorQueryResult(records=records, missing=missing)

    def upsert_chunks(
        self, chunks: Sequence[ExpertChunkEmbeddingInput]
    ) -> VectorIngestionResult:
        self.upsert_calls.append(list(chunks))
        if self.embedding_worker is not None:
            self.embedding_worker.embed(list(chunks))
        return VectorIngestionResult(
            results=[
                VectorItemResult(vector_doc_id=c.vector_doc_id, status="embedded")
                for c in chunks
            ]
        )

    def update_metadata(
        self, chunks: Sequence[ExpertChunkEmbeddingInput]
    ) -> VectorIngestionResult:
        self.update_calls.append(list(chunks))
        if self.update_mode == "gate_fail":
            return VectorIngestionResult(
                results=[
                    VectorItemResult(
                        vector_doc_id=c.vector_doc_id,
                        status="failed",
                        error_message="旧向量 model_space 不一致",
                        recommendation="call_upsert",
                    )
                    for c in chunks
                ]
            )
        if self.update_mode == "raise_inconsistent":
            raise VectorContractError("update_metadata 只允许 metadata-only 更新")
        return VectorIngestionResult(
            results=[
                VectorItemResult(
                    vector_doc_id=c.vector_doc_id, status="metadata_updated"
                )
                for c in chunks
            ]
        )

    def disable_vectors(self, vector_doc_ids: Sequence[str]) -> VectorDeletionResult:
        self.disable_calls.append(list(vector_doc_ids))
        return VectorDeletionResult(
            requested_count=len(vector_doc_ids),
            deleted_count=len(vector_doc_ids),
            not_found_count=0,
        )


class FakeEventOrder:
    def __init__(self, last_by_vid: dict[str, int]) -> None:
        self._last = last_by_vid

    def last_applied_source_event_id(self, vector_doc_id: str) -> int | None:
        return self._last.get(vector_doc_id)


def _core(store: FakeVectorStore, event_order: object | None = None):
    return ContractConsumerCore(
        vector_store=store,  # type: ignore[arg-type]
        event_order=event_order,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 标准 upsert 输入
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_type", ["profile_summary", "employment", "project"])
def test_core_upsert_new_chunk_embeds_for_all_types(chunk_type: str) -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)

    result = _core(store).process([_intent(chunk_type)])[0]

    assert result.status == "committed"
    assert result.result_code == "embedded"
    assert result.embedding_performed is True
    assert result.metadata_only is False
    assert result.error_code is None
    assert result.retryable is False
    assert len(worker.embed_calls) == 1


def test_core_content_hash_change_reembeds() -> None:
    store = FakeVectorStore()
    vid = _intent().vector_doc_id
    store.seed(_stored_record(vid))

    result = _core(store).process([_intent(text=_CHANGED_TEXT)])[0]

    assert result.status == "committed"
    assert result.result_code == "embedded"
    assert result.embedding_performed is True
    assert len(store.upsert_calls) == 1


def test_core_metadata_update_intent_reuses_stored_text_without_embedding() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    stored = _stored_record(template.vector_doc_id)
    store.seed(stored)

    def factory(stored_text: str, stored_hash: str) -> AdaptedContract:
        assert stored_text == _DEFAULT_TEXT
        stored_meta = stored["metadata"]
        assert isinstance(stored_meta, dict)
        assert stored_hash == stored_meta["content_hash"]
        # 元数据变化：metadata_hash 与已存不同，正文与 content_hash 不变。
        return AdaptedContract(
            contract=template.contract.model_copy(update={"metadata_hash": "d" * 64})
        )

    intent = ContractMetadataUpdateIntent(
        vector_doc_id=template.vector_doc_id,
        adapt_from_stored=factory,
    )
    result = _core(store).process([intent])[0]

    assert result.status == "committed"
    assert result.result_code == "metadata_updated"
    assert result.metadata_only is True
    assert result.embedding_performed is False
    assert CONSUMER_ACTION_SYNTHESIZE_FROM_STORED in result.actions
    assert CONSUMER_ACTION_METADATA_ONLY_UPDATE in result.actions
    assert worker.embed_calls == []  # metadata-only 零 Embedding
    assert store.upsert_calls == []


def test_core_metadata_factory_failure_is_rejected_deterministically() -> None:
    from rag_vector_store.incremental import ContractSynthesisError

    store = FakeVectorStore()
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    def factory(stored_text: str, stored_hash: str) -> AdaptedContract:
        raise ContractSynthesisError("缺少结构保留键 char_count")

    intent = ContractMetadataUpdateIntent(
        vector_doc_id=template.vector_doc_id,
        adapt_from_stored=factory,
    )
    result = _core(store).process([intent])[0]

    assert result.status == "rejected"
    assert result.error_code == "contract_adaptation_failed"
    assert result.retryable is False
    assert store.upsert_calls == []


# ---------------------------------------------------------------------------
# D3A：metadata-only 安全门失败 → 确定性拒绝（零 Embedding、零写入）
# ---------------------------------------------------------------------------
def test_core_gate_blocked_rejects_with_model_space_mismatch() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker, update_mode="gate_fail")
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))
    intent = ContractMetadataUpdateIntent(
        vector_doc_id=template.vector_doc_id,
        adapt_from_stored=lambda text, hsh: AdaptedContract(
            contract=template.contract.model_copy(update={"metadata_hash": "d" * 64})
        ),
    )

    result = _core(store).process([intent])[0]

    # 确定性模型空间错误：rejected、稳定 error_code、不可重试。
    assert result.status == "rejected"
    assert result.error_code == "model_space_mismatch"
    assert result.retryable is False
    assert result.fallback_used is False
    assert result.embedding_performed is False
    assert result.metadata_only is False
    assert result.result_code is None
    assert CONSUMER_ACTION_EMBEDDING_UPSERT not in result.actions
    # 零 Embedding、零 upsert、零 disable：绝不重新嵌入、绝不复用/改写旧向量。
    assert worker.embed_calls == []
    assert store.upsert_calls == []
    assert store.disable_calls == []
    assert len(store.update_calls) == 1  # 仅一次安全门探测


def test_core_gate_blocked_on_reactivation_path_also_rejects() -> None:
    """重新激活守卫路由到 metadata-only 后遇安全门失败同样确定性拒绝。"""

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker, update_mode="gate_fail")
    template = _intent()  # 输入 active
    store.seed(_stored_record(template.vector_doc_id, is_active=False))

    result = _core(store).process([template])[0]

    assert result.status == "rejected"
    assert result.error_code == "model_space_mismatch"
    assert store.upsert_calls == [] and worker.embed_calls == []


# ---------------------------------------------------------------------------
# disable / stale 子块 / Eligibility / skip
# ---------------------------------------------------------------------------
def test_core_disable_targets_are_logically_offline() -> None:
    store = FakeVectorStore()
    main = _intent().vector_doc_id
    stale_a = f"{main.rsplit(':', 1)[0]}:1"

    result = _core(store).process(
        [ContractDisableInput(vector_doc_id=main,
                              target_vector_doc_ids=(main, stale_a))]
    )[0]

    assert result.status == "committed"
    assert result.result_code == "targets_disabled"
    assert result.disabled_ids == (main, stale_a)
    assert store.disable_calls == [[main, stale_a]]
    assert store.upsert_calls == [] and store.update_calls == []


def test_core_eligibility_failure_routes_to_disable_not_embedding() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)

    result = _core(store).process(
        [_intent(eligibility_skip_reason="is_active=false")]
    )[0]

    assert result.status == "committed"
    assert result.result_code == "targets_disabled"
    assert CONSUMER_ACTION_DISABLE_TARGETS in result.actions
    assert store.disable_calls != []
    assert worker.embed_calls == []


def test_core_hash_unchanged_skips_without_writes() -> None:
    store = FakeVectorStore()
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process([template])[0]

    assert result.status == "skipped"
    assert result.result_code == "unchanged_skipped"
    assert result.error_code is None
    assert CONSUMER_ACTION_SKIP_UNCHANGED in result.actions


# ---------------------------------------------------------------------------
# C1：source_hash 投影与三 Hash 动作矩阵
# ---------------------------------------------------------------------------
def _intent_with_source_hash(source_hash: str) -> ContractUpsertIntent:
    """构造一个仅 source_hash 与默认不同的 upsert 意图（content/metadata 不变）。"""

    template = _intent()
    new_source = template.contract.source.model_copy(update={"source_hash": source_hash})
    return ContractUpsertIntent(
        contract=template.contract.model_copy(update={"source": new_source}),
        eligibility_skip_reason=None,
        stale_vector_doc_ids=(),
        source_event_id=None,
    )


def test_core_source_hash_only_change_routes_to_metadata_only() -> None:
    """content_hash 不变、仅 source_hash 变化 → metadata-only 更新。"""

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process([_intent_with_source_hash("e" * 64)])[0]

    assert result.status == "committed"
    assert result.result_code == "metadata_updated"
    assert result.metadata_only is True
    assert result.embedding_performed is False
    assert CONSUMER_ACTION_METADATA_ONLY_UPDATE in result.actions


def test_core_source_hash_only_change_has_zero_embedding() -> None:
    """source_hash-only 变化不得调用 Embedding。"""

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    _core(store).process([_intent_with_source_hash("e" * 64)])[0]

    assert worker.embed_calls == []
    assert store.upsert_calls == []
    assert len(store.update_calls) == 1


def test_core_metadata_hash_only_change_routes_to_metadata_only() -> None:
    """content_hash/source_hash 不变、仅 metadata_hash 变化 → metadata-only。"""

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    changed = ContractUpsertIntent(
        contract=template.contract.model_copy(update={"metadata_hash": "f" * 64}),
        eligibility_skip_reason=None,
        stale_vector_doc_ids=(),
        source_event_id=None,
    )
    result = _core(store).process([changed])[0]

    assert result.status == "committed"
    assert result.result_code == "metadata_updated"
    assert result.metadata_only is True
    assert result.embedding_performed is False
    assert worker.embed_calls == []
    assert store.upsert_calls == []


def test_core_content_hash_change_still_reembeds() -> None:
    """content_hash 变化仍重新 Embedding（三 Hash 矩阵中 content 优先级最高）。"""

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process([_intent(text=_CHANGED_TEXT)])[0]

    assert result.status == "committed"
    assert result.result_code == "embedded"
    assert result.embedding_performed is True
    assert len(worker.embed_calls) == 1


def test_core_absent_stored_source_hash_routes_to_metadata_only_not_skip() -> None:
    """已存 source_hash 缺失（旧版行）不得误判为三 Hash 全同 Skip。

    content_hash 相同且旧 source_hash 缺失 → 路由到 metadata-only 收敛 previous state，
    不调用 Embedding。
    """

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    # 已存记录不带 source_hash（模拟旧版行）
    store.seed(_stored_record(template.vector_doc_id, source_hash=None))

    result = _core(store).process([template])[0]

    assert result.status == "committed"
    assert result.result_code == "metadata_updated"
    assert result.metadata_only is True
    assert result.embedding_performed is False
    assert worker.embed_calls == []
    assert store.upsert_calls == []


def test_core_three_hash_equal_inactive_reactivates_via_metadata_only() -> None:
    """三 Hash 全等但已存 inactive、本次 active → metadata-only 恢复，不得误 Skip。"""

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()  # 输入 active
    store.seed(_stored_record(template.vector_doc_id, is_active=False))

    result = _core(store).process([template])[0]

    assert result.status == "committed"
    assert result.result_code == "metadata_updated"
    assert result.metadata_only is True
    assert result.embedding_performed is False
    assert worker.embed_calls == []
    assert CONSUMER_ACTION_SKIP_UNCHANGED not in result.actions


def test_core_source_hash_malformed_fails_closed() -> None:
    """已存 source_hash 存在但类型非法（空字符串）→ fail-closed，不静默当缺失。"""

    store = FakeVectorStore()
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id, source_hash=""))

    result = _core(store).process([template])[0]

    assert result.status == "rejected"
    assert result.error_code == "existing_state_invalid"


def test_core_gate_blocked_on_source_hash_change_path_also_rejects() -> None:
    """source_hash-only 变化走 metadata-only 遇安全门失败 → 确定性拒绝零 Embedding。"""

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker, update_mode="gate_fail")
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process([_intent_with_source_hash("e" * 64)])[0]

    assert result.status == "rejected"
    assert result.error_code == "model_space_mismatch"
    assert result.embedding_performed is False
    assert worker.embed_calls == []
    assert store.upsert_calls == []


def test_core_upsert_disables_stale_subchunks_after_commit() -> None:
    store = FakeVectorStore()
    template = _intent()
    stale_a = f"{template.vector_doc_id.rsplit(':', 1)[0]}:1"

    result = _core(store).process([_intent(stale_ids=(stale_a,))])[0]

    assert result.status == "committed"
    assert store.disable_calls == [[stale_a]]
    assert result.disabled_ids == (stale_a,)


def test_core_skip_upgrades_to_committed_when_stale_disabled() -> None:
    store = FakeVectorStore()
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))
    stale_a = f"{template.vector_doc_id.rsplit(':', 1)[0]}:1"

    result = _core(store).process([_intent(stale_ids=(stale_a,))])[0]

    assert result.status == "committed"
    assert result.result_code == "targets_disabled"  # 净效果是纯逻辑下线
    assert CONSUMER_ACTION_SKIP_UNCHANGED in result.actions


class _FailingDisableStore(FakeVectorStore):
    def disable_vectors(
        self, vector_doc_ids: Sequence[str]
    ) -> VectorDeletionResult:
        raise VectorMilvusError("milvus down")


def test_core_stale_disable_failure_keeps_main_effect_and_marks_retryable() -> None:
    worker = FakeEmbeddingWorker()
    store = _FailingDisableStore(embedding_worker=worker)
    template = _intent()
    stale_a = f"{template.vector_doc_id.rsplit(':', 1)[0]}:1"

    result = _core(store).process([_intent(stale_ids=(stale_a,))])[0]

    assert result.status == "failed"
    assert "主操作已完成" in (result.reason or "")
    assert result.error_code == "stale_disable_failed"
    assert result.retryable is True
    assert result.result_code is None
    assert len(store.upsert_calls) == 1  # 主写入效果保持


# ---------------------------------------------------------------------------
# 旧事件拦截 / PreRejectedInput / plan_only / 幂等裁决
# ---------------------------------------------------------------------------
def test_core_stale_source_event_rejected_before_any_action() -> None:
    store = FakeVectorStore()
    template = _intent(event_id=9)
    order = FakeEventOrder({template.vector_doc_id: 10})

    result = _core(store, order).process([template])[0]

    assert result.status == "rejected"
    assert result.error_code == "stale_source_event_id"
    assert result.retryable is False
    assert store.get_calls == []  # 拦截发生在任何查询之前


def test_core_pre_rejected_passthrough_after_stale_check() -> None:
    store = FakeVectorStore()
    rejected = PreRejectedInput(
        vector_doc_id="expert:900001:employment:21307:0",
        operation="disable",
        detail="disable_adaptation_failed：主 id 不符合正式公式",
        error_code="disable_adaptation_failed",
        source_event_id=None,
    )

    result = _core(store).process([rejected])[0]

    assert result.status == "rejected"
    assert result.error_code == "disable_adaptation_failed"
    assert "disable_adaptation_failed" in (result.reason or "")
    assert store.get_calls == []


def test_core_pre_rejected_yields_to_stale_interception() -> None:
    """透传拒绝仍先做旧事件拦截，保持既有判定顺序。"""

    store = FakeVectorStore()
    vid = "expert:900001:employment:21307:0"
    rejected = PreRejectedInput(
        vector_doc_id=vid,
        operation="upsert",
        detail="contract_adaptation_failed：x",
        error_code="contract_adaptation_failed",
        source_event_id=3,
    )
    order = FakeEventOrder({vid: 5})

    result = _core(store, order).process([rejected])[0]

    assert result.error_code == "stale_source_event_id"


def test_core_plan_only_produces_planned_results_without_mutations() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    results = _core(store).process(
        [template, ContractDisableInput(vector_doc_id="expert:1:project:2:0")],
        plan_only=True,
    )

    assert all(r.status == "planned" for r in results)
    assert all(r.result_code is None and r.error_code is None for r in results)
    assert CONSUMER_ACTION_SKIP_UNCHANGED in results[0].actions
    assert _mutation_counts(store) == 0
    assert worker.embed_calls == []


def test_core_duplicate_inputs_adjudicate_by_event_order() -> None:
    store = FakeVectorStore()
    old = _intent(text=_DEFAULT_TEXT, event_id=5)
    new = _intent(text=_CHANGED_TEXT, event_id=7)

    results = _core(store).process([old, new])

    assert [r.status for r in results] == ["skipped", "committed"]
    assert results[0].result_code == "duplicate_superseded"
    assert results[0].error_code is None
    assert len(store.upsert_calls) == 1
    assert store.upsert_calls[0][0].content_hash == new.contract.content_hash


def test_resolve_winner_indices_ranking_rules() -> None:
    mapping = resolve_winner_indices(
        [("a", 1), ("a", 3), ("b", None), ("b", 2), ("c", None)]
    )
    assert mapping == {0: 1, 1: 1, 2: 3, 3: 3, 4: 4}


def _mutation_counts(store: FakeVectorStore) -> int:
    return len(store.upsert_calls) + len(store.update_calls) + len(store.disable_calls)


# ---------------------------------------------------------------------------
# ExecutionResult 新字段不变量
# ---------------------------------------------------------------------------
def _base_result(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "vector_doc_id": "expert:1:employment:1:0",
        "operation": "upsert",
        "status": "committed",
        "result_code": "embedded",
        "embedding_performed": True,
    }
    base.update(overrides)
    return base


def test_result_invariants_valid_combinations_pass() -> None:
    from rag_vector_store.incremental import ExecutionResult

    # embedded 提交
    ExecutionResult.model_validate(_base_result())
    # metadata-only 提交
    ExecutionResult.model_validate(
        _base_result(result_code="metadata_updated",
                     embedding_performed=False, metadata_only=True)
    )
    # skipped 幂等落选
    ExecutionResult.model_validate(
        _base_result(status="skipped", result_code="duplicate_superseded",
                     embedding_performed=False)
    )
    # planned 双码皆无
    ExecutionResult.model_validate(
        _base_result(status="planned", result_code=None,
                     embedding_performed=False)
    )
    # rejected 确定性拒绝
    ExecutionResult.model_validate(
        _base_result(status="rejected", result_code=None,
                     embedding_performed=False, error_code="model_space_mismatch",
                     retryable=False)
    )
    # failed 可重试基础设施异常
    ExecutionResult.model_validate(
        _base_result(status="failed", result_code=None,
                     embedding_performed=False,
                     error_code="upsert_infra_failed", retryable=True)
    )


@pytest.mark.parametrize(
    ("overrides",),
    [
        ({"retryable": True},),
        ({"embedding_performed": True},),
    ],
)
def test_result_invariants_rejected_must_be_non_retryable_and_clean(
    overrides: dict[str, object],
) -> None:
    from rag_vector_store.incremental import ExecutionResult

    payload = _base_result(
        status="rejected",
        result_code=None,
        embedding_performed=False,
        error_code="contract_adaptation_failed",
    )
    payload.update(overrides)
    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(payload)


def test_result_invariants_committed_requires_result_code() -> None:
    from rag_vector_store.incremental import ExecutionResult

    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(_base_result(result_code=None))


def test_result_invariants_rejected_requires_error_code() -> None:
    from rag_vector_store.incremental import ExecutionResult

    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(
            _base_result(status="rejected", result_code=None,
                         embedding_performed=False)
        )


def test_result_invariants_error_code_forbidden_on_success() -> None:
    from rag_vector_store.incremental import ExecutionResult

    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(
            _base_result(error_code="model_space_mismatch")
        )


def test_result_invariants_result_code_forbidden_on_failure() -> None:
    from rag_vector_store.incremental import ExecutionResult

    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(
            _base_result(status="failed", error_code="upsert_infra_failed",
                         retryable=True)
        )


def test_result_invariants_embedding_requires_confirmation_on_commit() -> None:
    from rag_vector_store.incremental import ExecutionResult

    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(
            _base_result(status="skipped", result_code="unchanged_skipped",
                         embedding_performed=True)
        )


def test_result_invariants_metadata_only_exclusive_and_committed_only() -> None:
    from rag_vector_store.incremental import ExecutionResult

    with pytest.raises(ValidationError):  # 与 embedding_performed 互斥
        ExecutionResult.model_validate(
            _base_result(metadata_only=True)
        )
    with pytest.raises(ValidationError):  # 只允许 committed
        ExecutionResult.model_validate(
            _base_result(status="planned", result_code=None,
                         embedding_performed=False, metadata_only=True)
        )


def test_result_invariants_unknown_action_fail_closed() -> None:
    from rag_vector_store.incremental import ExecutionResult

    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(_base_result(actions=("not_an_action",)))


def test_ingestion_error_family_maps_to_retryable_failed() -> None:
    """VectorIngestionError（非 Contract）在读取阶段转 failed 且可重试。"""

    class FailingQueryStore(FakeVectorStore):
        def get_vectors(
            self, vector_doc_ids: Sequence[str]
        ) -> VectorQueryResult:
            raise VectorIngestionError("network glitch")

    store = FailingQueryStore()
    result = _core(store).process([_intent()])[0]

    assert result.status == "failed"
    assert result.error_code == "existing_query_failed"
    assert result.retryable is True
