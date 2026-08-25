"""Phase 3A：通用 Work Item Consumer 纯业务核心的单元测试。

只测试业务编排逻辑：全部依赖 Fake（Fake Contract Adapter / Fake Vector Store /
Fake Embedding Worker），不连接 MySQL、不连接 Milvus、不调用真实 Embedding、
不做任何状态表写入；同时以结构性断言锁定 Consumer 不引入任何持久化依赖。

Fake Adapter 采用“模板派生”构造：以共享 fixture 的合法契约为模板，按 Work Item
的 vector_doc_id / chunk_text / 声明 hash / is_active 覆盖后交给真实
ContractValidator 校验。这与真实适配器“从 Work Item 无损派生 Contract”的方向
一致，保证差分判定的哈希链路自洽，同时不把被测逻辑换成真实适配器。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from expert_contracts.models import ExpertChunkEmbeddingInput
from rag_vector_store.chunking.contract_pipeline import (
    ContractValidator,
    EmbeddingEligibilityValidator,
)
from rag_vector_store.incremental import (
    CONSUMER_ACTION_EMBEDDING_UPSERT,
    CONSUMER_ACTION_METADATA_ONLY_UPDATE,
    CONSUMER_ACTION_SKIP_UNCHANGED,
    DisableWorkItem,
    V15VectorizationWorkItem,
    V15WorkItemConsumer,
    WorkItemAdapterPort,
    WorkItemContractAdaptation,
    WorkItemContractAdapterError,
)
from rag_vector_store.ingestion.errors import (
    VectorContractError,
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

# 与已存记录/Work Item 对齐的默认正文与哈希常量。
_DEFAULT_TEXT = "上游传入的最终正文，本模块不改写"
_CHANGED_TEXT = "正文发生变化后的新文本"
_CONTENT_HASH = hashlib.sha256(_DEFAULT_TEXT.encode("utf-8")).hexdigest()
_NEW_CONTENT_HASH = hashlib.sha256(_CHANGED_TEXT.encode("utf-8")).hexdigest()
_METADATA_HASH = "c" * 64
_NEW_METADATA_HASH = "d" * 64


# ---------------------------------------------------------------------------
# 夹具构造工具
# ---------------------------------------------------------------------------
def _fixture_contracts() -> dict[str, dict[str, object]]:
    """读取共享 fixture 的三类合法契约载荷。"""

    with _FIXTURE_PATH.open(encoding="utf-8") as stream:
        payloads = [json.loads(line) for line in stream if line.strip()]
    return {str(p["chunk_type"]): p for p in payloads}


_FIXTURES = _fixture_contracts()

# 从各类型模板提取测试身份常量（vid 家族前缀共享，便于构造 stale 子块）。
_TEMPLATE_VID: dict[str, str] = {
    ct: str(p["vector_doc_id"]) for ct, p in _FIXTURES.items()
}
_EMPLOYMENT_VID = _TEMPLATE_VID["employment"]
_EMPLOYMENT_PREFIX = _EMPLOYMENT_VID.rsplit(":", 1)[0]
_STALE_ID_A = f"{_EMPLOYMENT_PREFIX}:1"
_STALE_ID_B = f"{_EMPLOYMENT_PREFIX}:2"


def _real_contract(chunk_type: str) -> ExpertChunkEmbeddingInput:
    """由 fixture 构造指定类型的真实契约实例（作为 Fake Adapter 模板）。"""

    return ExpertChunkEmbeddingInput.model_validate(_FIXTURES[chunk_type])


def _structure_keys(contract_dict: dict[str, object]) -> dict[str, object]:
    """把契约载荷拆成 Phase 2 约定的结构保留键集合。"""

    source = contract_dict["source"]
    versions = contract_dict["versions"]
    assert isinstance(source, dict) and isinstance(versions, dict)
    return {
        "chunk_subtype": contract_dict["chunk_subtype"],
        "chunk_index": contract_dict["chunk_index"],
        "chunk_total": contract_dict["chunk_total"],
        "char_count": contract_dict["char_count"],
        "serial_number": contract_dict["serial_number"],
        "updated_at": contract_dict["updated_at"],
        "source_field": source["source_field"],
        "source_version": source["source_version"],
        "source_hash": source["source_hash"],
        "metadata_schema_version": versions["metadata_schema_version"],
    }


def _work_item(
    operation: str = "upsert",
    vector_doc_id: str | None = None,
    *,
    chunk_type: str = "employment",
    chunk_text: str = _DEFAULT_TEXT,
    **overrides: Any,
) -> V15VectorizationWorkItem:
    """以对应类型合法契约为底稿构造 Work Item 完整行（身份列取自模板）。"""

    template = _FIXTURES[chunk_type]
    if vector_doc_id is None:
        vector_doc_id = _TEMPLATE_VID[chunk_type]
    source = template["source"]
    assert isinstance(source, dict)
    base: dict[str, Any] = {
        "work_item_id": 1,
        "batch_id": 1,
        "status": "created",
        "operation": operation,
        "vector_doc_id": vector_doc_id,
        "expert_id": template["expert_id"],
        "chunk_type": chunk_type,
        "source_table": (
            "experts_work_experience"
            if chunk_type == "employment"
            else ("experts" if chunk_type == "profile_summary"
                  else "experts_project_experience")
        ),
        "source_field_id": source["source_field_id"],
        "metadata": _structure_keys(template),
        **overrides,
    }
    if operation == "upsert":
        base.setdefault("chunk_text", chunk_text)
        base.setdefault(
            "content_hash",
            hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
        )
        base.setdefault("metadata_hash", _METADATA_HASH)
        base.setdefault("is_active", True)
    elif operation == "metadata_update":
        base.setdefault("metadata_hash", _NEW_METADATA_HASH)
    else:  # disable
        base.setdefault("is_active", False)
    merged = {**base, **overrides}
    if operation == "disable":
        merged["is_active"] = False  # disable 行模型强制 False
    return V15VectorizationWorkItem.model_validate(merged)


def _stored_record(
    vector_doc_id: str,
    *,
    content_hash: str = _CONTENT_HASH,
    metadata_hash: str = _METADATA_HASH,
    is_active: bool = True,
    chunk_text: str = _DEFAULT_TEXT,
) -> dict[str, object]:
    """构造 Fake 向量存储中的已存记录（V15 形态：hash 平铺在 metadata JSON）。"""

    return {
        "vector_doc_id": vector_doc_id,
        "chunk_text": chunk_text,
        "metadata": {
            "content_hash": content_hash,
            "metadata_hash": metadata_hash,
            "is_active": is_active,
        },
    }


class FakeEmbeddingWorker:
    """记录 embed 调用次数的假 Worker（模拟 upsert_chunks 内部编排）。"""

    def __init__(self) -> None:
        self.embed_calls: list[int] = []

    def embed(self, chunks: list[ExpertChunkEmbeddingInput]) -> dict[str, None]:
        self.embed_calls.append(len(chunks))
        return {}


class FakeVectorStore:
    """满足最小向量能力面的假入口，记录全部调用并可脚本化分支。

    刻意不提供任何物理删除方法，与生产 ``VectorMutationPort`` 形状一致。
    """

    def __init__(
        self,
        *,
        embedding_worker: FakeEmbeddingWorker | None = None,
        update_mode: str = "ok",
    ) -> None:
        self.embedding_worker = embedding_worker
        self.update_mode = update_mode
        self.existing: dict[str, dict[str, object]] = {}
        self.disable_missing: set[str] = set()
        self.raise_on_disable: Exception | None = None
        self.upsert_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.update_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.disable_calls: list[list[str]] = []
        self.get_calls: list[list[str]] = []

    def seed(self, record: dict[str, object]) -> None:
        vid = str(record["vector_doc_id"])
        self.existing[vid] = record

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
        if self.raise_on_disable is not None:
            raise self.raise_on_disable
        deleted = [vid for vid in vector_doc_ids if vid not in self.disable_missing]
        not_found = len(vector_doc_ids) - len(deleted)
        return VectorDeletionResult(
            requested_count=len(vector_doc_ids),
            deleted_count=len(deleted),
            not_found_count=not_found,
        )


class FakeContractAdapter:
    """脚本化假适配器：从 Work Item 派生合法 Contract 并经真实 Validator 校验。"""

    def __init__(
        self,
        *,
        upsert_error: Exception | None = None,
        eligibility_skip_reason: str | None = None,
    ) -> None:
        self.upsert_error = upsert_error
        self.eligibility_skip_reason = eligibility_skip_reason
        self.templates: dict[str, ExpertChunkEmbeddingInput] = {
            ct: _real_contract(ct) for ct in _FIXTURES
        }
        self.upsert_calls: list[V15VectorizationWorkItem] = []
        self.disable_calls: list[V15VectorizationWorkItem] = []

    def adapt_upsert(
        self, work_item: V15VectorizationWorkItem
    ) -> WorkItemContractAdaptation:
        self.upsert_calls.append(work_item)
        if self.upsert_error is not None:
            raise self.upsert_error
        template = self.templates[work_item.chunk_type or "employment"]
        text = work_item.chunk_text or ""
        contract = template.model_copy(
            update={
                "vector_doc_id": work_item.vector_doc_id,
                "chunk_text": text,
                "char_count": len(text),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata_hash": work_item.metadata_hash,
                "is_active": work_item.is_active,
            }
        )
        prepared = ContractValidator().validate(contract)
        if self.eligibility_skip_reason is not None:
            skip_reason: str | None = self.eligibility_skip_reason
        else:
            skip_reason = EmbeddingEligibilityValidator().get_skip_reason(prepared)
        return WorkItemContractAdaptation(
            contract=contract,
            prepared=prepared,
            eligibility_skip_reason=skip_reason,
        )

    def adapt_disable(self, work_item: V15VectorizationWorkItem) -> DisableWorkItem:
        self.disable_calls.append(work_item)
        ordered: list[str] = []
        for vid in [work_item.vector_doc_id, *work_item.stale_vector_doc_ids]:
            if vid not in ordered:
                ordered.append(vid)
        return DisableWorkItem(vector_doc_ids=tuple(ordered))


class FakeEventOrder:
    """固定映射的事件顺序视图。"""

    def __init__(self, last_by_vid: dict[str, int]) -> None:
        self._last = last_by_vid

    def last_applied_source_event_id(self, vector_doc_id: str) -> int | None:
        return self._last.get(vector_doc_id)


def _consumer(
    store: FakeVectorStore,
    adapter: FakeContractAdapter | None = None,
    event_order: Any = None,
) -> V15WorkItemConsumer:
    return V15WorkItemConsumer(
        adapter=adapter if adapter is not None else FakeContractAdapter(),
        vector_store=store,
        event_order=event_order,
    )


def _mutation_counts(store: FakeVectorStore) -> int:
    return len(store.upsert_calls) + len(store.update_calls) + len(store.disable_calls)


_ = WorkItemAdapterPort  # 协议仅作静态类型用途


# ---------------------------------------------------------------------------
# 必需覆盖 1-3：三种 chunk_type / upsert / 新 Chunk
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "chunk_type",
    ["profile_summary", "employment", "project"],
)
def test_upsert_new_chunk_commits_for_all_three_chunk_types(chunk_type: str) -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    item = _work_item("upsert", chunk_type=chunk_type)

    result = _consumer(store).process([item])[0]

    assert result.status == "committed"
    assert result.operation == "upsert"
    assert CONSUMER_ACTION_EMBEDDING_UPSERT in result.actions
    assert len(store.upsert_calls) == 1
    assert len(worker.embed_calls) == 1  # Embedding 经 upsert 编排恰好一次
    assert result.fallback_used is False


def test_content_hash_change_reembeds_and_upserts() -> None:
    store = FakeVectorStore()
    store.seed(_stored_record(_EMPLOYMENT_VID))
    item = _work_item(
        "upsert",
        chunk_text=_CHANGED_TEXT,
        content_hash=_NEW_CONTENT_HASH,
    )

    result = _consumer(store).process([item])[0]

    assert result.status == "committed"
    assert CONSUMER_ACTION_EMBEDDING_UPSERT in result.actions
    assert len(store.upsert_calls) == 1
    assert store.update_calls == []


# ---------------------------------------------------------------------------
# 必需覆盖 4：metadata_update 与安全门失败语义（D3A 修正后）
# ---------------------------------------------------------------------------
def test_metadata_only_update_reuses_stored_text_without_embedding() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    store.seed(_stored_record(_EMPLOYMENT_VID))  # 已存为旧 metadata_hash
    item = _work_item("metadata_update")

    result = _consumer(store).process([item])[0]

    assert result.status == "committed"
    assert CONSUMER_ACTION_METADATA_ONLY_UPDATE in result.actions
    assert len(store.update_calls) == 1
    assert store.upsert_calls == []
    assert worker.embed_calls == []  # metadata-only 零 Embedding


def test_metadata_update_gate_failure_rejects_without_any_writes() -> None:
    """D3A 修正：安全门失败（model_space/dim 不一致）→ 确定性拒绝。

    禁止降级重新 Embedding、禁止调用 upsert_chunks、禁止逻辑下线、
    禁止复用不匹配旧向量；error_code=model_space_mismatch、retryable=False、
    fallback_used=False。
    """

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker, update_mode="gate_fail")
    store.seed(_stored_record(_EMPLOYMENT_VID))
    item = _work_item("metadata_update")

    result = _consumer(store).process([item])[0]

    assert result.status == "rejected"
    assert result.error_code == "model_space_mismatch"
    assert result.retryable is False
    assert result.fallback_used is False
    assert result.embedding_performed is False
    assert result.metadata_only is False
    assert CONSUMER_ACTION_METADATA_ONLY_UPDATE in result.actions
    assert CONSUMER_ACTION_EMBEDDING_UPSERT not in result.actions
    # 零 Embedding、零 upsert、零 disable；仅允许一次 update_metadata 探测。
    assert len(store.update_calls) == 1
    assert store.upsert_calls == []
    assert store.disable_calls == []
    assert worker.embed_calls == []  # 旧向量绝不复用，也绝不重新 Embedding


def test_metadata_update_target_missing_is_rejected_without_fallback() -> None:
    store = FakeVectorStore()
    item = _work_item("metadata_update")

    result = _consumer(store).process([item])[0]

    assert result.status == "rejected"
    assert "metadata_target_missing" in (result.reason or "")
    assert _mutation_counts(store) == 0


def test_metadata_update_inconsistency_raise_maps_to_rejected() -> None:
    store = FakeVectorStore(update_mode="raise_inconsistent")
    store.seed(_stored_record(_EMPLOYMENT_VID))
    item = _work_item("metadata_update")

    result = _consumer(store).process([item])[0]

    assert result.status == "rejected"
    assert "metadata_only_inconsistent" in (result.reason or "")
    assert store.upsert_calls == []  # 一致性失败不降级


# ---------------------------------------------------------------------------
# 必需覆盖 5-6：disable 与 stale 子块
# ---------------------------------------------------------------------------
def test_disable_uses_logical_offline_with_deduped_family() -> None:
    store = FakeVectorStore()
    item = _work_item(
        "disable",
        stale_vector_doc_ids=(_STALE_ID_A, _STALE_ID_A, _STALE_ID_B),
    )

    result = _consumer(store).process([item])[0]

    assert result.status == "committed"
    assert store.disable_calls == [[_EMPLOYMENT_VID, _STALE_ID_A, _STALE_ID_B]]
    assert result.disabled_ids == (_EMPLOYMENT_VID, _STALE_ID_A, _STALE_ID_B)
    assert store.upsert_calls == [] and store.update_calls == []
    assert not hasattr(store, "delete_vectors")  # 端口无物理删除能力


def test_upsert_disables_stale_subchunks_after_main_success() -> None:
    store = FakeVectorStore()
    item = _work_item(
        "upsert", stale_vector_doc_ids=(_STALE_ID_A, _STALE_ID_B)
    )

    result = _consumer(store).process([item])[0]

    assert result.status == "committed"
    assert len(store.upsert_calls) == 1
    assert store.disable_calls == [[_STALE_ID_A, _STALE_ID_B]]
    assert result.disabled_ids == (_STALE_ID_A, _STALE_ID_B)


def test_stale_disable_failure_marks_retryable_failed_but_keeps_main_effect() -> None:
    store = FakeVectorStore()
    store.raise_on_disable = VectorMilvusError("milvus down")
    item = _work_item("upsert", stale_vector_doc_ids=(_STALE_ID_A,))

    result = _consumer(store).process([item])[0]

    assert result.status == "failed"
    assert "主操作已完成" in (result.reason or "")
    assert len(store.upsert_calls) == 1  # 主写入效果保持


def test_stale_disables_run_even_when_main_skips_as_hash_unchanged() -> None:
    store = FakeVectorStore()
    store.seed(_stored_record(_EMPLOYMENT_VID))
    item = _work_item("upsert", stale_vector_doc_ids=(_STALE_ID_A,))

    result = _consumer(store).process([item])[0]

    # 主路径 skip（双 hash 一致），stale 仍被独立下线 → 存在写入效果升级 committed
    assert result.status == "committed"
    assert CONSUMER_ACTION_SKIP_UNCHANGED in result.actions
    assert store.disable_calls == [[_STALE_ID_A]]
    assert store.upsert_calls == []


# ---------------------------------------------------------------------------
# 必需覆盖 7：hash unchanged skip + 重新激活守卫
# ---------------------------------------------------------------------------
def test_hash_unchanged_active_entity_skips_without_any_write() -> None:
    store = FakeVectorStore()
    store.seed(_stored_record(_EMPLOYMENT_VID))
    item = _work_item("upsert")

    result = _consumer(store).process([item])[0]

    assert result.status == "skipped"
    assert CONSUMER_ACTION_SKIP_UNCHANGED in result.actions
    assert _mutation_counts(store) == 0


def test_reactivation_of_disabled_entity_routes_to_metadata_only_update() -> None:
    """重新激活守卫：hash 全等但已存 inactive、输入 active → metadata-only 更新。"""

    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    store.seed(_stored_record(_EMPLOYMENT_VID, is_active=False))
    item = _work_item("upsert")  # 输入 is_active=True、hash 与已存全等

    result = _consumer(store).process([item])[0]

    assert result.status == "committed"
    assert CONSUMER_ACTION_METADATA_ONLY_UPDATE in result.actions
    assert store.upsert_calls == []
    assert worker.embed_calls == []  # 重新激活复用旧向量


# ---------------------------------------------------------------------------
# 必需覆盖 8：旧 source_event_id 拦截
# ---------------------------------------------------------------------------
def test_stale_source_event_id_rejected_without_writes() -> None:
    store = FakeVectorStore()
    order = FakeEventOrder({_EMPLOYMENT_VID: 10})
    item = _work_item("upsert", source_event_id=9)

    result = _consumer(store, event_order=order).process([item])[0]

    assert result.status == "rejected"
    assert "stale_source_event_id" in (result.reason or "")
    assert _mutation_counts(store) == 0


def test_equal_source_event_id_passes_for_idempotent_replay() -> None:
    store = FakeVectorStore()
    store.seed(_stored_record(_EMPLOYMENT_VID))
    order = FakeEventOrder({_EMPLOYMENT_VID: 10})
    item = _work_item("upsert", source_event_id=10)

    result = _consumer(store, event_order=order).process([item])[0]

    # 同事件重放放行，交由 hash 幂等收敛为 skipped
    assert result.status == "skipped"


# ---------------------------------------------------------------------------
# 必需覆盖 9：重复输入的幂等执行计划
# ---------------------------------------------------------------------------
def test_duplicate_inputs_adjudicate_to_single_execution() -> None:
    store = FakeVectorStore()
    older = _work_item("upsert", source_event_id=5)
    newer = _work_item(
        "upsert",
        chunk_text=_CHANGED_TEXT,
        content_hash=_NEW_CONTENT_HASH,
        source_event_id=7,
    )

    results = _consumer(store).process([older, newer])

    assert [r.status for r in results] == ["skipped", "committed"]
    assert "duplicate_superseded" in (results[0].reason or "")
    assert len(store.upsert_calls) == 1
    assert store.upsert_calls[0][0].content_hash == _NEW_CONTENT_HASH


def test_duplicate_inputs_with_equal_events_keep_later_occurrence() -> None:
    store = FakeVectorStore()
    first = _work_item("upsert", source_event_id=None)
    second = _work_item("upsert", source_event_id=None)

    results = _consumer(store).process([first, second])

    assert [r.status for r in results] == ["skipped", "committed"]
    assert len(store.upsert_calls) == 1


# ---------------------------------------------------------------------------
# 必需覆盖 10-11：Contract 失败与 Eligibility 不通过
# ---------------------------------------------------------------------------
def test_contract_adaptation_failure_rejects_deterministically() -> None:
    store = FakeVectorStore()
    adapter = FakeContractAdapter(
        upsert_error=WorkItemContractAdapterError("缺少结构保留键 char_count")
    )
    item = _work_item("upsert")

    result = _consumer(store, adapter).process([item])[0]

    assert result.status == "rejected"
    assert "contract_adaptation_failed" in (result.reason or "")
    assert _mutation_counts(store) == 0


def test_eligibility_failure_routes_to_logical_disable_not_skip() -> None:
    store = FakeVectorStore()
    adapter = FakeContractAdapter(eligibility_skip_reason="is_active=false")
    item = _work_item("upsert")

    result = _consumer(store, adapter).process([item])[0]

    assert result.status == "committed"
    assert store.disable_calls == [[_EMPLOYMENT_VID]]
    assert store.upsert_calls == []  # 绝不进入 Embedding


def test_eligibility_rejected_status_derives_from_prepared_contract() -> None:
    """无脚本注入时 Eligibility 结果来自 prepared 契约本身（is_active=false 输入）。"""

    store = FakeVectorStore()
    item = _work_item("upsert", is_active=False)

    result = _consumer(store).process([item])[0]

    assert result.status == "committed"
    assert store.disable_calls == [[_EMPLOYMENT_VID]]
    assert store.upsert_calls == []


def test_disable_adaptation_failure_rejects() -> None:
    store = FakeVectorStore()

    class StrictDisableAdapter(FakeContractAdapter):
        def adapt_disable(
            self, work_item: V15VectorizationWorkItem
        ) -> DisableWorkItem:
            raise WorkItemContractAdapterError("主 id 不符合正式公式")

    item = _work_item("disable")
    result = _consumer(store, StrictDisableAdapter()).process([item])[0]

    assert result.status == "rejected"
    assert _mutation_counts(store) == 0


# ---------------------------------------------------------------------------
# plan_only 只读规划
# ---------------------------------------------------------------------------
def test_plan_only_produces_planned_results_without_mutations() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    store.seed(_stored_record(_EMPLOYMENT_VID))
    items = [
        _work_item("upsert"),  # → skip 计划
        _work_item("disable", vector_doc_id=_STALE_ID_B),  # → disable 计划
    ]

    results = _consumer(store).process(items, plan_only=True)

    assert all(r.status == "planned" for r in results)
    assert CONSUMER_ACTION_SKIP_UNCHANGED in results[0].actions
    assert _mutation_counts(store) == 0
    assert worker.embed_calls == []


# ---------------------------------------------------------------------------
# 必需覆盖 12：不发生任何状态表写入（结构性断言）
# ---------------------------------------------------------------------------
def test_consumer_has_no_state_persistence_dependencies() -> None:
    """Consumer 模块不得引入任何状态仓储实现或 MySQL 依赖。"""

    incremental_dir = (
        Path(__file__).resolve().parents[1] / "src" / "rag_vector_store" / "incremental"
    )
    sources = "\n".join(
        (incremental_dir / name).read_text(encoding="utf-8")
        for name in (
            "consumer.py",
            "consumer_models.py",
            "consumer_core.py",
        )
    )

    forbidden_tokens = (
        "pymysql",
        "MySQLV15VectorizationRepository",
        "from .repository",
        "import repository",
        "INSERT INTO",
        "UPDATE v15_vectorization_work_item",
        "connector",
    )
    for token in forbidden_tokens:
        assert token not in sources, f"禁止出现 {token!r}"

    # 核心判定文件不依赖 Work Item 行模型（D2A 抽取边界）。
    core_source = (incremental_dir / "consumer_core.py").read_text(encoding="utf-8")
    work_item_tokens = (
        "V15VectorizationWorkItem",
        "V15WorkItemInput",
        "from .models",
        "from .contract_adapter",
        "WorkItemAdapterPort",
        "adapt_upsert(\n        self, work_item",
    )
    for token in work_item_tokens:
        assert token not in core_source, f"核心不得出现 Work Item 依赖 {token!r}"

    # 运行时形态：兼容入口仅持有适配器与核心，无任何连接器/仓储属性。
    consumer = _consumer(FakeVectorStore())
    private_state = vars(consumer)
    assert set(private_state.keys()) == {"_adapter", "_core"}
    core_state = vars(private_state["_core"])
    assert set(core_state.keys()) == {"_vector_store", "_event_order"}
