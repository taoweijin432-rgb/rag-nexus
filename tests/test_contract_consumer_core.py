"""ContractConsumerCore 正式输入模型的单元测试。

只验证标准 Contract / Disable 输入到向量能力面的差分决策和结构化结果。
全部依赖 Fake，不连接 MySQL、Milvus 或真实 Embedding。范围 stale 收敛由
``test_scope_handoff_consumer.py`` 单独覆盖。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Literal

import pytest
from pydantic import ValidationError

from expert_contracts.models import ExpertChunkEmbeddingInput
from rag_vector_store.incremental import (
    CONSUMER_ACTION_DISABLE_TARGETS,
    CONSUMER_ACTION_EMBEDDING_UPSERT,
    CONSUMER_ACTION_METADATA_ONLY_UPDATE,
    CONSUMER_ACTION_SKIP_UNCHANGED,
    ContractConsumerCore,
    ContractDisableInput,
    ContractUpsertIntent,
    ExecutionResult,
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
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)
_DEFAULT_TEXT = "上游传入的最终正文，本模块不改写"
_CHANGED_TEXT = "正文发生变化后的新文本"
_METADATA_HASH = "c" * 64


class _UnsetSourceHash:
    """区分“使用 fixture 默认值”和“显式缺少 source_hash”两种场景。"""


_USE_DEFAULT_SOURCE_HASH: Final = _UnsetSourceHash()


def _fixture_contract(chunk_type: str) -> ExpertChunkEmbeddingInput:
    with _FIXTURE_PATH.open(encoding="utf-8") as stream:
        payloads = [json.loads(line) for line in stream if line.strip()]
    by_type = {str(payload["chunk_type"]): payload for payload in payloads}
    return ExpertChunkEmbeddingInput.model_validate(by_type[chunk_type])


def _intent(
    chunk_type: str = "employment",
    *,
    text: str = _DEFAULT_TEXT,
    metadata_hash: str = _METADATA_HASH,
    eligibility_skip_reason: str | None = None,
) -> ContractUpsertIntent:
    """基于合法 fixture 构造正文和 content_hash 自洽的正式意图。"""

    contract = _fixture_contract(chunk_type).model_copy(
        update={
            "chunk_text": text,
            "char_count": len(text),
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            "metadata_hash": metadata_hash,
        }
    )
    return ContractUpsertIntent(
        contract=contract,
        eligibility_skip_reason=eligibility_skip_reason,
    )


def _intent_with_source_hash(source_hash: str) -> ContractUpsertIntent:
    template = _intent()
    source = template.contract.source.model_copy(update={"source_hash": source_hash})
    return ContractUpsertIntent(
        contract=template.contract.model_copy(update={"source": source}),
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
    if isinstance(source_hash, _UnsetSourceHash):
        source_hash = _fixture_contract("employment").source.source_hash
    metadata: dict[str, object] = {
        "content_hash": content_hash
        or hashlib.sha256(chunk_text.encode()).hexdigest(),
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
        self.calls: list[list[ExpertChunkEmbeddingInput]] = []

    def embed(self, chunks: Sequence[ExpertChunkEmbeddingInput]) -> None:
        self.calls.append(list(chunks))


class FakeVectorStore:
    """记录核心产生的全部读写，并可脚本化各基础设施分支。"""

    def __init__(
        self,
        *,
        embedding_worker: FakeEmbeddingWorker | None = None,
        update_mode: Literal["ok", "gate_fail", "inconsistent", "infra_fail"] = "ok",
        upsert_mode: Literal[
            "ok", "skipped", "contract_fail", "infra_fail", "unexpected"
        ] = "ok",
        query_fails: bool = False,
        disable_fails: bool = False,
    ) -> None:
        self.embedding_worker = embedding_worker
        self.update_mode = update_mode
        self.upsert_mode = upsert_mode
        self.query_fails = query_fails
        self.disable_fails = disable_fails
        self.existing: dict[str, dict[str, object]] = {}
        self.get_calls: list[list[str]] = []
        self.upsert_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.update_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.disable_calls: list[list[str]] = []

    @property
    def mutation_count(self) -> int:
        return len(self.upsert_calls) + len(self.update_calls) + len(self.disable_calls)

    def seed(self, record: dict[str, object]) -> None:
        self.existing[str(record["vector_doc_id"])] = record

    def get_vectors(self, vector_doc_ids: Sequence[str]) -> VectorQueryResult:
        self.get_calls.append(list(vector_doc_ids))
        if self.query_fails:
            raise VectorMilvusError("Milvus 查询失败")
        return VectorQueryResult(
            records=[
                self.existing[vector_doc_id]
                for vector_doc_id in vector_doc_ids
                if vector_doc_id in self.existing
            ],
            missing=[
                vector_doc_id
                for vector_doc_id in vector_doc_ids
                if vector_doc_id not in self.existing
            ],
        )

    def upsert_chunks(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult:
        self.upsert_calls.append(list(chunks))
        if self.upsert_mode == "contract_fail":
            raise VectorContractError("Contract 非法")
        if self.upsert_mode == "infra_fail":
            raise VectorMilvusError("Milvus 写入失败")
        if self.embedding_worker is not None:
            self.embedding_worker.embed(chunks)
        status = {
            "ok": "embedded",
            "skipped": "skipped",
            "unexpected": "failed",
        }[self.upsert_mode]
        return VectorIngestionResult(
            results=[
                VectorItemResult(
                    vector_doc_id=chunk.vector_doc_id,
                    status=status,
                    error_message="测试返回状态" if status != "embedded" else None,
                )
                for chunk in chunks
            ]
        )

    def update_metadata(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult:
        self.update_calls.append(list(chunks))
        if self.update_mode == "inconsistent":
            raise VectorContractError("正文与已存实体不一致")
        if self.update_mode == "infra_fail":
            raise VectorMilvusError("Milvus 更新失败")
        if self.update_mode == "gate_fail":
            return VectorIngestionResult(
                results=[
                    VectorItemResult(
                        vector_doc_id=chunk.vector_doc_id,
                        status="failed",
                        error_message="旧向量模型空间不一致",
                        recommendation="call_upsert",
                    )
                    for chunk in chunks
                ]
            )
        return VectorIngestionResult(
            results=[
                VectorItemResult(
                    vector_doc_id=chunk.vector_doc_id,
                    status="metadata_updated",
                )
                for chunk in chunks
            ]
        )

    def disable_vectors(
        self,
        vector_doc_ids: Sequence[str],
    ) -> VectorDeletionResult:
        self.disable_calls.append(list(vector_doc_ids))
        if self.disable_fails:
            raise VectorMilvusError("Milvus 下线失败")
        return VectorDeletionResult(
            requested_count=len(vector_doc_ids),
            deleted_count=len(vector_doc_ids),
            not_found_count=0,
        )


def _core(
    store: FakeVectorStore,
    *,
    eligibility: Callable[[ExpertChunkEmbeddingInput], str | None] | None = None,
) -> ContractConsumerCore:
    return ContractConsumerCore(
        vector_store=store,  # type: ignore[arg-type]
        eligibility_skip_of=eligibility,
    )


@pytest.mark.parametrize("chunk_type", ["profile_summary", "employment", "project"])
def test_new_contract_embeds_for_all_chunk_types(chunk_type: str) -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)

    result = _core(store).process([_intent(chunk_type)])[0]

    assert result.status == "committed"
    assert result.result_code == "embedded"
    assert result.embedding_performed is True
    assert len(worker.calls) == 1


class _MalformedDownstreamStore(FakeVectorStore):
    """返回结构合法但与请求不对齐的下游结果。"""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def get_vectors(self, vector_doc_ids: Sequence[str]) -> VectorQueryResult:
        if self.mode == "query_type":
            return object()  # type: ignore[return-value]
        if self.mode == "query_extra":
            return VectorQueryResult(
                records=[_stored_record("expert:999:employment:extra:0")],
                missing=[],
            )
        return super().get_vectors(vector_doc_ids)

    def upsert_chunks(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult:
        if self.mode == "upsert_empty":
            return VectorIngestionResult(results=[])
        if self.mode == "upsert_wrong_id":
            return VectorIngestionResult(
                results=[
                    VectorItemResult(
                        vector_doc_id="expert:999:employment:wrong:0",
                        status="embedded",
                    )
                ]
            )
        return super().upsert_chunks(chunks)

    def update_metadata(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult:
        if self.mode == "update_multiple":
            return VectorIngestionResult(
                results=[
                    VectorItemResult(
                        vector_doc_id=chunk.vector_doc_id,
                        status="metadata_updated",
                    )
                    for chunk in chunks
                ]
                + [
                    VectorItemResult(
                        vector_doc_id="expert:999:employment:extra:0",
                        status="metadata_updated",
                    )
                ]
            )
        return super().update_metadata(chunks)

    def disable_vectors(
        self,
        vector_doc_ids: Sequence[str],
    ) -> VectorDeletionResult:
        if self.mode == "disable_count":
            return VectorDeletionResult(
                requested_count=0,
                deleted_count=0,
                not_found_count=0,
            )
        return super().disable_vectors(vector_doc_ids)


def test_consumer_returns_retryable_failure_for_mismatched_upsert_result() -> None:
    for mode in ("upsert_empty", "upsert_wrong_id"):
        result = _core(_MalformedDownstreamStore(mode)).process([_intent()])[0]
        assert result.status == "failed"
        assert result.error_code == "downstream_result_mismatch"
        assert result.retryable is True


def test_consumer_returns_retryable_failure_for_mismatched_update_result() -> None:
    store = _MalformedDownstreamStore("update_multiple")
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))
    result = _core(store).process(
        [_intent(metadata_hash="f" * 64)]
    )[0]
    assert result.status == "failed"
    assert result.error_code == "downstream_result_mismatch"
    assert result.retryable is True


def test_consumer_returns_retryable_failure_for_mismatched_disable_count() -> None:
    result = _core(_MalformedDownstreamStore("disable_count")).process(
        [ContractDisableInput(vector_doc_id="expert:1:employment:1:0")]
    )[0]
    assert result.status == "failed"
    assert result.error_code == "downstream_result_mismatch"
    assert result.retryable is True


def test_consumer_returns_retryable_failure_for_malformed_query_result() -> None:
    for mode in ("query_type", "query_extra"):
        result = _core(_MalformedDownstreamStore(mode)).process([_intent()])[0]
        assert result.status == "failed"
        assert result.error_code == "downstream_result_mismatch"
        assert result.retryable is True


def test_content_hash_change_reembeds() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process([_intent(text=_CHANGED_TEXT)])[0]

    assert result.result_code == "embedded"
    assert CONSUMER_ACTION_EMBEDDING_UPSERT in result.actions
    assert len(worker.calls) == 1


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(
            lambda: _intent(metadata_hash="f" * 64),
            id="metadata_hash",
        ),
        pytest.param(
            lambda: _intent_with_source_hash("e" * 64),
            id="source_hash",
        ),
    ],
)
def test_non_content_hash_change_is_metadata_only(
    changed: Callable[[], ContractUpsertIntent],
) -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process([changed()])[0]

    assert result.status == "committed"
    assert result.result_code == "metadata_updated"
    assert result.metadata_only is True
    assert CONSUMER_ACTION_METADATA_ONLY_UPDATE in result.actions
    assert worker.calls == []
    assert store.upsert_calls == []
    assert len(store.update_calls) == 1


def test_missing_stored_source_hash_is_metadata_only() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id, source_hash=None))

    result = _core(store).process([template])[0]

    assert result.result_code == "metadata_updated"
    assert result.embedding_performed is False
    assert worker.calls == []


def test_inactive_equal_hash_record_reactivates_via_metadata_only() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id, is_active=False))

    result = _core(store).process([template])[0]

    assert result.result_code == "metadata_updated"
    assert result.metadata_only is True
    assert CONSUMER_ACTION_SKIP_UNCHANGED not in result.actions
    assert worker.calls == []


def test_equal_hash_active_record_skips_without_mutation() -> None:
    store = FakeVectorStore()
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process([template])[0]

    assert result.status == "skipped"
    assert result.result_code == "unchanged_skipped"
    assert CONSUMER_ACTION_SKIP_UNCHANGED in result.actions
    assert store.mutation_count == 0


def test_malformed_existing_source_hash_fails_closed() -> None:
    store = FakeVectorStore()
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id, source_hash=""))

    result = _core(store).process([template])[0]

    assert result.status == "rejected"
    assert result.error_code == "existing_state_invalid"
    assert store.mutation_count == 0


def test_metadata_only_model_space_gate_rejects_without_embedding() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker, update_mode="gate_fail")
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process(
        [_intent(metadata_hash="f" * 64)]
    )[0]

    assert result.status == "rejected"
    assert result.error_code == "model_space_mismatch"
    assert result.retryable is False
    assert result.embedding_performed is False
    assert worker.calls == []
    assert store.upsert_calls == []
    assert store.disable_calls == []


def test_metadata_only_contract_inconsistency_is_rejected() -> None:
    store = FakeVectorStore(update_mode="inconsistent")
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process(
        [_intent(metadata_hash="f" * 64)]
    )[0]

    assert result.status == "rejected"
    assert result.error_code == "metadata_only_inconsistent"
    assert result.retryable is False


def test_metadata_only_infrastructure_error_is_retryable() -> None:
    store = FakeVectorStore(update_mode="infra_fail")
    template = _intent()
    store.seed(_stored_record(template.vector_doc_id))

    result = _core(store).process(
        [_intent(metadata_hash="f" * 64)]
    )[0]

    assert result.status == "failed"
    assert result.error_code == "metadata_update_infra_failed"
    assert result.retryable is True


@pytest.mark.parametrize(
    ("mode", "status", "error_code", "retryable"),
    [
        ("contract_fail", "rejected", "upsert_contract_failed", False),
        ("infra_fail", "failed", "upsert_infra_failed", True),
        ("unexpected", "failed", "upsert_infra_failed", True),
    ],
)
def test_upsert_failures_are_classified(
    mode: Literal["contract_fail", "infra_fail", "unexpected"],
    status: str,
    error_code: str,
    retryable: bool,
) -> None:
    store = FakeVectorStore(upsert_mode=mode)

    result = _core(store).process([_intent()])[0]

    assert result.status == status
    assert result.error_code == error_code
    assert result.retryable is retryable


def test_defensive_upsert_skip_remains_non_error_result() -> None:
    store = FakeVectorStore(upsert_mode="skipped")

    result = _core(store).process([_intent()])[0]

    assert result.status == "skipped"
    assert result.result_code == "upsert_skipped"


def test_query_infrastructure_error_is_retryable() -> None:
    store = FakeVectorStore(query_fails=True)

    result = _core(store).process([_intent()])[0]

    assert result.status == "failed"
    assert result.error_code == "existing_query_failed"
    assert result.retryable is True
    assert store.mutation_count == 0


def test_eligibility_failure_routes_to_disable() -> None:
    worker = FakeEmbeddingWorker()
    store = FakeVectorStore(embedding_worker=worker)

    result = _core(store).process(
        [_intent(eligibility_skip_reason="is_active=false")]
    )[0]

    assert result.status == "committed"
    assert result.result_code == "targets_disabled"
    assert CONSUMER_ACTION_DISABLE_TARGETS in result.actions
    assert store.disable_calls
    assert worker.calls == []


def test_eligibility_disable_failure_is_retryable() -> None:
    store = FakeVectorStore(disable_fails=True)

    result = _core(store).process(
        [_intent(eligibility_skip_reason="verification_status=rejected")]
    )[0]

    assert result.status == "failed"
    assert result.error_code == "eligibility_disable_failed"
    assert result.retryable is True


def test_explicit_disable_deduplicates_targets_in_order() -> None:
    store = FakeVectorStore()
    main = _intent().vector_doc_id
    stale = f"{main.rsplit(':', 1)[0]}:1"

    result = _core(store).process(
        [
            ContractDisableInput(
                vector_doc_id=main,
                target_vector_doc_ids=(main, stale, main),
            )
        ]
    )[0]

    assert result.status == "committed"
    assert result.result_code == "targets_disabled"
    assert result.disabled_ids == (main, stale)
    assert store.disable_calls == [[main, stale]]


def test_explicit_disable_failure_is_retryable() -> None:
    store = FakeVectorStore(disable_fails=True)
    vector_doc_id = _intent().vector_doc_id

    result = _core(store).process(
        [ContractDisableInput(vector_doc_id=vector_doc_id)]
    )[0]

    assert result.status == "failed"
    assert result.error_code == "disable_infra_failed"
    assert result.retryable is True


def test_duplicate_vector_doc_ids_fail_before_any_read_or_write() -> None:
    store = FakeVectorStore()
    duplicate = _intent()

    with pytest.raises(ValueError, match="重复 vector_doc_id"):
        _core(store).process([duplicate, duplicate])

    assert store.get_calls == []
    assert store.mutation_count == 0


def test_process_contracts_uses_injected_eligibility() -> None:
    store = FakeVectorStore()
    contract = _intent().contract

    result = _core(
        store,
        eligibility=lambda _: "verification_status=rejected",
    ).process_contracts([contract])[0]

    assert result.result_code == "targets_disabled"
    assert store.disable_calls == [[contract.vector_doc_id]]


def test_process_contracts_without_eligibility_fails_closed() -> None:
    store = FakeVectorStore()

    with pytest.raises(RuntimeError, match="eligibility_skip_of"):
        _core(store).process_contracts([_intent().contract])

    assert store.get_calls == []
    assert store.mutation_count == 0


def _base_result(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "vector_doc_id": "expert:1:employment:1:0",
        "operation": "upsert",
        "status": "committed",
        "result_code": "embedded",
        "embedding_performed": True,
    }
    payload.update(overrides)
    return payload


def test_execution_result_accepts_current_valid_states() -> None:
    ExecutionResult.model_validate(_base_result())
    ExecutionResult.model_validate(
        _base_result(
            result_code="metadata_updated",
            embedding_performed=False,
            metadata_only=True,
        )
    )
    ExecutionResult.model_validate(
        _base_result(
            status="skipped",
            result_code="unchanged_skipped",
            embedding_performed=False,
        )
    )
    ExecutionResult.model_validate(
        _base_result(
            status="rejected",
            result_code=None,
            embedding_performed=False,
            error_code="model_space_mismatch",
        )
    )
    ExecutionResult.model_validate(
        _base_result(
            status="failed",
            result_code=None,
            embedding_performed=False,
            error_code="upsert_infra_failed",
            retryable=True,
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "committed", "result_code": None},
        {
            "status": "rejected",
            "result_code": None,
            "embedding_performed": False,
            "error_code": None,
        },
        {"status": "committed", "error_code": "unexpected"},
        {
            "status": "failed",
            "error_code": "upsert_infra_failed",
            "retryable": True,
        },
        {
            "status": "skipped",
            "result_code": "unchanged_skipped",
            "embedding_performed": True,
        },
        {
            "status": "failed",
            "result_code": None,
            "error_code": "upsert_infra_failed",
            "embedding_performed": False,
            "metadata_only": True,
        },
        {"actions": ("not_an_action",)},
    ],
)
def test_execution_result_rejects_invalid_field_combinations(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ExecutionResult.model_validate(_base_result(**overrides))
