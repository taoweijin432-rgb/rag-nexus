from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import cast

import pytest

from rag_vector_store import (
    ContractEmbeddingResult,
    ContractEmbeddingWorker,
    ContractIngestionError,
    ContractIngestionService,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
    MutationResult,
    SkippedContractChunk,
)


class FakeEmbeddingModel:
    """只在内存中记录文本调用并返回可控向量。"""

    dimension = 3
    model_alias = "fake-model"
    model_version = "fake-v1"

    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.mode == "error":
            raise RuntimeError("fake embedding failure")
        if self.mode == "wrong_dimension":
            return [[0.1, 0.2] for _ in texts]
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeContractRepository:
    """只在内存中记录正式 Contract upsert。"""

    def __init__(self, result: MutationResult | None = None) -> None:
        self.calls: list[list[EmbeddedContractPreparedChunk]] = []
        self.result = result
        self.error: Exception | None = None

    def upsert_batch(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> MutationResult:
        items = list(records)
        self.calls.append(items)
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return MutationResult(
            affected_count=len(items),
            primary_keys=[item.vector_doc_id for item in items],
        )


class StubEmbeddingWorker:
    """返回预设结果，用于测试单条结果一致性保护。"""

    def __init__(self, result: ContractEmbeddingResult) -> None:
        self.result = result
        self.calls: list[list[ExpertChunkEmbeddingInput]] = []

    def embed(
        self,
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> ContractEmbeddingResult:
        self.calls.append(list(records))
        return self.result


def fixed_clock() -> datetime:
    """返回带时区的固定时间，避免测试依赖系统时钟。"""

    return datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def make_record() -> ExpertChunkEmbeddingInput:
    """构造当前正式 Contract 的最小合法内存对象。"""

    text = "[专家能力概述]\n虚构测试专家具备知识检索系统建设经验。"
    vector_doc_id = "expert:900001:profile_summary:900001:0"
    return ExpertChunkEmbeddingInput(
        contract_version="2.0.0",
        chunk_id="expert:900001:profile_summary:test-profile:0",
        vector_doc_id=vector_doc_id,
        expert_id=900001,
        serial_number="DEV900001",
        chunk_type="profile_summary",
        chunk_subtype=None,
        chunk_index=0,
        chunk_total=1,
        chunk_text=text,
        char_count=len(text),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        metadata={
            "vector_doc_id": vector_doc_id,
            "company_name_raw": None,
            "company_name_norm": None,
            "company_normalization_status": None,
            "role_name_raw": None,
            "role_code": None,
            "role_level": None,
            "start_month": None,
            "end_month": None,
            "is_current": None,
            "industry_codes": ["artificial_intelligence"],
            "topic_codes": ["knowledge_retrieval"],
            "origin_type": "source_template",
            "verification_status": "source_evidence",
            "weight_tier": "B",
        },
        source={
            "source_table": "dev_experts",
            "source_item_key": "test-profile",
            "source_field": "background",
            "source_field_id": "dev_experts:900001:background",
            "source_version": "2026-08-12T10:00:00+08:00",
            "source_updated_at": "2026-08-12T10:00:00+08:00",
            "source_hash": "a" * 64,
        },
        versions={
            "data_version": "dev_20260812_01",
            "chunk_policy_version": "1.2",
            "metadata_schema_version": "2.0",
            "dictionary_version": "dev_2026.08",
        },
        is_active=True,
        updated_at="2026-08-12T10:05:00+08:00",
    )


def make_service(
    model: FakeEmbeddingModel | None = None,
    repository: FakeContractRepository | None = None,
) -> tuple[ContractIngestionService, FakeEmbeddingModel, FakeContractRepository]:
    """组装不连接外部系统的正式 Contract ingestion 服务。"""

    actual_model = model or FakeEmbeddingModel()
    actual_repository = repository or FakeContractRepository()
    worker = ContractEmbeddingWorker(
        actual_model,
        expected_dimension=3,
        clock=fixed_clock,
    )
    return (
        ContractIngestionService(worker, actual_repository),
        actual_model,
        actual_repository,
    )


def embedded_chunk(record: ExpertChunkEmbeddingInput) -> EmbeddedContractPreparedChunk:
    """通过正式 Worker 生成一个测试用 embedded 结果。"""

    model = FakeEmbeddingModel()
    worker = ContractEmbeddingWorker(model, expected_dimension=3, clock=fixed_clock)
    result = worker.embed([record])
    assert len(result.embedded_chunks) == 1
    return result.embedded_chunks[0]


def test_ingest_one_embedded_and_only_embeds_chunk_text() -> None:
    service, model, repository = make_service()
    record = make_record()

    result = service.ingest_one(record)

    assert result.status == "embedded"
    assert result.chunk_id == record.chunk_id
    assert result.vector_doc_id == record.vector_doc_id
    assert result.affected_count == 1
    assert result.skip_reason is None
    assert model.calls == [[record.chunk_text]]
    assert len(repository.calls) == 1
    assert repository.calls[0][0].chunk_id == record.chunk_id


def test_ingest_one_skips_ineligible_record_without_external_calls() -> None:
    service, model, repository = make_service()
    record = make_record().model_copy(update={"is_active": False})

    result = service.ingest_one(record)

    assert result.status == "skipped"
    assert result.chunk_id == record.chunk_id
    assert result.vector_doc_id is None
    assert result.affected_count == 0
    assert result.skip_reason == "is_active=false"
    assert model.calls == []
    assert repository.calls == []


def test_ingest_one_contract_validation_failure_does_not_call_dependencies() -> None:
    service, model, repository = make_service()
    record = make_record().model_copy(update={"content_hash": "0" * 64})

    with pytest.raises(ValueError, match="content_hash"):
        service.ingest_one(record)

    assert model.calls == []
    assert repository.calls == []


@pytest.mark.parametrize("mode", ["error", "wrong_dimension"])
def test_ingest_one_embedding_failure_does_not_call_repository(mode: str) -> None:
    model = FakeEmbeddingModel(mode)
    service, _, repository = make_service(model=model)

    with pytest.raises((RuntimeError, ValueError), match="fake embedding failure|dimension"):
        service.ingest_one(make_record())

    assert len(model.calls) == 1
    assert repository.calls == []


def test_ingest_one_repository_failure_is_not_converted_to_skip() -> None:
    repository = FakeContractRepository()
    repository.error = RuntimeError("fake repository failure")
    service, model, _ = make_service(repository=repository)

    with pytest.raises(RuntimeError, match="fake repository failure"):
        service.ingest_one(make_record())

    assert model.calls == [[make_record().chunk_text]]
    assert len(repository.calls) == 1


@pytest.mark.parametrize(
    "worker_result_factory",
    [
        lambda embedded: ContractEmbeddingResult([], []),
        lambda embedded: ContractEmbeddingResult([embedded, embedded], []),
        lambda embedded: ContractEmbeddingResult(
            [embedded],
            [SkippedContractChunk(embedded.chunk_id, "测试冲突")],
        ),
    ],
)
def test_ingest_one_rejects_empty_or_multiple_worker_results(worker_result_factory) -> None:
    record = make_record()
    embedded = embedded_chunk(record)
    result = worker_result_factory(embedded)
    worker = cast(ContractEmbeddingWorker, StubEmbeddingWorker(result))
    repository = FakeContractRepository()
    service = ContractIngestionService(worker, repository)

    with pytest.raises(ContractIngestionError, match="只能得到一个"):
        service.ingest_one(record)

    assert repository.calls == []


def test_ingest_one_rejects_repository_primary_key_mismatch() -> None:
    repository = FakeContractRepository(
        MutationResult(affected_count=1, primary_keys=["wrong-vector-doc-id"])
    )
    service, model, _ = make_service(repository=repository)

    with pytest.raises(ContractIngestionError, match="primary_keys"):
        service.ingest_one(make_record())

    assert model.calls == [[make_record().chunk_text]]
