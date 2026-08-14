"""Contract 4.0.0 编排层的单条/批量 ingestion 单元测试。

总流程以 ``vector_doc_id`` 为唯一身份，彻底不依赖 ``chunk_id``。覆盖单条、
批量、skip 语义、fail-closed 身份校验与“不循环调用单条入口”的批次行为。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import cast

import pytest

from expert_contracts.models import ExpertChunkEmbeddingInput
from rag_vector_store import (
    ContractBatchIngestionService,
    ContractIngestionError,
    ContractIngestionService,
    EmbeddedContractPreparedChunk,
    MutationResult,
)
from rag_vector_store.chunking.contract_pipeline import (
    ContractEmbeddingResult,
    ContractEmbeddingWorker,
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
    """返回预设结果，用于测试编排层的身份一致性保护。"""

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
    return datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_record(index: int = 0) -> ExpertChunkEmbeddingInput:
    """构造 4.0.0 的最小合法内存记录（以 vector_doc_id 为身份，无 chunk_id）。"""

    text = "[专家能力概述]\n虚构测试专家具备知识检索系统建设经验。"
    # 用可回查的 source_field_id 区分不同记录，chunk_index 保持 0 < chunk_total，
    # 使每条都满足 Contract 合法且 vector_doc_id 唯一。
    source_field_id = f"profile-{index}"
    vector_doc_id = f"expert:900001:profile_summary:{source_field_id}:0"
    return ExpertChunkEmbeddingInput(
        contract_version="4.0.0",
        vector_doc_id=vector_doc_id,
        expert_id=900001,
        serial_number="DEV900001",
        chunk_type="profile_summary",
        chunk_subtype=None,
        chunk_index=0,
        chunk_total=1,
        chunk_text=text,
        char_count=len(text),
        content_hash=_sha(text),
        metadata_hash="b" * 64,
        metadata={
            "company_name_raw": None,
            "company_name_norm": None,
            "role_name_raw": None,
            "role_code": None,
            "role_level": None,
            "job_status_norm": "active",
            "abroad_flag_norm": False,
            "work_years": None,
            "project_name": None,
            "start_month": None,
            "end_month": None,
            "is_current": None,
            "industry_codes": ["artificial_intelligence"],
            "function_codes": [],
            "topic_codes": ["knowledge_retrieval"],
            "school_names_raw": [],
            "major_names_raw": [],
            "eligible_for_recommendation": True,
            "origin_type": "compressed_from_source",
            "verification_status": "source_summary",
            "weight_tier": "B",
        },
        source={
            "source_table": "experts",
            "source_field": "company+job_name+background",
            "source_field_id": source_field_id,
            "source_version": "2026-08-12T10:00:00+08:00",
            "source_updated_at": "2026-08-12T10:00:00+08:00",
            "source_hash": "a" * 64,
        },
        versions={"chunk_policy_version": "1.3", "metadata_schema_version": "3.0"},
        is_active=True,
        updated_at="2026-08-12T10:05:00+08:00",
    )


def make_records(count: int) -> list[ExpertChunkEmbeddingInput]:
    """构造多条主键各异的正式 Contract 测试数据。"""

    return [make_record(index) for index in range(count)]


def make_service(
    model: FakeEmbeddingModel | None = None,
    repository: FakeContractRepository | None = None,
) -> tuple[ContractIngestionService, FakeEmbeddingModel, FakeContractRepository]:
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


def make_batch_service(
    model: FakeEmbeddingModel | None = None,
    repository: FakeContractRepository | None = None,
) -> tuple[
    ContractBatchIngestionService,
    FakeEmbeddingModel,
    FakeContractRepository,
]:
    actual_model = model or FakeEmbeddingModel()
    actual_repository = repository or FakeContractRepository()
    worker = ContractEmbeddingWorker(
        actual_model,
        expected_dimension=3,
        clock=fixed_clock,
    )
    return (
        ContractBatchIngestionService(worker, actual_repository),
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


def with_worker(service: ContractIngestionService, result: ContractEmbeddingResult):
    """把单条服务的 Worker 替换为预设结果。"""

    stub = cast(ContractEmbeddingWorker, StubEmbeddingWorker(result))
    return ContractIngestionService(stub, service.repository), stub


def with_batch_worker(
    service: ContractBatchIngestionService,
    result: ContractEmbeddingResult,
):
    stub = cast(ContractEmbeddingWorker, StubEmbeddingWorker(result))
    return ContractBatchIngestionService(stub, service.repository), stub


# ---- Single Ingestion ----

def test_single_eligible_embeds_and_reports_vector_doc_id() -> None:
    service, model, repository = make_service()
    record = make_record()

    result = service.ingest_one(record)

    assert result.status == "embedded"
    assert result.vector_doc_id == record.vector_doc_id
    assert result.affected_count == 1
    assert result.skip_reason is None
    assert model.calls == [[record.chunk_text]]
    assert len(repository.calls) == 1
    assert repository.calls[0][0].vector_doc_id == record.vector_doc_id


def test_single_skipped_reports_vector_doc_id_without_repository() -> None:
    service, model, repository = make_service()
    record = make_record().model_copy(update={"is_active": False})

    result = service.ingest_one(record)

    assert result.status == "skipped"
    assert result.vector_doc_id == record.vector_doc_id
    assert result.affected_count == 0
    assert result.skip_reason == "is_active=false"
    assert model.calls == []
    assert repository.calls == []


def test_single_rejected_is_skipped_without_repository() -> None:
    service, _, repository = make_service()
    record = make_record().model_copy(
        update={
            "metadata": make_record().metadata.model_copy(
                update={"verification_status": "rejected"}
            )
        }
    )

    result = service.ingest_one(record)

    assert result.status == "skipped"
    assert result.vector_doc_id == record.vector_doc_id
    assert "rejected" in result.skip_reason
    assert repository.calls == []


def test_single_contract_validation_failure_does_not_call_dependencies() -> None:
    service, model, repository = make_service()
    record = make_record().model_copy(update={"content_hash": "0" * 64})

    with pytest.raises(ValueError, match="content_hash"):
        service.ingest_one(record)

    assert model.calls == []
    assert repository.calls == []


@pytest.mark.parametrize("mode", ["error", "wrong_dimension"])
def test_single_embedding_failure_does_not_call_repository(mode: str) -> None:
    model = FakeEmbeddingModel(mode)
    service, _, repository = make_service(model=model)

    with pytest.raises((RuntimeError, ValueError), match="fake embedding failure|dimension"):
        service.ingest_one(make_record())

    assert len(model.calls) == 1
    assert repository.calls == []


def test_single_repository_failure_is_not_converted_to_skip() -> None:
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
        lambda e: ContractEmbeddingResult([], []),
        lambda e: ContractEmbeddingResult([e, e], []),
        lambda e: ContractEmbeddingResult([e], [SkippedContractChunk(e.vector_doc_id, "冲突")]),
    ],
)
def test_single_rejects_empty_or_multiple_worker_results(worker_result_factory) -> None:
    record = make_record()
    embedded = embedded_chunk(record)
    service, _ = with_worker(make_service()[0], worker_result_factory(embedded))
    repository = service.repository

    with pytest.raises(ContractIngestionError, match="只能得到一个"):
        service.ingest_one(record)

    assert repository.calls == []


def test_single_rejects_worker_result_identity_conflict() -> None:
    record = make_record()
    embedded = embedded_chunk(record)
    wrong = embedded.model_copy(update={"vector_doc_id": "expert:900001:profile_summary:999:0"})
    service, _ = with_worker(make_service()[0], ContractEmbeddingResult([wrong], []))
    repository = service.repository

    with pytest.raises(ContractIngestionError, match="vector_doc_id"):
        service.ingest_one(record)

    assert repository.calls == []


def test_single_rejects_repository_primary_key_mismatch() -> None:
    repository = FakeContractRepository(
        MutationResult(affected_count=1, primary_keys=["wrong-vector-doc-id"])
    )
    service, model, _ = make_service(repository=repository)

    with pytest.raises(ContractIngestionError, match="primary_keys"):
        service.ingest_one(make_record())

    assert model.calls == [[make_record().chunk_text]]


# ---- Batch Ingestion ----

def test_batch_all_eligible_single_upsert() -> None:
    service, model, repository = make_batch_service()
    records = make_records(120)

    result = service.ingest_batch(records)

    assert result.total_count == 120
    assert result.embedded_count == 120
    assert result.skipped_count == 0
    assert result.affected_count == 120
    assert [item.vector_doc_id for item in result.results] == [
        record.vector_doc_id for record in records
    ]
    assert [item.status for item in result.results] == ["embedded"] * 120
    assert model.calls == [[record.chunk_text for record in records]]
    # 真正批量：一次 Repository.upsert_batch。
    assert len(repository.calls) == 1
    assert len(repository.calls[0]) == 120


def test_batch_mixed_eligible_and_skipped_single_upsert() -> None:
    service, model, repository = make_batch_service()
    records = make_records(205)
    skipped_indexes = {10, 100, 204}
    records = [
        record.model_copy(update={"is_active": False})
        if index in skipped_indexes
        else record
        for index, record in enumerate(records)
    ]

    result = service.ingest_batch(records)

    assert result.total_count == 205
    assert result.embedded_count == 202
    assert result.skipped_count == 3
    assert result.affected_count == 202
    assert [item.vector_doc_id for item in result.results] == [
        record.vector_doc_id for record in records
    ]
    assert [item.status for item in result.results] == [
        "skipped" if index in skipped_indexes else "embedded"
        for index in range(205)
    ]
    assert model.calls == [
        [
            record.chunk_text
            for index, record in enumerate(records)
            if index not in skipped_indexes
        ]
    ]
    assert len(repository.calls) == 1
    assert len(repository.calls[0]) == 202
    assert [item.vector_doc_id for item in repository.calls[0]] == [
        record.vector_doc_id
        for index, record in enumerate(records)
        if index not in skipped_indexes
    ]


def test_batch_rejected_records_are_skipped() -> None:
    service, model, repository = make_batch_service()
    records = make_records(5)
    records = [
        record.model_copy(
            update={
                "metadata": record.metadata.model_copy(
                    update={"verification_status": "rejected"}
                )
            }
        )
        if index == 2
        else record
        for index, record in enumerate(records)
    ]

    result = service.ingest_batch(records)

    assert result.embedded_count == 4
    assert result.skipped_count == 1
    assert result.results[2].status == "skipped"
    assert "rejected" in result.results[2].skip_reason
    assert model.calls == [[records[i].chunk_text for i in (0, 1, 3, 4)]]
    assert len(repository.calls) == 1


def test_batch_inactive_records_are_skipped() -> None:
    service, model, _ = make_batch_service()
    records = make_records(3)
    records = [records[0].model_copy(update={"is_active": False}), records[1], records[2]]

    result = service.ingest_batch(records)

    assert result.results[0].status == "skipped"
    assert result.results[0].skip_reason == "is_active=false"
    assert result.embedded_count == 2
    assert model.calls == [[records[1].chunk_text, records[2].chunk_text]]


def test_batch_duplicate_vector_doc_id_fails_closed() -> None:
    service, model, repository = make_batch_service()
    records = make_records(2)
    records[1] = records[1].model_copy(update={"vector_doc_id": records[0].vector_doc_id})

    with pytest.raises(ContractIngestionError, match="重复 vector_doc_id"):
        service.ingest_batch(records)

    assert model.calls == []
    assert repository.calls == []


def test_batch_worker_missing_input_vector_doc_id_fails_closed() -> None:
    records = make_records(2)
    a = records[0]
    embedded = embedded_chunk(a)
    extra = SkippedContractChunk("expert:900001:profile_summary:999:0", "跳过")
    service = make_batch_service()[0]
    service, _ = with_batch_worker(service, ContractEmbeddingResult([embedded], [extra]))

    with pytest.raises(ContractIngestionError, match="vector_doc_id"):
        service.ingest_batch(records)

    assert service.repository.calls == []


def test_batch_worker_extra_vector_doc_id_fails_closed() -> None:
    records = make_records(2)
    a = records[0]
    embedded_a = embedded_chunk(a)
    extra = embedded_chunk(a).model_copy(
        update={"vector_doc_id": "expert:900001:profile_summary:999:0"}
    )
    service = make_batch_service()[0]
    service, _ = with_batch_worker(
        service, ContractEmbeddingResult([embedded_a, extra], [])
    )

    with pytest.raises(ContractIngestionError, match="vector_doc_id"):
        service.ingest_batch(records)

    assert service.repository.calls == []


def test_batch_worker_skip_embed_identity_conflict_fails_closed() -> None:
    records = make_records(2)
    a = records[0]
    # 同一 vector_doc_id 同时出现在 embedded 与 skipped —— 身份冲突。
    embedded_a = embedded_chunk(a)
    dup_skip = SkippedContractChunk(a.vector_doc_id, "冲突")
    service = make_batch_service()[0]
    service, _ = with_batch_worker(service, ContractEmbeddingResult([embedded_a], [dup_skip]))

    with pytest.raises(ContractIngestionError, match="重复 vector_doc_id"):
        service.ingest_batch(records)

    assert service.repository.calls == []


def test_batch_invalid_input_fails_closed_before_embedding_and_upsert() -> None:
    service, model, repository = make_batch_service()
    records = make_records(201)
    records[100] = records[100].model_copy(update={"content_hash": "0" * 64})

    with pytest.raises(ValueError, match="content_hash"):
        service.ingest_batch(records)

    assert model.calls == []
    assert repository.calls == []


def test_batch_does_not_upsert_when_embedding_fails() -> None:
    model = FakeEmbeddingModel("error")
    service, _, repository = make_batch_service(model=model)

    with pytest.raises(RuntimeError, match="fake embedding failure"):
        service.ingest_batch(make_records(120))

    assert len(model.calls) == 1
    assert repository.calls == []


# ---- 全流程无 chunk_id 依赖 ----

def test_full_flow_has_no_chunk_id_dependency() -> None:
    assert not hasattr(make_record(), "chunk_id")
    service, _, repository = make_service()
    record = make_record()
    result = service.ingest_one(record)
    assert not hasattr(result, "chunk_id")
    assert repository.calls[0][0].vector_doc_id == record.vector_doc_id
    assert not hasattr(repository.calls[0][0], "chunk_id")


def test_batch_does_not_loop_single_ingestion() -> None:
    service, model, repository = make_batch_service()
    records = make_records(150)

    result = service.ingest_batch(records)

    # 一次 embedding 批调用 + 一次 Repository.upsert_batch，而非 150 次单条。
    assert len(model.calls) == 1
    assert len(repository.calls) == 1
    assert result.total_count == 150
    assert result.affected_count == 150