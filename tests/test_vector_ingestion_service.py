"""VectorIngestionService 公共 Facade 测试（Fake，不联网）。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from expert_contracts import adapt_expert_chunk_mapping
from rag_vector_store import (
    ContractEmbeddingWorker,
    EmbeddedContractPreparedChunk,
    MilvusV15RowMapper,
    MutationResult,
    VectorIngestionService,
    VectorOwnerList,
    VectorQueryResult,
)
from rag_vector_store.chunking.contract_pipeline import ContractValidator
from rag_vector_store.core.ingestion import DomainIngestionService
from rag_vector_store.domains import ExpertContractAdapter
from rag_vector_store.ingestion.errors import VectorContractError

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)


class FakeEmbeddingModel:
    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("fake embedding failure")
        return [[0.1] * 1024 for _ in texts]


class FakeRepository:
    """保存 V15 行并实现 Facade 所需的最小 Repository。"""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.upsert_calls: list[list[EmbeddedContractPreparedChunk]] = []

    def get_many_by_vector_doc_ids(
        self,
        vector_doc_ids: Sequence[str],
        *,
        include_embedding: bool = True,
    ) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for vector_doc_id in vector_doc_ids:
            row = self.rows.get(vector_doc_id)
            if row is None:
                continue
            result[vector_doc_id] = (
                dict(row)
                if include_embedding
                else {key: value for key, value in row.items() if key != "embedding"}
            )
        return result

    def upsert_batch(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> MutationResult:
        items = list(records)
        self.upsert_calls.append(items)
        for record in items:
            self.rows[record.vector_doc_id] = MilvusV15RowMapper().to_row(record)
        return MutationResult(
            affected_count=len(items),
            primary_keys=[item.vector_doc_id for item in items],
        )

    def query_by_owner_id(self, owner_id: int) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for vector_doc_id, row in self.rows.items():
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("expert_id") == owner_id:
                rows.append({"vector_doc_id": vector_doc_id})
        return rows


def make_service(
    *,
    model: FakeEmbeddingModel | None = None,
) -> tuple[VectorIngestionService, FakeEmbeddingModel, FakeRepository]:
    actual_model = model or FakeEmbeddingModel()
    validator = ContractValidator()
    adapter = ExpertContractAdapter(validator)
    worker = ContractEmbeddingWorker(
        actual_model,
        expected_dimension=1024,
        validator=validator,
        clock=lambda: datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc),
    )
    repository = FakeRepository()
    ingestion = DomainIngestionService(worker, repository, adapter)
    service = VectorIngestionService(
        ingestion_service=ingestion,
        repository=repository,  # type: ignore[arg-type]
        validator=validator,
        row_mapper=MilvusV15RowMapper(adapter),
        adapter=adapter,
    )
    return service, actual_model, repository


def records() -> list[object]:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        return [
            adapt_expert_chunk_mapping(json.loads(line))
            for line in stream
            if line.strip()
        ]


def test_upsert_uses_domain_ingestion_and_only_sends_chunk_text() -> None:
    service, model, repository = make_service()
    items = records()[:2]

    result = service.upsert_chunks(items)

    assert result.embedded_count == 2
    assert model.calls == [[item.chunk_text for item in items]]
    assert len(repository.upsert_calls) == 1


def test_metadata_only_reuses_existing_vector() -> None:
    service, model, repository = make_service()
    item = records()[0]
    service.upsert_chunks([item])
    original_embedding = list(repository.rows[item.vector_doc_id]["embedding"])  # type: ignore[arg-type]
    changed = item.model_copy(update={"metadata_hash": "f" * 64})

    result = service.update_metadata([changed])

    assert result.metadata_updated_count == 1
    assert model.calls == [[item.chunk_text]]
    assert repository.rows[item.vector_doc_id]["embedding"] == original_embedding


def test_disable_is_logical_and_idempotent() -> None:
    service, _model, repository = make_service()
    item = records()[0]
    service.upsert_chunks([item])

    first = service.disable_vectors([item.vector_doc_id])
    second = service.disable_vectors([item.vector_doc_id])

    assert first.deleted_count == 1
    assert second.deleted_count == 1
    metadata = repository.rows[item.vector_doc_id]["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["is_active"] is False


def test_get_and_list_owner_return_strict_public_models() -> None:
    service, _model, _repository = make_service()
    items = records()[:2]
    service.upsert_chunks(items)

    query = service.get_vectors([items[0].vector_doc_id, "expert:999:missing:0:0"])
    listing = service.list_vectors_by_owner(items[0].expert_id)

    assert isinstance(query, VectorQueryResult)
    assert query.missing == ["expert:999:missing:0:0"]
    assert isinstance(listing, VectorOwnerList)
    assert listing.owner_id == items[0].expert_id
    assert listing.count == 2


def test_service_rejects_resume_prefixed_expert_input() -> None:
    service, _model, _repository = make_service()
    item = records()[0].model_copy(
        update={"vector_doc_id": "resume:103:profile_summary:1:0"}
    )
    with pytest.raises(VectorContractError):
        service.upsert_chunks([item])


def test_embedding_failure_does_not_write_repository() -> None:
    service, model, repository = make_service(model=FakeEmbeddingModel(fail=True))
    with pytest.raises(Exception, match="fake embedding failure"):
        service.upsert_chunks(records()[:1])
    assert model.calls
    assert repository.upsert_calls == []
