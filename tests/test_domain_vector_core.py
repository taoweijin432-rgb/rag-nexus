"""领域无关向量内核的隔离与结果对齐测试。

Fixture 只使用测试专用的通用记录，不引入专家或 Resume 业务字段。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from rag_vector_store.core import (
    DomainEmbeddingError,
    DomainEmbeddingResult,
    DomainEmbeddingWorker,
    DomainIngestionError,
    DomainIngestionService,
    DomainSkippedVector,
    DomainVectorLifecycleService,
    VectorContractAdapter,
)
from rag_vector_store.incremental.consumer_core import ContractConsumerCore, ContractUpsertIntent
from rag_vector_store.ingestion.errors import VectorIngestionError
from rag_vector_store.ingestion.models import (
    VectorDeletionResult,
    VectorIngestionResult,
    VectorItemResult,
    VectorQueryResult,
)
from rag_vector_store.models import MutationResult


@dataclass(frozen=True)
class InputRecord:
    owner: str
    vector_doc_id: str
    text: str
    content_hash: str = "content"
    metadata_hash: str = "metadata"
    source_hash: str | None = "source"
    active: bool = True


@dataclass(frozen=True)
class PreparedRecord(InputRecord):
    pass


@dataclass(frozen=True)
class EmbeddedRecord(PreparedRecord):
    embedding: tuple[float, ...] = ()
    model_alias: str = "test-model"
    model_version: str = "v1"
    embedded_at: str = ""


class FakeAdapter:
    def __init__(self, owner_field: str, prefix: str, owner_value: str) -> None:
        self._owner_field = owner_field
        self._prefix = prefix
        self._owner_value = owner_value

    def is_input(self, value: object) -> bool:
        return (
            type(value) is InputRecord
            and value.owner == self._owner_value
        )

    def validate(self, value: InputRecord) -> PreparedRecord:
        return PreparedRecord(**value.__dict__)

    def vector_doc_id(self, value: object) -> str:
        return value.vector_doc_id  # type: ignore[attr-defined]

    def chunk_text(self, value: PreparedRecord) -> str:
        return value.text

    def skip_reason(self, value: PreparedRecord) -> str | None:
        return None if value.active else "inactive"

    def attach_embedding(
        self,
        value: PreparedRecord,
        embedding: list[float],
        *,
        embedding_model_alias: str,
        embedding_model_version: str,
        embedded_at: str,
    ) -> EmbeddedRecord:
        return EmbeddedRecord(
            **value.__dict__,
            embedding=tuple(embedding),
            model_alias=embedding_model_alias,
            model_version=embedding_model_version,
            embedded_at=embedded_at,
        )

    def is_embedded(self, value: object) -> bool:
        return type(value) is EmbeddedRecord

    def is_prepared(self, value: object) -> bool:
        return type(value) is PreparedRecord

    def model_space(self, value: EmbeddedRecord) -> dict[str, object]:
        return {
            "embedding_model_alias": value.model_alias,
            "embedding_model_version": value.model_version,
            "embedding_dim": len(value.embedding),
        }

    def embedding(self, value: EmbeddedRecord) -> tuple[float, ...]:
        return value.embedding

    def embedded_at(self, value: EmbeddedRecord) -> str:
        return value.embedded_at

    def metadata(self, value: EmbeddedRecord) -> dict[str, object]:
        return {self._owner_field: value.owner, "metadata_hash": value.metadata_hash}

    def source(self, value: EmbeddedRecord) -> dict[str, object]:
        return {"source_hash": value.source_hash}

    def versions(self, value: EmbeddedRecord) -> dict[str, object]:
        return {"version": "test"}

    def from_embedded_row(self, row: dict[str, object]) -> EmbeddedRecord:
        raise NotImplementedError

    def with_active(self, value: EmbeddedRecord, is_active: bool) -> EmbeddedRecord:
        return replace(value, active=is_active)

    def content_hash(self, value: object) -> str:
        return value.content_hash  # type: ignore[attr-defined]

    def metadata_hash(self, value: object) -> str:
        return value.metadata_hash  # type: ignore[attr-defined]

    def source_hash(self, value: object) -> str | None:
        return value.source_hash  # type: ignore[attr-defined]

    def text_and_identity_equal(self, left: object, right: object) -> bool:
        return (
            self.vector_doc_id(left) == self.vector_doc_id(right)
            and left.text == right.text  # type: ignore[attr-defined]
            and self.content_hash(left) == self.content_hash(right)
        )

    def identity(self, value: object) -> str:
        return self.vector_doc_id(value)

    def owner_id(self, value: object) -> str:
        return value.owner  # type: ignore[attr-defined]

    def owner_field(self) -> str:
        return self._owner_field

    def id_prefix(self) -> str:
        return self._prefix


class FakeModel:
    model_alias = "test-model"
    model_version = "v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 2.0, 3.0] for _ in texts]


class FakeRepository:
    def __init__(self) -> None:
        self.calls: list[list[EmbeddedRecord]] = []

    def upsert_batch(self, records: list[EmbeddedRecord]) -> MutationResult:
        self.calls.append(list(records))
        return MutationResult(
            affected_count=len(records),
            primary_keys=[record.vector_doc_id for record in records],
        )


class FakeLifecycleRepository(FakeRepository):
    def __init__(self, row: EmbeddedRecord | None) -> None:
        super().__init__()
        self.row = row

    def get_by_vector_doc_id(
        self,
        vector_doc_id: str,
        *,
        include_embedding: bool = True,
    ) -> dict[str, object] | None:
        if self.row is None or self.row.vector_doc_id != vector_doc_id:
            return None
        return {"record": self.row}

    def get_many_by_vector_doc_ids(
        self,
        vector_doc_ids: list[str],
        *,
        include_embedding: bool = True,
    ) -> dict[str, dict[str, object]]:
        return {
            vector_doc_id: {"record": self.row}
            for vector_doc_id in vector_doc_ids
            if self.row is not None and self.row.vector_doc_id == vector_doc_id
        }


class FakeBatchLifecycleRepository(FakeRepository):
    def __init__(self, rows: list[EmbeddedRecord]) -> None:
        super().__init__()
        self.rows = {row.vector_doc_id: row for row in rows}
        self.read_calls: list[list[str]] = []

    def get_many_by_vector_doc_ids(
        self,
        vector_doc_ids: list[str],
        *,
        include_embedding: bool = True,
    ) -> dict[str, dict[str, object]]:
        self.read_calls.append(list(vector_doc_ids))
        return {
            vector_doc_id: {"record": self.rows[vector_doc_id]}
            for vector_doc_id in vector_doc_ids
            if vector_doc_id in self.rows
        }


class FakeRowMapper:
    def from_row(self, row: dict[str, object]) -> EmbeddedRecord:
        return row["record"]  # type: ignore[return-value]


def test_generic_worker_only_embeds_eligible_text() -> None:
    model = FakeModel()
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    worker = DomainEmbeddingWorker(
        model,
        adapter,
        expected_dimension=3,
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    result = worker.embed(
        [
            InputRecord("a", "alpha:1", "first"),
            InputRecord("a", "alpha:2", "skip", active=False),
        ]
    )

    assert model.calls == [["first"]]
    assert [item.vector_doc_id for item in result.embedded_chunks] == ["alpha:1"]
    assert result.skipped_chunks == [DomainSkippedVector("alpha:2", "inactive")]
    assert result.embedded_chunks[0].embedded_at.startswith("2026-01-01T00:00:00")


def test_generic_worker_rejects_embedding_count_mismatch() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")

    class WrongCountModel(FakeModel):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return []

    worker = DomainEmbeddingWorker(
        WrongCountModel(),
        adapter,
        expected_dimension=3,
    )

    with pytest.raises(DomainEmbeddingError, match="数量"):
        worker.embed([InputRecord("a", "alpha:1", "first")])


def test_generic_worker_rejects_embedding_dimension_mismatch() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")

    class WrongDimensionModel(FakeModel):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[1.0, 2.0] for _ in texts]

    worker = DomainEmbeddingWorker(
        WrongDimensionModel(),
        adapter,
        expected_dimension=3,
    )

    with pytest.raises(DomainEmbeddingError, match="dimension mismatch"):
        worker.embed([InputRecord("a", "alpha:1", "first")])


def test_generic_ingestion_rejects_cross_owner_result() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    model = FakeModel()
    repository = FakeRepository()
    service = DomainIngestionService(
        DomainEmbeddingWorker(model, adapter, expected_dimension=3),
        repository,
        adapter,
    )

    with pytest.raises(TypeError, match="当前领域"):
        service.ingest_one(InputRecord("b", "beta:1", "wrong"))


def test_two_domain_adapters_have_independent_ids_and_results() -> None:
    model_a = FakeModel()
    model_b = FakeModel()
    adapter_a = FakeAdapter("owner_a", "alpha:", "a")
    adapter_b = FakeAdapter("owner_b", "beta:", "b")
    repo_a = FakeRepository()
    repo_b = FakeRepository()
    service_a = DomainIngestionService(
        DomainEmbeddingWorker(model_a, adapter_a, expected_dimension=3),
        repo_a,
        adapter_a,
    )
    service_b = DomainIngestionService(
        DomainEmbeddingWorker(model_b, adapter_b, expected_dimension=3),
        repo_b,
        adapter_b,
    )

    result_a = service_a.ingest_one(InputRecord("a", "alpha:1", "A"))
    result_b = service_b.ingest_one(InputRecord("b", "beta:1", "B"))

    assert result_a.vector_doc_id == "alpha:1"
    assert result_b.vector_doc_id == "beta:1"
    assert model_a.calls == [["A"]]
    assert model_b.calls == [["B"]]
    assert [row.vector_doc_id for row in repo_a.calls[0]] == ["alpha:1"]
    assert [row.vector_doc_id for row in repo_b.calls[0]] == ["beta:1"]


def test_generic_ingestion_fails_closed_on_result_identity_mismatch() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    repository = FakeRepository()

    class WrongWorker:
        def embed(self, records: Any) -> DomainEmbeddingResult[Any, EmbeddedRecord]:
            return DomainEmbeddingResult(
                embedded_chunks=[
                    EmbeddedRecord(
                        owner="a",
                        vector_doc_id="alpha:other",
                        text="wrong",
                    )
                ],
                skipped_chunks=[],
            )

    service = DomainIngestionService(WrongWorker(), repository, adapter)  # type: ignore[arg-type]
    with pytest.raises(DomainIngestionError, match="vector_doc_id"):
        service.ingest_one(InputRecord("a", "alpha:1", "expected"))


def test_generic_lifecycle_reuses_vector_and_disables_idempotently() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    old = EmbeddedRecord(
        owner="a",
        vector_doc_id="alpha:1",
        text="same",
        embedding=(1.0, 2.0, 3.0),
        embedded_at="old-time",
    )
    repository = FakeLifecycleRepository(old)
    lifecycle = DomainVectorLifecycleService(
        repository,
        FakeRowMapper(),
        adapter,
        expected_dimension=3,
        formal_model_space_id="test-model:3:v1",
    )

    updated = lifecycle.update_metadata(
        [
            InputRecord(
                "a",
                "alpha:1",
                "same",
                metadata_hash="new-metadata",
            )
        ]
    )
    assert updated.results[0].status == "metadata_updated"
    assert repository.calls[0][0].embedding == old.embedding
    assert repository.calls[0][0].embedded_at == old.embedded_at

    disabled = lifecycle.disable_vectors(["alpha:1"])
    assert disabled.deleted_count == 1
    assert repository.calls[-1][0].active is False


def test_generic_lifecycle_rejects_metadata_identity_change() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    old = EmbeddedRecord(
        owner="a",
        vector_doc_id="alpha:1",
        text="old",
        embedding=(1.0, 2.0, 3.0),
    )
    lifecycle = DomainVectorLifecycleService(
        FakeLifecycleRepository(old),
        FakeRowMapper(),
        adapter,
        expected_dimension=3,
        formal_model_space_id="test-model:3:v1",
    )
    with pytest.raises(ValueError, match="chunk_text"):
        lifecycle.update_metadata([InputRecord("a", "alpha:1", "new")])


def test_generic_lifecycle_batches_metadata_updates_after_full_validation() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    old_rows = [
        EmbeddedRecord(
            owner="a",
            vector_doc_id=f"alpha:{index}",
            text=f"text-{index}",
            embedding=(1.0, 2.0, 3.0),
            embedded_at="old-time",
        )
        for index in range(401)
    ]
    repository = FakeBatchLifecycleRepository(old_rows)
    lifecycle = DomainVectorLifecycleService(
        repository,
        FakeRowMapper(),
        adapter,
        expected_dimension=3,
        formal_model_space_id="test-model:3:v1",
    )

    result = lifecycle.update_metadata(
        [
            InputRecord(
                "a",
                f"alpha:{index}",
                f"text-{index}",
                metadata_hash=f"metadata-{index}",
            )
            for index in range(401)
        ]
    )

    assert len(result.results) == 401
    assert len(repository.read_calls) == 1
    assert [len(batch) for batch in repository.calls] == [200, 200, 1]


def test_generic_lifecycle_batches_disable_updates() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    old_rows = [
        EmbeddedRecord(
            owner="a",
            vector_doc_id=f"alpha:{index}",
            text=f"text-{index}",
            embedding=(1.0, 2.0, 3.0),
        )
        for index in range(401)
    ]
    repository = FakeBatchLifecycleRepository(old_rows)
    lifecycle = DomainVectorLifecycleService(
        repository,
        FakeRowMapper(),
        adapter,
        expected_dimension=3,
        formal_model_space_id="test-model:3:v1",
    )

    result = lifecycle.disable_vectors(
        [f"alpha:{index}" for index in range(401)]
    )

    assert result.deleted_count == 401
    assert len(repository.read_calls) == 1
    assert [len(batch) for batch in repository.calls] == [200, 200, 1]
    assert all(not record.active for batch in repository.calls for record in batch)


def test_generic_lifecycle_identity_failure_has_zero_writes() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    repository = FakeBatchLifecycleRepository(
        [
            EmbeddedRecord(
                owner="a",
                vector_doc_id="alpha:1",
                text="same",
                embedding=(1.0, 2.0, 3.0),
            ),
            EmbeddedRecord(
                owner="a",
                vector_doc_id="alpha:2",
                text="old",
                embedding=(1.0, 2.0, 3.0),
            ),
        ]
    )
    lifecycle = DomainVectorLifecycleService(
        repository,
        FakeRowMapper(),
        adapter,
        expected_dimension=3,
        formal_model_space_id="test-model:3:v1",
    )

    with pytest.raises(ValueError, match="chunk_text"):
        lifecycle.update_metadata(
            [
                InputRecord("a", "alpha:1", "same"),
                InputRecord("a", "alpha:2", "new"),
            ]
        )

    assert repository.calls == []


def test_generic_adapter_protocol_does_not_require_expert_type() -> None:
    # 运行时结构检查：测试 Adapter 满足协议所需的领域无关属性，且没有专家字段。
    assert isinstance(FakeAdapter("owner", "x:", "x"), VectorContractAdapter)


def test_core_modules_do_not_import_expert_contracts() -> None:
    core_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rag_vector_store"
        / "core"
    )
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("expert_contracts")
                    for alias in node.names
                ), f"{path.name} 不得直接导入 expert_contracts"
            elif isinstance(node, ast.ImportFrom):
                assert not (
                    node.module
                    and node.module.startswith("expert_contracts")
                ), f"{path.name} 不得直接导入 expert_contracts"


def test_retired_work_item_runtime_is_not_publicly_exported() -> None:
    from rag_vector_store import incremental

    retired = (
        "V15WorkItemConsumer",
        "V15WorkItemContractAdapter",
        "V15VectorizationWorkItem",
        "MySQLV15VectorizationRepository",
    )
    assert all(not hasattr(incremental, name) for name in retired)


class FakeConsumerStore:
    def __init__(self) -> None:
        self.upserted: list[Any] = []

    def get_vectors(self, vector_doc_ids: list[str]) -> VectorQueryResult:
        return VectorQueryResult(records=[], missing=vector_doc_ids)

    def upsert_chunks(self, chunks: list[Any]) -> VectorIngestionResult:
        self.upserted.extend(chunks)
        return VectorIngestionResult(
            results=[
                VectorItemResult(vector_doc_id=chunk.vector_doc_id, status="embedded")
                for chunk in chunks
            ]
        )

    def update_metadata(self, chunks: list[Any]) -> VectorIngestionResult:
        raise VectorIngestionError("not used")

    def disable_vectors(self, vector_doc_ids: list[str]) -> VectorDeletionResult:
        raise VectorIngestionError("not used")


def test_incremental_consumer_accepts_injected_domain_adapter() -> None:
    adapter = FakeAdapter("owner_a", "alpha:", "a")
    store = FakeConsumerStore()
    core = ContractConsumerCore(
        vector_store=store,  # type: ignore[arg-type]
        contract_adapter=adapter,
    )

    result = core.process(
        [ContractUpsertIntent(contract=InputRecord("a", "alpha:1", "text"))]
    )[0]

    assert result.status == "committed"
    assert result.result_code == "embedded"
    assert store.upserted == [InputRecord("a", "alpha:1", "text")]
