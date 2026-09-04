"""不绑定业务领域的单条/批量入库编排。"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from ..models import MutationResult
from .adapters import VectorContractAdapter
from .embedding import DomainEmbeddingResult, DomainSkippedVector
from .guards import require_adapter_input

InputT = TypeVar("InputT")
PreparedT = TypeVar("PreparedT")
EmbeddedT = TypeVar("EmbeddedT")


class DomainEmbeddingPort(Protocol[InputT, PreparedT, EmbeddedT]):
    def embed(self, records: Iterable[InputT]) -> DomainEmbeddingResult[PreparedT, EmbeddedT]:
        ...


class DomainRepository(Protocol[EmbeddedT]):
    def upsert_batch(self, records: Sequence[EmbeddedT]) -> MutationResult:
        ...


class DomainIngestionError(RuntimeError):
    """通用编排结果无法与输入或 Repository 对齐。"""


@dataclass(frozen=True)
class DomainIngestionResult:
    status: Literal["embedded", "skipped"]
    vector_doc_id: str
    affected_count: int = 0
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("embedded", "skipped"):
            raise ValueError("status 必须是 embedded 或 skipped")
        if not isinstance(self.vector_doc_id, str) or not self.vector_doc_id.strip():
            raise ValueError("vector_doc_id 必须是非空字符串")
        if isinstance(self.affected_count, bool) or not isinstance(self.affected_count, int):
            raise ValueError("affected_count 必须是非负整数")  # noqa: TRY004
        if self.affected_count < 0:
            raise ValueError("affected_count 必须是非负整数")
        if self.status == "embedded":
            if self.skip_reason is not None:
                raise ValueError("embedded 结果不能包含 skip_reason")
            if self.affected_count != 1:
                raise ValueError("单条 embedded 结果的 affected_count 必须是 1")
        else:
            if self.affected_count != 0:
                raise ValueError("skipped 结果的 affected_count 必须是 0")
            if not isinstance(self.skip_reason, str) or not self.skip_reason.strip():
                raise ValueError("skipped 结果必须包含 skip_reason")


@dataclass(frozen=True)
class DomainBatchIngestionResult:
    results: list[DomainIngestionResult]
    embedded_count: int
    skipped_count: int
    affected_count: int

    def __post_init__(self) -> None:
        if any(not isinstance(item, DomainIngestionResult) for item in self.results):
            raise ValueError("results 必须只包含 DomainIngestionResult")
        embedded_count = sum(item.status == "embedded" for item in self.results)
        skipped_count = sum(item.status == "skipped" for item in self.results)
        if self.embedded_count != embedded_count:
            raise ValueError("embedded_count 与 results 不一致")
        if self.skipped_count != skipped_count:
            raise ValueError("skipped_count 与 results 不一致")
        if self.embedded_count + self.skipped_count != len(self.results):
            raise ValueError("批量汇总数量与 results 不一致")
        if self.affected_count != self.embedded_count:
            raise ValueError("affected_count 必须等于 embedded_count")

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def item_results(self) -> list[DomainIngestionResult]:
        return self.results


class DomainIngestionService(Generic[InputT, PreparedT, EmbeddedT]):
    """领域无关的单条和批量入库编排。"""

    def __init__(
        self,
        embedding_worker: DomainEmbeddingPort[InputT, PreparedT, EmbeddedT],
        repository: DomainRepository[EmbeddedT],
        adapter: VectorContractAdapter[InputT, PreparedT, EmbeddedT],
    ) -> None:
        self.embedding_worker = embedding_worker
        self.repository = repository
        self.adapter = adapter

    def ingest_one(self, record: InputT) -> DomainIngestionResult:
        require_adapter_input(self.adapter, record, operation="ingest_one")
        result = self.embedding_worker.embed([record])
        embedded, skipped = self._single_worker_result(record, result)
        if skipped is not None:
            return DomainIngestionResult(
                status="skipped",
                vector_doc_id=self.adapter.vector_doc_id(record),
                skip_reason=skipped.reason,
            )
        if embedded is None:
            raise DomainIngestionError("Worker 未返回 embedded 或 skipped 结果")
        mutation = self.repository.upsert_batch([embedded])
        self._validate_repository_result(
            [embedded], mutation, self.adapter
        )
        return DomainIngestionResult(
            status="embedded",
            vector_doc_id=self.adapter.vector_doc_id(embedded),
            affected_count=1,
        )

    def ingest_batch(self, records: Iterable[InputT]) -> DomainBatchIngestionResult:
        items = list(records)
        self._validate_input_records(items)
        result = self.embedding_worker.embed(items)
        embedded, skipped = self._validate_worker_result(items, result)
        affected_count = 0
        if embedded:
            mutation = self.repository.upsert_batch(embedded)
            self._validate_repository_result(embedded, mutation, self.adapter)
            affected_count = mutation.affected_count

        embedded_by_id = {self.adapter.vector_doc_id(item): item for item in embedded}
        skipped_by_id = {item.vector_doc_id: item for item in skipped}
        results: list[DomainIngestionResult] = []
        for item in items:
            vector_doc_id = self.adapter.vector_doc_id(item)
            if vector_doc_id in embedded_by_id:
                results.append(
                    DomainIngestionResult(
                        status="embedded",
                        vector_doc_id=vector_doc_id,
                        affected_count=1,
                    )
                )
            else:
                skipped_item = skipped_by_id.get(vector_doc_id)
                if skipped_item is None:
                    raise DomainIngestionError(
                        f"Worker 结果缺少输入 vector_doc_id: {vector_doc_id}"
                    )
                results.append(
                    DomainIngestionResult(
                        status="skipped",
                        vector_doc_id=vector_doc_id,
                        skip_reason=skipped_item.reason,
                    )
                )
        return DomainBatchIngestionResult(
            results=results,
            embedded_count=len(embedded),
            skipped_count=len(skipped),
            affected_count=affected_count,
        )

    def _validate_input_records(self, records: Sequence[InputT]) -> None:
        for item in records:
            require_adapter_input(self.adapter, item, operation="ingest_batch")
        ids = [self.adapter.vector_doc_id(item) for item in records]
        if len(ids) != len(set(ids)):
            raise DomainIngestionError("一个批次不能包含重复 vector_doc_id")

    def _single_worker_result(
        self,
        record: InputT,
        result: DomainEmbeddingResult[PreparedT, EmbeddedT],
    ) -> tuple[EmbeddedT | None, DomainSkippedVector | None]:
        if not isinstance(result, DomainEmbeddingResult):
            raise DomainIngestionError("Worker 返回了非法 DomainEmbeddingResult")
        if len(result.embedded_chunks) + len(result.skipped_chunks) != 1:
            raise DomainIngestionError(
                "单条 ingestion 必须得到且只能得到一个 embedded 或 skipped 结果"
            )
        input_id = self.adapter.vector_doc_id(record)
        if result.embedded_chunks:
            embedded = result.embedded_chunks[0]
            if not self.adapter.is_embedded(embedded):
                raise DomainIngestionError("Worker 返回了非法 embedded Contract")
            if self.adapter.vector_doc_id(embedded) != input_id:
                raise DomainIngestionError("Worker 返回的 vector_doc_id 与输入不一致")
            return embedded, None
        skipped = result.skipped_chunks[0]
        if skipped.vector_doc_id != input_id:
            raise DomainIngestionError("Worker 返回的 skipped vector_doc_id 与输入不一致")
        return None, skipped

    def _validate_worker_result(
        self,
        records: Sequence[InputT],
        result: DomainEmbeddingResult[PreparedT, EmbeddedT],
    ) -> tuple[list[EmbeddedT], list[DomainSkippedVector]]:
        if not isinstance(result, DomainEmbeddingResult):
            raise DomainIngestionError("Worker 返回了非法 DomainEmbeddingResult")
        embedded = result.embedded_chunks
        skipped = result.skipped_chunks
        if len(embedded) + len(skipped) != len(records):
            raise DomainIngestionError("批量 ingestion 返回结果数量与输入不一致")
        if any(not self.adapter.is_embedded(item) for item in embedded):
            raise DomainIngestionError("Worker 返回了非法 embedded Contract")
        ids = [self.adapter.vector_doc_id(item) for item in embedded] + [
            item.vector_doc_id for item in skipped
        ]
        input_ids = {self.adapter.vector_doc_id(item) for item in records}
        if len(ids) != len(set(ids)):
            raise DomainIngestionError("Worker 返回了重复 vector_doc_id")
        if set(ids) != input_ids:
            raise DomainIngestionError(
                "Worker 返回的 vector_doc_id 与输入不一致 "
                f"(missing={sorted(input_ids - set(ids))}, "
                f"extra={sorted(set(ids) - input_ids)})"
            )
        return embedded, skipped

    @staticmethod
    def _validate_repository_result(
        embedded: Sequence[EmbeddedT],
        mutation: MutationResult,
        adapter: VectorContractAdapter[Any, Any, EmbeddedT],
    ) -> None:
        if not isinstance(mutation, MutationResult):
            raise DomainIngestionError("Repository 返回了非法 MutationResult")
        if mutation.affected_count != len(embedded):
            raise DomainIngestionError("Repository affected_count 与 embedded 数量不一致")
        expected_ids = [adapter.vector_doc_id(item) for item in embedded]
        if mutation.primary_keys != expected_ids:
            raise DomainIngestionError("Repository 返回的 primary_keys 与 embedded 记录不一致")


class DomainBatchIngestionService(
    DomainIngestionService[InputT, PreparedT, EmbeddedT]
):
    """已弃用的批量类名；行为完全委托 ``DomainIngestionService``。"""

    def __init__(
        self,
        embedding_worker: DomainEmbeddingPort[InputT, PreparedT, EmbeddedT],
        repository: DomainRepository[EmbeddedT],
        adapter: VectorContractAdapter[InputT, PreparedT, EmbeddedT],
    ) -> None:
        warnings.warn(
            "DomainBatchIngestionService 已弃用；请使用 DomainIngestionService",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(embedding_worker, repository, adapter)


__all__ = [
    "DomainBatchIngestionResult",
    "DomainBatchIngestionService",
    "DomainEmbeddingPort",
    "DomainIngestionError",
    "DomainIngestionResult",
    "DomainIngestionService",
    "DomainRepository",
]
