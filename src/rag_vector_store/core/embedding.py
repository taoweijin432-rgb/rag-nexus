"""不绑定业务领域的 Embedding Worker。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from ..constants import FORMAL_EMBEDDING_DIMENSION
from ..embedding import EmbeddingModel
from .adapters import VectorContractAdapter
from .embedding_validation import validate_embedding_vector
from .guards import require_adapter_input

PreparedT = TypeVar("PreparedT")
EmbeddedT = TypeVar("EmbeddedT")
InputT = TypeVar("InputT")


@dataclass(frozen=True)
class DomainSkippedVector:
    """合法但不进入 Embedding 的领域记录。"""

    vector_doc_id: str
    reason: str


class DomainEmbeddingError(RuntimeError):
    """Embedding 返回结果不符合通用 Worker 协议。"""


@dataclass(frozen=True)
class DomainEmbeddingResult(Generic[PreparedT, EmbeddedT]):
    """通用 Worker 输出。"""

    embedded_chunks: list[EmbeddedT]
    skipped_chunks: list[DomainSkippedVector]

    @property
    def eligible_count(self) -> int:
        return len(self.embedded_chunks)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_chunks)


class DomainEmbeddingWorker(Generic[InputT, PreparedT, EmbeddedT]):
    """只依赖领域 Adapter 的批量 Embedding 编排。"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        adapter: VectorContractAdapter[InputT, PreparedT, EmbeddedT],
        *,
        embedding_model_alias: str | None = None,
        embedding_model_version: str | None = None,
        expected_dimension: int = FORMAL_EMBEDDING_DIMENSION,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(expected_dimension, bool) or not isinstance(expected_dimension, int):
            raise ValueError("expected_dimension 必须是正整数")  # noqa: TRY004
        if expected_dimension < 1:
            raise ValueError("expected_dimension 必须是正整数")
        self.embedding_model = embedding_model
        self.adapter = adapter
        self.embedding_model_alias = embedding_model_alias or _required_model_text(
            embedding_model, "model_alias"
        )
        self.embedding_model_version = embedding_model_version or _required_model_text(
            embedding_model, "model_version"
        )
        self.expected_dimension = expected_dimension
        self.clock = clock or _utc_now

    def embed(
        self,
        records: Iterable[InputT],
    ) -> DomainEmbeddingResult[PreparedT, EmbeddedT]:
        """校验、筛选并批量向量化，只发送领域 Adapter 提供的正文。"""

        eligible: list[PreparedT] = []
        skipped: list[DomainSkippedVector] = []
        for record in records:
            require_adapter_input(self.adapter, record, operation="Embedding")
            prepared = self.adapter.validate(record)
            if not self.adapter.is_prepared(prepared):
                raise TypeError("Contract Adapter 返回了非法 Prepared Contract")
            reason = self.adapter.skip_reason(prepared)
            if reason is None:
                eligible.append(prepared)
            else:
                skipped.append(
                    DomainSkippedVector(
                        vector_doc_id=self.adapter.vector_doc_id(prepared),
                        reason=reason,
                    )
                )

        if not eligible:
            return DomainEmbeddingResult(embedded_chunks=[], skipped_chunks=skipped)

        texts = [self.adapter.chunk_text(record) for record in eligible]
        raw_result = self.embedding_model.embed(texts)
        try:
            raw_embeddings = list(raw_result)
        except (TypeError, ValueError) as error:
            raise DomainEmbeddingError("Embedding 返回值必须是可迭代的向量集合") from error
        if len(raw_embeddings) != len(eligible):
            raise DomainEmbeddingError(
                "Embedding 返回数量与符合资格的输入记录数量不一致"
            )

        embedded_at = _format_embedded_at(self.clock())
        embedded: list[EmbeddedT] = []
        for prepared, raw_embedding in zip(eligible, raw_embeddings):
            try:
                embedding = validate_embedding_vector(
                    raw_embedding,
                    expected_dimension=self.expected_dimension,
                    label=(
                        f"vector_doc_id "
                        f"{self.adapter.vector_doc_id(prepared)!r} 的 Embedding"
                    ),
                )
            except (TypeError, ValueError) as error:
                raise DomainEmbeddingError(str(error)) from error
            try:
                item = self.adapter.attach_embedding(
                    prepared,
                    embedding,
                    embedding_model_alias=self.embedding_model_alias,
                    embedding_model_version=self.embedding_model_version,
                    embedded_at=embedded_at,
                )
            except (TypeError, ValueError) as error:
                raise DomainEmbeddingError(
                    "Contract Adapter 无法回填合法 Embedded Contract"
                ) from error
            if not self.adapter.is_embedded(item):
                raise DomainEmbeddingError(
                    "Contract Adapter 返回了非法 Embedded Contract"
                )
            embedded.append(item)
        return DomainEmbeddingResult(
            embedded_chunks=embedded,
            skipped_chunks=skipped,
        )


def _required_model_text(model: Any, name: str) -> str:
    value = getattr(model, name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Embedding model 缺少合法的 {name}")
    return value


def _format_embedded_at(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("clock 必须返回 datetime")
    aware = value
    if aware.tzinfo is None or aware.utcoffset() is None:
        aware = aware.replace(tzinfo=timezone.utc)
    return aware.isoformat(timespec="microseconds")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "DomainEmbeddingError",
    "DomainEmbeddingResult",
    "DomainEmbeddingWorker",
    "DomainSkippedVector",
]
