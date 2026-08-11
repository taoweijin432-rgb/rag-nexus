"""chunk、embedding 和 repository 之间的批处理编排。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from typing import Any

from .chunking.contracts import EmbeddedPreparedChunk, PreparedChunk
from .embedding import EmbeddingModel
from .models import Chunk, EmbeddedChunk
from .ports import ChunkRepository

_NO_CHUNK = object()


@dataclass(frozen=True)
class IngestionResult:
    """一次入库运行的统计结果。"""

    total_chunks: int
    total_batches: int
    inserted_count: int


class ChunkIngestionService:
    """按批次向量化 chunk，并把每批交给 repository。"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        repository: ChunkRepository,
        *,
        batch_size: int = 64,
    ) -> None:
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.embedding_model = embedding_model
        self.repository = repository
        self.batch_size = batch_size

    def ingest(
        self,
        chunks: Iterable[Chunk | PreparedChunk | Mapping[str, Any]],
    ) -> IngestionResult:
        """兼容旧 Chunk，并自动分派 PreparedChunk 入库路径。"""

        iterator = iter(chunks)
        first = next(iterator, _NO_CHUNK)
        if first is _NO_CHUNK:
            return self._ingest_legacy(())
        if isinstance(first, PreparedChunk):
            return self.ingest_prepared(chain((first,), iterator))
        return self._ingest_legacy(chain((first,), iterator))

    def ingest_prepared(self, chunks: Iterable[PreparedChunk]) -> IngestionResult:
        """批量向量化 PreparedChunk，并通过 upsert_batch 写入 repository。"""

        dimension = _get_embedding_dimension(self.embedding_model)
        self.repository.initialize(dimension)

        total_chunks = 0
        total_batches = 0
        inserted_count = 0
        iterator = iter(chunks)

        while True:
            batch = list(islice(iterator, self.batch_size))
            if not batch:
                break

            if not all(isinstance(item, PreparedChunk) for item in batch):
                raise TypeError("ingest_prepared 只接受 PreparedChunk")
            texts = [item.chunk_text for item in batch]
            raw_embeddings = self.embedding_model.embed(texts)
            embedded_chunks = _make_embedded_prepared_chunks(
                batch,
                raw_embeddings,
                dimension,
            )
            result = self.repository.upsert_batch(embedded_chunks)

            total_chunks += len(embedded_chunks)
            total_batches += 1
            inserted_count += result.affected_count

        return IngestionResult(
            total_chunks=total_chunks,
            total_batches=total_batches,
            inserted_count=inserted_count,
        )

    def _ingest_legacy(
        self,
        chunks: Iterable[Chunk | Mapping[str, Any]],
    ) -> IngestionResult:
        """保留旧 Chunk 的 insert_batch 入库行为。"""

        dimension = _get_embedding_dimension(self.embedding_model)
        self.repository.initialize(dimension)

        total_chunks = 0
        total_batches = 0
        inserted_count = 0
        iterator = iter(chunks)

        while True:
            batch = list(islice(iterator, self.batch_size))
            if not batch:
                break

            normalized_chunks = [_coerce_chunk(item) for item in batch]
            texts = [item.text for item in normalized_chunks]
            raw_embeddings = self.embedding_model.embed(texts)
            embedded_chunks = _make_embedded_chunks(
                normalized_chunks,
                raw_embeddings,
                dimension,
            )
            result = self.repository.insert_batch(embedded_chunks)

            total_chunks += len(embedded_chunks)
            total_batches += 1
            inserted_count += result.affected_count

        return IngestionResult(
            total_chunks=total_chunks,
            total_batches=total_batches,
            inserted_count=inserted_count,
        )


def _coerce_chunk(value: Chunk | Mapping[str, Any]) -> Chunk:
    if isinstance(value, Chunk):
        return value
    return Chunk.model_validate(value)


def _get_embedding_dimension(embedding_model: EmbeddingModel) -> int:
    dimension = getattr(embedding_model, "dimension", None)
    if callable(dimension):
        dimension = dimension()
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("embedding_model.dimension must be a positive integer")
    return dimension


def _make_embedded_chunks(
    chunks: Sequence[Chunk],
    raw_embeddings: Sequence[Sequence[float]],
    dimension: int,
) -> list[EmbeddedChunk]:
    embeddings = list(raw_embeddings)
    if len(embeddings) != len(chunks):
        raise ValueError(
            "embedding_model.embed must return exactly one vector per input text "
            f"(expected {len(chunks)}, got {len(embeddings)})"
        )

    embedded_chunks: list[EmbeddedChunk] = []
    for chunk, raw_embedding in zip(chunks, embeddings):
        embedding = [float(value) for value in raw_embedding]
        if len(embedding) != dimension:
            raise ValueError(
                f"embedding dimension mismatch for chunk {chunk.chunk_id!r}: "
                f"expected {dimension}, got {len(embedding)}"
            )
        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                embedding=embedding,
            )
        )
    return embedded_chunks


def _make_embedded_prepared_chunks(
    chunks: Sequence[PreparedChunk],
    raw_embeddings: Sequence[Sequence[float]],
    dimension: int,
) -> list[EmbeddedPreparedChunk]:
    embeddings = list(raw_embeddings)
    if len(embeddings) != len(chunks):
        raise ValueError(
            "embedding_model.embed 必须为每条 PreparedChunk 返回一个向量 "
            f"(expected {len(chunks)}, got {len(embeddings)})"
        )

    embedded_chunks: list[EmbeddedPreparedChunk] = []
    for chunk, raw_embedding in zip(chunks, embeddings):
        embedding = [float(value) for value in raw_embedding]
        if len(embedding) != dimension:
            raise ValueError(
                f"chunk {chunk.vector_doc_id!r} 的 embedding 维度不匹配: "
                f"expected {dimension}, got {len(embedding)}"
            )
        embedded_chunks.append(
            EmbeddedPreparedChunk(
                **chunk.model_dump(),
                embedding=embedding,
            )
        )
    return embedded_chunks
