"""让入库服务与 Milvus 解耦的 repository 接口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .chunking.contracts import EmbeddedPreparedChunk
from .models import EmbeddedChunk, MutationResult

RepositoryRecord = EmbeddedChunk | EmbeddedPreparedChunk


class ChunkRepository(Protocol):
    """由 :class:`ChunkIngestionService` 使用的持久化边界。"""

    def initialize(self, vector_dimension: int) -> None:
        """建立连接，并确保目标集合 schema 兼容。"""

        ...

    def insert_batch(self, records: Sequence[RepositoryRecord]) -> MutationResult:
        """持久化一批已经完成向量化的 chunk。"""

        ...

    def upsert_batch(self, records: Sequence[RepositoryRecord]) -> MutationResult:
        """根据主键插入或替换一批记录。"""

        ...

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> MutationResult:
        """根据稳定 ID 删除 chunk。"""

        ...
