"""上游依赖的向量能力端口。

生产环境由 ``rag_vector_store.VectorIngestionService`` 实现，单元测试可注入纯内存
Fake。Pipeline 因而不依赖 Milvus Repository、Embedding Worker 等下游内部类型。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from expert_contracts import ExpertChunkEmbeddingInput
from rag_vector_store import (
    VectorDeletionResult,
    VectorExpertList,
    VectorIngestionResult,
    VectorQueryResult,
)

if TYPE_CHECKING:
    from .vector_sync import ExpertVectorSyncResult


@dataclass(frozen=True)
class ExpertBatchSyncInput:
    """批量同步的单专家输入。"""

    expert_id: int
    records: Sequence[ExpertChunkEmbeddingInput]


class VectorStorePort(Protocol):
    """下游公开五个 Python 方法的最小结构协议。"""

    def upsert_chunks(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult: ...

    def update_metadata(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult: ...

    def delete_vectors(
        self,
        vector_doc_ids: Sequence[str],
    ) -> VectorDeletionResult: ...

    def get_vectors(
        self,
        vector_doc_ids: Sequence[str],
    ) -> VectorQueryResult: ...

    def list_vectors_by_expert(self, expert_id: int) -> VectorExpertList: ...


class ExpertVectorSyncPort(Protocol):
    """Pipeline 依赖的专家快照同步端口。"""

    def sync_expert(
        self,
        expert_id: int,
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> ExpertVectorSyncResult: ...

    def sync_expert_batch(
        self,
        experts: Sequence[ExpertBatchSyncInput],
    ) -> list[ExpertVectorSyncResult]: ...


__all__ = ["ExpertBatchSyncInput", "ExpertVectorSyncPort", "VectorStorePort"]
