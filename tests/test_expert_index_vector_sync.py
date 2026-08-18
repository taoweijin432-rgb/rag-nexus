"""上游向量差分编排的小型纯内存测试。"""

from __future__ import annotations

import json
from pathlib import Path

from expert_contracts import ExpertChunkEmbeddingInput, adapt_expert_chunk_mapping
from expert_index.vector_sync import ExpertVectorSynchronizer
from rag_vector_store import (
    VectorDeletionResult,
    VectorExpertList,
    VectorIngestionResult,
    VectorItemResult,
    VectorQueryResult,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "expert_chunk_embedding_input_v4.jsonl"


def _records() -> list[ExpertChunkEmbeddingInput]:
    with FIXTURE.open(encoding="utf-8") as stream:
        return [
            adapt_expert_chunk_mapping(json.loads(line))
            for line in stream
            if line.strip()
        ]


class FakeVectorStore:
    """按公共返回契约记录调用，不生成向量或连接 Milvus。"""

    def __init__(
        self,
        rows: dict[str, dict[str, object]],
        *,
        metadata_fallback_ids: set[str] | None = None,
    ) -> None:
        self.rows = rows
        self.metadata_fallback_ids = metadata_fallback_ids or set()
        self.upsert_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.metadata_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.delete_calls: list[list[str]] = []

    def list_vectors_by_expert(self, expert_id: int) -> VectorExpertList:
        ids = [
            vector_doc_id
            for vector_doc_id, row in self.rows.items()
            if row["expert_id"] == expert_id
        ]
        return VectorExpertList(expert_id=expert_id, vector_doc_ids=ids, count=len(ids))

    def get_vectors(self, vector_doc_ids: list[str]) -> VectorQueryResult:
        return VectorQueryResult(
            records=[self.rows[item] for item in vector_doc_ids if item in self.rows],
            missing=[item for item in vector_doc_ids if item not in self.rows],
        )

    def upsert_chunks(
        self,
        chunks: list[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult:
        self.upsert_calls.append(list(chunks))
        return VectorIngestionResult(
            results=[
                VectorItemResult(vector_doc_id=item.vector_doc_id, status="embedded")
                for item in chunks
            ]
        )

    def update_metadata(
        self,
        chunks: list[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult:
        self.metadata_calls.append(list(chunks))
        return VectorIngestionResult(
            results=[
                VectorItemResult(
                    vector_doc_id=item.vector_doc_id,
                    status=(
                        "failed"
                        if item.vector_doc_id in self.metadata_fallback_ids
                        else "metadata_updated"
                    ),
                    recommendation=(
                        "call_upsert"
                        if item.vector_doc_id in self.metadata_fallback_ids
                        else None
                    ),
                )
                for item in chunks
            ]
        )

    def delete_vectors(self, vector_doc_ids: list[str]) -> VectorDeletionResult:
        self.delete_calls.append(list(vector_doc_ids))
        deleted = sum(item in self.rows for item in vector_doc_ids)
        return VectorDeletionResult(
            requested_count=len(vector_doc_ids),
            deleted_count=deleted,
            not_found_count=len(vector_doc_ids) - deleted,
        )


def _row(record: ExpertChunkEmbeddingInput, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "vector_doc_id": record.vector_doc_id,
        "expert_id": record.expert_id,
        "content_hash": record.content_hash,
        "metadata_hash": record.metadata_hash,
    }
    row.update(changes)
    return row


def _copy_with_source_id(
    record: ExpertChunkEmbeddingInput,
    source_field_id: str,
) -> ExpertChunkEmbeddingInput:
    """构造另一条公式仍合法的虚构 Contract，供新增分支测试。"""

    payload = record.model_dump()
    source = dict(payload["source"])
    source["source_field_id"] = source_field_id
    payload["source"] = source
    payload["vector_doc_id"] = (
        f"expert:{record.expert_id}:{record.chunk_type}:{source_field_id}:"
        f"{record.chunk_index}"
    )
    return ExpertChunkEmbeddingInput.model_validate(payload)


def test_sync_expert_routes_hash_changes_and_deletes_stale_vectors() -> None:
    unchanged, metadata_changed, content_changed = _records()
    new_record = _copy_with_source_id(content_changed, "162")
    records = [unchanged, metadata_changed, content_changed, new_record]
    stale_id = "expert:900001:employment:99999:0"
    store = FakeVectorStore(
        {
            unchanged.vector_doc_id: _row(unchanged),
            metadata_changed.vector_doc_id: _row(
                metadata_changed,
                metadata_hash="0" * 64,
            ),
            content_changed.vector_doc_id: _row(
                content_changed,
                content_hash="1" * 64,
            ),
            stale_id: {
                "vector_doc_id": stale_id,
                "expert_id": 900001,
                "content_hash": "2" * 64,
                "metadata_hash": "3" * 64,
            },
        }
    )

    result = ExpertVectorSynchronizer(store).sync_expert(900001, records)

    assert result.embedded_count == 2
    assert result.metadata_updated_count == 1
    assert result.unchanged_count == 1
    assert result.deleted_count == 1
    assert [item.vector_doc_id for item in store.upsert_calls[0]] == [
        content_changed.vector_doc_id,
        new_record.vector_doc_id,
    ]
    assert [item.vector_doc_id for item in store.metadata_calls[0]] == [
        metadata_changed.vector_doc_id
    ]
    assert store.delete_calls == [[stale_id]]


def test_sync_expert_falls_back_to_upsert_and_deletes_rejected_record() -> None:
    metadata_changed = _records()[1]
    rejected = _records()[0]
    rejected = rejected.model_copy(
        update={
            "metadata": rejected.metadata.model_copy(
                update={"verification_status": "rejected"}
            ),
            "metadata_hash": "4" * 64,
        }
    )
    store = FakeVectorStore(
        {
            metadata_changed.vector_doc_id: _row(
                metadata_changed,
                metadata_hash="0" * 64,
            ),
            rejected.vector_doc_id: _row(rejected),
        },
        metadata_fallback_ids={metadata_changed.vector_doc_id},
    )

    result = ExpertVectorSynchronizer(store).sync_expert(
        900001,
        [metadata_changed, rejected],
    )

    assert result.embedded_count == 1
    assert result.metadata_updated_count == 0
    assert [item.vector_doc_id for item in store.upsert_calls[0]] == [
        metadata_changed.vector_doc_id
    ]
    assert store.delete_calls == [[rejected.vector_doc_id]]
