"""VectorIngestionService 公共 Facade 测试。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from expert_contracts import adapt_expert_chunk_mapping
from rag_vector_store import (
    CompanyCloudEmbeddingError,
    ContractBatchIngestionResult,
    ContractBatchIngestionService,
    ContractIngestionResult,
    ContractValidationError,
    EmbeddedContractPreparedChunk,
    MilvusV15RowMapper,
    MutationResult,
    VectorContractError,
    VectorDeletionResult,
    VectorEmbeddingError,
    VectorIngestionResult,
    VectorIngestionService,
    VectorMilvusError,
    VectorQueryResult,
)
from rag_vector_store.chunking.contract_pipeline import ContractEmbeddingWorker
from rag_vector_store.model_space import build_model_space_id

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)


class FakeEmbeddingModel:
    """与仓库测试一致的最小 Embedding 模型（1024 维）。"""

    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


def _row_expert_id(row: dict[str, Any]) -> Any:
    """从 fake row 提取 expert_id，兼容 V15（位于 metadata JSON）。"""

    if "expert_id" in row:
        return row["expert_id"]
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get("expert_id")
    return None


class FakeMilvusRepository:
    """最小 Fake，覆盖 Facade 用到的 Repository 方法。"""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[list[EmbeddedContractPreparedChunk]] = []
        self.delete_calls: list[list[str]] = []
        self.get_calls: list[str] = []
        self.query_calls: list[int] = []
        self.get_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.query_error: Exception | None = None
        self.upsert_error: Exception | None = None

    def get_by_vector_doc_id(
        self,
        vector_doc_id: str,
        *,
        include_embedding: bool = True,
    ) -> dict[str, Any] | None:
        self.get_calls.append(vector_doc_id)
        if self.get_error is not None:
            raise self.get_error
        row = self.rows.get(vector_doc_id)
        if row is None:
            return None
        if not include_embedding:
            return {k: v for k, v in row.items() if k != "embedding"}
        return dict(row)

    def delete_by_vector_doc_ids(
        self,
        vector_doc_ids: Sequence[str],
    ) -> MutationResult:
        self.delete_calls.append(list(vector_doc_ids))
        if self.delete_error is not None:
            raise self.delete_error
        ids = list(vector_doc_ids)
        deleted = [i for i in ids if i in self.rows]
        for i in deleted:
            del self.rows[i]
        return MutationResult(affected_count=len(deleted))

    def upsert_batch(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> MutationResult:
        self.upsert_calls.append(list(records))
        if self.upsert_error is not None:
            raise self.upsert_error
        for record in records:
            self.rows[record.vector_doc_id] = MilvusV15RowMapper().to_row(record)
        return MutationResult(
            affected_count=len(records),
            primary_keys=[r.vector_doc_id for r in records],
        )

    def query_by_expert_id(
        self,
        expert_id: int,
        *,
        output_fields: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        self.query_calls.append(expert_id)
        if self.query_error is not None:
            raise self.query_error
        fields = (
            list(output_fields)
            if output_fields is not None
            else ["vector_doc_id"]
        )
        return [
            {
                field: row[field]
                for field in fields
                if field in row
            }
            for row in self.rows.values()
            if _row_expert_id(row) == expert_id
        ]


def input_records() -> list[Any]:
    import json

    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        return [
            adapt_expert_chunk_mapping(obj)
            for obj in (
                json.loads(line) for line in stream if line.strip()
            )
        ]


def embedded_chunks() -> list[EmbeddedContractPreparedChunk]:
    records = input_records()
    return ContractEmbeddingWorker(
        FakeEmbeddingModel(),
        clock=lambda: datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
    ).embed(records).embedded_chunks


def make_service(
    repo: FakeMilvusRepository,
) -> VectorIngestionService:
    return VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------
def test_upsert_chunks_maps_embedded_and_skipped() -> None:
    repo = FakeMilvusRepository()
    worker = ContractEmbeddingWorker(FakeEmbeddingModel())
    # expert 900001 全部 eligible，no skip
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(worker, repo),
        repository=repo,  # type: ignore[arg-type]
    )
    chunks = input_records()

    result = service.upsert_chunks(chunks)

    assert isinstance(result, VectorIngestionResult)
    assert result.embedded_count == len(chunks)
    assert result.skipped_count == 0
    assert result.total_count == len(chunks)
    assert all(item.status == "embedded" for item in result.results)
    # 向量被真正写入 Fake
    assert len(repo.upsert_calls) == 1
    assert len(repo.upsert_calls[0]) == len(chunks)


def test_upsert_chunks_maps_skipped_via_stub_ingestion() -> None:
    repo = FakeMilvusRepository()
    # 直接给 Facade 注入预设的 ingestion 结果，验证公共映射
    skipped_result = ContractBatchIngestionResult(
        results=[
            ContractIngestionResult(
                status="skipped",
                vector_doc_id="expert:900001:profile_summary:900001:0",
                skip_reason="verification_status=rejected",
            )
        ],
        embedded_count=0,
        skipped_count=1,
        affected_count=0,
    )

    class StubIngestion:
        def ingest_batch(self, records: Sequence[Any]) -> Any:
            return skipped_result

    service = VectorIngestionService(
        ingestion_service=StubIngestion(),  # type: ignore[arg-type]
        repository=repo,  # type: ignore[arg-type]
    )

    result = service.upsert_chunks(input_records())

    assert result.skipped_count == 1
    assert result.results[0].status == "skipped"
    assert result.results[0].error_message == "verification_status=rejected"
    assert repo.upsert_calls == []


def test_upsert_chunks_converts_contract_error() -> None:
    repo = FakeMilvusRepository()

    class StubIngestion:
        def ingest_batch(self, records: Sequence[Any]) -> Any:
            raise ContractValidationError("非法 contract")

    service = VectorIngestionService(
        ingestion_service=StubIngestion(),  # type: ignore[arg-type]
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorContractError):
        service.upsert_chunks(input_records())


def test_upsert_chunks_converts_embedding_error() -> None:
    repo = FakeMilvusRepository()

    class StubIngestion:
        def ingest_batch(self, records: Sequence[Any]) -> Any:
            raise CompanyCloudEmbeddingError("API 超时")

    service = VectorIngestionService(
        ingestion_service=StubIngestion(),  # type: ignore[arg-type]
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorEmbeddingError):
        service.upsert_chunks(input_records())


def test_upsert_chunks_converts_milvus_error() -> None:
    repo = FakeMilvusRepository()

    class StubIngestion:
        def ingest_batch(self, records: Sequence[Any]) -> Any:
            raise RuntimeError("Milvus upsert failed")

    service = VectorIngestionService(
        ingestion_service=StubIngestion(),  # type: ignore[arg-type]
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.upsert_chunks(input_records())


# ---------------------------------------------------------------------------
# delete_vectors
# ---------------------------------------------------------------------------
def test_delete_vectors_success() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunks = embedded_chunks()
    for c in chunks:
        repo.rows[c.vector_doc_id] = MilvusV15RowMapper().to_row(c)
    ids = [c.vector_doc_id for c in chunks]

    result = service.delete_vectors(ids)

    assert isinstance(result, VectorDeletionResult)
    assert result.requested_count == len(ids)
    assert result.deleted_count == len(ids)
    assert result.not_found_count == 0


def test_delete_vectors_missing() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunks = embedded_chunks()
    for c in chunks:
        repo.rows[c.vector_doc_id] = MilvusV15RowMapper().to_row(c)
    present = chunks[0].vector_doc_id
    missing = "expert:999:profile_summary:999:0"

    result = service.delete_vectors([present, missing])

    assert result.requested_count == 2
    assert result.deleted_count == 1
    # pymilvus delete 只返回删除条数，不返回具体未命中主键：
    # 公共结果只报告可靠计数，不伪造具体未命中的 vector_doc_id
    assert result.not_found_count == result.requested_count - result.deleted_count == 1


def test_delete_vectors_not_found_count_only_reports_count() -> None:
    # 未命中 ID 位于输入前部时，仍只报告可靠的数量，不猜测具体身份
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunks = embedded_chunks()
    for c in chunks:
        repo.rows[c.vector_doc_id] = MilvusV15RowMapper().to_row(c)
    missing = "expert:999:profile_summary:999:0"
    present = chunks[0].vector_doc_id

    result = service.delete_vectors([missing, present])

    assert result.requested_count == 2
    assert result.deleted_count == 1
    # 数量精确，不依赖于输入顺序或伪造缺失 ID
    assert result.not_found_count == 1
    assert getattr(result, "not_found", None) is None  # 无伪造具体 ID 字段


def test_delete_vectors_empty_is_safe() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    result = service.delete_vectors([])
    assert result.requested_count == 0
    assert result.deleted_count == 0
    assert result.not_found_count == 0
    assert repo.delete_calls == []


def test_delete_vectors_dedupes_via_disable() -> None:
    """delete_vectors 去重且逐条只读一次（逻辑下线路径）。"""

    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunk = embedded_chunks()[0]
    repo.rows[chunk.vector_doc_id] = MilvusV15RowMapper().to_row(chunk)

    result = service.delete_vectors([chunk.vector_doc_id, chunk.vector_doc_id])

    assert result.requested_count == 1
    assert result.deleted_count == 1
    # 不触发物理删除；去重后只读取一次。
    assert repo.delete_calls == []
    assert repo.get_calls == [chunk.vector_doc_id]
    assert chunk.vector_doc_id in repo.rows


def test_disable_upsert_error_converts_to_milvus_error() -> None:
    """逻辑下线回写失败时按 Milvus 层故障路由公共异常。"""

    repo = FakeMilvusRepository()
    repo.upsert_error = RuntimeError("galaxy down")
    service = make_service(repo)
    chunk = embedded_chunks()[0]
    repo.rows[chunk.vector_doc_id] = MilvusV15RowMapper().to_row(chunk)

    with pytest.raises(VectorMilvusError):
        service.delete_vectors([chunk.vector_doc_id])


# ---------------------------------------------------------------------------
# get_vectors
# ---------------------------------------------------------------------------
def test_get_vectors_existing_and_missing() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunks = embedded_chunks()
    for c in chunks:
        repo.rows[c.vector_doc_id] = MilvusV15RowMapper().to_row(c)
    missing = "expert:999:profile_summary:999:0"
    ids = [chunks[0].vector_doc_id, missing, chunks[1].vector_doc_id]

    result = service.get_vectors(ids)

    assert isinstance(result, VectorQueryResult)
    assert [r["vector_doc_id"] for r in result.records] == [
        chunks[0].vector_doc_id,
        chunks[1].vector_doc_id,
    ]
    assert result.missing == [missing]
    # 不暴露 embedding
    assert all("embedding" not in record for record in result.records)


def test_get_vectors_dedupes_and_orders() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunks = embedded_chunks()
    for c in chunks:
        repo.rows[c.vector_doc_id] = MilvusV15RowMapper().to_row(c)
    ids = [chunks[1].vector_doc_id, chunks[0].vector_doc_id, chunks[1].vector_doc_id]

    result = service.get_vectors(ids)

    assert [r["vector_doc_id"] for r in result.records] == [
        chunks[1].vector_doc_id,
        chunks[0].vector_doc_id,
    ]


def _v15_repo_with(chunks: list[EmbeddedContractPreparedChunk]) -> FakeMilvusRepository:
    """以 V15 物理 row 形态存储：hash 仅存在于 metadata JSON，顶层严格 4 字段。"""

    repo = FakeMilvusRepository()
    mapper = MilvusV15RowMapper()
    for c in chunks:
        repo.rows[c.vector_doc_id] = mapper.to_row(c)
    return repo


def test_get_vectors_v15_projects_hash_from_metadata_to_top() -> None:
    """V15 物理 row hash 位于 metadata JSON；get_vectors 逻辑返回顶层应含哈希。"""

    repo = _v15_repo_with(embedded_chunks())
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunk = embedded_chunks()[0]

    result = service.get_vectors([chunk.vector_doc_id])

    assert len(result.records) == 1
    record = result.records[0]
    # 物理顶层仍不含哈希（投影只作用于逻辑返回，不改物理 row）
    assert "content_hash" not in repo.rows[chunk.vector_doc_id]
    assert "metadata_hash" not in repo.rows[chunk.vector_doc_id]
    assert "source_hash" not in repo.rows[chunk.vector_doc_id]
    # 逻辑返回顶层已投影三个哈希，且与 metadata 中存储的值一致
    flat = repo.rows[chunk.vector_doc_id]["metadata"]
    assert record["content_hash"] == flat["content_hash"]
    assert record["metadata_hash"] == flat["metadata_hash"]
    assert record["source_hash"] == flat["source_hash"]
    assert record["content_hash"] == chunk.content_hash
    assert record["metadata_hash"] == chunk.metadata_hash
    assert record["source_hash"] == chunk.source.source_hash
    # 仍不暴露 embedding
    assert "embedding" not in record
    # 只投影三个哈希，不平铺其它 metadata 字段
    for other in ("expert_id", "chunk_type", "is_active"):
        assert other not in record


def test_get_vectors_v15_supports_existing_rows_idempotent_read() -> None:
    """V15 get_vectors 逻辑返回可被 ExpertVectorSynchronizer._existing_rows 读取。"""

    from expert_index.vector_sync import ExpertVectorSynchronizer

    chunks = embedded_chunks()
    repo = _v15_repo_with(chunks)
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    ids = sorted(c.vector_doc_id for c in chunks)

    result = service.get_vectors(ids)

    # 上游增量幂等判断不再因顶层缺哈希而抛错
    existing_by_id = ExpertVectorSynchronizer._existing_rows(result.records)
    assert set(existing_by_id) == set(ids)
    for vector_doc_id, item in existing_by_id.items():
        original = next(c for c in chunks if c.vector_doc_id == vector_doc_id)
        assert item["content_hash"] == original.content_hash
        assert item["metadata_hash"] == original.metadata_hash


def test_get_vectors_v15_missing_content_hash_fails_closed() -> None:
    """metadata JSON 缺 content_hash 时，顶层无哈希，上游 _existing_rows fail-closed。"""

    from expert_index.vector_sync import ExpertVectorSynchronizer, VectorSyncError

    chunks = embedded_chunks()
    repo = _v15_repo_with(chunks)
    # 破坏第一个 chunk 物理 row 的 metadata，移除 content_hash
    first_id = chunks[0].vector_doc_id
    repo.rows[first_id]["metadata"] = {
        k: v for k, v in dict(repo.rows[first_id]["metadata"]).items()
        if k != "content_hash"
    }
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )

    result = service.get_vectors([first_id])

    # 投影不补造哈希：逻辑返回顶层仍缺 content_hash
    assert "content_hash" not in result.records[0]
    with pytest.raises(VectorSyncError):
        ExpertVectorSynchronizer._existing_rows(result.records)


def test_get_vectors_v15_missing_metadata_hash_fails_closed() -> None:
    """metadata JSON 缺 metadata_hash 时，顶层无哈希，上游 _existing_rows fail-closed。"""

    from expert_index.vector_sync import ExpertVectorSynchronizer, VectorSyncError

    chunks = embedded_chunks()
    repo = _v15_repo_with(chunks)
    first_id = chunks[0].vector_doc_id
    repo.rows[first_id]["metadata"] = {
        k: v for k, v in dict(repo.rows[first_id]["metadata"]).items()
        if k != "metadata_hash"
    }
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )

    result = service.get_vectors([first_id])

    assert "metadata_hash" not in result.records[0]
    with pytest.raises(VectorSyncError):
        ExpertVectorSynchronizer._existing_rows(result.records)


def test_get_vectors_v15_does_not_mutate_physical_row() -> None:
    """投影只作用于逻辑返回；原始 V15 repository row 顶层仍严格无哈希。"""

    chunks = embedded_chunks()
    repo = _v15_repo_with(chunks)
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunk = chunks[0]
    physical_before = dict(repo.rows[chunk.vector_doc_id])
    metadata_before = dict(physical_before["metadata"])

    service.get_vectors([chunk.vector_doc_id])

    physical_after = repo.rows[chunk.vector_doc_id]
    assert set(physical_after.keys()) == set(physical_before.keys())
    assert "content_hash" not in physical_after
    assert "metadata_hash" not in physical_after
    assert dict(physical_after["metadata"]) == metadata_before


# ---------------------------------------------------------------------------
# update_metadata
# ---------------------------------------------------------------------------
def _repo_with_one(chunk: EmbeddedContractPreparedChunk) -> FakeMilvusRepository:
    repo = FakeMilvusRepository()
    repo.rows[chunk.vector_doc_id] = MilvusV15RowMapper().to_row(chunk)
    return repo


def test_update_metadata_success_reuses_old_vector() -> None:
    old_chunk = embedded_chunks()[0]
    repo = _repo_with_one(old_chunk)
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunk = input_records()[0]
    # 模拟 metadata-only 变更：metadata_hash 变化（chunk_text / content_hash 不变）
    new_metadata_hash = "1" * 64
    new_chunk = chunk.model_copy(update={"metadata_hash": new_metadata_hash})

    result = service.update_metadata([new_chunk])

    assert result.results[0].status == "metadata_updated"
    assert result.metadata_updated_count == 1
    assert result.failed_count == 0
    assert len(repo.upsert_calls) == 1
    written = repo.upsert_calls[0][0]
    # 真实复用旧 vector：写入的 embedding / 模型信息来自旧 Milvus Entity
    assert written.embedding == old_chunk.embedding
    assert written.embedding_dim == old_chunk.embedding_dim
    assert written.embedding_model_alias == old_chunk.embedding_model_alias
    assert written.embedding_model_version == old_chunk.embedding_model_version
    # 写入新的 metadata_hash / content_hash：来自本次调用传入的新 Chunk
    assert written.metadata_hash == new_metadata_hash
    assert written.metadata_hash != old_chunk.metadata_hash
    assert written.content_hash == chunk.content_hash
    # 旧 metadata 不被错误保留，使用新 Chunk 的 metadata
    assert written.metadata.model_dump() == chunk.metadata.model_dump()
    assert written.vector_doc_id == chunk.vector_doc_id


def test_update_metadata_does_not_call_embedding() -> None:
    repo = _repo_with_one(embedded_chunks()[0])
    embed_calls: list[int] = []

    class TrackingWorker(ContractEmbeddingWorker):
        def embed(self, records: Any) -> Any:
            embed_calls.append(1)
            return super().embed(records)

    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            TrackingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )

    service.update_metadata(input_records())

    assert embed_calls == []


def test_update_metadata_entity_missing() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunk = input_records()[0]

    result = service.update_metadata([chunk])

    assert result.results[0].status == "failed"
    assert result.results[0].recommendation == "call_upsert"
    assert repo.upsert_calls == []


def test_update_metadata_model_space_mismatch() -> None:
    old = embedded_chunks()[0]
    old = old.model_copy(
        update={"embedding_model_version": "v2"}
    )
    repo = _repo_with_one(old)
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )

    result = service.update_metadata([input_records()[0]])

    assert result.results[0].status == "failed"
    assert result.results[0].recommendation == "call_upsert"
    assert "model_space" in (result.results[0].error_message or "")
    assert repo.upsert_calls == []


def test_update_metadata_dimension_mismatch() -> None:
    old = embedded_chunks()[0]
    repo = _repo_with_one(old)
    # 注入的正式维度与 model_space 编码维度不一致，隔离测试独立的维度安全门。
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
        expected_dimension=768,
        formal_model_space_id=build_model_space_id(
            model_alias="alibaba-bailian:text-embedding-v4",
            dimension=1024,
            model_version="v1",
        ),
    )

    result = service.update_metadata([input_records()[0]])

    assert result.results[0].status == "failed"
    assert result.results[0].recommendation == "call_upsert"
    assert "embedding_dim" in (result.results[0].error_message or "")
    assert repo.upsert_calls == []


def test_update_metadata_converts_contract_error() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    # 非 Contract 输入（不是 ExpertChunkEmbeddingInput）
    with pytest.raises(VectorContractError):
        service.update_metadata(["not-a-chunk"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# update_metadata：metadata-only 安全边界（Task6-1A）
# ---------------------------------------------------------------------------
def test_update_metadata_content_hash_change_fails_closed() -> None:
    """输入的 content_hash 与 existing 不一致 → 必须抛 VectorContractError。

    输入 chunk 本身是合法 Contract（content_hash == sha256(chunk_text)）；
    existing 实体的 content_hash 被异常篡改，导致两者不一致。
    """

    old_chunk = embedded_chunks()[0]
    repo = _repo_with_one(old_chunk)
    # 篕改 existing row 的 content_hash（V15 中位于 metadata JSON），模拟 existing
    # 实体状态异常
    tampered_row = dict(repo.rows[old_chunk.vector_doc_id])
    tampered_metadata = dict(tampered_row["metadata"])
    tampered_metadata["content_hash"] = "b" * 64
    tampered_row["metadata"] = tampered_metadata
    repo.rows[old_chunk.vector_doc_id] = tampered_row
    service = make_service(repo)

    with pytest.raises(VectorContractError) as exc_info:
        service.update_metadata([input_records()[0]])

    assert "content_hash 不一致" in str(exc_info.value)
    assert "metadata-only" in str(exc_info.value)
    assert "upsert_chunks" in str(exc_info.value)
    # 未复用旧向量写入、未调用 Embedding
    assert repo.upsert_calls == []


def test_update_metadata_chunk_text_change_fails_closed() -> None:
    """输入的 chunk_text 与 existing 不一致 → 必须抛 VectorContractError。

    输入 chunk 本身是合法 Contract；existing 实体的 chunk_text 被异常篡改。
    """

    old_chunk = embedded_chunks()[0]
    repo = _repo_with_one(old_chunk)
    # 篕改 existing row 的 chunk_text，模拟 existing 实体状态异常
    tampered_row = dict(repo.rows[old_chunk.vector_doc_id])
    tampered_row["chunk_text"] = old_chunk.chunk_text + " 篕改"
    repo.rows[old_chunk.vector_doc_id] = tampered_row
    service = make_service(repo)

    with pytest.raises(VectorContractError) as exc_info:
        service.update_metadata([input_records()[0]])

    assert "chunk_text 不一致" in str(exc_info.value)
    assert "metadata-only" in str(exc_info.value)
    assert "upsert_chunks" in str(exc_info.value)
    assert repo.upsert_calls == []


def test_update_metadata_metadata_only_change_succeeds() -> None:
    """正常 metadata-only 更新（metadata_hash 变化、正文/主键不变）→ 成功。"""

    old_chunk = embedded_chunks()[0]
    repo = _repo_with_one(old_chunk)
    service = make_service(repo)
    chunk = input_records()[0]
    new_metadata_hash = "1" * 64
    new_chunk = chunk.model_copy(update={"metadata_hash": new_metadata_hash})

    result = service.update_metadata([new_chunk])

    assert result.results[0].status == "metadata_updated"
    assert result.metadata_updated_count == 1
    assert result.failed_count == 0
    assert len(repo.upsert_calls) == 1
    written = repo.upsert_calls[0][0]
    # 正文 / 主键 / content_hash 保持与 existing 一致
    assert written.vector_doc_id == old_chunk.vector_doc_id
    assert written.chunk_text == old_chunk.chunk_text
    assert written.content_hash == old_chunk.content_hash
    # 新 metadata_hash 生效
    assert written.metadata_hash == new_metadata_hash


def test_update_metadata_metadata_only_reuses_existing_embedding() -> None:
    """正常 metadata-only 更新 → 复用 existing embedding / 模型字段。"""

    old_chunk = embedded_chunks()[0]
    repo = _repo_with_one(old_chunk)
    service = make_service(repo)
    chunk = input_records()[0]
    new_chunk = chunk.model_copy(update={"metadata_hash": "1" * 64})

    service.update_metadata([new_chunk])

    written = repo.upsert_calls[0][0]
    assert written.embedding == old_chunk.embedding
    assert written.embedding_dim == old_chunk.embedding_dim
    assert written.embedding_model_alias == old_chunk.embedding_model_alias
    assert written.embedding_model_version == old_chunk.embedding_model_version
    assert written.embedded_at == old_chunk.embedded_at


def test_update_metadata_metadata_only_does_not_call_embedding() -> None:
    """正常 metadata-only 更新 → Embedding 调用次数为 0。"""

    old_chunk = embedded_chunks()[0]
    repo = _repo_with_one(old_chunk)
    embed_calls: list[int] = []

    class TrackingWorker(ContractEmbeddingWorker):
        def embed(self, records: Any) -> Any:
            embed_calls.append(1)
            return super().embed(records)

    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            TrackingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunk = input_records()[0]
    new_chunk = chunk.model_copy(update={"metadata_hash": "1" * 64})

    service.update_metadata([new_chunk])

    assert embed_calls == []


# ---------------------------------------------------------------------------
# list_vectors_by_expert
# ---------------------------------------------------------------------------
def test_list_vectors_by_expert() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    chunks = embedded_chunks()
    for c in chunks:
        repo.rows[c.vector_doc_id] = MilvusV15RowMapper().to_row(c)

    result = service.list_vectors_by_expert(900001)

    assert result.expert_id == 900001
    assert result.count == len(chunks)
    assert set(result.vector_doc_ids) == {c.vector_doc_id for c in chunks}


def test_list_vectors_by_expert_empty() -> None:
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    result = service.list_vectors_by_expert(12345)
    assert result.vector_doc_ids == []
    assert result.count == 0


def test_list_vectors_by_expert_converts_error() -> None:
    repo = FakeMilvusRepository()
    repo.query_error = RuntimeError("query failed")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.list_vectors_by_expert(900001)

# ---------------------------------------------------------------------------
# P0 异常边界：TypeError / RuntimeError / ValidationError 统一映射公共异常
# ---------------------------------------------------------------------------
def test_upsert_chunks_type_error_from_repo_converted() -> None:
    repo = FakeMilvusRepository()
    repo.upsert_error = TypeError("Milvus mutation response is not a Mapping")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.upsert_chunks(input_records())


def test_upsert_chunks_runtime_error_from_repo_converted() -> None:
    repo = FakeMilvusRepository()
    repo.upsert_error = RuntimeError("Milvus upsert failed: collection not loaded")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.upsert_chunks(input_records())


def test_delete_vectors_get_type_error_converted() -> None:
    """逻辑下线读取旧实体失败时按 Milvus 层故障路由公共异常。"""

    repo = FakeMilvusRepository()
    repo.get_error = TypeError("Milvus query returned invalid entity structure")
    service = make_service(repo)
    with pytest.raises(VectorMilvusError):
        service.delete_vectors(["expert:900001:profile_summary:900001:0"])


def test_delete_vectors_get_value_error_is_milvus_error() -> None:
    """逻辑下线读取旧实体的 ValueError 同样路由为 Milvus 层故障。"""

    repo = FakeMilvusRepository()
    repo.get_error = ValueError("Milvus delete 失败")
    service = make_service(repo)
    with pytest.raises(VectorMilvusError):
        service.delete_vectors(["expert:900001:profile_summary:900001:0"])


# ---------------------------------------------------------------------------
# disable_vectors（统一逻辑下线）
# ---------------------------------------------------------------------------
def test_disable_vectors_marks_inactive_and_preserves_payload() -> None:
    """disable 成功：实体保留，仅 is_active 变化，向量/文本/历史 metadata 不变。"""

    repo = FakeMilvusRepository()
    service = make_service(repo)
    chunk = embedded_chunks()[0]
    original_row = MilvusV15RowMapper().to_row(chunk)
    repo.rows[chunk.vector_doc_id] = dict(original_row)

    result = service.disable_vectors([chunk.vector_doc_id])

    assert isinstance(result, VectorDeletionResult)
    assert result.requested_count == 1
    assert result.deleted_count == 1
    assert result.not_found_count == 0
    # 实体仍在，无物理删除调用。
    assert repo.delete_calls == []
    new_row = repo.rows[chunk.vector_doc_id]
    assert new_row is not None
    restored = MilvusV15RowMapper().from_row(new_row)
    assert restored.is_active is False
    # 向量、正文与全部历史内容保持不变。
    assert restored.embedding == chunk.embedding
    assert restored.chunk_text == chunk.chunk_text
    assert restored.content_hash == chunk.content_hash
    assert restored.metadata_hash == chunk.metadata_hash
    assert restored.metadata == chunk.metadata
    assert restored.embedding_model_alias == chunk.embedding_model_alias
    assert restored.embedding_dim == chunk.embedding_dim


def test_disable_vectors_is_idempotent_on_replay() -> None:
    """重复 disable 幂等：已下线实体不再触发写入。"""

    repo = FakeMilvusRepository()
    service = make_service(repo)
    chunk = embedded_chunks()[0]
    repo.rows[chunk.vector_doc_id] = MilvusV15RowMapper().to_row(chunk)

    first = service.disable_vectors([chunk.vector_doc_id])
    upserts_after_first = len(repo.upsert_calls)
    second = service.disable_vectors([chunk.vector_doc_id])

    assert first.deleted_count == 1
    assert second.deleted_count == 1
    assert second.not_found_count == 0
    # 重放零写入。
    assert len(repo.upsert_calls) == upserts_after_first


def test_disable_vectors_missing_ids_counted_not_found() -> None:
    """不存在的 id 计入 not_found，不报错（幂等语义）。"""

    repo = FakeMilvusRepository()
    service = make_service(repo)

    result = service.disable_vectors(
        ["expert:999:profile_summary:999:0", "expert:998:employment:998:0"]
    )

    assert result.requested_count == 2
    assert result.deleted_count == 0
    assert result.not_found_count == 2
    assert repo.upsert_calls == []


def test_disabled_chunk_can_be_reupserted_active() -> None:
    """已 disable 的同 vector_doc_id 重新 upsert 后恢复 active。

    覆盖"disabled 数据不阻止同 vector_doc_id 重新出现"：upsert 以主键
    覆盖写回，is_active 随新输入恢复为 true。
    """

    repo = FakeMilvusRepository()
    service = make_service(repo)
    chunks = embedded_chunks()
    inputs = input_records()
    target_input = inputs[0]
    target_id = chunks[0].vector_doc_id
    repo.rows[target_id] = MilvusV15RowMapper().to_row(chunks[0])

    disabled = service.disable_vectors([target_id])
    assert disabled.deleted_count == 1
    assert (
        MilvusV15RowMapper().from_row(repo.rows[target_id]).is_active is False
    )

    reupsert = service.upsert_chunks([target_input])

    assert reupsert.embedded_count == 1
    restored = MilvusV15RowMapper().from_row(repo.rows[target_id])
    assert restored.is_active is True
    assert restored.content_hash == chunks[0].content_hash


def test_list_vectors_by_expert_includes_disabled_without_blocking_upsert() -> None:
    """list 默认结果包含 disabled id，但不阻止同一 Chunk 重新 upsert。"""

    repo = FakeMilvusRepository()
    service = make_service(repo)
    chunks = embedded_chunks()
    inputs = input_records()
    for c in chunks:
        repo.rows[c.vector_doc_id] = MilvusV15RowMapper().to_row(c)
    target_id = chunks[0].vector_doc_id
    expert_id = chunks[0].expert_id

    service.disable_vectors([target_id])

    listed = service.list_vectors_by_expert(expert_id)
    # disabled 实体仍被列出（保留在 Collection 内），供 diff 与审计。
    assert target_id in listed.vector_doc_ids

    # 列出并不阻止恢复：重新 upsert 成功。
    reupsert = service.upsert_chunks([inputs[0]])
    assert reupsert.embedded_count == 1
    assert (
        MilvusV15RowMapper().from_row(repo.rows[target_id]).is_active is True
    )


def test_get_vectors_type_error_from_repo_converted() -> None:
    repo = FakeMilvusRepository()
    repo.get_error = TypeError("Milvus query returned invalid entity structure")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.get_vectors(["expert:900001:profile_summary:900001:0"])


def test_list_vectors_by_expert_type_error_converted() -> None:
    repo = FakeMilvusRepository()
    repo.query_error = TypeError("Milvus expert_id iterator returned invalid batch")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.list_vectors_by_expert(900001)


def _repo_with_corrupted_row(chunk: EmbeddedContractPreparedChunk) -> FakeMilvusRepository:
    repo = FakeMilvusRepository()
    row = MilvusV15RowMapper().to_row(chunk)
    row.pop("metadata")  # 破坏反向映射所需的 flat metadata JSON
    repo.rows[chunk.vector_doc_id] = row
    return repo


def test_update_metadata_from_row_type_error_converted() -> None:
    repo = _repo_with_corrupted_row(embedded_chunks()[0])
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    # Milvus 行损坏时 from_row 抛 TypeError，应统一转换为公共异常而非泄漏
    with pytest.raises(VectorMilvusError):
        service.update_metadata([input_records()[0]])


def test_update_metadata_upsert_batch_runtime_error_converted() -> None:
    repo = _repo_with_one(embedded_chunks()[0])
    repo.upsert_error = RuntimeError("Milvus upsert failed: collection not loaded")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.update_metadata([input_records()[0]])


def test_update_metadata_upsert_batch_type_error_converted() -> None:
    repo = _repo_with_one(embedded_chunks()[0])
    repo.upsert_error = TypeError("Milvus mutation response is not a Mapping")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.update_metadata([input_records()[0]])


def test_update_metadata_upsert_batch_validation_error_converted() -> None:
    from pydantic import ValidationError

    repo = _repo_with_one(embedded_chunks()[0])
    try:
        MutationResult(affected_count=-1)
    except ValidationError as exc:
        repo.upsert_error = exc
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.update_metadata([input_records()[0]])


def test_public_error_not_re_wrapped() -> None:
    """已属于公共异常的异常应当原样透传，不重复包装。"""
    repo = FakeMilvusRepository()

    class StubIngestion:
        def ingest_batch(self, records: Sequence[Any]) -> Any:
            raise VectorMilvusError("已转换的 Milvus 错误")

    service = VectorIngestionService(
        ingestion_service=StubIngestion(),  # type: ignore[arg-type]
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError) as excinfo:
        service.upsert_chunks(input_records())
    assert isinstance(excinfo.value, VectorMilvusError)


# ---------------------------------------------------------------------------
# 异常分类收紧：调用参数 / Contract 错误不能映射为 Milvus 错误；
# repository / Milvus 层 ValueError 才映射为 VectorMilvusError
# ---------------------------------------------------------------------------
def test_upsert_chunks_illegal_input_is_contract_error() -> None:
    """非 Contract 输入（调用方参数错误）必须是 VectorContractError，而非 VectorMilvusError。"""
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorContractError):
        service.upsert_chunks([{"not": "a contract"}])  # type: ignore[list-item]


def test_upsert_chunks_contract_validation_failure_is_contract_error() -> None:
    """Contract 校验失败（如 content_hash 与 chunk_text 不一致）映射为
    VectorContractError 而非 VectorMilvusError。"""

    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    bad = input_records()[0].model_copy(update={"content_hash": "0" * 64})
    with pytest.raises(VectorContractError):
        service.upsert_chunks([bad])


def test_get_vectors_repository_value_error_is_milvus_error() -> None:
    """repository 层 ValueError 映射为 VectorMilvusError。"""
    repo = FakeMilvusRepository()
    repo.get_error = ValueError("Milvus query 非法 Entity")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.get_vectors(["expert:900001:profile_summary:900001:0"])


def test_list_vectors_by_expert_bad_expert_id_is_contract_error() -> None:
    """非法 expert_id（调用方参数错误）映射为 VectorContractError，而非 VectorMilvusError。"""
    repo = FakeMilvusRepository()
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorContractError):
        service.list_vectors_by_expert("900001")  # type: ignore[arg-type]
    with pytest.raises(VectorContractError):
        service.list_vectors_by_expert(0)


def test_list_vectors_by_expert_repository_value_error_is_milvus_error() -> None:
    repo = FakeMilvusRepository()
    repo.query_error = ValueError("Milvus query 非法")
    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.list_vectors_by_expert(900001)


def test_update_metadata_from_row_validation_error_is_milvus_error() -> None:
    """损坏 Milvus 行导致 from_row 抛出异常（TypeError/ValidationError）时映射为
    VectorMilvusError，不泄漏原始异常。"""

    repo = FakeMilvusRepository()
    # 构造一个 from_row 会抛异常的非法行（缺 flat metadata JSON）
    chunk = embedded_chunks()[0]
    bad_row = {
        "vector_doc_id": chunk.vector_doc_id,
        "contract_version": "4.1.0",
    }
    repo.rows[chunk.vector_doc_id] = bad_row

    service = VectorIngestionService(
        ingestion_service=ContractBatchIngestionService(
            ContractEmbeddingWorker(FakeEmbeddingModel()), repo
        ),
        repository=repo,  # type: ignore[arg-type]
    )
    with pytest.raises(VectorMilvusError):
        service.update_metadata([input_records()[0]])
