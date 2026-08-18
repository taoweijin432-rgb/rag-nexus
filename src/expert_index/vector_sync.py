"""依据上游当前快照调用下游公共方法同步一位专家的向量。

本模块只做增量决策与结果对齐：正文变化走完整向量化，metadata-only 复用旧向量，
不变记录跳过，陈旧或不再 eligible 的记录删除。它不读取业务 MySQL，也不直接调用
Milvus Repository。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from expert_contracts import ExpertChunkEmbeddingInput
from rag_vector_store import VectorIngestionResult, VectorItemResult

from .ports import VectorStorePort


class VectorSyncError(RuntimeError):
    """下游结果无法与本次专家快照可靠对齐。"""


@dataclass(frozen=True)
class ExpertVectorSyncResult:
    """一位专家完成向量同步后的安全数量摘要。"""

    expert_id: int
    current_count: int
    eligible_count: int
    embedded_count: int
    metadata_updated_count: int
    unchanged_count: int
    deleted_count: int
    delete_not_found_count: int


class ExpertVectorSynchronizer:
    """把一位专家的完整 Contract 快照同步到下游向量服务。"""

    def __init__(self, store: VectorStorePort) -> None:
        self.store = store

    def sync_expert(
        self,
        expert_id: int,
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> ExpertVectorSyncResult:
        """计算完整差分并按 upsert → metadata → delete 顺序执行。"""

        if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id < 1:
            raise ValueError("expert_id 必须是正整数")

        items = list(records)
        if any(not isinstance(item, ExpertChunkEmbeddingInput) for item in items):
            raise TypeError("records 只接受 ExpertChunkEmbeddingInput")
        if any(item.expert_id != expert_id for item in items):
            raise VectorSyncError("Contract expert_id 与同步目标不一致")

        current_by_id = {item.vector_doc_id: item for item in items}
        if len(current_by_id) != len(items):
            raise VectorSyncError("同一专家当前快照包含重复 vector_doc_id")

        # rejected/inactive 是合法 Contract，但不应保留在正常向量集合；如果此前存在
        # 同 ID，它会落入 stale_ids 并在写入成功后删除。
        eligible_by_id = {
            vector_doc_id: item
            for vector_doc_id, item in current_by_id.items()
            if item.is_active and item.metadata.verification_status != "rejected"
        }

        listed = self.store.list_vectors_by_expert(expert_id)
        existing_ids = set(listed.vector_doc_ids)
        query = self.store.get_vectors(sorted(existing_ids)) if existing_ids else None
        existing_by_id = self._existing_rows(query.records if query is not None else [])

        to_upsert: list[ExpertChunkEmbeddingInput] = []
        to_update: list[ExpertChunkEmbeddingInput] = []
        unchanged_count = 0
        for vector_doc_id, record in eligible_by_id.items():
            old = existing_by_id.get(vector_doc_id)
            if old is None or old["content_hash"] != record.content_hash:
                to_upsert.append(record)
            elif old["metadata_hash"] != record.metadata_hash:
                to_update.append(record)
            else:
                unchanged_count += 1

        embedded_count = self._upsert(to_upsert)
        metadata_updated_count, fallback = self._update_metadata(to_update)
        embedded_count += self._upsert(fallback)

        stale_ids = sorted(existing_ids - set(eligible_by_id))
        deletion = self.store.delete_vectors(stale_ids)
        if deletion.requested_count != len(stale_ids):
            raise VectorSyncError("delete_vectors 返回的 requested_count 与请求不一致")

        return ExpertVectorSyncResult(
            expert_id=expert_id,
            current_count=len(items),
            eligible_count=len(eligible_by_id),
            embedded_count=embedded_count,
            metadata_updated_count=metadata_updated_count,
            unchanged_count=unchanged_count,
            deleted_count=deletion.deleted_count,
            delete_not_found_count=deletion.not_found_count,
        )

    @staticmethod
    def _existing_rows(
        rows: Sequence[Mapping[str, object]],
    ) -> dict[str, dict[str, str]]:
        """只提取增量判断需要的 ID 与两个哈希，并拒绝不完整返回。"""

        result: dict[str, dict[str, str]] = {}
        for row in rows:
            vector_doc_id = row.get("vector_doc_id")
            content_hash = row.get("content_hash")
            metadata_hash = row.get("metadata_hash")
            if not all(
                isinstance(value, str) and value
                for value in (vector_doc_id, content_hash, metadata_hash)
            ):
                raise VectorSyncError("get_vectors 返回记录缺少 ID 或哈希")
            assert isinstance(vector_doc_id, str)
            assert isinstance(content_hash, str)
            assert isinstance(metadata_hash, str)
            if vector_doc_id in result:
                raise VectorSyncError("get_vectors 返回重复 vector_doc_id")
            result[vector_doc_id] = {
                "content_hash": content_hash,
                "metadata_hash": metadata_hash,
            }
        return result

    def _upsert(self, records: Sequence[ExpertChunkEmbeddingInput]) -> int:
        """执行完整向量化并要求每条 eligible 输入都返回 embedded。"""

        if not records:
            return 0
        result = self.store.upsert_chunks(records)
        self._require_statuses(result, records, expected_status="embedded")
        return len(records)

    def _update_metadata(
        self,
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> tuple[int, list[ExpertChunkEmbeddingInput]]:
        """执行 metadata-only；安全门失败的记录返回给完整 upsert 回退。"""

        if not records:
            return 0, []
        result = self.store.update_metadata(records)
        by_id = self._items_by_id(result, records)
        records_by_id = {item.vector_doc_id: item for item in records}
        updated_count = 0
        fallback: list[ExpertChunkEmbeddingInput] = []
        for vector_doc_id, record in records_by_id.items():
            item = by_id[vector_doc_id]
            if item.status == "metadata_updated":
                updated_count += 1
            elif item.status == "failed" and item.recommendation == "call_upsert":
                fallback.append(record)
            else:
                raise VectorSyncError("update_metadata 返回了不支持的条目状态")
        return updated_count, fallback

    def _require_statuses(
        self,
        result: VectorIngestionResult,
        records: Sequence[ExpertChunkEmbeddingInput],
        *,
        expected_status: str,
    ) -> None:
        by_id = self._items_by_id(result, records)
        if any(item.status != expected_status for item in by_id.values()):
            raise VectorSyncError(f"下游返回状态不是预期的 {expected_status}")

    @staticmethod
    def _items_by_id(
        result: VectorIngestionResult,
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> dict[str, VectorItemResult]:
        expected_ids = {item.vector_doc_id for item in records}
        by_id = {item.vector_doc_id: item for item in result.results}
        if len(by_id) != len(result.results) or set(by_id) != expected_ids:
            raise VectorSyncError("下游逐条结果与请求 vector_doc_id 不一致")
        return by_id


__all__ = [
    "ExpertVectorSyncResult",
    "ExpertVectorSynchronizer",
    "VectorSyncError",
]
