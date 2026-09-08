"""依据上游当前快照调用下游公共方法同步一位专家的向量。

本模块只做增量决策与结果对齐：正文变化走完整向量化，metadata-only 复用旧向量，
不变记录跳过，陈旧或不再 eligible 的记录删除。它不读取业务 MySQL，也不直接调用
Milvus Repository。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from expert_contracts import ExpertChunkEmbeddingInput
from rag_vector_store import (
    VectorIngestionResult,
    VectorItemResult,
    VectorOwnerList,
    VectorQueryResult,
)

from .ports import ExpertBatchSyncInput, VectorStorePort


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


@dataclass(frozen=True)
class _ExpertDiff:
    """一位专家的差分计算结果，不含任何写入执行。"""

    expert_id: int
    current_count: int
    eligible_count: int
    to_upsert: list[ExpertChunkEmbeddingInput] = field(default_factory=list)
    to_update: list[ExpertChunkEmbeddingInput] = field(default_factory=list)
    unchanged_count: int = 0
    stale_ids: list[str] = field(default_factory=list)


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

        diff = self._compute_expert_diff(expert_id, records)

        embedded_count = self._upsert(diff.to_upsert)
        metadata_updated_count, fallback = self._update_metadata(diff.to_update)
        embedded_count += self._upsert(fallback)

        deletion = self.store.disable_vectors(diff.stale_ids)
        if deletion.requested_count != len(diff.stale_ids):
            raise VectorSyncError("disable_vectors 返回的 requested_count 与请求不一致")

        return ExpertVectorSyncResult(
            expert_id=diff.expert_id,
            current_count=diff.current_count,
            eligible_count=diff.eligible_count,
            embedded_count=embedded_count,
            metadata_updated_count=metadata_updated_count,
            unchanged_count=diff.unchanged_count,
            deleted_count=deletion.deleted_count,
            delete_not_found_count=deletion.not_found_count,
        )

    def sync_expert_batch(
        self,
        experts: Sequence[ExpertBatchSyncInput],
    ) -> list[ExpertVectorSyncResult]:
        """跨专家批量同步，合并 Embedding 与 Milvus 写入。

        与 :meth:`sync_expert` 语义等价，但把多位专家的 ``to_upsert`` 合并为一次
        ``store.upsert_chunks``（一次 Embedding 批 + 一次 Milvus upsert），减少
        per-expert 小批量的调用开销。各专家的 list/get/diff 仍按专家维度独立执行，
        保证幂等与陈旧删除语义不变。

        执行顺序：
        1. 逐专家计算差分（list → get → diff，不触发 Embedding）；
        2. 合并全部专家的 ``to_upsert`` 一次写入（一次 Embedding + 一次 Milvus）；
        3. 逐专家执行 metadata-only 更新（安全门失败的记录回退到 upsert）；
        4. 合并全部回退记录再一次写入（一次 Embedding + 一次 Milvus）；
        5. 逐专家删除陈旧向量，汇总各专家结果。

        任何一步失败都 fail-closed，整批抛出，不会返回部分专家结果。
        """

        if not all(isinstance(e, ExpertBatchSyncInput) for e in experts):
            raise TypeError("experts 只接受 ExpertBatchSyncInput")
        diffs = [self._compute_expert_diff(e.expert_id, e.records) for e in experts]

        # 阶段 1：跨专家合并首次 upsert（一次 Embedding + 一次 Milvus 写入）
        first_upsert: list[ExpertChunkEmbeddingInput] = []
        for diff in diffs:
            first_upsert.extend(diff.to_upsert)
        first_embedded_vids = self._upsert_tracking(first_upsert)

        # 阶段 2：逐专家 metadata-only 更新，收集安全门回退记录
        per_expert_metadata: dict[int, int] = {}
        fallback_upsert: list[ExpertChunkEmbeddingInput] = []
        for diff in diffs:
            updated, fallback = self._update_metadata(diff.to_update)
            per_expert_metadata[diff.expert_id] = updated
            fallback_upsert.extend(fallback)

        # 阶段 3：合并全部回退记录再一次 Embedding + 写入
        fallback_embedded_vids = self._upsert_tracking(fallback_upsert)

        # 阶段 4：逐专家删除陈旧向量并汇总结果
        results: list[ExpertVectorSyncResult] = []
        for diff in diffs:
            deletion = self.store.disable_vectors(diff.stale_ids)
            if deletion.requested_count != len(diff.stale_ids):
                raise VectorSyncError(
                    "disable_vectors 返回的 requested_count 与请求不一致"
                )
            embedded_count = (
                sum(1 for r in diff.to_upsert if r.vector_doc_id in first_embedded_vids)
                + sum(
                    1
                    for r in diff.to_update
                    if r.vector_doc_id in fallback_embedded_vids
                )
            )
            results.append(
                ExpertVectorSyncResult(
                    expert_id=diff.expert_id,
                    current_count=diff.current_count,
                    eligible_count=diff.eligible_count,
                    embedded_count=embedded_count,
                    metadata_updated_count=per_expert_metadata.get(diff.expert_id, 0),
                    unchanged_count=diff.unchanged_count,
                    deleted_count=deletion.deleted_count,
                    delete_not_found_count=deletion.not_found_count,
                )
            )
        return results

    def _compute_expert_diff(
        self,
        expert_id: int,
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> _ExpertDiff:
        """计算一位专家的差分，不触发任何 Embedding 或写入。"""

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

        listed = self.store.list_vectors_by_owner(expert_id)
        self._validate_owner_listing(listed, expert_id)
        existing_ids = set(listed.vector_doc_ids)
        query = self.store.get_vectors(sorted(existing_ids)) if existing_ids else None
        existing_by_id = self._existing_rows(
            query,
            expected_ids=existing_ids,
        )

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

        stale_ids = sorted(existing_ids - set(eligible_by_id))
        return _ExpertDiff(
            expert_id=expert_id,
            current_count=len(items),
            eligible_count=len(eligible_by_id),
            to_upsert=to_upsert,
            to_update=to_update,
            unchanged_count=unchanged_count,
            stale_ids=stale_ids,
        )

    def _upsert_tracking(
        self,
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> set[str]:
        """批量 upsert 并返回成功嵌入的 vector_doc_id 集合，用于跨专家计数。"""

        if not records:
            return set()
        self._upsert(records)
        return {item.vector_doc_id for item in records}

    @staticmethod
    def _validate_owner_listing(
        listing: object,
        expected_owner_id: int,
    ) -> None:
        """校验 owner 列表，防止错误集合参与差分与 stale 计算。"""

        if not isinstance(listing, VectorOwnerList):
            raise VectorSyncError(
                "list_vectors_by_owner 返回值不是 VectorOwnerList"
            )
        if listing.owner_id != expected_owner_id:
            raise VectorSyncError(
                "list_vectors_by_owner 返回 owner_id 与同步目标不一致"
            )
        ids = listing.vector_doc_ids
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise VectorSyncError(
                "list_vectors_by_owner 返回空或非法 vector_doc_id"
            )
        if len(ids) != len(set(ids)) or listing.count != len(ids):
            raise VectorSyncError(
                "list_vectors_by_owner 返回重复 ID 或 count 不一致"
            )

    @staticmethod
    def _existing_rows(
        query: object,
        *,
        expected_ids: set[str],
    ) -> dict[str, dict[str, str]]:
        """只提取增量判断需要的 ID 与两个哈希，并拒绝不完整返回。"""

        if not isinstance(query, VectorQueryResult):
            if expected_ids:
                raise VectorSyncError(
                    "get_vectors 返回值不是 VectorQueryResult"
                )
            return {}
        rows = query.records
        returned_ids = {
            row.get("vector_doc_id")
            for row in rows
            if isinstance(row.get("vector_doc_id"), str)
        }
        missing_ids = set(query.missing)
        if returned_ids | missing_ids != expected_ids:
            raise VectorSyncError(
                "get_vectors 返回 ID 集合与 list_vectors_by_owner 不一致"
            )
        if returned_ids & missing_ids:
            raise VectorSyncError(
                "get_vectors records 与 missing 存在重复 ID"
            )
        if missing_ids:
            raise VectorSyncError(
                "get_vectors 未命中 list_vectors_by_owner 声明存在的记录"
            )

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
