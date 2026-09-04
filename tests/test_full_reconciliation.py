"""全量向量 dry-run 对账 Service 的只读行为测试。

重点验证：
    - 五类差异分类正确；
    - skipped / rejected / is_active=false 不进入 desired（其旧向量若仍为
      active 则归入 stale）；
    - Milvus 中 is_active=false 的逻辑下线实体只作统计审计：不参与差异分类、
      不判为多余 Entity、不参与终验失败判定；
    - fail-closed：Contract 无效、desired 重复、Milvus list 异常、状态不可信；
    - 空快照正确求 diff 但不执行任何删除；
    - 全过程零 Embedding / 零物理删除操作。
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from expert_contracts import ExpertChunkEmbeddingInput
from rag_vector_store.ingestion.errors import (
    VectorContractError,
    VectorEmbeddingError,
    VectorInconsistentStateError,
    VectorIngestionError,
    VectorMilvusError,
)
from rag_vector_store.ingestion.models import (
    VectorDeletionResult,
    VectorIngestionResult,
    VectorItemResult,
)
from rag_vector_store.reconciliation import (
    FullVectorReconciliationService,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_record(
    index: int = 0,
    *,
    content_hash: str | None = None,
    metadata_hash: str | None = None,
    is_active: bool = True,
    verification_status: str = "source_summary",
    vector_doc_id: str | None = None,
) -> ExpertChunkEmbeddingInput:
    """构造 4.0.0 最小合法内存记录。"""
    text = "[专家能力概述]\n虚构测试专家具备知识检索系统建设经验。"
    source_field_id = f"profile-{index}"
    return ExpertChunkEmbeddingInput(
        contract_version="4.1.0",
        vector_doc_id=vector_doc_id
        or f"expert:900001:profile_summary:{source_field_id}:0",
        expert_id=900001,
        serial_number="DEV900001",
        chunk_type="profile_summary",
        chunk_subtype=None,
        chunk_index=0,
        chunk_total=1,
        chunk_text=text,
        char_count=len(text),
        content_hash=content_hash or _sha(text),
        metadata_hash=metadata_hash or "b" * 64,
        metadata={
            "company_name_raw": None,
            "company_name_norm": None,
            "role_name_raw": None,
            "role_code": None,
            "role_level": None,
            "job_status_norm": "active",
            "abroad_flag_norm": False,
            "work_years": None,
            "industry_codes": ["artificial_intelligence"],
            "function_codes": [],
            "topic_codes": ["knowledge_retrieval"],
            "school_names_raw": [],
            "major_names_raw": [],
            "eligible_for_recommendation": True,
            "origin_type": "compressed_from_source",
            "verification_status": verification_status,
            "weight_tier": "B",
        },
        source={
            "source_table": "experts",
            "source_field": "company+job_name+background",
            "source_field_id": source_field_id,
            "source_version": "2026-08-12T10:00:00+08:00",
            "source_updated_at": "2026-08-12T10:00:00+08:00",
            "source_hash": "a" * 64,
        },
        versions={"chunk_policy_version": "1.3", "metadata_schema_version": "4.0"},
        is_active=is_active,
        updated_at="2026-08-12T10:05:00+08:00",
    )


def default_records(count: int = 3) -> list[ExpertChunkEmbeddingInput]:
    return [make_record(index) for index in range(count)]


class FakeReconRepository:
    """暴露 ``list_all_vector_states`` 的只读 Fake（始终不出现任何写调用）。"""

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        list_error: Exception | None = None,
    ) -> None:
        self.rows = rows if rows is not None else []
        self.list_error = list_error
        self.list_calls: list[dict[str, Any]] = []

    def list_all_vector_states(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.list_calls.append(kwargs)
        if self.list_error is not None:
            raise self.list_error
        return [dict(row) for row in self.rows]

    def _activity_rows(self) -> list[dict[str, Any]]:
        """活动状态扫描行源：与全量扫描同源，缺省补 is_active=True。"""

        if self.list_error is not None:
            raise self.list_error
        rows = []
        for row in self.rows:
            item = dict(row)
            item.setdefault("is_active", True)
            rows.append(item)
        return rows

    def list_all_vector_activity_states(self) -> list[dict[str, Any]]:
        """Service 实际使用的活动状态扫描（与全量扫描同源同帧）。"""

        self.list_calls.append({})
        return self._activity_rows()


def existing_row(
    vector_doc_id: str,
    content_hash: str,
    metadata_hash: str,
    *,
    is_active: bool = True,
) -> dict[str, Any]:
    return {
        "vector_doc_id": vector_doc_id,
        "content_hash": content_hash,
        "metadata_hash": metadata_hash,
        "is_active": is_active,
    }


def make_service(repo: FakeReconRepository) -> FullVectorReconciliationService:
    return FullVectorReconciliationService(repo)

# ---------------------------------------------------------------------------
# 正常分类
# ---------------------------------------------------------------------------


def test_reconcile_all_classifies_all_five_categories() -> None:
    records = default_records(4)
    # r0: unchanged; r1: content_changed; r2: metadata_changed; r3: missing
    r0 = records[0]
    r1 = records[1]
    r1_content = "1" * 64
    r2 = records[2]
    r2_metadata = "2" * 64

    repo = FakeReconRepository(
        [
            existing_row(r0.vector_doc_id, r0.content_hash, r0.metadata_hash),
            existing_row(r1.vector_doc_id, r1_content, r1.metadata_hash),
            existing_row(r2.vector_doc_id, r2.content_hash, r2_metadata),
        ]
    )
    # 额外 stale 一条（desired 中不存在）
    repo.rows.append(existing_row("expert:900001:profile_summary:stale:0", "s" * 64, "s" * 64))

    service = make_service(repo)
    result = service.reconcile_all(records)

    assert result.input_count == 4
    assert result.eligible_count == 4
    assert result.skipped_count == 0
    assert result.desired_count == 4
    assert result.existing_count == 4
    assert result.missing_count == 1
    assert result.content_changed_count == 1
    assert result.metadata_changed_count == 1
    assert result.unchanged_count == 1
    assert result.stale_count == 1


def test_reconcile_all_all_unchanged() -> None:
    records = default_records(3)
    repo = FakeReconRepository(
        [
            existing_row(r.vector_doc_id, r.content_hash, r.metadata_hash)
            for r in records
        ]
    )
    result = make_service(repo).reconcile_all(records)

    assert result.input_count == 3
    assert result.unchanged_count == 3
    assert result.missing_count == 0
    assert result.content_changed_count == 0
    assert result.metadata_changed_count == 0
    assert result.stale_count == 0


def test_reconcile_all_missing() -> None:
    repo = FakeReconRepository([])
    records = default_records(2)
    result = make_service(repo).reconcile_all(records)

    assert result.missing_count == 2
    assert result.existing_count == 0
    assert result.stale_count == 0


def test_reconcile_all_stale_sample_capped_at_20() -> None:
    repo = FakeReconRepository(
        [
            existing_row(
                f"expert:900001:project:{i}:0", f"{i:064x}", f"{i:064x}"
            )
            for i in range(30)
        ]
    )
    # 空 desired -> 全部 existing 均为 stale
    result = make_service(repo).reconcile_all([])

    assert result.input_count == 0
    assert result.eligible_count == 0
    assert result.desired_count == 0
    assert result.existing_count == 30
    assert result.stale_count == 30
    assert len(result.stale_sample) == 20


# ---------------------------------------------------------------------------
# Eligibility / is_active / rejected 不进入 desired
# ---------------------------------------------------------------------------


def test_reconcile_all_skipped_inactive_and_rejected_are_stale() -> None:
    active = make_record(0)
    inactive = make_record(1, is_active=False)
    rejected = make_record(2, verification_status="rejected")

    repo = FakeReconRepository(
        [
            existing_row(active.vector_doc_id, active.content_hash, active.metadata_hash),
            existing_row(inactive.vector_doc_id, inactive.content_hash, inactive.metadata_hash),
            existing_row(rejected.vector_doc_id, rejected.content_hash, rejected.metadata_hash),
        ]
    )
    result = make_service(repo).reconcile_all([active, inactive, rejected])

    # inactive / rejected 被 skip，不进入 desired；其旧向量被识别为 stale
    assert result.input_count == 3
    assert result.eligible_count == 1
    assert result.skipped_count == 2
    assert result.desired_count == 1
    assert result.unchanged_count == 1
    assert result.stale_count == 2
    assert set(result.stale_sample) == {
        inactive.vector_doc_id,
        rejected.vector_doc_id,
    }


# ---------------------------------------------------------------------------
# fail-closed
# ---------------------------------------------------------------------------


def test_reconcile_all_fails_on_contract_invalid() -> None:
    bad = make_record(0, content_hash="f" * 64)  # content_hash 与 chunk_text 不符
    repo = FakeReconRepository()

    with pytest.raises(VectorContractError):
        make_service(repo).reconcile_all([bad])

    # 未触碰 Milvus（input 无效时不读）
    assert repo.list_calls == []


def test_reconcile_all_fails_on_duplicate_desired_id() -> None:
    r0 = make_record(0)
    dup = make_record(0)  # 相同 vector_doc_id
    repo = FakeReconRepository()

    with pytest.raises(VectorInconsistentStateError, match="重复 vector_doc_id"):
        make_service(repo).reconcile_all([r0, dup])

    # 未触碰 Milvus
    assert repo.list_calls == []


def test_reconcile_all_fails_on_repository_list_error() -> None:
    repo = FakeReconRepository(list_error=RuntimeError("milvus list failed"))

    with pytest.raises(VectorMilvusError, match="milvus list failed"):
        make_service(repo).reconcile_all(default_records(1))


def test_reconcile_all_fails_on_untrustworthy_existing_state() -> None:
    repo = FakeReconRepository(
        [{"vector_doc_id": "x", "content_hash": None, "metadata_hash": None}]
    )

    with pytest.raises(VectorInconsistentStateError, match="状态字段不可信"):
        make_service(repo).reconcile_all(default_records(1))


# ---------------------------------------------------------------------------
# 只读保证
# ---------------------------------------------------------------------------


def test_reconcile_all_never_writes_or_embeds() -> None:
    records = default_records(2)
    repo = FakeReconRepository(
        [
            existing_row(r.vector_doc_id, r.content_hash, r.metadata_hash)
            for r in records
        ]
    )

    result = make_service(repo).reconcile_all(records)

    assert result.unchanged_count == 2
    # Fake 只暴露 list_all_vector_states：任何写/删/Embedding 调用都会 AttributeError
    assert not hasattr(repo, "upsert_batch")
    assert not hasattr(repo, "delete_by_vector_doc_ids")
    assert not hasattr(repo, "assert_destructive_maintenance_allowed")
    assert repo.list_calls == [{}]  # 仅一次只读 list


def test_reconcile_all_empty_snapshot_is_pure_diff() -> None:
    repo = FakeReconRepository(
        [existing_row("expert:900001:profile_summary:old:0", "a" * 64, "b" * 64)]
    )

    result = make_service(repo).reconcile_all([])

    assert result.stale_count == 1
    assert result.stale_sample == ("expert:900001:profile_summary:old:0",)
    # dry-run 阶段不执行任何删除（Fake 无删除能力）

# ---------------------------------------------------------------------------
# 正向同步 apply_positive_changes
# ---------------------------------------------------------------------------


class FakePositiveRepository(FakeReconRepository):
    """在只读 Fake 基础上增加可注入的安全守卫（用于正向同步测试）。

    ``list_rows_sequence`` 支持多次扫描调用返回不同结果
    （写后重读、删除后终验），每次调用消耗一帧；耗尽后回落到 ``rows``。
    全量扫描与活动状态扫描共享同一帧序列。
    """

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        list_error: Exception | None = None,
        guard_error: Exception | None = None,
        list_rows_sequence: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__(rows, list_error=list_error)
        self.guard_error = guard_error
        self.guard_calls: list[dict[str, Any]] = []
        self._list_frames: list[list[dict[str, Any]]] = (
            list(list_rows_sequence) if list_rows_sequence is not None else []
        )

    def _next_rows(self) -> list[dict[str, Any]]:
        if self._list_frames:
            return [dict(r) for r in self._list_frames.pop(0)]
        return [dict(r) for r in self.rows]

    def list_all_vector_states(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.list_calls.append(kwargs)
        if self.list_error is not None:
            raise self.list_error
        return self._next_rows()

    def _activity_rows(self) -> list[dict[str, Any]]:
        if self.list_error is not None:
            raise self.list_error
        rows = []
        for row in self._next_rows():
            item = dict(row)
            item.setdefault("is_active", True)
            rows.append(item)
        return rows

    def assert_destructive_maintenance_allowed(self, **kwargs: Any) -> dict[str, Any]:
        self.guard_calls.append(kwargs)
        if self.guard_error is not None:
            raise self.guard_error
        return {"collection_name": "expert_candidates_v14_contract_dev", "authorized": True}


class FakeIngestionService:
    """记录 upsert / update_metadata 调用的 Fake（内部可注入失败与定制结果）。"""

    def __init__(
        self,
        *,
        upsert_result: Any | None = None,
        metadata_result: Any | None = None,
        upsert_error: Exception | None = None,
        metadata_error: Exception | None = None,
        delete_error: Exception | None = None,
        delete_result: Any | None = None,
    ) -> None:
        self.upsert_result = upsert_result
        self.metadata_result = metadata_result
        self.upsert_error = upsert_error
        self.metadata_error = metadata_error
        self.delete_error = delete_error
        self.delete_result = delete_result
        self.upsert_calls: list[list[Any]] = []
        self.metadata_calls: list[list[Any]] = []
        self.delete_calls: list[list[str]] = []

    def upsert_chunks(self, chunks: list[Any]) -> VectorIngestionResult:
        self.upsert_calls.append(list(chunks))
        if self.upsert_error is not None:
            raise self.upsert_error
        if self.upsert_result is not None:
            return self.upsert_result
        return VectorIngestionResult(
            results=[
                VectorItemResult(vector_doc_id=c.vector_doc_id, status="embedded")
                for c in chunks
            ]
        )

    def update_metadata(self, chunks: list[Any]) -> VectorIngestionResult:
        self.metadata_calls.append(list(chunks))
        if self.metadata_error is not None:
            raise self.metadata_error
        if self.metadata_result is not None:
            return self.metadata_result
        return VectorIngestionResult(
            results=[
                VectorItemResult(
                    vector_doc_id=c.vector_doc_id,
                    status="metadata_updated",
                )
                for c in chunks
            ]
        )

    def disable_vectors(self, vector_doc_ids: list[str]) -> VectorDeletionResult:
        ids = list(vector_doc_ids)
        self.delete_calls.append(ids)
        if self.delete_error is not None:
            raise self.delete_error
        if self.delete_result is not None:
            return self.delete_result
        return VectorDeletionResult(
            requested_count=len(ids),
            deleted_count=len(ids),
        )


def make_positive_service(
    repo: FakePositiveRepository,
    ingestion: FakeIngestionService,
) -> FullVectorReconciliationService:
    return FullVectorReconciliationService(repo, ingestion=ingestion)


def _upsert_with_failed(chunks: list[Any]) -> VectorIngestionResult:
    return VectorIngestionResult(
        results=[
            VectorItemResult(
                vector_doc_id=c.vector_doc_id,
                status="failed",
                error_message="boom",
            )
            for c in chunks
        ]
    )


def _empty_result() -> VectorIngestionResult:
    return VectorIngestionResult(results=[])


def test_apply_missing_routes_to_upsert() -> None:
    record = make_record(0)  # Milvus 中不存在 -> missing
    repo = FakePositiveRepository([])
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_positive_changes([record])

    assert result.missing_count == 1
    assert result.embedded_count == 1
    assert ib_ids(ing.upsert_calls) == [[record.vector_doc_id]]
    assert ing.metadata_calls == []
    assert repo.guard_calls == [{}]


def test_apply_content_changed_routes_to_upsert() -> None:
    record = make_record(0)
    repo = FakePositiveRepository(
        [existing_row(record.vector_doc_id, "1" * 64, record.metadata_hash)]
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_positive_changes([record])

    assert result.content_changed_count == 1
    assert result.embedded_count == 1
    assert ib_ids(ing.upsert_calls) == [[record.vector_doc_id]]
    assert ing.metadata_calls == []


def test_apply_metadata_changed_routes_to_update_metadata() -> None:
    record = make_record(0)
    repo = FakePositiveRepository(
        [existing_row(record.vector_doc_id, record.content_hash, "2" * 64)]
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_positive_changes([record])

    assert result.metadata_changed_count == 1
    assert result.metadata_updated_count == 1
    assert ing.upsert_calls == []
    assert ib_ids(ing.metadata_calls) == [[record.vector_doc_id]]


def test_apply_unchanged_is_zero_write() -> None:
    record = make_record(0)
    repo = FakePositiveRepository(
        [existing_row(record.vector_doc_id, record.content_hash, record.metadata_hash)]
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_positive_changes([record])

    assert result.unchanged_count == 1
    assert result.embedded_count == 0
    assert result.metadata_updated_count == 0
    assert ing.upsert_calls == []
    assert ing.metadata_calls == []
    # 守卫仍被调用（本次 Full Reconciliation 运行在授权的维护 Collection 上）
    assert repo.guard_calls == [{}]


def test_apply_stale_is_never_deleted() -> None:
    record = make_record(0)
    repo = FakePositiveRepository(
        [
            existing_row(record.vector_doc_id, record.content_hash, record.metadata_hash),
            existing_row("expert:900001:profile_summary:stale:0", "s" * 64, "s" * 64),
        ]
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_positive_changes([record])

    assert result.unchanged_count == 1
    assert result.stale_count == 1
    # stale 只统计，绝不删除：Fake 虽提供删除能力，但 apply_positive_changes 不会调用
    assert not hasattr(repo, "delete_by_vector_doc_ids")
    assert ing.delete_calls == []


def test_apply_mixed_diff() -> None:
    records = default_records(4)
    missing = records[0]
    content_changed = records[1]
    metadata_changed = records[2]
    unchanged = records[3]
    repo = FakePositiveRepository(
        [
            existing_row(content_changed.vector_doc_id, "1" * 64, content_changed.metadata_hash),
            existing_row(metadata_changed.vector_doc_id, metadata_changed.content_hash, "2" * 64),
            existing_row(unchanged.vector_doc_id, unchanged.content_hash, unchanged.metadata_hash),
            existing_row("expert:900001:profile_summary:stale:0", "s" * 64, "s" * 64),
        ]
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_positive_changes(records)

    assert result.missing_count == 1
    assert result.content_changed_count == 1
    assert result.metadata_changed_count == 1
    assert result.unchanged_count == 1
    assert result.stale_count == 1
    assert result.embedded_count == 2
    assert result.metadata_updated_count == 1
    # 顺序：missing 先、content_changed 后，一并 upsert
    assert ib_ids(ing.upsert_calls) == [
        [missing.vector_doc_id, content_changed.vector_doc_id]
    ]
    assert ib_ids(ing.metadata_calls) == [[metadata_changed.vector_doc_id]]
    # stale 未删除
    assert not hasattr(repo, "delete_by_vector_doc_ids")


def test_apply_upsert_failure_is_fail_closed() -> None:
    record = make_record(0)
    repo = FakePositiveRepository([])
    ing = FakeIngestionService(
        upsert_result=_upsert_with_failed([record])
    )

    with pytest.raises(VectorInconsistentStateError, match="出现失败项|不一致"):
        make_positive_service(repo, ing).apply_positive_changes([record])

    # 失败后不再执行 delete（fake 无删除能力）
    assert not hasattr(repo, "delete_by_vector_doc_ids")


def test_apply_upsert_count_mismatch_is_fail_closed() -> None:
    record = make_record(0)
    repo = FakePositiveRepository([])
    ing = FakeIngestionService(upsert_result=_empty_result())  # 0 成功 vs 计划 1

    with pytest.raises(VectorInconsistentStateError, match="实际成功数量与计划不一致"):
        make_positive_service(repo, ing).apply_positive_changes([record])


def test_apply_metadata_failure_is_fail_closed() -> None:
    record = make_record(0)
    repo = FakePositiveRepository(
        [existing_row(record.vector_doc_id, record.content_hash, "2" * 64)]
    )
    ing = FakeIngestionService(
        metadata_result=_upsert_with_failed([record])
    )

    with pytest.raises(VectorInconsistentStateError, match="失败项|不一致"):
        make_positive_service(repo, ing).apply_positive_changes([record])


def test_apply_metadata_count_mismatch_is_fail_closed() -> None:
    record = make_record(0)
    repo = FakePositiveRepository(
        [existing_row(record.vector_doc_id, record.content_hash, "2" * 64)]
    )
    ing = FakeIngestionService(metadata_result=_empty_result())

    with pytest.raises(VectorInconsistentStateError, match="实际成功数量与计划不一致"):
        make_positive_service(repo, ing).apply_positive_changes([record])


def test_apply_guard_failure_is_zero_write() -> None:
    record = make_record(0)  # missing -> 本应 upsert
    repo = FakePositiveRepository([], guard_error=RuntimeError("未获授权"))
    ing = FakeIngestionService()

    with pytest.raises(VectorMilvusError, match="未获授权"):
        make_positive_service(repo, ing).apply_positive_changes([record])

    # 任何写操作前守卫失败 -> 零写入
    assert ing.upsert_calls == []
    assert ing.metadata_calls == []
    assert repo.guard_calls == [{}]


def test_apply_requires_ingestion() -> None:
    repo = FakePositiveRepository([])
    service = FullVectorReconciliationService(repo)  # 未注入 ingestion

    with pytest.raises(VectorIngestionError, match="ingestion"):
        service.apply_positive_changes([make_record(0)])


def ib_ids(calls: list[list[Any]]) -> list[list[str]]:
    """从 upsert / update_metadata 调用记录中抽取 vector_doc_id 列表。

    计划中的记录是 ``ContractPreparedChunk``（与原始 ``ExpertChunkEmbeddingInput``
    是不同实例），因此断言用 ``vector_doc_id`` 而非对象相等。
    """

    return [[c.vector_doc_id for c in batch] for batch in calls]


# ---------------------------------------------------------------------------
# 全库 reconciliation apply_full_reconciliation
# ---------------------------------------------------------------------------


def test_full_recon_normal_stale_delete() -> None:
    record = make_record(0)
    desired_row = existing_row(record.vector_doc_id, record.content_hash, record.metadata_hash)
    stale_row = existing_row("expert:900001:profile_summary:stale:0", "s" * 64, "s" * 64)
    # 写前：仅 stale（record 缺失 -> upsert）；写后：record 已存在 + stale 仍在；终验：stale 已删
    repo = FakePositiveRepository(
        rows=[stale_row],
        list_rows_sequence=[
            [stale_row],
            [desired_row, stale_row],
            [desired_row],
        ],
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_full_reconciliation([record])

    assert result.embedded_count == 1  # missing -> upsert
    assert result.deleted_count == 1
    assert result.final_existing_count == 1  # desired 仍存在
    assert result.consistent is True
    assert len(ing.delete_calls) == 1
    assert ing.delete_calls[0] == ["expert:900001:profile_summary:stale:0"]


def test_full_recon_multi_batch_stale_delete() -> None:
    # delete_batch_size=2，stale 3 条 -> 2 批
    desired = default_records(1)
    desired_row = existing_row(desired[0].vector_doc_id, desired[0].content_hash, desired[0].metadata_hash)
    stale = [
        existing_row(f"expert:900001:profile_summary:stale{i}:0", "s" * 64, "s" * 64)
        for i in range(3)
    ]
    repo = FakePositiveRepository(
        rows=[stale[0]],
        list_rows_sequence=[stale, [desired_row] + stale, [desired_row]],
    )
    ing = FakeIngestionService()
    service = FullVectorReconciliationService(
        repo, ingestion=ing, delete_batch_size=2
    )
    result = service.apply_full_reconciliation(desired)

    assert result.deleted_count == 3
    assert len(ing.delete_calls) == 2
    assert len(ing.delete_calls[0]) == 2
    assert len(ing.delete_calls[1]) == 1
    assert result.consistent is True


def test_full_recon_no_stale_zero_delete() -> None:
    record = make_record(0)
    # 写前已有且不变；写后、终验均不变
    row = existing_row(record.vector_doc_id, record.content_hash, record.metadata_hash)
    repo = FakePositiveRepository(rows=[row], list_rows_sequence=[[row], [row], [row]])
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_full_reconciliation([record])

    assert result.unchanged_count == 1
    assert result.deleted_count == 0
    assert ing.delete_calls == []
    assert result.consistent is True


def test_full_recon_positive_write_failure_no_delete() -> None:
    record = make_record(0)
    repo = FakePositiveRepository(
        rows=[],
        list_rows_sequence=[[], []],  # 写后、终验
    )
    ing = FakeIngestionService(upsert_result=_upsert_with_failed([record]))

    with pytest.raises(VectorInconsistentStateError, match="失败项"):
        make_positive_service(repo, ing).apply_full_reconciliation([record])

    assert ing.delete_calls == []


def test_full_recon_existing_after_write_read_failure_no_delete() -> None:
    record = make_record(0)
    # 第一次读（plan）：返回空；第二次读（写后）：抛异常
    repo = FakePositiveRepository(
        rows=[],
        list_rows_sequence=[],
    )
    # 让第二次 list 抛异常：先正常返回一帧，再注入错误
    repo._list_frames = []
    repo.list_error = RuntimeError("milvus list failed after write")
    ing = FakeIngestionService()

    with pytest.raises(VectorMilvusError, match="milvus list failed"):
        make_positive_service(repo, ing).apply_full_reconciliation([record])

    assert ing.delete_calls == []


def test_full_recon_guard_failure_zero_mutation() -> None:
    record = make_record(0)
    repo = FakePositiveRepository([], guard_error=RuntimeError("未获授权"))
    ing = FakeIngestionService()

    with pytest.raises(VectorMilvusError, match="未获授权"):
        make_positive_service(repo, ing).apply_full_reconciliation([record])

    assert ing.upsert_calls == []
    assert ing.metadata_calls == []
    assert ing.delete_calls == []


def test_full_recon_empty_snapshot_non_empty_collection_fails() -> None:
    repo = FakePositiveRepository(
        rows=[existing_row("expert:900001:profile_summary:stale:0", "s" * 64, "s" * 64)],
    )
    ing = FakeIngestionService()

    with pytest.raises(VectorInconsistentStateError, match="空快照"):
        make_positive_service(repo, ing).apply_full_reconciliation([])

    # 守卫在空快照检查之后才调用，故 guard/upsert/delete 均未触发
    assert repo.guard_calls == []
    assert ing.upsert_calls == []
    assert ing.delete_calls == []


def test_full_recon_all_skipped_non_empty_snapshot_does_not_fail() -> None:
    """chunks 非空但全部被 Eligibility 跳过 → 不是「空 snapshot」，走正常 reconciliation。

    全 skipped 时 desired_ids 为空；existing 全部为 stale → 正常删除并终验。
    不触发空快照保护。
    """
    inactive = make_record(0, is_active=False)
    stale_row = existing_row(inactive.vector_doc_id, inactive.content_hash, inactive.metadata_hash)
    repo = FakePositiveRepository(
        rows=[stale_row],
        list_rows_sequence=[
            [stale_row],       # plan 读
            [stale_row],       # 写后重读（无正向写，desired 空）
            [],                 # 删后终验
        ],
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_full_reconciliation([inactive])

    assert result.input_count == 1
    assert result.eligible_count == 0
    assert result.skipped_count == 1
    assert result.desired_count == 0
    # existing 中 1 条全部 stale，被删除，终验为空 == desired_ids（空集）
    assert result.deleted_count == 1
    assert result.final_existing_count == 0
    assert result.consistent is True


def test_full_recon_empty_snapshot_empty_collection_succeeds() -> None:
    """空 snapshot + 空 Collection → 正常返回，不触发保护。"""
    repo = FakePositiveRepository(rows=[])
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_full_reconciliation([])

    assert result.input_count == 0
    assert result.deleted_count == 0
    assert result.final_existing_count == 0
    assert result.consistent is True


def test_full_recon_delete_batch_failure_stops_subsequent() -> None:
    desired = default_records(1)
    desired_row = existing_row(desired[0].vector_doc_id, desired[0].content_hash, desired[0].metadata_hash)
    stale = [
        existing_row(f"expert:900001:profile_summary:stale{i}:0", "s" * 64, "s" * 64)
        for i in range(3)
    ]
    repo = FakePositiveRepository(
        rows=[stale[0]],
        list_rows_sequence=[stale, [desired_row] + stale, [desired_row]],
    )
    ing = FakeIngestionService(
        delete_error=VectorMilvusError("delete batch failed"),
    )
    service = FullVectorReconciliationService(
        repo, ingestion=ing, delete_batch_size=1
    )

    with pytest.raises(VectorMilvusError, match="delete batch failed"):
        service.apply_full_reconciliation(desired)

    # 第一批失败后立即停止后续批次；VectorMilvusError 原样保留不被吞掉
    assert len(ing.delete_calls) == 1


def test_full_recon_final_equals_desired_success() -> None:
    record = make_record(0)
    row = existing_row(record.vector_doc_id, record.content_hash, record.metadata_hash)
    stale = existing_row("expert:900001:profile_summary:stale:0", "s" * 64, "s" * 64)
    repo = FakePositiveRepository(
        rows=[],
        list_rows_sequence=[[row, stale], [row, stale], [row]],
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_full_reconciliation([record])

    assert result.deleted_count == 1
    assert result.final_existing_count == 1
    assert result.consistent is True


def test_full_recon_final_mismatch_fail_closed() -> None:
    record = make_record(0)
    stale = existing_row("expert:900001:profile_summary:stale:0", "s" * 64, "s" * 64)
    # 删除后重读仍返回 stale（模拟删除未生效），终验不一致
    repo = FakePositiveRepository(
        rows=[],
        list_rows_sequence=[[stale], [stale], [stale]],
    )
    ing = FakeIngestionService()
    with pytest.raises(VectorInconsistentStateError, match="最终一致性验证失败"):
        make_positive_service(repo, ing).apply_full_reconciliation([record])


def test_full_recon_dry_run_behavior_unchanged() -> None:
    record = make_record(0)
    repo = FakeReconRepository(
        [existing_row(record.vector_doc_id, record.content_hash, record.metadata_hash)]
    )
    result = FullVectorReconciliationService(repo).reconcile_all([record])

    assert result.embedded_count == 0
    assert result.metadata_updated_count == 0
    assert result.deleted_count == 0
    assert result.final_existing_count == 0
    assert result.consistent is True


def test_full_recon_apply_positive_behavior_unchanged() -> None:
    record = make_record(0)
    repo = FakePositiveRepository([])
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_positive_changes([record])

    assert result.embedded_count == 1
    assert result.deleted_count == 0
    assert result.final_existing_count == 0
    assert result.consistent is True
    assert ing.delete_calls == []


# ---------------------------------------------------------------------------
# 公共异常边界
# ---------------------------------------------------------------------------


def test_ingestion_vector_milvus_error_passes_through_unchanged() -> None:
    """VectorIngestionService 已抛出的 VectorMilvusError 原样保留，不重复包装。"""

    record = make_record(0)
    repo = FakePositiveRepository([])
    ing = FakeIngestionService(upsert_error=VectorMilvusError("upsert infra failed"))

    with pytest.raises(VectorMilvusError, match="upsert infra failed"):
        make_positive_service(repo, ing).apply_positive_changes([record])

    # 不再继续任何后续动作（delete 未执行）
    assert ing.delete_calls == []


def test_ingestion_vector_embedding_error_passes_through_unchanged() -> None:
    """Embedding 层已有 VectorEmbeddingError 原样保留，不重复包装。"""

    record = make_record(0)
    repo = FakePositiveRepository([])
    ing = FakeIngestionService(upsert_error=VectorEmbeddingError("embed failed"))

    with pytest.raises(VectorEmbeddingError, match="embed failed"):
        make_positive_service(repo, ing).apply_positive_changes([record])


def test_ingestion_vector_error_is_not_swallowed_by_disable() -> None:
    """disable_vectors 阶段已抛出的 VectorMilvusError 不被 swallow。"""

    desired = default_records(1)
    desired_row = existing_row(
        desired[0].vector_doc_id, desired[0].content_hash, desired[0].metadata_hash
    )
    stale_row = existing_row(
        "expert:900001:profile_summary:stale:0", "s" * 64, "s" * 64
    )
    repo = FakePositiveRepository(
        rows=[],
        list_rows_sequence=[
            [desired_row, stale_row],
            [desired_row, stale_row],
            [desired_row],
        ],
    )
    ing = FakeIngestionService(
        delete_error=VectorMilvusError("delete infra failed"),
    )

    with pytest.raises(VectorMilvusError, match="delete infra failed"):
        make_positive_service(repo, ing).apply_full_reconciliation(desired)

    # 已尝试删除（非零执行），但错误不被吞掉
    assert len(ing.delete_calls) >= 1


def test_contract_error_preserves_original_cause() -> None:
    """Contract 非法被包装为 VectorContractError 且保留原始异常作为 cause。"""

    # 直接构造一条会触发 TypeError 的输入（非 ExpertChunkEmbeddingInput）
    repo = FakePositiveRepository()
    with pytest.raises(VectorContractError) as excinfo:
        make_service(repo).reconcile_all([{"not": "a contract"}])

    assert isinstance(excinfo.value.__cause__, TypeError)


def test_milvus_error_preserves_original_cause() -> None:
    """Milvus 扫描失败被包装为 VectorMilvusError 且保留原始异常作为 cause。"""

    original = RuntimeError("milvus list failed")
    repo = FakeReconRepository(list_error=original)

    with pytest.raises(VectorMilvusError) as excinfo:
        make_service(repo).reconcile_all(default_records(1))

    assert excinfo.value.__cause__ is original


def test_guard_error_preserves_original_cause() -> None:
    """destructive guard 失败被包装为 VectorMilvusError 且保留原始 cause。"""

    original = RuntimeError("未获授权")
    repo = FakePositiveRepository([], guard_error=original)
    ing = FakeIngestionService()

    with pytest.raises(VectorMilvusError) as excinfo:
        make_positive_service(repo, ing).apply_positive_changes([make_record(0)])

    assert excinfo.value.__cause__ is original


def test_duplicate_desired_is_inconsistent_state_error() -> None:
    """重复 desired vector_doc_id -> VectorInconsistentStateError（无 Milvus 触碰）。"""

    r0 = make_record(0)
    repo = FakePositiveRepository()

    with pytest.raises(VectorInconsistentStateError, match="重复 vector_doc_id"):
        make_service(repo).reconcile_all([r0, make_record(0)])

    assert repo.list_calls == []


def test_malformed_existing_state_is_inconsistent_state_error() -> None:
    """existing 状态字段不可信 -> VectorInconsistentStateError。"""

    repo = FakePositiveRepository(
        [{"vector_doc_id": "x", "content_hash": "c", "metadata_hash": ""}]
    )

    with pytest.raises(VectorInconsistentStateError, match="状态字段不可信"):
        make_service(repo).reconcile_all(default_records(1))


def test_wiring_error_is_vector_ingestion_error() -> None:
    """未注入 ingestion 时属于服务装配错误 -> VectorIngestionError（非底层异常泄漏）。"""

    repo = FakePositiveRepository([])
    service = FullVectorReconciliationService(repo)

    with pytest.raises(VectorIngestionError, match="ingestion"):
        service.apply_full_reconciliation([])


# ---------------------------------------------------------------------------
# 逻辑下线（inactive Entity）与 Full Reconciliation 兼容性
# ---------------------------------------------------------------------------


def test_reconcile_with_inactive_stale_entity_passes() -> None:
    """存在 inactive 多余实体时 reconciliation 通过：不判 stale、不触发下线。"""

    records = default_records(2)  # r0 unchanged；r1 missing -> upsert
    active_row = existing_row(
        records[0].vector_doc_id, records[0].content_hash, records[0].metadata_hash
    )
    inactive_extra = existing_row(
        "expert:900001:profile_summary:disabled:0",
        "i" * 64,
        "j" * 64,
        is_active=False,
    )
    after_write = [active_row, existing_row(
        records[1].vector_doc_id, records[1].content_hash, records[1].metadata_hash
    ), inactive_extra]
    repo = FakePositiveRepository(
        rows=[active_row],
        list_rows_sequence=[
            [active_row, inactive_extra],  # plan 读：r1 缺失 + 1 条 inactive
            after_write,                    # 写后重读
            after_write,                    # 终验
        ],
    )
    ing = FakeIngestionService()
    result = make_positive_service(repo, ing).apply_full_reconciliation(records)

    # dry-run 口径（同一 plan）：inactive 不参与分类
    assert result.stale_count == 0
    assert result.stale_sample == ()
    assert result.inactive_entity_count == 1
    assert result.active_entity_count == 1  # plan 时点：仅 r0 为 active
    assert result.existing_count == 2       # total = active + inactive
    # 终读口径：成功时 active/desired ID 集一致，mismatch 恒为 0
    assert result.active_mismatch_count == 0
    assert result.embedded_count == 1
    # 关键：inactive 实体未被重复下线，终验通过
    assert ing.delete_calls == []
    assert result.deleted_count == 0
    assert result.final_existing_count == 2
    assert result.consistent is True


def test_full_recon_final_missing_active_fails_closed() -> None:
    """终验时 active 实体缺失（写入未生效模拟）-> fail-closed。"""

    record = make_record(0)
    # 三次读取均不含 desired（upsert 声称成功但数据丢失）
    repo = FakePositiveRepository(rows=[], list_rows_sequence=[[], [], []])
    ing = FakeIngestionService()

    with pytest.raises(VectorInconsistentStateError) as excinfo:
        make_positive_service(repo, ing).apply_full_reconciliation([record])

    assert "最终一致性验证失败" in str(excinfo.value)
    assert "缺失=1" in str(excinfo.value)
    assert "多余=0" in str(excinfo.value)


def test_full_recon_final_active_hash_mismatch_fails_closed() -> None:
    """终验时 active 实体 hash 与 desired 不一致 -> fail-closed。"""

    record = make_record(0)
    wrong_hash_row = existing_row(
        record.vector_doc_id,
        "f" * 64,  # content_hash 与 desired 不一致（写入被篡改模拟）
        record.metadata_hash,
    )
    repo = FakePositiveRepository(
        rows=[], list_rows_sequence=[[wrong_hash_row], [wrong_hash_row], [wrong_hash_row]],
    )
    ing = FakeIngestionService()

    with pytest.raises(VectorInconsistentStateError) as excinfo:
        make_positive_service(repo, ing).apply_full_reconciliation([record])

    assert "最终一致性验证失败" in str(excinfo.value)
    assert "hash不一致=1" in str(excinfo.value)


def test_inactive_entities_never_counted_as_stale() -> None:
    """仅剩 inactive 实体时 dry-run 不报任何 stale / mismatch（纯审计计数）。"""

    record = make_record(0)
    inactive_a = existing_row(
        "expert:900001:profile_summary:old-a:0", "a" * 64, "b" * 64, is_active=False
    )
    inactive_b = existing_row(
        "expert:900001:profile_summary:old-b:0", "c" * 64, "d" * 64, is_active=False
    )
    repo = FakeReconRepository([inactive_a, inactive_b])
    result = make_service(repo).reconcile_all([record])

    # record 缺失（其唯一实体 inactive），但 inactive 永不入 stale
    assert result.missing_count == 1
    assert result.stale_count == 0
    assert result.stale_sample == ()
    assert result.active_entity_count == 0
    assert result.inactive_entity_count == 2
    assert result.existing_count == 2
    assert result.active_mismatch_count == 1  # 仅缺失，无多余


def test_reconciliation_after_disable_is_stable_across_runs() -> None:
    """下线后重复执行 apply_full_reconciliation 结果稳定（幂等零写入）。"""

    record = make_record(0)
    desired_row = existing_row(
        record.vector_doc_id, record.content_hash, record.metadata_hash
    )
    inactive_entity = existing_row(
        "expert:900001:profile_summary:disabled:0",
        "i" * 64,
        "j" * 64,
        is_active=False,
    )

    def run(frames: list[list[dict[str, Any]]]) -> Any:
        repo = FakePositiveRepository(
            rows=[desired_row, inactive_entity],
            list_rows_sequence=[list(frame) for frame in frames],
        )
        ing = FakeIngestionService()
        result = make_positive_service(repo, ing).apply_full_reconciliation([record])
        return result, ing

    # 第一次：record 尚未写入（missing -> upsert）；inactive 保持不动
    first, ing1 = run(
        [
            [inactive_entity],             # plan 读：只有 inactive
            [desired_row, inactive_entity],  # 写后
            [desired_row, inactive_entity],  # 终验
        ]
    )
    # 第二次：全部就位（全 unchanged），结果与第一次一致且零写入
    second, ing2 = run(
        [
            [desired_row, inactive_entity],
            [desired_row, inactive_entity],
            [desired_row, inactive_entity],
        ]
    )

    for result in (first, second):
        assert result.consistent is True
        assert result.deleted_count == 0
        assert result.stale_count == 0
        assert result.inactive_entity_count == 1
        assert result.final_existing_count == 1
    assert len(ing1.delete_calls) == 0
    assert len(ing2.delete_calls) == 0
    assert len(ing2.upsert_calls) == 0  # 第二次完全幂等零写


def test_untrustworthy_is_active_fails_closed() -> None:
    """is_active 非布尔值 -> VectorInconsistentStateError（不默认为 active）。"""

    bad_row = {
        "vector_doc_id": "expert:900001:profile_summary:p-0:0",
        "content_hash": "a" * 64,
        "metadata_hash": "b" * 64,
        "is_active": "yes",  # 类型错误
    }
    repo = FakeReconRepository([bad_row])

    with pytest.raises(VectorInconsistentStateError, match="is_active 必须是布尔值"):
        make_service(repo).reconcile_all(default_records(1))
