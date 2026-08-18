"""全量向量 dry-run 对账 Service 的只读行为测试。

重点验证：
    - 五类差异分类正确；
    - skipped / rejected / is_active=false 不进入 desired（其旧向量归入 stale）；
    - fail-closed：Contract 无效、desired 重复、Milvus list 异常、状态不可信；
    - 空快照正确求 diff 但不执行任何删除；
    - 全过程零 Embedding / 零 Milvus 写操作。
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from expert_contracts import ExpertChunkEmbeddingInput
from rag_vector_store.chunking.contract_pipeline import (
    ContractValidationError,
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
        contract_version="4.0.0",
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
            "project_name": None,
            "start_month": None,
            "end_month": None,
            "is_current": None,
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
        versions={"chunk_policy_version": "1.3", "metadata_schema_version": "3.0"},
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


def existing_row(
    vector_doc_id: str,
    content_hash: str,
    metadata_hash: str,
) -> dict[str, Any]:
    return {
        "vector_doc_id": vector_doc_id,
        "content_hash": content_hash,
        "metadata_hash": metadata_hash,
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

    with pytest.raises(ContractValidationError):
        make_service(repo).reconcile_all([bad])

    # 未触碰 Milvus（input 无效时不读）
    assert repo.list_calls == []


def test_reconcile_all_fails_on_duplicate_desired_id() -> None:
    r0 = make_record(0)
    dup = make_record(0)  # 相同 vector_doc_id
    repo = FakeReconRepository()

    with pytest.raises(ValueError, match="重复 vector_doc_id"):
        make_service(repo).reconcile_all([r0, dup])

    # 未触碰 Milvus
    assert repo.list_calls == []


def test_reconcile_all_fails_on_repository_list_error() -> None:
    repo = FakeReconRepository(list_error=RuntimeError("milvus list failed"))

    with pytest.raises(RuntimeError, match="milvus list failed"):
        make_service(repo).reconcile_all(default_records(1))


def test_reconcile_all_fails_on_untrustworthy_existing_state() -> None:
    repo = FakeReconRepository(
        [{"vector_doc_id": "x", "content_hash": None, "metadata_hash": None}]
    )

    with pytest.raises(RuntimeError, match="状态字段不可信"):
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