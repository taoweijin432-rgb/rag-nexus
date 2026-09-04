"""ContractValidator / EmbeddingEligibilityValidator 的 4.1.0 消费单元测试。

覆盖：三类 chunk_type × origin_type × verification_status 填充合法性、
非 profile 的 eligible_for_recommendation 约束、Eligibility 只保留
is_active / verification_status=rejected 两条规则，以及 Worker 在跳过时
不触碰 Embedding 服务。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from expert_contracts.models import ExpertChunkEmbeddingInput
from rag_vector_store.chunking.contract_pipeline import (
    ContractValidationError,
    ContractValidator,
    EmbeddingEligibilityValidator,
    compare_contract_output_schema,
    expected_contract_output_fields,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(chunk_type: str) -> str:
    return {
        "profile_summary": "[专家能力概述]\n虚构测试专家具备检索系统建设经验。",
        "employment": "[任职经历]\n公司:某公司\n岗位:产品经理\n职责简述。",
        "project": "[项目经历]\n项目名称:某项目\n目标与动作概述。",
    }[chunk_type]


def _metadata(chunk_type: str) -> dict[str, Any]:
    """按 chunk_type 构造对应 Metadata 字段（不输出其他类型的占位）。"""
    common: dict[str, Any] = {
        "company_name_raw": "某公司",
        "company_name_norm": "某标准公司",
        "industry_codes": [],
        "topic_codes": [],
        "origin_type": "source_parsed",
        "verification_status": "verified",
        "weight_tier": "A",
    }
    if chunk_type == "profile_summary":
        common.update(
            {
                "company_name_raw": None,
                "company_name_norm": None,
                "role_name_raw": None,
                "role_code": None,
                "role_level": None,
                "job_status_norm": "active",
                "abroad_flag_norm": False,
                "work_years": Decimal("12.00"),
                "function_codes": ["product_management"],
                "school_names_raw": ["测试大学"],
                "major_names_raw": ["软件工程"],
                "eligible_for_recommendation": True,
                "origin_type": "compressed_from_source",
                "verification_status": "source_summary",
                "weight_tier": "B",
            }
        )
    elif chunk_type == "employment":
        common.update(
            {
                "role_name_raw": "产品经理",
                "role_code": "product_manager",
                "role_level": "manager",
                "start_month": "202301",
                "end_month": None,
                "is_current": True,
                "industry_codes": ["software"],
            }
        )
    elif chunk_type == "project":
        common.update(
            {
                "project_name": "某项目",
                "start_month": "202401",
                "end_month": "202406",
            }
        )
    return common


def make_record(
    chunk_type: str = "profile_summary",
    *,
    source_field_id: str | None = None,
    verification_status: str | None = None,
    origin_type: str | None = None,
    eligible: Any = None,
    is_active: bool = True,
    weight_tier: str | None = None,
    _set_eligible: bool = False,
) -> ExpertChunkEmbeddingInput:
    text = _text(chunk_type)
    sid = source_field_id or {
        "profile_summary": "900001",
        "employment": "21307",
        "project": "161",
    }[chunk_type]
    meta = _metadata(chunk_type)
    if origin_type is not None:
        meta["origin_type"] = origin_type
    if verification_status is not None:
        meta["verification_status"] = verification_status
    if weight_tier is not None:
        meta["weight_tier"] = weight_tier
    if _set_eligible:
        meta["eligible_for_recommendation"] = eligible
    return ExpertChunkEmbeddingInput(
        contract_version="4.1.0",
        vector_doc_id=f"expert:7:{chunk_type}:{sid}:0",
        expert_id=7,
        serial_number="SN-7",
        chunk_type=chunk_type,
        chunk_subtype=None,
        chunk_index=0,
        chunk_total=1,
        chunk_text=text,
        char_count=len(text),
        content_hash=_sha(text),
        metadata_hash="b" * 64,
        metadata=meta,
        source={
            "source_table": "experts",
            "source_field": "background",
            "source_field_id": sid,
            "source_version": "2026-08-11T10:00:00+08:00",
            "source_updated_at": "2026-08-11T10:00:00+08:00",
            "source_hash": "a" * 64,
        },
        versions={"chunk_policy_version": "1.3", "metadata_schema_version": "4.0"},
        is_active=is_active,
        updated_at="2026-08-11T10:05:00+08:00",
    )


def _validator() -> ContractValidator:
    return ContractValidator()


def _eligibility() -> EmbeddingEligibilityValidator:
    return EmbeddingEligibilityValidator()


def _contract_schema_description(vector_dimension: int) -> dict[str, object]:
    fields: list[dict[str, object]] = []
    for name, expected in expected_contract_output_fields(vector_dimension).items():
        params = {
            key: expected[key]
            for key in ("max_length", "dim")
            if expected[key] is not None
        }
        fields.append(
            {
                "name": name,
                "type": expected["datatype"],
                "nullable": expected["nullable"],
                "is_primary": expected["primary_key"],
                "params": params,
            }
        )
    return {
        "auto_id": False,
        "enable_dynamic_field": False,
        "fields": fields,
    }


# ---- 合法 + eligible ----

def test_profile_source_summary_is_valid_and_eligible() -> None:
    prepared = _validator().validate(make_record("profile_summary"))

    assert _eligibility().is_eligible(prepared) is True
    assert _eligibility().get_skip_reason(prepared) is None


def test_employment_verified_is_valid_and_eligible() -> None:
    prepared = _validator().validate(make_record("employment"))

    assert _eligibility().is_eligible(prepared) is True


def test_project_verified_is_valid_and_eligible() -> None:
    prepared = _validator().validate(make_record("project"))

    assert _eligibility().is_eligible(prepared) is True


# ---- rejected 是合法但 skip ----

def test_profile_rejected_is_valid_but_skipped() -> None:
    record = make_record("profile_summary", verification_status="rejected")
    prepared = _validator().validate(record)  # 不抛 ContractValidationError

    assert _eligibility().is_eligible(prepared) is False
    assert "rejected" in _eligibility().get_skip_reason(prepared)


def test_is_active_false_is_valid_but_skipped() -> None:
    record = make_record("profile_summary", is_active=False)
    prepared = _validator().validate(record)

    assert _eligibility().is_eligible(prepared) is False
    assert _eligibility().get_skip_reason(prepared) == "is_active=false"


# ---- 非法（ContractValidator fail-closed）----

def test_employment_with_source_summary_is_invalid() -> None:
    with pytest.raises(ContractValidationError, match="verification_status"):
        _validator().validate(make_record("employment", verification_status="source_summary"))


def test_project_with_rejected_is_invalid() -> None:
    with pytest.raises(ContractValidationError, match="verification_status"):
        _validator().validate(make_record("project", verification_status="rejected"))


def test_profile_with_verified_is_invalid() -> None:
    with pytest.raises(ContractValidationError, match="verification_status"):
        _validator().validate(make_record("profile_summary", verification_status="verified"))


def test_non_profile_with_eligible_for_recommendation_is_invalid() -> None:
    # per-type 后 employment Metadata 不含 eligible_for_recommendation 字段，
    # 传入该字段在 Contract 构造层即被 extra=forbid 拒绝（fail-closed 语义不变，
    # 从“Validator 拒绝”提前到“Contract 构造拒绝”）。
    with pytest.raises(ValidationError, match="extra_forbidden"):
        make_record("employment", _set_eligible=True, eligible=True)


def test_wrong_origin_type_is_invalid() -> None:
    with pytest.raises(ContractValidationError, match="origin_type"):
        _validator().validate(
            make_record("profile_summary", origin_type="source_parsed")
        )


# ---- weight_tier / eligible_for_recommendation 不再影响 Eligibility ----

def test_weight_tier_does_not_affect_eligibility() -> None:
    # weight_tier 是检索权重字段：合法枚举值任意都不影响是否向量化。
    prepared = _validator().validate(
        make_record("employment", weight_tier="C")
    )
    assert _eligibility().is_eligible(prepared) is True


def test_eligible_for_recommendation_false_does_not_affect_eligibility() -> None:
    # profile 的 eligible_for_recommendation 只是推荐资格，false 仍可向量化。
    prepared = _validator().validate(
        make_record("profile_summary", _set_eligible=True, eligible=False)
    )
    assert _eligibility().is_eligible(prepared) is True


def test_eligible_for_recommendation_none_does_not_affect_eligibility() -> None:
    prepared = _validator().validate(
        make_record("profile_summary", _set_eligible=True, eligible=None)
    )
    assert _eligibility().is_eligible(prepared) is True


# ---- Worker：skip 时不触碰 Embedding 服务 ----

class FakeEmbeddingModel:
    dimension = 4
    model_alias = "fake-model"
    model_version = "fake-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_worker_skips_rejected_without_embedding_call() -> None:
    from rag_vector_store.chunking.contract_pipeline import ContractEmbeddingWorker

    model = FakeEmbeddingModel()
    worker = ContractEmbeddingWorker(model, expected_dimension=4)

    record = make_record("profile_summary", verification_status="rejected")
    result = worker.embed([record])

    assert result.embedded_chunks == []
    assert result.skipped_count == 1
    assert model.calls == []
    assert result.skipped_chunks[0].vector_doc_id == record.vector_doc_id


def test_worker_embeds_eligible_and_uses_vector_doc_id() -> None:
    from rag_vector_store.chunking.contract_pipeline import ContractEmbeddingWorker

    model = FakeEmbeddingModel()
    worker = ContractEmbeddingWorker(model, expected_dimension=4)

    record = make_record("employment")
    result = worker.embed([record])

    assert result.eligible_count == 1
    assert model.calls == [[record.chunk_text]]
    embedded = result.embedded_chunks[0]
    assert embedded.vector_doc_id == record.vector_doc_id
    assert not hasattr(embedded, "chunk_id")


# ---- Schema 比较：旧公开入口复用正式 Compatibility Gate ----


def test_contract_schema_comparison_accepts_exact_v15_schema() -> None:
    result = compare_contract_output_schema(
        _contract_schema_description(4),
        vector_dimension=4,
    )

    assert result["compatible"] is True
    assert result["missing_fields"] == []
    assert result["unexpected_fields"] == []
    assert result["changed_fields"] == {}
    assert result["collection_options"] == {}
    assert result["incompatible_fields"] == {}


def test_contract_schema_comparison_rejects_unexpected_fields_and_options() -> None:
    description = _contract_schema_description(4)
    fields = description["fields"]
    assert isinstance(fields, list)
    fields.append(
        {
            "name": "legacy_expert_id",
            "type": "INT64",
            "nullable": False,
            "is_primary": False,
            "params": {},
        }
    )
    description["enable_dynamic_field"] = True

    result = compare_contract_output_schema(description, vector_dimension=4)

    assert result["compatible"] is False
    assert result["unexpected_fields"] == ["legacy_expert_id"]
    assert result["collection_options"] == {
        "enable_dynamic_field": {"expected": False, "actual": True}
    }


def test_contract_schema_comparison_reports_primary_key_mismatch() -> None:
    description = _contract_schema_description(4)
    fields = description["fields"]
    assert isinstance(fields, list)
    vector_doc_id = next(field for field in fields if field["name"] == "vector_doc_id")
    vector_doc_id["is_primary"] = False

    result = compare_contract_output_schema(description, vector_dimension=4)

    assert result["compatible"] is False
    assert result["incompatible_fields"] == {
        "vector_doc_id": {
            "primary_key": {"expected": True, "actual": False}
        }
    }


# ---- 正式表 NULLABLE 字段不因旧 v2 必填规则 fail ----

def _with_metadata(record: ExpertChunkEmbeddingInput, **updates: Any) -> ExpertChunkEmbeddingInput:
    return record.model_copy(
        update={"metadata": record.metadata.model_copy(update=updates)}
    )


def test_employment_start_month_none_is_valid() -> None:
    record = _with_metadata(make_record("employment"), start_month=None)

    prepared = _validator().validate(record)  # 不抛 ContractValidationError
    assert prepared.metadata.start_month is None


def test_employment_is_current_none_is_valid() -> None:
    record = _with_metadata(make_record("employment"), is_current=None)

    prepared = _validator().validate(record)
    assert prepared.metadata.is_current is None


def test_employment_nullable_company_and_role_do_not_fail() -> None:
    record = _with_metadata(
        make_record("employment"),
        company_name_raw=None,
        company_name_norm=None,
        role_name_raw=None,
        role_code=None,
        role_level=None,
    )

    prepared = _validator().validate(record)
    assert prepared.metadata.company_name_raw is None
    assert prepared.metadata.role_name_raw is None


def test_project_project_name_none_is_valid() -> None:
    record = _with_metadata(make_record("project"), project_name=None)

    prepared = _validator().validate(record)  # 不受旧 v2 project_name 必填规则影响
    assert prepared.metadata.project_name is None


def test_is_current_true_with_end_month_is_invalid() -> None:
    record = _with_metadata(make_record("employment"), is_current=True, end_month="202312")

    with pytest.raises(ContractValidationError, match="is_current"):
        _validator().validate(record)
