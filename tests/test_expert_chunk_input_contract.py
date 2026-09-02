"""专家 Chunk 输入契约 4.0.0 的结构与字段语义单元测试。

本文件只验证 ``expert_contracts`` 的 Contract 模型、Adapter 和正式 fixture，
不依赖 Validator / Eligibility / Milvus 等下游模块（它们在各自任务中适配）。
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from expert_contracts import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    OriginType,
    VerificationStatus,
    WeightTier,
    adapt_expert_chunk_mapping,
)
from expert_contracts.models import ExpertChunkEmbeddingInput

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)

_REMOVED_TOP = {"chunk_id", "source_item_key", "data_version", "dictionary_version"}
_REMOVED_METADATA = {"vector_doc_id", "project_role", "company_normalization_status"}
_REMOVED_SOURCE = {"source_item_key", "source_record_id"}
_ADDED_METADATA = {
    "job_status_norm",
    "abroad_flag_norm",
    "work_years",
    "function_codes",
    "school_names_raw",
    "major_names_raw",
    "eligible_for_recommendation",
    "metadata_hash",
}


def fixture_records() -> list[ExpertChunkEmbeddingInput]:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        return [
            adapt_expert_chunk_mapping(json.loads(line))
            for line in stream
            if line.strip()
        ]


def fixture_payload(chunk_type: str = "profile_summary") -> dict[str, object]:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        payloads = [json.loads(line) for line in stream]
    return next(payload for payload in payloads if payload["chunk_type"] == chunk_type)


def test_contract_version_is_400() -> None:
    assert EXPERT_CHUNK_CONTRACT_VERSION == "4.1.0"


def test_three_contract_fixture_types_pass() -> None:
    records = fixture_records()

    assert len(records) == 3
    assert [record.chunk_type for record in records] == [
        "profile_summary",
        "employment",
        "project",
    ]
    assert {record.contract_version for record in records} == {"4.1.0"}


def test_generated_json_schema_is_strict_contract_v400() -> None:
    schema = ExpertChunkEmbeddingInput.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["contract_version"]["const"] == "4.1.0"
    assert "vector_doc_id" in schema["properties"]
    assert "chunk_id" not in schema["properties"]
    assert set(schema["required"]) >= {
        "metadata_hash",
        "metadata",
        "source",
        "versions",
        "vector_doc_id",
        "serial_number",
    }

    # per-type Metadata：三个独立模型，不再有统一 superset 模型。
    for model_name in (
        "ProfileSummaryMetadata",
        "EmploymentMetadata",
        "ProjectMetadata",
    ):
        assert model_name in schema["$defs"], model_name
    metadata_schema = schema["$defs"]["ProfileSummaryMetadata"]
    assert "vector_doc_id" not in metadata_schema["required"]
    assert "vector_doc_id" not in metadata_schema["properties"]
    assert "project_role" not in metadata_schema["properties"]
    assert "company_normalization_status" not in metadata_schema["properties"]
    assert set(_ADDED_METADATA - {"metadata_hash"}) <= set(
        metadata_schema["properties"]
    )
    assert "work_years" in metadata_schema["properties"]
    # Employment/Project 模型不得包含 profile 专属字段。
    employment_schema = schema["$defs"]["EmploymentMetadata"]
    assert "work_years" not in employment_schema["properties"]
    assert "school_names_raw" not in employment_schema["properties"]
    assert "eligible_for_recommendation" not in employment_schema["properties"]
    project_schema = schema["$defs"]["ProjectMetadata"]
    assert "role_name_raw" not in project_schema["properties"]
    assert "is_current" not in project_schema["properties"]
    assert "eligible_for_recommendation" not in project_schema["properties"]

    source_schema = schema["$defs"]["ExpertChunkSource"]
    assert "source_record_id" not in source_schema["properties"]
    assert "source_item_key" not in source_schema["properties"]
    assert "source_field_id" in source_schema["required"]
    assert source_schema["properties"]["source_field"]["maxLength"] == 512

    versions_schema = schema["$defs"]["ExpertChunkVersions"]
    assert set(versions_schema["required"]) == {
        "chunk_policy_version",
        "metadata_schema_version",
    }
    assert "data_version" not in versions_schema["properties"]
    assert "dictionary_version" not in versions_schema["properties"]


def test_removed_fields_are_absent_from_contract() -> None:
    record = fixture_records()[0]
    top_dump = record.model_dump()
    metadata_dump = record.metadata.model_dump()
    source_dump = record.source.model_dump()
    versions_dump = record.versions.model_dump()

    assert not (set(top_dump) & _REMOVED_TOP)
    assert not (set(metadata_dump) & _REMOVED_METADATA)
    assert not (set(source_dump) & _REMOVED_SOURCE)
    assert set(versions_dump) == {"chunk_policy_version", "metadata_schema_version"}
    assert top_dump["vector_doc_id"] == record.vector_doc_id


def test_vector_doc_id_only_on_top_level_and_matches_formula() -> None:
    record = fixture_records()[0]

    # 顶层为主键，metadata 不再携带副本。
    assert record.vector_doc_id
    assert not hasattr(record.metadata, "vector_doc_id")

    formula = (
        f"expert:{record.expert_id}:{record.chunk_type}:"
        f"{record.source.source_field_id}:{record.chunk_index}"
    )
    assert record.vector_doc_id == formula


def test_vector_doc_id_formula_mismatch_is_rejected() -> None:
    payload = fixture_payload()
    payload["vector_doc_id"] = "expert:999999:profile_summary:999999:0"

    with pytest.raises(ValueError, match="vector_doc_id"):
        ExpertChunkEmbeddingInput.model_validate(payload)


def test_resume_source_uses_resume_vector_doc_id_namespace() -> None:
    """共享 Contract 按 Resume 冻结来源标签接受 ``resume:`` 主键。"""

    payload = fixture_payload("employment")
    payload["source"]["source_table"] = "resume_work_experience"
    payload["vector_doc_id"] = "resume:900001:employment:21307:0"

    record = ExpertChunkEmbeddingInput.model_validate(payload)

    assert record.vector_doc_id == "resume:900001:employment:21307:0"


def test_contract_version_error_is_rejected() -> None:
    payload = fixture_payload()
    payload["contract_version"] = "2.0.0"

    with pytest.raises(ValueError, match="contract_version"):
        ExpertChunkEmbeddingInput.model_validate(payload)


def test_work_years_is_decimal_not_float() -> None:
    record = fixture_records()[0]

    assert isinstance(record.metadata.work_years, Decimal)
    assert not isinstance(record.metadata.work_years, float)
    assert record.metadata.work_years == Decimal("12.00")


def test_month_format_is_yyyymm() -> None:
    employment = fixture_records()[1]
    project = fixture_records()[2]

    assert employment.metadata.start_month == "202301"
    assert project.metadata.start_month == "202401"
    assert project.metadata.end_month == "202406"


def test_verification_status_is_source_summary_rejected_verified() -> None:
    values = set(VerificationStatus.__args__)

    assert values == {"source_summary", "rejected", "verified"}
    # 旧的 source_evidence/validated 值必须在契约层被拒绝。
    old_payload = fixture_payload("employment")
    old_payload["metadata"]["verification_status"] = "source_evidence"
    with pytest.raises(ValueError, match="verification_status"):
        ExpertChunkEmbeddingInput.model_validate(old_payload)


def test_origin_type_is_compressed_from_source_or_source_parsed() -> None:
    values = set(OriginType.__args__)

    assert values == {"compressed_from_source", "source_parsed"}
    old_payload = fixture_payload("profile_summary")
    old_payload["metadata"]["origin_type"] = "source_template"
    with pytest.raises(ValueError, match="origin_type"):
        ExpertChunkEmbeddingInput.model_validate(old_payload)


def test_weight_tier_type_retains_known_values() -> None:
    # 正式约束前不擅自删除可能存在的合法取值。
    assert set(WeightTier.__args__) == {"A", "B", "C"}


def test_profile_only_fields_are_present() -> None:
    profile = fixture_records()[0]

    assert profile.metadata.eligible_for_recommendation is True
    assert profile.metadata.job_status_norm == "active"
    assert profile.metadata.abroad_flag_norm is False
    assert profile.metadata.school_names_raw == ["华东流程测试大学"]
    assert profile.metadata.major_names_raw == ["信息管理与信息系统"]
    assert profile.metadata.function_codes == ["product_management"]


def test_non_profile_fields_are_absent_not_null() -> None:
    """per-type 后：employment/project 的 metadata 不包含 profile 专属字段。"""
    employment = fixture_records()[1]
    employment_dump = employment.metadata.model_dump()
    for field in (
        "eligible_for_recommendation",
        "job_status_norm",
        "abroad_flag_norm",
        "work_years",
        "function_codes",
        "school_names_raw",
        "major_names_raw",
    ):
        assert field not in employment_dump, field
        assert not hasattr(employment.metadata, field), field

    project = fixture_records()[2]
    project_dump = project.metadata.model_dump()
    for field in (
        "role_name_raw",
        "role_code",
        "role_level",
        "is_current",
        "eligible_for_recommendation",
        "school_names_raw",
        "major_names_raw",
    ):
        assert field not in project_dump, field
        assert not hasattr(project.metadata, field), field

    profile = fixture_records()[0]
    profile_dump = profile.metadata.model_dump()
    for field in ("project_name", "start_month", "end_month", "is_current"):
        assert field not in profile_dump, field
        assert not hasattr(profile.metadata, field), field


def test_metadata_type_must_match_chunk_type_fail_closed() -> None:
    """chunk_type 与 Metadata 类型不匹配必须被拒绝（fail-closed）。"""
    employment = fixture_records()[1]
    project = fixture_records()[2]

    # employment + ProfileSummaryMetadata（把 metadata 换成 profile 字段集）
    bad_employment = employment.model_copy(
        update={"metadata": fixture_records()[0].metadata}
    )
    with pytest.raises(ValueError, match="metadata 必须是 EmploymentMetadata"):
        ExpertChunkEmbeddingInput.model_validate(
            bad_employment.model_dump(mode="json")
        )

    # project + EmploymentMetadata
    bad_project = project.model_copy(
        update={"metadata": employment.metadata}
    )
    with pytest.raises(ValueError, match="metadata 必须是 ProjectMetadata"):
        ExpertChunkEmbeddingInput.model_validate(
            bad_project.model_dump(mode="json")
        )

    # profile_summary + ProjectMetadata
    profile = fixture_records()[0]
    bad_profile = profile.model_copy(update={"metadata": project.metadata})
    with pytest.raises(ValueError, match="metadata 必须是 ProfileSummaryMetadata"):
        ExpertChunkEmbeddingInput.model_validate(
            bad_profile.model_dump(mode="json")
        )


def test_metadata_hash_is_sha256_and_versions_are_13_30() -> None:
    for record in fixture_records():
        assert len(record.metadata_hash) == 64
        assert set(record.metadata_hash) <= set("0123456789abcdef")
        assert record.versions.chunk_policy_version == "1.3"
        assert record.versions.metadata_schema_version == "4.0"


def test_hash_is_computed_from_chunk_text_in_fixture() -> None:
    record = fixture_records()[0]
    assert record.content_hash == hashlib.sha256(
        record.chunk_text.encode("utf-8")
    ).hexdigest()


def test_all_records_agree_on_vector_doc_id_with_source() -> None:
    for record in fixture_records():
        assert record.vector_doc_id.startswith(
            f"expert:{record.expert_id}:{record.chunk_type}:"
        )
        assert record.vector_doc_id.endswith(f":{record.chunk_index}")
