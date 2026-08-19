"""共享 Contract / Adapter / Bridge 层的闭合单元测试（导出契约 4.0.0）。

本文件验证 ``ExpertChunkMappingAdapter`` 如何把 JSON-compatible Mapping 无损
适配为 ``ExpertChunkEmbeddingInput``（4.0.0），并保证旧结构字段在契约层已被
彻底移除。不依赖 Ingestion / Validator / Milvus 等下游（由各自任务适配）。
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from expert_contracts import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    ExpertChunkEmbeddingInput,
    ExpertChunkMappingAdapter,
    ExpertChunkMappingError,
)


def expert_chunk_mapping() -> dict[str, object]:
    text = "[专家能力概述]\n上游生成的共享 Contract 测试记录。"
    return {
        # vector_doc_id 只存在顶层，且满足正式公式。
        "contract_version": "4.1.0",
        "vector_doc_id": "expert:900001:profile_summary:900001:0",
        "expert_id": 900001,
        "serial_number": "DEV900001",
        "chunk_type": "profile_summary",
        "chunk_subtype": None,
        "chunk_index": 0,
        "chunk_total": 1,
        "chunk_text": text,
        "char_count": len(text),
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "metadata_hash": "a" * 64,
        "metadata": {
            "company_name_raw": None,
            "company_name_norm": None,
            "role_name_raw": None,
            "role_code": None,
            "role_level": None,
            "job_status_norm": "active",
            "abroad_flag_norm": False,
            "work_years": "9.50",
            "industry_codes": ["artificial_intelligence"],
            "function_codes": ["product_management"],
            "topic_codes": ["knowledge_retrieval"],
            "school_names_raw": ["测试大学"],
            "major_names_raw": ["软件工程"],
            "eligible_for_recommendation": True,
            "origin_type": "compressed_from_source",
            "verification_status": "source_summary",
            "weight_tier": "B",
        },
        "source": {
            "source_table": "experts",
            "source_field": "company+job_name+background",
            "source_field_id": "900001",
            "source_version": "2026-08-12T10:00:00+08:00",
            "source_updated_at": "2026-08-12T10:00:00+08:00",
            "source_hash": "a" * 64,
        },
        "versions": {
            "chunk_policy_version": "1.3",
            "metadata_schema_version": "4.0",
        },
        "is_active": True,
        "updated_at": "2026-08-12T10:05:00+08:00",
    }


def test_adapter_maps_json_mapping_to_contract_v400() -> None:
    mapping = expert_chunk_mapping()
    metadata = dict(mapping["metadata"])
    source = dict(mapping["source"])
    versions = dict(mapping["versions"])

    contract = ExpertChunkMappingAdapter().from_mapping(mapping)

    assert isinstance(contract, ExpertChunkEmbeddingInput)
    assert contract.contract_version == EXPERT_CHUNK_CONTRACT_VERSION == "4.1.0"
    # vector_doc_id 只存在顶层，metadata 不再携带副本。
    assert contract.vector_doc_id == "expert:900001:profile_summary:900001:0"
    assert not hasattr(contract.metadata, "vector_doc_id")
    # 扁平字段、来源与版本无损适配；work_years 会被规范为 Decimal（非 float）。
    contract_metadata = contract.metadata.model_dump(exclude_unset=True)
    raw_metadata = dict(metadata)
    raw_metadata["work_years"] = "9.50"
    for key, value in raw_metadata.items():
        if key == "work_years":
            continue
        assert contract_metadata[key] == value, key
    assert contract.metadata.work_years == Decimal("9.50")
    assert not isinstance(contract.metadata.work_years, float)
    assert contract.source.model_dump() == source
    assert contract.versions.model_dump() == versions
    # 新增字段类型正确：work_years 是 Decimal（非 float）。
    assert str(contract.metadata.work_years) == "9.50"


def test_adapter_rejects_top_level_vector_doc_id_not_matching_formula() -> None:
    mapping = expert_chunk_mapping()
    mapping["vector_doc_id"] = "different-upstream-id"

    with pytest.raises(ExpertChunkMappingError, match="vector_doc_id"):
        ExpertChunkMappingAdapter().from_mapping(mapping)


def test_adapter_requires_top_level_vector_doc_id() -> None:
    mapping = expert_chunk_mapping()
    del mapping["vector_doc_id"]

    with pytest.raises(ExpertChunkMappingError, match="vector_doc_id"):
        ExpertChunkMappingAdapter().from_mapping(mapping)


def test_adapter_rejects_deprecated_fields() -> None:
    # 旧结构字段在 4.0.0 契约中已不存在，必须 fail-closed。
    cases: list[tuple[str, dict[str, object]]] = [
        ("source_record_id", {"source": {"source_record_id": "deprecated"}}),
        ("chunk_id", {"chunk_id": "expert:900001:profile_summary:old:0"}),
        ("metadata.vector_doc_id", {"metadata": {"vector_doc_id": "old"}}),
        ("source.source_item_key", {"source": {"source_item_key": "old"}}),
        ("versions.data_version", {"versions": {"data_version": "old"}}),
        ("versions.dictionary_version", {"versions": {"dictionary_version": "old"}}),
        ("metadata.project_role", {"metadata": {"project_role": "负责人"}}),
        (
            "metadata.company_normalization_status",
            {"metadata": {"company_normalization_status": "verified"}},
        ),
    ]
    for _label, update in cases:
        mapping = expert_chunk_mapping()
        _apply_nested(mapping, update)
        with pytest.raises(ExpertChunkMappingError):
            ExpertChunkMappingAdapter().from_mapping(mapping)


def test_adapter_accepts_rejected_and_verified_statuses() -> None:
    # rejected 是合法业务状态；verified 是 employment/project 的固定值。
    for status in ("rejected", "verified"):
        mapping = expert_chunk_mapping()
        mapping["metadata"]["verification_status"] = status
        contract = ExpertChunkMappingAdapter().from_mapping(mapping)
        assert contract.metadata.verification_status == status


def _apply_nested(target: dict[str, object], update: dict[str, object]) -> None:
    for key, value in update.items():
        if isinstance(value, dict):
            inner = target.get(key)
            if isinstance(inner, dict):
                _apply_nested(inner, value)
                continue
        target[key] = value