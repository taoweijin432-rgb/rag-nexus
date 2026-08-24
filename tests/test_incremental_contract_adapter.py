"""Phase 2：Work Item → V15 正式 Contract 薄适配器的单元测试。

只验证 ``rag_vector_store.incremental.contract_adapter`` 的映射与 fail-closed
行为；不连接 MySQL、不调用 Milvus、不调用 Embedding、不修改同事代码。
Contract 合法字段值取自共享 fixture（与 Contract 模型测试同源），保证夹具
本身可通过全部既有校验。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from expert_contracts.models import (
    EXPERT_CHUNK_METADATA_BY_TYPE,
    EmploymentMetadata,
    ProfileSummaryMetadata,
    ProjectMetadata,
)
from rag_vector_store.incremental import (
    DisableWorkItem,
    V15WorkItemContractAdapter,
    WorkItemContractAdaptation,
    WorkItemContractAdapterError,
)
from rag_vector_store.incremental.models import V15VectorizationWorkItem

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)


def _fixture_contract(chunk_type: str) -> dict[str, object]:
    """读取共享 fixture 中指定 chunk_type 的合法契约载荷。"""

    with _FIXTURE_PATH.open(encoding="utf-8") as stream:
        payloads = [json.loads(line) for line in stream if line.strip()]
    return next(p for p in payloads if p["chunk_type"] == chunk_type)


def _structure_keys(contract_dict: dict[str, object]) -> dict[str, object]:
    """把契约载荷拆成 Phase 2 约定的结构保留键集合。"""

    source = contract_dict["source"]
    versions = contract_dict["versions"]
    assert isinstance(source, dict) and isinstance(versions, dict)
    return {
        "chunk_subtype": contract_dict["chunk_subtype"],
        "chunk_index": contract_dict["chunk_index"],
        "chunk_total": contract_dict["chunk_total"],
        "char_count": contract_dict["char_count"],
        "serial_number": contract_dict["serial_number"],
        "updated_at": contract_dict["updated_at"],
        "source_field": source["source_field"],
        "source_version": source["source_version"],
        "source_hash": source["source_hash"],
        "metadata_schema_version": versions["metadata_schema_version"],
    }


def _work_item(
    chunk_type: str = "employment",
    *,
    metadata_extra: dict[str, object] | None = None,
    metadata_remove: tuple[str, ...] = (),
    **overrides: Any,
) -> V15VectorizationWorkItem:
    """构造一条以合法契约为底稿的 Work Item 完整行。"""

    contract_dict = _fixture_contract(chunk_type)
    assert isinstance(contract_dict["metadata"], dict)
    metadata: dict[str, object] = {
        **contract_dict["metadata"],  # type: ignore[dict-item]
        **_structure_keys(contract_dict),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    for key in metadata_remove:
        metadata.pop(key, None)

    source = contract_dict["source"]
    assert isinstance(source, dict)
    defaults: dict[str, object] = {
        "operation": "upsert",
        "vector_doc_id": contract_dict["vector_doc_id"],
        "chunk_text": contract_dict["chunk_text"],
        "metadata": metadata,
        "content_hash": contract_dict["content_hash"],
        "metadata_hash": contract_dict["metadata_hash"],
        "chunk_policy_version": "1.3",
        "embedding_model_alias": None,
        "embedding_model_version": None,
        "is_active": True,
        "expert_id": contract_dict["expert_id"],
        "chunk_type": chunk_type,
        "source_table": source["source_table"],
        "source_field_id": source["source_field_id"],
        "source_event_id": 1042,
        "source_updated_at": datetime(2026, 8, 11, 10, 0, 0,
                                      tzinfo=timezone(timedelta(hours=8))),
        "stale_vector_doc_ids": (),
        "status": "created",
        "work_item_id": 1,
        "batch_id": 1,
    }
    defaults.update(overrides)
    return V15VectorizationWorkItem(**defaults)  # type: ignore[arg-type]


def _disable_item(**overrides: Any) -> V15VectorizationWorkItem:
    """构造 disable Work Item 完整行。

    真实入库路径中 is_active=False 由 ``V15WorkItemInput`` 的 operation
    校验器强制；全行读取模型不再重复该校验，因此这里显式传入。
    """

    return _work_item(
        "employment",
        operation="disable",
        chunk_text=None,
        content_hash=None,
        metadata_hash=None,
        metadata=None,
        is_active=False,
        **overrides,
    )


# ---------------------------------------------------------------------------
# 三类 chunk_type 成功映射
# ---------------------------------------------------------------------------


def test_upsert_employment_maps_to_full_contract() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item("employment")

    result = adapter.adapt_upsert(work_item)

    assert isinstance(result, WorkItemContractAdaptation)
    contract = result.contract
    assert contract.contract_version == "4.1.0"
    assert contract.chunk_type == "employment"
    assert isinstance(contract.metadata, EmploymentMetadata)
    assert contract.expert_id == 900001
    # 来源定位统一使用 source_field_id。
    assert contract.source.source_field_id == "21307"
    assert contract.source.source_table == "experts_work_experience"
    assert contract.versions.chunk_policy_version == "1.3"
    assert contract.versions.metadata_schema_version == "4.0"
    # 合法且 eligible 的记录：Eligibility 预判为允许 Embedding。
    assert result.eligibility_skip_reason is None
    assert result.prepared.vector_doc_id == contract.vector_doc_id


def test_upsert_project_maps_source_field_and_chunk_index() -> None:
    adapter = V15WorkItemContractAdapter()
    # 模拟 re-chunk 后的第二个子块：公式与 chunk_index 同步变化。
    work_item = _work_item(
        "project",
        vector_doc_id="expert:900001:project:161:1",
        metadata_extra={"chunk_index": 1, "chunk_total": 3,
                        "chunk_subtype": "result"},
    )

    result = adapter.adapt_upsert(work_item)

    contract = result.contract
    assert contract.chunk_index == 1
    assert contract.chunk_total == 3
    assert contract.chunk_subtype == "result"
    assert contract.vector_doc_id == "expert:900001:project:161:1"
    assert contract.source.source_field_id == "161"
    assert isinstance(contract.metadata, ProjectMetadata)


def test_upsert_profile_summary_keeps_education_inside_metadata_only() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item("profile_summary")

    result = adapter.adapt_upsert(work_item)

    metadata = result.contract.metadata
    assert isinstance(metadata, ProfileSummaryMetadata)
    # 教育信息只作为 profile_summary 的 Metadata 来源（学校/专业数组），
    # 不存在也不允许出现独立的教育 Chunk 类型。
    assert metadata.school_names_raw == ["华东流程测试大学"]
    assert metadata.major_names_raw == ["信息管理与信息系统"]
    assert set(EXPERT_CHUNK_METADATA_BY_TYPE) == {
        "profile_summary", "employment", "project",
    }
    # 模型层：education 不是合法 Chunk 类型（Literal 三值约束拒绝）。
    education_row = _work_item("profile_summary").model_dump()
    education_row["chunk_type"] = "education"
    with pytest.raises(ValidationError, match="chunk_type"):
        V15VectorizationWorkItem.model_validate(education_row)


def test_upsert_requires_expert_identity_and_metadata_hash() -> None:
    adapter = V15WorkItemContractAdapter()

    with pytest.raises(WorkItemContractAdapterError, match="expert_id"):
        adapter.adapt_upsert(_work_item(expert_id=None))
    with pytest.raises(WorkItemContractAdapterError, match="expert_id"):
        adapter.adapt_upsert(_work_item(expert_id=0))
    with pytest.raises(WorkItemContractAdapterError, match="metadata_hash"):
        adapter.adapt_upsert(_work_item(metadata_hash=None))
    with pytest.raises(WorkItemContractAdapterError, match="source_field_id"):
        adapter.adapt_upsert(_work_item(source_field_id=None))


# ---------------------------------------------------------------------------
# chunk_text 原样透传与完整性校验
# ---------------------------------------------------------------------------


def test_chunk_text_is_passed_through_verbatim() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item("employment")
    original_text = work_item.chunk_text
    assert original_text is not None and "\n" in original_text

    contract = adapter.adapt_upsert(work_item).contract

    # 只做透传：对象为同一值，未做任何拼接、截断或清洗。
    assert contract.chunk_text == original_text
    assert contract.char_count == len(original_text)


def test_char_count_mismatch_is_rejected() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item("employment", metadata_extra={"char_count": 9999})

    with pytest.raises(WorkItemContractAdapterError, match="char_count"):
        adapter.adapt_upsert(work_item)


def test_content_hash_mismatch_is_rejected() -> None:
    adapter = V15WorkItemContractAdapter()
    wrong_hash = "a" * 64  # 格式合法但与 chunk_text 不一致
    work_item = _work_item("employment", content_hash=wrong_hash)

    with pytest.raises(WorkItemContractAdapterError, match="content_hash"):
        adapter.adapt_upsert(work_item)


def test_malformed_metadata_hash_is_rejected_at_input_boundary() -> None:
    """hash 格式校验发生在创建入口（V15WorkItemInput），全行读取模型不重复。"""

    from rag_vector_store.incremental import V15WorkItemInput

    with pytest.raises(ValidationError):
        V15WorkItemInput(
            operation="upsert",
            vector_doc_id="expert:900001:employment:21307:0",
            chunk_text="任一非空文本",
            content_hash="a" * 64,
            metadata_hash="not-a-hash",
        )


# ---------------------------------------------------------------------------
# 公共与私有 Metadata 映射
# ---------------------------------------------------------------------------


def test_common_metadata_fields_are_mapped() -> None:
    adapter = V15WorkItemContractAdapter()
    contract = adapter.adapt_upsert(_work_item("employment")).contract

    metadata = contract.metadata
    assert metadata.company_name_raw == "测试科技有限公司"
    assert metadata.company_name_norm == "测试科技有限公司"
    assert metadata.industry_codes == ["artificial_intelligence",
                                       "knowledge_retrieval"]
    assert metadata.origin_type == "source_parsed"
    assert metadata.verification_status == "verified"
    assert metadata.weight_tier == "A"


def test_private_metadata_fields_are_type_specific() -> None:
    adapter = V15WorkItemContractAdapter()

    profile = adapter.adapt_upsert(_work_item("profile_summary")).contract.metadata
    assert isinstance(profile, ProfileSummaryMetadata)
    assert profile.work_years == Decimal("12.00")
    assert profile.job_status_norm == "active"
    assert profile.abroad_flag_norm is False
    assert profile.eligible_for_recommendation is True

    employment = adapter.adapt_upsert(_work_item("employment")).contract.metadata
    assert isinstance(employment, EmploymentMetadata)
    assert employment.role_name_raw == "人工智能平台负责人"
    assert employment.start_month == "202301"
    assert employment.end_month is None
    assert employment.is_current is True
    # per-type 模型：employment 不携带 profile/project 私有字段。
    assert not hasattr(employment, "school_names_raw")
    assert not hasattr(employment, "project_name")

    project = adapter.adapt_upsert(_work_item("project")).contract.metadata
    assert isinstance(project, ProjectMetadata)
    assert project.project_name == "企业知识检索验证项目"
    assert project.start_month == "202401"
    assert project.end_month == "202406"
    assert not hasattr(project, "role_name_raw")
    assert not hasattr(project, "is_current")


def test_serial_number_and_structure_keys_do_not_leak_into_metadata() -> None:
    adapter = V15WorkItemContractAdapter()
    contract = adapter.adapt_upsert(_work_item("employment")).contract

    assert contract.serial_number == "DEV900001"
    # 结构保留键被适配器消费，不允许残留在业务 Metadata 里。
    leaked = {"chunk_index", "chunk_total", "updated_at", "source_field",
              "source_hash", "metadata_schema_version"}
    assert not leaked & set(contract.metadata.model_dump())


# ---------------------------------------------------------------------------
# fail-closed：非法输入一律拒绝
# ---------------------------------------------------------------------------


def test_illegal_chunk_type_is_rejected() -> None:
    adapter = V15WorkItemContractAdapter()
    # 模型层：Literal 三值约束直接拒绝。
    dumped = _work_item("employment").model_dump()
    dumped["chunk_type"] = "employment_history"
    with pytest.raises(ValidationError):
        V15VectorizationWorkItem.model_validate(dumped)
    # 适配器层：绕过模型构造（model_copy 不触发校验）时仍 fail-closed。
    bypassed = _work_item("employment").model_copy(
        update={"chunk_type": "education"},
    )
    with pytest.raises(WorkItemContractAdapterError, match="非法 chunk_type"):
        adapter.adapt_upsert(bypassed)  # type: ignore[arg-type]


def test_missing_structure_keys_are_listed_fail_closed() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item(
        "employment",
        metadata_remove=("source_hash", "chunk_total"),
    )

    with pytest.raises(WorkItemContractAdapterError) as exc_info:
        adapter.adapt_upsert(work_item)
    message = str(exc_info.value)
    assert "source_hash" in message
    assert "chunk_total" in message


def test_source_record_id_is_rejected_explicitly() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item(
        "employment",
        metadata_extra={"source_record_id": "21307"},
    )

    with pytest.raises(WorkItemContractAdapterError) as exc_info:
        adapter.adapt_upsert(work_item)
    message = str(exc_info.value)
    assert "source_record_id" in message
    assert "source_field_id" in message


def test_empty_metadata_payload_is_rejected() -> None:
    adapter = V15WorkItemContractAdapter()

    with pytest.raises(WorkItemContractAdapterError, match="metadata 载荷"):
        adapter.adapt_upsert(_work_item(metadata=None))


# ---------------------------------------------------------------------------
# 时间字段与时区
# ---------------------------------------------------------------------------


def test_naive_source_updated_at_fails_closed_without_timezone_config() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item(
        "employment",
        source_updated_at=datetime(2026, 8, 11, 10, 0, 0),  # noqa: DTZ001 -- 故意 naive，触发 fail-closed 分支
    )
    work_item_md = work_item.metadata
    assert isinstance(work_item_md, dict)
    work_item_md.pop("source_updated_at", None)

    with pytest.raises(WorkItemContractAdapterError, match="business_timezone"):
        adapter.adapt_upsert(work_item)


def test_business_timezone_resolves_naive_column_value() -> None:
    adapter = V15WorkItemContractAdapter(
        business_timezone=ZoneInfo("Asia/Shanghai"),
    )
    work_item = _work_item(
        "employment",
        source_updated_at=datetime(2026, 8, 11, 10, 0, 0),  # noqa: DTZ001 -- 故意 naive，触发 fail-closed 分支
    )
    work_item_md = work_item.metadata
    assert isinstance(work_item_md, dict)
    work_item_md.pop("source_updated_at", None)

    contract = adapter.adapt_upsert(work_item).contract

    assert contract.source.source_updated_at is not None
    assert contract.source.source_updated_at.endswith("+08:00")


def test_payload_source_updated_at_takes_precedence_over_column() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item(
        "employment",
        metadata_extra={"source_updated_at": "2026-08-12T08:00:00+00:00"},
    )

    contract = adapter.adapt_upsert(work_item).contract

    assert contract.source.source_updated_at == "2026-08-12T08:00:00+00:00"


# ---------------------------------------------------------------------------
# Eligibility 复用
# ---------------------------------------------------------------------------


def test_eligibility_skip_reason_is_surfaced_for_inactive_upsert() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item("employment", is_active=False)

    result = adapter.adapt_upsert(work_item)

    # is_active=false 是合法 Contract，但 Eligibility 预判应跳过；
    # 是否执行跳过由 Consumer 决策，适配器只上报原因。
    assert result.contract.is_active is False
    assert result.eligibility_skip_reason == "is_active=false"


# ---------------------------------------------------------------------------
# disable 最小输入契约
# ---------------------------------------------------------------------------


def test_disable_minimal_input_yields_single_target_id() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _disable_item()

    result = adapter.adapt(work_item)

    assert isinstance(result, DisableWorkItem)
    assert work_item.is_active is False
    assert result.vector_doc_ids == ("expert:900001:employment:21307:0",)


def test_disable_collects_stale_family_ids_with_dedup_and_order() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _disable_item(
        stale_vector_doc_ids=(
            "expert:900001:employment:21307:1",
            "expert:900001:employment:21307:0",  # 与主 id 重复，应去重
            "expert:900001:employment:21307:2",
        ),
    )

    result = adapter.adapt_disable(work_item)

    assert result.vector_doc_ids == (
        "expert:900001:employment:21307:0",
        "expert:900001:employment:21307:1",
        "expert:900001:employment:21307:2",
    )


def test_disable_rejects_identity_conflict_with_main_vector_doc_id() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _disable_item(expert_id=999999)  # 与 id 公式中的 900001 冲突

    with pytest.raises(WorkItemContractAdapterError, match="expert_id"):
        adapter.adapt_disable(work_item)


def test_disable_rejects_stale_id_from_another_expert_family() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _disable_item(
        stale_vector_doc_ids=("expert:888888:employment:21307:1",),
    )

    with pytest.raises(WorkItemContractAdapterError, match="stale_vector_doc_id"):
        adapter.adapt_disable(work_item)


def test_disable_rejects_non_formula_vector_doc_id() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _disable_item(vector_doc_id="legacy-chunk-000123")

    with pytest.raises(WorkItemContractAdapterError, match="正式公式"):
        adapter.adapt_disable(work_item)


# ---------------------------------------------------------------------------
# metadata_update：本阶段固定 fail-closed
# ---------------------------------------------------------------------------


def test_metadata_update_fails_closed_with_execution_guidance() -> None:
    adapter = V15WorkItemContractAdapter()
    work_item = _work_item(
        "employment",
        operation="metadata_update",
        chunk_text=None,
        content_hash=None,
    )

    with pytest.raises(WorkItemContractAdapterError) as exc_info:
        adapter.adapt(work_item)
    message = str(exc_info.value)
    assert "metadata_update" in message
    assert "chunk_text" in message
    # 明确给出 Consumer 执行期路径指引，而不是静默失败。
    assert "get_by_vector_doc_id" in message

    with pytest.raises(WorkItemContractAdapterError):
        adapter.adapt_metadata_update(work_item)


# ---------------------------------------------------------------------------
# 操作分发
# ---------------------------------------------------------------------------


def test_adapt_dispatches_by_operation() -> None:
    adapter = V15WorkItemContractAdapter()

    upsert_result = adapter.adapt(_work_item("employment"))
    assert isinstance(upsert_result, WorkItemContractAdaptation)

    disable_result = adapter.adapt(_disable_item())
    assert isinstance(disable_result, DisableWorkItem)
