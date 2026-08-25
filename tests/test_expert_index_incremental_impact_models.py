"""P4.2 增量影响范围与内存期望 Chunk 集模型的最小契约测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from expert_index.models import (
    INCREMENTAL_IMPACT_CHUNK_TYPES,
    INCREMENTAL_IMPACT_SCOPE_TYPES,
    ChunkRecord,
    IncrementalImpactScope,
    PreparedChunkSet,
    impact_scope_from_target,
)

# ---------------------------------------------------------------------------
# 测试用 ChunkRecord 构造工具：只填足 scope 校验与排序所需的字段，其余用占位。
# ---------------------------------------------------------------------------

_BASE_CHUNK_FIELDS: dict[str, object] = {
    "serial_number": "S0001",
    "chunk_subtype": None,
    "chunk_total": 1,
    "chunk_text": "示例正文",
    "char_count": 4,
    "source_field": "company+job_name+background",
    "source_version": "2026-08-24T00:00:00+00:00",
    "source_updated_at": None,
    "source_hash": "a" * 64,
    "content_hash": "b" * 64,
    "metadata_hash": "c" * 64,
    "company_name_raw": None,
    "company_name_norm": None,
    "role_name_raw": None,
    "role_code": None,
    "role_level": None,
    "job_status_norm": None,
    "abroad_flag_norm": None,
    "work_years": None,
    "project_name": None,
    "start_month": None,
    "end_month": None,
    "is_current": None,
    "industry_codes": (),
    "function_codes": (),
    "topic_codes": (),
    "school_names_raw": (),
    "major_names_raw": (),
    "eligible_for_recommendation": None,
    "origin_type": "source_parsed",
    "verification_status": "verified",
    "weight_tier": "A",
    "chunk_policy_version": "1.3",
    "metadata_schema_version": "4.0",
    "embedding_model_alias": None,
    "embedding_model_version": None,
    "indexed_at": None,
    "is_active": True,
    "build_batch_id": 8,
}


def _chunk(
    *,
    expert_id: int = 100,
    chunk_type: str = "employment",
    source_table: str = "experts_work_experience",
    source_field_id: str = "5000",
    chunk_index: int = 0,
    vector_doc_id: str | None = None,
) -> ChunkRecord:
    """构造满足 PreparedChunkSet 范围校验的最小 ChunkRecord。"""

    if vector_doc_id is None:
        vector_doc_id = f"expert:{expert_id}:{chunk_type}:{source_field_id}:{chunk_index}"
    return ChunkRecord(
        expert_id=expert_id,
        chunk_type=chunk_type,
        chunk_index=chunk_index,
        vector_doc_id=vector_doc_id,
        source_table=source_table,
        source_field_id=source_field_id,
        **_BASE_CHUNK_FIELDS,  # type: ignore[arg-type]
    )


def _scope(
    scope_type: str = "employment",
    *,
    expert_id: int = 100,
    source_field_id: str | None = "5000",
) -> IncrementalImpactScope:
    """构造测试用影响范围。"""

    if scope_type == "expert":
        return IncrementalImpactScope(
            scope_type="expert", expert_id=expert_id,
            chunk_type=None, source_table=None, source_field_id=None,
        )
    if scope_type == "profile":
        return IncrementalImpactScope(
            scope_type="profile", expert_id=expert_id,
            chunk_type="profile_summary",
            source_table="experts+experts_education_experience",
            source_field_id=str(expert_id),
        )
    # employment / project
    source_table = (
        "experts_work_experience" if scope_type == "employment"
        else "experts_project_experience"
    )
    assert source_field_id is not None
    return IncrementalImpactScope(
        scope_type=scope_type, expert_id=expert_id,
        chunk_type=scope_type,
        source_table=source_table,
        source_field_id=source_field_id,
    )


# ---------------------------------------------------------------------------
# IncrementalImpactScope 构造与校验
# ---------------------------------------------------------------------------


class TestIncrementalImpactScope:
    def test_four_scope_types_are_supported(self) -> None:
        assert INCREMENTAL_IMPACT_SCOPE_TYPES == frozenset(
            {"expert", "profile", "employment", "project"}
        )

    def test_expert_scope_spans_three_chunk_types(self) -> None:
        scope = IncrementalImpactScope(
            scope_type="expert", expert_id=100,
            chunk_type=None, source_table=None, source_field_id=None,
        )
        assert scope.scope_type == "expert"
        assert scope.chunk_type is None

    def test_profile_scope_canonical_source_identity(self) -> None:
        scope = IncrementalImpactScope(
            scope_type="profile", expert_id=100,
            chunk_type="profile_summary",
            source_table="experts+experts_education_experience",
            source_field_id="100",
        )
        assert scope.source_field_id == str(scope.expert_id)

    def test_employment_scope_requires_source_field_id(self) -> None:
        scope = IncrementalImpactScope(
            scope_type="employment", expert_id=100,
            chunk_type="employment",
            source_table="experts_work_experience",
            source_field_id="5000",
        )
        assert scope.source_field_id == "5000"

    def test_project_scope_requires_source_field_id(self) -> None:
        scope = IncrementalImpactScope(
            scope_type="project", expert_id=100,
            chunk_type="project",
            source_table="experts_project_experience",
            source_field_id="6000",
        )
        assert scope.source_field_id == "6000"

    def test_expert_scope_with_any_scope_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="expert 范围必须跨三类"):
            IncrementalImpactScope(
                scope_type="expert", expert_id=100,
                chunk_type="employment", source_table=None, source_field_id=None,
            )

    def test_unknown_scope_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="不支持的影响范围类型"):
            IncrementalImpactScope(
                scope_type="education", expert_id=100,
                chunk_type="profile_summary", source_table="x", source_field_id="100",
            )

    def test_profile_wrong_chunk_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="profile 范围的 chunk_type 必须为 profile_summary"):
            IncrementalImpactScope(
                scope_type="profile", expert_id=100,
                chunk_type="employment",
                source_table="experts+experts_education_experience",
                source_field_id="100",
            )

    def test_profile_wrong_source_field_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="profile 范围的 source_field_id 必须等于"):
            IncrementalImpactScope(
                scope_type="profile", expert_id=100,
                chunk_type="profile_summary",
                source_table="experts+experts_education_experience",
                source_field_id="999",
            )

    def test_employment_wrong_source_table_rejected(self) -> None:
        with pytest.raises(ValueError, match="employment 范围的 source_table 必须为"):
            IncrementalImpactScope(
                scope_type="employment", expert_id=100,
                chunk_type="employment",
                source_table="experts_project_experience",
                source_field_id="5000",
            )

    def test_non_positive_expert_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="expert_id 必须是正整数"):
            IncrementalImpactScope(
                scope_type="employment", expert_id=0,
                chunk_type="employment",
                source_table="experts_work_experience",
                source_field_id="5000",
            )

    def test_empty_source_field_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_field_id 必须是非空字符串"):
            IncrementalImpactScope(
                scope_type="employment", expert_id=100,
                chunk_type="employment",
                source_table="experts_work_experience",
                source_field_id="",
            )

    def test_whitespace_source_field_id_rejected(self) -> None:
        # 含首尾空白的身份串会在后续范围精确匹配时失败，fail-closed 拒绝而非静默接受。
        with pytest.raises(ValueError, match="不得含首尾空白"):
            IncrementalImpactScope(
                scope_type="employment", expert_id=100,
                chunk_type="employment",
                source_table="experts_work_experience",
                source_field_id=" 5000 ",
            )


# ---------------------------------------------------------------------------
# impact_scope_from_target 映射
# ---------------------------------------------------------------------------


class TestImpactScopeFromTarget:
    def test_education_target_maps_to_profile_scope(self) -> None:
        scope = impact_scope_from_target(target_type="education", expert_id=100)
        assert scope.scope_type == "profile"
        assert scope.chunk_type == "profile_summary"
        assert scope.source_field_id == "100"
        assert scope.source_table == "experts+experts_education_experience"

    def test_expert_target_maps_to_expert_scope(self) -> None:
        scope = impact_scope_from_target(target_type="expert", expert_id=100)
        assert scope.scope_type == "expert"
        assert scope.chunk_type is None

    def test_expert_target_rejects_source_field_id(self) -> None:
        with pytest.raises(ValueError, match="expert 范围不接受 source_field_id"):
            impact_scope_from_target(target_type="expert", expert_id=100, source_field_id="5000")

    def test_profile_target_rejects_source_field_id(self) -> None:
        with pytest.raises(ValueError, match="profile 范围不接受 source_field_id"):
            impact_scope_from_target(target_type="profile", expert_id=100, source_field_id="100")

    def test_education_target_rejects_source_field_id(self) -> None:
        with pytest.raises(ValueError, match="education 范围不接受 source_field_id"):
            impact_scope_from_target(target_type="education", expert_id=100, source_field_id="100")

    @pytest.mark.parametrize("target", ["employment", "project"])
    def test_family_target_rejects_none_source_field_id(self, target: str) -> None:
        with pytest.raises(ValueError, match="必须提供非空白 source_field_id"):
            impact_scope_from_target(target_type=target, expert_id=100)

    @pytest.mark.parametrize("target", ["employment", "project"])
    def test_family_target_rejects_empty_source_field_id(self, target: str) -> None:
        with pytest.raises(ValueError, match="必须提供非空白 source_field_id"):
            impact_scope_from_target(target_type=target, expert_id=100, source_field_id="")

    @pytest.mark.parametrize("target", ["employment", "project"])
    def test_family_target_rejects_whitespace_source_field_id(self, target: str) -> None:
        with pytest.raises(ValueError, match="不得含首尾空白"):
            impact_scope_from_target(target_type=target, expert_id=100, source_field_id=" 5000 ")

    def test_unknown_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="不支持的同步任务目标类型"):
            impact_scope_from_target(target_type="unknown", expert_id=100)


# ---------------------------------------------------------------------------
# PreparedChunkSet 范围一致性、去重、排序与空集合
# ---------------------------------------------------------------------------


class TestPreparedChunkSetScopeContainment:
    def test_expert_scope_accepts_mixed_chunk_types(self) -> None:
        scope = _scope("expert", expert_id=100)
        chunks = (
            _chunk(expert_id=100, chunk_type="profile_summary",
                   source_table="experts+experts_education_experience", source_field_id="100"),
            _chunk(expert_id=100, chunk_type="employment",
                   source_table="experts_work_experience", source_field_id="5000"),
            _chunk(expert_id=100, chunk_type="project",
                   source_table="experts_project_experience", source_field_id="6000"),
        )
        prepared = PreparedChunkSet(
            task_id=1, batch_id=8, data_version="20260824_v15",
            impact_scope=scope, desired_chunks=chunks,
        )
        assert len(prepared.desired_chunks) == 3

    def test_cross_expert_chunk_rejected(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        with pytest.raises(ValueError, match="不属于范围 expert_id"):
            PreparedChunkSet(
                task_id=1, batch_id=8, data_version="v",
                impact_scope=scope,
                desired_chunks=(_chunk(expert_id=999, source_field_id="5000"),),
            )

    def test_out_of_scope_chunk_type_rejected(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        with pytest.raises(ValueError, match="employment 范围不允许 chunk_type"):
            PreparedChunkSet(
                task_id=1, batch_id=8, data_version="v",
                impact_scope=scope,
                desired_chunks=(_chunk(chunk_type="project", source_table="experts_project_experience"),),
            )

    def test_out_of_scope_source_field_id_rejected(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        with pytest.raises(ValueError, match="source_field_id=.*与范围不一致"):
            PreparedChunkSet(
                task_id=1, batch_id=8, data_version="v",
                impact_scope=scope,
                desired_chunks=(_chunk(source_field_id="5001"),),
            )


class TestPreparedChunkSetDedupAndSort:
    def test_identical_duplicate_collapses(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        chunk = _chunk()
        prepared = PreparedChunkSet(
            task_id=1, batch_id=8, data_version="v",
            impact_scope=scope, desired_chunks=(chunk, chunk),
        )
        assert len(prepared.desired_chunks) == 1

    def test_conflicting_duplicate_rejected(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        c1 = _chunk()
        c2 = replace(c1, chunk_text="不同正文", char_count=4, content_hash="d" * 64)
        with pytest.raises(ValueError, match="存在不一致的重复 Chunk"):
            PreparedChunkSet(
                task_id=1, batch_id=8, data_version="v",
                impact_scope=scope, desired_chunks=(c1, c2),
            )

    def test_deterministic_sort_by_type_then_field_then_index(self) -> None:
        scope = _scope("expert", expert_id=100)
        # 故意乱序：project 在前、employment 多分片在后、profile 居中。
        chunks = (
            _chunk(expert_id=100, chunk_type="project",
                   source_table="experts_project_experience", source_field_id="6000", chunk_index=0),
            _chunk(expert_id=100, chunk_type="employment",
                   source_table="experts_work_experience", source_field_id="5000", chunk_index=1),
            _chunk(expert_id=100, chunk_type="employment",
                   source_table="experts_work_experience", source_field_id="5000", chunk_index=0),
            _chunk(expert_id=100, chunk_type="profile_summary",
                   source_table="experts+experts_education_experience", source_field_id="100", chunk_index=0),
        )
        prepared = PreparedChunkSet(
            task_id=1, batch_id=8, data_version="v",
            impact_scope=scope, desired_chunks=chunks,
        )
        types = [c.chunk_type for c in prepared.desired_chunks]
        ids = [c.source_field_id for c in prepared.desired_chunks]
        idxs = [c.chunk_index for c in prepared.desired_chunks]
        # 排序键 (chunk_type, source_field_id, chunk_index, vector_doc_id)
        assert types == ["employment", "employment", "profile_summary", "project"]
        assert ids == ["5000", "5000", "100", "6000"]
        assert idxs == [0, 1, 0, 0]

    def test_desired_vector_doc_ids_stable_order(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        c0 = _chunk(chunk_index=0)
        c1 = _chunk(chunk_index=1)
        prepared = PreparedChunkSet(
            task_id=1, batch_id=8, data_version="v",
            impact_scope=scope, desired_chunks=(c1, c0),
        )
        assert prepared.desired_vector_doc_ids == (
            "expert:100:employment:5000:0",
            "expert:100:employment:5000:1",
        )


class TestPreparedChunkSetEmptyAndValidation:
    def test_empty_desired_set_is_valid_clear_semantics(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        prepared = PreparedChunkSet(
            task_id=1, batch_id=8, data_version="v",
            impact_scope=scope, desired_chunks=(),
        )
        assert prepared.is_empty is True
        assert prepared.desired_vector_doc_ids == ()

    def test_empty_expert_scope_preserves_scope(self) -> None:
        scope = _scope("expert", expert_id=100)
        prepared = PreparedChunkSet(
            task_id=1, batch_id=8, data_version="v",
            impact_scope=scope, desired_chunks=(),
        )
        assert prepared.is_empty is True
        assert prepared.impact_scope.scope_type == "expert"

    def test_non_positive_task_id_rejected(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        with pytest.raises(ValueError, match="task_id 必须是正整数"):
            PreparedChunkSet(
                task_id=0, batch_id=8, data_version="v",
                impact_scope=scope, desired_chunks=(),
            )

    def test_blank_data_version_rejected(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        with pytest.raises(ValueError, match="data_version 必须是非空字符串"):
            PreparedChunkSet(
                task_id=1, batch_id=8, data_version="",
                impact_scope=scope, desired_chunks=(),
            )

    def test_list_input_converted_to_immutable_tuple(self) -> None:
        scope = _scope("employment", expert_id=100, source_field_id="5000")
        prepared = PreparedChunkSet(
            task_id=1, batch_id=8, data_version="v",
            impact_scope=scope, desired_chunks=[_chunk()],  # type: ignore[arg-type]
        )
        assert isinstance(prepared.desired_chunks, tuple)
        assert len(prepared.desired_chunks) == 1


# ---------------------------------------------------------------------------
# source_record_id 禁止项与冻结常量
# ---------------------------------------------------------------------------


class TestSourceFieldIdContract:
    def test_source_record_id_absent_from_scope_fields(self) -> None:
        field_names = {f.name for f in __import__("dataclasses").fields(IncrementalImpactScope)}
        assert "source_record_id" not in field_names
        assert "source_field_id" in field_names

    def test_source_record_id_absent_from_prepared_fields(self) -> None:
        field_names = {f.name for f in __import__("dataclasses").fields(PreparedChunkSet)}
        assert "source_record_id" not in field_names

    def test_source_record_id_passed_via_kwargs_to_scope_raises_type_error(self) -> None:
        # frozen dataclass 无该字段，传入必须 fail-closed（TypeError），不得静默忽略。
        with pytest.raises(TypeError):
            IncrementalImpactScope(
                scope_type="employment",
                expert_id=100,
                chunk_type="employment",
                source_table="experts_work_experience",
                source_field_id="5000",
                source_record_id="5000",  # type: ignore[call-arg]
            )

    def test_impact_chunk_types_match_three_types(self) -> None:
        assert INCREMENTAL_IMPACT_CHUNK_TYPES == frozenset(
            {"profile_summary", "employment", "project"}
        )
