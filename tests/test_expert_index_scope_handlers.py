"""P4.5~P4.7 Scope Handler 测试。

覆盖四类 scope 正常生成、education→profile 映射、父专家删除升级 expert 空 desired、
子行逻辑删除空 desired、N→M 分块只保留当前 desired、输出范围隔离、稳定排序、
校验失败 fail-closed。全部使用内存模型与真实 ChunkFactory，不连接数据库。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.models import (
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
    ScopedCurrentFacts,
    impact_scope_from_target,
)
from expert_index.normalize import normalize_aggregate
from expert_index.scope_handlers import (
    handle_employment_scope,
    handle_expert_scope,
    handle_profile_scope,
    handle_project_scope,
)
from expert_index.validators import ChunkValidationError, validate_chunks

CREATED = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc)
DELETE_TS = datetime(2026, 2, 4, 20, 27, 49, 613000, tzinfo=timezone.utc)

TASK_ID = 4242
BATCH_ID = 987654
DV = "dv-handler-test"


def _settings() -> Settings:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _factory() -> ChunkFactory:
    return ChunkFactory(_settings())


# ---------------------------------------------------------------------------
# 来源行工厂：复用 golden 一致的真实数据
# ---------------------------------------------------------------------------


def _expert(expert_id: int = 880001, *, deleted_at: datetime | None = None) -> ExpertSource:
    return ExpertSource(
        id=expert_id, serial_number="GOLDEN-专家-001", company="ＡＩ未来有限公司",
        job_status=1, job_name="首席执行官", contact_status=3, doubtful_flag=1,
        abroad_flag=2, background="专注知识图谱与检索增强，支持多语言语义分析。",
        work_years=8.0, updated_at=UPDATED, created_at=CREATED, deleted_at=deleted_at,
    )


def _education(edu_id: int, expert_id: int = 880001, *, deleted_at: datetime | None = None,
                school: str = "清华大学", major: str = "计算机") -> EducationSource:
    return EducationSource(
        id=edu_id, expert_id=expert_id, school_name=school, major=major,
        updated_at=UPDATED, created_at=CREATED, deleted_at=deleted_at,
    )


def _employment(emp_id: int = 9201, expert_id: int = 880001, *, deleted_at: datetime | None = None) -> EmploymentSource:
    return EmploymentSource(
        id=emp_id, expert_id=expert_id, company_name="北京京东世纪贸易有限公司",
        position_title="项目经理", start_date=datetime(2020, 2, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 12, 1, tzinfo=timezone.utc),
        work_description="工作职责：负责推荐系统交付与跨团队协作。\n代表性成果：检索准确率提升18%。",
        is_employed=1, updated_at=UPDATED, created_at=CREATED, deleted_at=deleted_at,
    )


def _long_employment(emp_id: int = 9202, expert_id: int = 880001) -> EmploymentSource:
    """职责超长，强制切分多片，用于 N→M 测试。"""

    return EmploymentSource(
        id=emp_id, expert_id=expert_id, company_name="京东", position_title="架构师",
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        work_description="工作职责：" + "负责系统架构设计与核心模块交付。" * 60,
        is_employed=1, updated_at=UPDATED, created_at=CREATED, deleted_at=None,
    )


def _project(proj_id: int = 9301, expert_id: int = 880001, *, deleted_at: datetime | None = None) -> ProjectSource:
    return ProjectSource(
        id=proj_id, expert_id=expert_id, project_title="推荐引擎重构", company_name="京东",
        start_date=datetime(2021, 3, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        work_description="项目目标：提升吞吐。\n关键动作：重写召回层。\n项目结果：QPS提升3倍。",
        updated_at=UPDATED, created_at=CREATED, deleted_at=deleted_at,
    )


def _expert_facts(expert: ExpertSource, employments=(), projects=(), educations=()) -> ScopedCurrentFacts:
    scope = impact_scope_from_target(target_type="expert", expert_id=expert.id)
    return ScopedCurrentFacts(
        scope=scope, expert=expert,
        employments=tuple(employments), projects=tuple(projects), educations=tuple(educations),
    )


def _profile_facts(expert: ExpertSource, educations=()) -> ScopedCurrentFacts:
    scope = impact_scope_from_target(target_type="profile", expert_id=expert.id)
    return ScopedCurrentFacts(scope=scope, expert=expert, educations=tuple(educations))


def _employment_facts(expert: ExpertSource, employment: EmploymentSource) -> ScopedCurrentFacts:
    scope = impact_scope_from_target(
        target_type="employment", expert_id=expert.id, source_field_id=str(employment.id)
    )
    return ScopedCurrentFacts(scope=scope, expert=expert, employments=(employment,))


def _project_facts(expert: ExpertSource, project: ProjectSource) -> ScopedCurrentFacts:
    scope = impact_scope_from_target(
        target_type="project", expert_id=expert.id, source_field_id=str(project.id)
    )
    return ScopedCurrentFacts(scope=scope, expert=expert, projects=(project,))


def _call(handler, facts):
    return handler(facts, _factory(), task_id=TASK_ID, batch_id=BATCH_ID, data_version=DV)


# ===========================================================================
# P4.7 Expert scope
# ===========================================================================


class TestExpertScope:
    def test_active_expert_builds_full_set(self):
        expert = _expert()
        emps = [_employment(), _long_employment()]
        projs = [_project()]
        edus = [_education(1), _education(2, school="北京大学", major="AI")]
        facts = _expert_facts(expert, emps, projs, edus)
        result = _call(handle_expert_scope, facts)
        assert result.impact_scope.scope_type == "expert"
        types = {c.chunk_type for c in result.desired_chunks}
        assert types == {"profile_summary", "employment", "project"}
        # 全部归属同一专家
        assert all(c.expert_id == expert.id for c in result.desired_chunks)

    def test_expert_deleted_returns_expert_empty(self):
        expert = _expert(deleted_at=DELETE_TS)
        # P4.3：已删除专家子表必须为空
        facts = _expert_facts(expert, [], [], [])
        result = _call(handle_expert_scope, facts)
        assert result.impact_scope.scope_type == "expert"
        assert result.is_empty

    def test_expert_scope_output_equals_full_build(self):
        # Handler 输出必须与 normalize_aggregate + build_all 完全一致
        expert = _expert()
        emps = [_employment(), _long_employment()]
        projs = [_project()]
        edus = [_education(1), _education(2, school="北京大学", major="AI")]
        facts = _expert_facts(expert, emps, projs, edus)
        result = _call(handle_expert_scope, facts)
        # 对照：直接调用正式主链路
        aggregate = ExpertAggregate(
            profile=expert, employments=tuple(emps), projects=tuple(projs), educations=tuple(edus),
        )
        profile_stage, emp_stages, proj_stages = normalize_aggregate(
            aggregate, data_version=DV, build_batch_id=BATCH_ID
        )
        expected = _factory().build_all(profile_stage, emp_stages, proj_stages)
        validate_chunks(expected)
        # 逐 Chunk 比较（PreparedChunkSet 会排序，对照也按同样 key 排序）
        key = lambda c: (c.chunk_type, c.source_field_id, c.chunk_index, c.vector_doc_id)
        assert sorted(result.desired_chunks, key=key) == sorted(expected, key=key)

    def test_wrong_scope_type_fail_closed(self):
        facts = _profile_facts(_expert(), [])
        with pytest.raises(ValueError, match="期望范围 expert"):
            _call(handle_expert_scope, facts)


# ===========================================================================
# P4.5 Profile / Education scope
# ===========================================================================


class TestProfileScope:
    def test_active_profile_builds_profile_summary(self):
        facts = _profile_facts(_expert(), [_education(1), _education(2, school="北京大学", major="AI")])
        result = _call(handle_profile_scope, facts)
        assert result.impact_scope.scope_type == "profile"
        assert len(result.desired_chunks) == 1
        assert result.desired_chunks[0].chunk_type == "profile_summary"
        assert result.desired_chunks[0].source_field_id == "880001"

    def test_parent_deleted_upgrades_to_expert_empty(self):
        # P4.3：已删除专家 profile 范围教育必须为空
        facts = _profile_facts(_expert(deleted_at=DELETE_TS), [])
        result = _call(handle_profile_scope, facts)
        assert result.impact_scope.scope_type == "expert"
        assert result.is_empty

    def test_no_background_returns_profile_empty(self):
        # 公司/岗位/背景全空时 build_profile 返回空，合法清空
        empty_expert = ExpertSource(
            id=880001, serial_number=None, company=None, job_status=None, job_name=None,
            contact_status=None, doubtful_flag=None, abroad_flag=None, background=None,
            work_years=None, updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        facts = _profile_facts(empty_expert, [_education(1)])
        result = _call(handle_profile_scope, facts)
        assert result.impact_scope.scope_type == "profile"
        assert result.is_empty

    def test_education_trigger_same_path_as_profile(self):
        # education 触发经 impact_scope_from_target 映射为 profile scope，Handler 同处理
        edu = _education(101)
        scope = impact_scope_from_target(target_type="education", expert_id=880001)
        # education 的 ScopedCurrentFacts：profile 范围 + 教育
        facts = ScopedCurrentFacts(scope=scope, expert=_expert(), educations=(edu,))
        result = _call(handle_profile_scope, facts)
        assert result.impact_scope.scope_type == "profile"
        assert len(result.desired_chunks) == 1

    def test_education_metadata_only_keeps_content_hash(self):
        # 仅教育 Metadata 变化时 content_hash 不变（验证 P4.5 第 4 条规则）
        edu_a = [_education(1, school="清华大学", major="计算机")]
        edu_b = [_education(1, school="清华大学", major="人工智能")]  # 同学校，专业变
        r1 = _call(handle_profile_scope, _profile_facts(_expert(), edu_a))
        r2 = _call(handle_profile_scope, _profile_facts(_expert(), edu_b))
        assert len(r1.desired_chunks) == 1 and len(r2.desired_chunks) == 1
        assert r1.desired_chunks[0].content_hash == r2.desired_chunks[0].content_hash
        # metadata_hash 应变化（专业进入 metadata）
        assert r1.desired_chunks[0].metadata_hash != r2.desired_chunks[0].metadata_hash


# ===========================================================================
# P4.6 Employment / Project scope
# ===========================================================================


class TestEmploymentScope:
    def test_active_employment_builds_single_family(self):
        emp = _employment()
        result = _call(handle_employment_scope, _employment_facts(_expert(), emp))
        assert result.impact_scope.scope_type == "employment"
        assert all(c.chunk_type == "employment" for c in result.desired_chunks)
        assert all(c.source_field_id == "9201" for c in result.desired_chunks)

    def test_company_is_nullable_when_role_or_description_exists(self):
        """公司为空但岗位/职责存在时，Builder 与 Contract 应允许生成 Chunk。"""

        emp = replace(_employment(), company_name=None)
        result = _call(handle_employment_scope, _employment_facts(_expert(), emp))

        assert len(result.desired_chunks) == 1
        chunk = result.desired_chunks[0]
        assert chunk.chunk_type == "employment"
        assert chunk.company_name_raw is None
        assert chunk.role_name_raw == "项目经理"

    def test_parent_deleted_upgrades_to_expert_empty(self):
        emp = _employment()
        facts = _employment_facts(_expert(deleted_at=DELETE_TS), emp)
        result = _call(handle_employment_scope, facts)
        assert result.impact_scope.scope_type == "expert"
        assert result.is_empty

    def test_row_deleted_returns_employment_empty(self):
        emp = _employment(deleted_at=DELETE_TS)
        result = _call(handle_employment_scope, _employment_facts(_expert(), emp))
        assert result.impact_scope.scope_type == "employment"
        assert result.is_empty

    def test_n_to_m_split_keeps_current_only(self):
        # 超长职责强制多分片，desired 只含当前 0..M-1
        emp = _long_employment()
        result = _call(handle_employment_scope, _employment_facts(_expert(), emp))
        indices = sorted(c.chunk_index for c in result.desired_chunks)
        assert indices == list(range(len(indices)))
        assert len(indices) >= 2  # 确实被切分
        # 全部同一 source_field_id
        assert all(c.source_field_id == str(emp.id) for c in result.desired_chunks)

    def test_missing_fields_returns_employment_empty(self):
        # 公司岗位都缺，build_employment 返回空，合法清空
        empty_emp = EmploymentSource(
            id=9299, expert_id=880001, company_name=None, position_title=None,
            start_date=None, end_date=None, work_description=None, is_employed=None,
            updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        result = _call(handle_employment_scope, _employment_facts(_expert(), empty_emp))
        assert result.impact_scope.scope_type == "employment"
        assert result.is_empty

    def test_child_expert_id_mismatch_fail_closed(self):
        emp = EmploymentSource(
            id=9201, expert_id=999999,  # 不属于该专家
            company_name="X", position_title="Y", start_date=None, end_date=None,
            work_description=None, is_employed=None, updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        # scope 用 expert_id=880001 source_field_id=9201，但行 expert_id=999999
        scope = impact_scope_from_target(target_type="employment", expert_id=880001, source_field_id="9201")
        facts = ScopedCurrentFacts(scope=scope, expert=_expert(), employments=(emp,))
        with pytest.raises(ValueError, match="子表行 expert_id"):
            _call(handle_employment_scope, facts)

    def test_child_id_mismatch_fail_closed(self):
        emp = _employment(9201)  # id=9201
        scope = impact_scope_from_target(target_type="employment", expert_id=880001, source_field_id="8888")
        facts = ScopedCurrentFacts(scope=scope, expert=_expert(), employments=(emp,))
        with pytest.raises(ValueError, match="子表行 id"):
            _call(handle_employment_scope, facts)

    def test_output_matches_full_build_family(self):
        # employment Handler 输出必须等于 Full Build 中该 family 的子集
        emp = _long_employment()
        projs = [_project()]
        edus = [_education(1)]
        facts_expert = _expert_facts(_expert(), [emp], projs, edus)
        full = _call(handle_expert_scope, facts_expert)
        single = _call(handle_employment_scope, _employment_facts(_expert(), emp))
        full_family = tuple(
            c for c in full.desired_chunks if c.chunk_type == "employment" and c.source_field_id == str(emp.id)
        )
        key = lambda c: (c.chunk_index, c.vector_doc_id)
        assert sorted(single.desired_chunks, key=key) == sorted(full_family, key=key)


class TestProjectScope:
    def test_active_project_builds_single_family(self):
        proj = _project()
        result = _call(handle_project_scope, _project_facts(_expert(), proj))
        assert result.impact_scope.scope_type == "project"
        assert all(c.chunk_type == "project" for c in result.desired_chunks)
        assert all(c.source_field_id == "9301" for c in result.desired_chunks)

    def test_parent_deleted_upgrades_to_expert_empty(self):
        proj = _project()
        facts = _project_facts(_expert(deleted_at=DELETE_TS), proj)
        result = _call(handle_project_scope, facts)
        assert result.impact_scope.scope_type == "expert"
        assert result.is_empty

    def test_row_deleted_returns_project_empty(self):
        proj = _project(deleted_at=DELETE_TS)
        result = _call(handle_project_scope, _project_facts(_expert(), proj))
        assert result.impact_scope.scope_type == "project"
        assert result.is_empty

    def test_missing_fields_returns_project_empty(self):
        empty_proj = ProjectSource(
            id=9399, expert_id=880001, project_title=None, company_name=None,
            start_date=None, end_date=None, work_description=None,
            updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        result = _call(handle_project_scope, _project_facts(_expert(), empty_proj))
        assert result.impact_scope.scope_type == "project"
        assert result.is_empty


# ===========================================================================
# 输出隔离、稳定排序、fail-closed
# ===========================================================================


class TestIsolationAndStability:
    def test_numeric_serial_number_matches_full_and_incremental_outputs(self):
        """17 位业务编号在全量及两个经历增量范围中必须逐值一致。"""

        expert = replace(_expert(), serial_number="０１２３４５６７８９０１２３４５６")
        employment = _employment()
        project = _project()
        full = _call(
            handle_expert_scope,
            _expert_facts(expert, [employment], [project], [_education(1)]),
        )
        employment_incremental = _call(
            handle_employment_scope,
            _employment_facts(expert, employment),
        )
        project_incremental = _call(
            handle_project_scope,
            _project_facts(expert, project),
        )

        expected = "01234567890123456"
        assert all(chunk.serial_number == expected for chunk in full.desired_chunks)
        assert all(
            chunk.serial_number == expected
            for chunk in employment_incremental.desired_chunks
        )
        assert all(
            chunk.serial_number == expected
            for chunk in project_incremental.desired_chunks
        )

    def test_employment_scope_excludes_other_families(self):
        # scope 只输出触发行；ScopedCurrentFacts employment 范围 shape 只允许单条触发行
        emp_target = _employment(9201)
        result = _call(handle_employment_scope, _employment_facts(_expert(), emp_target))
        assert all(c.source_field_id == "9201" for c in result.desired_chunks)
        assert len(result.desired_chunks) >= 1

    def test_stable_sorted_and_repeatable(self):
        facts = _expert_facts(_expert(), [_employment(), _long_employment()], [_project()], [_education(1)])
        r1 = _call(handle_expert_scope, facts)
        r2 = _call(handle_expert_scope, facts)
        assert r1.desired_chunks == r2.desired_chunks
        # PreparedChunkSet 已按 (chunk_type, source_field_id, chunk_index, vector_doc_id) 排序
        keys = [(c.chunk_type, c.source_field_id, c.chunk_index, c.vector_doc_id) for c in r1.desired_chunks]
        assert keys == sorted(keys)

    def test_validate_failure_propagates_fail_closed(self, monkeypatch):
        # monkeypatch validate_chunks 抛错，Handler 不吞异常
        facts = _expert_facts(_expert(), [_employment()], [_project()], [_education(1)])

        def boom(_chunks):
            raise ChunkValidationError("vector_doc_id=demo", "故意失败")

        import expert_index.scope_handlers as mod
        monkeypatch.setattr(mod, "validate_chunks", boom)
        with pytest.raises(ChunkValidationError):
            _call(handle_expert_scope, facts)


class TestFactoryNotInvokedOnDelete:
    """父专家删除/行逻辑删除时 Factory 零调用（spy ChunkFactory）。"""

    def _spy_factory(self):
        class _SpyFactory:
            def __init__(self):
                self.calls = 0
            def build_all(self, *a, **k):
                self.calls += 1
                raise AssertionError("expert 删除不应调用 build_all")
            def build_profile(self, *a, **k):
                self.calls += 1
                raise AssertionError("profile 删除/空不应调用 build_profile")
            def build_employment(self, *a, **k):
                self.calls += 1
                raise AssertionError("employment 删除不应调用 build_employment")
            def build_project(self, *a, **k):
                self.calls += 1
                raise AssertionError("project 删除不应调用 build_project")
        return _SpyFactory()

    def _call_with(self, handler, facts, factory):
        return handler(facts, factory, task_id=TASK_ID, batch_id=BATCH_ID, data_version=DV)

    def test_expert_deleted_zero_factory_calls(self):
        spy = self._spy_factory()
        facts = _expert_facts(_expert(deleted_at=DELETE_TS), [], [], [])
        result = self._call_with(handle_expert_scope, facts, spy)
        assert result.is_empty and spy.calls == 0

    def test_profile_parent_deleted_zero_factory_calls(self):
        spy = self._spy_factory()
        facts = _profile_facts(_expert(deleted_at=DELETE_TS), [])
        result = self._call_with(handle_profile_scope, facts, spy)
        assert result.impact_scope.scope_type == "expert" and spy.calls == 0

    def test_employment_parent_deleted_zero_factory_calls(self):
        spy = self._spy_factory()
        emp = _employment()
        facts = _employment_facts(_expert(deleted_at=DELETE_TS), emp)
        result = self._call_with(handle_employment_scope, facts, spy)
        assert result.impact_scope.scope_type == "expert" and spy.calls == 0

    def test_employment_row_deleted_zero_factory_calls(self):
        spy = self._spy_factory()
        emp = _employment(deleted_at=DELETE_TS)
        facts = _employment_facts(_expert(), emp)
        result = self._call_with(handle_employment_scope, facts, spy)
        assert result.impact_scope.scope_type == "employment" and spy.calls == 0

    def test_project_parent_deleted_zero_factory_calls(self):
        spy = self._spy_factory()
        proj = _project()
        facts = _project_facts(_expert(deleted_at=DELETE_TS), proj)
        result = self._call_with(handle_project_scope, facts, spy)
        assert result.impact_scope.scope_type == "expert" and spy.calls == 0

    def test_project_row_deleted_zero_factory_calls(self):
        spy = self._spy_factory()
        proj = _project(deleted_at=DELETE_TS)
        facts = _project_facts(_expert(), proj)
        result = self._call_with(handle_project_scope, facts, spy)
        assert result.impact_scope.scope_type == "project" and spy.calls == 0


class TestOwnershipValidation:
    """聚合归属校验：子表行必须归属当前 scope 专家。"""

    def test_expert_scope_mismatched_employment_owner_fail_closed(self):
        emp = EmploymentSource(
            id=9201, expert_id=999999, company_name="X", position_title="Y",
            start_date=None, end_date=None, work_description=None, is_employed=None,
            updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        facts = _expert_facts(_expert(), [emp], [], [])
        with pytest.raises(ValueError, match="工作行 expert_id"):
            _call(handle_expert_scope, facts)

    def test_expert_scope_mismatched_project_owner_fail_closed(self):
        proj = ProjectSource(
            id=9301, expert_id=999999, project_title="X", company_name=None,
            start_date=None, end_date=None, work_description=None,
            updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        facts = _expert_facts(_expert(), [], [proj], [])
        with pytest.raises(ValueError, match="项目行 expert_id"):
            _call(handle_expert_scope, facts)

    def test_expert_scope_mismatched_education_owner_fail_closed(self):
        edu = _education(1, expert_id=999999)
        facts = _expert_facts(_expert(), [], [], [edu])
        with pytest.raises(ValueError, match="教育行 expert_id"):
            _call(handle_expert_scope, facts)

    def test_profile_scope_mismatched_education_owner_fail_closed(self):
        edu = _education(1, expert_id=999999)
        facts = _profile_facts(_expert(), [edu])
        with pytest.raises(ValueError, match="教育行 expert_id"):
            _call(handle_profile_scope, facts)

    def test_project_child_expert_id_mismatch_fail_closed(self):
        proj = ProjectSource(
            id=9301, expert_id=999999, project_title="X", company_name=None,
            start_date=None, end_date=None, work_description=None,
            updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        scope = impact_scope_from_target(target_type="project", expert_id=880001, source_field_id="9301")
        facts = ScopedCurrentFacts(scope=scope, expert=_expert(), projects=(proj,))
        with pytest.raises(ValueError, match="子表行 expert_id"):
            _call(handle_project_scope, facts)

    def test_project_child_id_mismatch_fail_closed(self):
        proj = _project(9301)
        scope = impact_scope_from_target(target_type="project", expert_id=880001, source_field_id="8888")
        facts = ScopedCurrentFacts(scope=scope, expert=_expert(), projects=(proj,))
        with pytest.raises(ValueError, match="子表行 id"):
            _call(handle_project_scope, facts)

    def test_employment_parent_deleted_still_validates_child_identity(self):
        # 父删除不应掩盖子行身份错误：身份错误必须先 fail-closed
        emp = EmploymentSource(
            id=9201, expert_id=999999, company_name="X", position_title="Y",
            start_date=None, end_date=None, work_description=None, is_employed=None,
            updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        scope = impact_scope_from_target(target_type="employment", expert_id=880001, source_field_id="9201")
        facts = ScopedCurrentFacts(scope=scope, expert=_expert(deleted_at=DELETE_TS), employments=(emp,))
        with pytest.raises(ValueError, match="子表行 expert_id"):
            _call(handle_employment_scope, facts)

    def test_project_parent_deleted_still_validates_child_identity(self):
        proj = ProjectSource(
            id=9301, expert_id=999999, project_title="X", company_name=None,
            start_date=None, end_date=None, work_description=None,
            updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        scope = impact_scope_from_target(target_type="project", expert_id=880001, source_field_id="9301")
        facts = ScopedCurrentFacts(scope=scope, expert=_expert(deleted_at=DELETE_TS), projects=(proj,))
        with pytest.raises(ValueError, match="子表行 expert_id"):
            _call(handle_project_scope, facts)


class TestProjectFamilyParity:
    def test_project_output_matches_full_build_family(self):
        proj = _project()
        emps = [_employment()]
        edus = [_education(1)]
        facts_expert = _expert_facts(_expert(), emps, [proj], edus)
        full = _call(handle_expert_scope, facts_expert)
        single = _call(handle_project_scope, _project_facts(_expert(), proj))
        full_family = tuple(
            c for c in full.desired_chunks if c.chunk_type == "project" and c.source_field_id == str(proj.id)
        )
        key = lambda c: (c.chunk_index, c.vector_doc_id)
        assert sorted(single.desired_chunks, key=key) == sorted(full_family, key=key)

    def test_project_n_to_m_split_keeps_current_only(self):
        # 超长项目强制多分片
        long_proj = ProjectSource(
            id=9302, expert_id=880001, project_title="大规模重构", company_name="京东",
            start_date=datetime(2021, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            work_description="项目目标：" + "完成核心系统重构并提升整体性能指标。" * 60,
            updated_at=UPDATED, created_at=CREATED, deleted_at=None,
        )
        result = _call(handle_project_scope, _project_facts(_expert(), long_proj))
        indices = sorted(c.chunk_index for c in result.desired_chunks)
        assert indices == list(range(len(indices)))
        assert len(indices) >= 2
        assert all(c.source_field_id == str(long_proj.id) for c in result.desired_chunks)


# ===========================================================================
# 结构断言
# ===========================================================================


class TestStructure:
    def test_no_source_record_id_in_handlers(self):
        import expert_index.scope_handlers as mod
        with open(mod.__file__, encoding="utf-8") as fh:
            source = fh.read()
        assert "source_record_id" not in source

    def test_four_handlers_exported(self):
        import expert_index.scope_handlers as mod
        for name in ("handle_expert_scope", "handle_profile_scope",
                     "handle_employment_scope", "handle_project_scope"):
            assert hasattr(mod, name)
