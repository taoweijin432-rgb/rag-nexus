"""P4.4 单对象 Normalize 入口测试。

验证 normalize_profile/normalize_employment/normalize_project 三个公开入口的
单对象语义，以及 normalize_aggregate 组合三者后与改造前完全一致。所有断言
基于确定性清洗规则，不连接真实数据库。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from expert_index.models import (
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
)
from expert_index.normalize import (
    clean_text,
    normalize_aggregate,
    normalize_employment,
    normalize_profile,
    normalize_project,
    normalize_serial_number,
)

# ---------------------------------------------------------------------------
# 来源行工厂
# ---------------------------------------------------------------------------


def _profile(expert_id: int = 880001) -> ExpertSource:
    return ExpertSource(
        id=expert_id,
        serial_number="S-001",
        company="ＡＩ未来有限公司",
        job_status=1,
        job_name="首席执行官",
        contact_status=3,
        doubtful_flag=1,
        abroad_flag=2,
        background="专注知识图谱。",
        work_years=8.0,
        updated_at=datetime(2026, 1, 2, 3, 4, 5, 123456),  # noqa: DTZ001
        created_at=datetime(2026, 1, 2, 3, 4, 5, 123456),  # noqa: DTZ001
        deleted_at=None,
    )


def _education(edu_id: int, expert_id: int = 880001, *,
                school: str = "清华大学", major: str = "计算机",
                updated_at: datetime | None = None) -> EducationSource:
    return EducationSource(
        id=edu_id, expert_id=expert_id, school_name=school, major=major,
        updated_at=updated_at, created_at=datetime(2020, 9, 1),  # noqa: DTZ001
        deleted_at=None,
    )


def _employment(emp_id: int = 9201, expert_id: int = 880001) -> EmploymentSource:
    return EmploymentSource(
        id=emp_id, expert_id=expert_id, company_name="北京京东世纪贸易有限公司",
        position_title="项目经理", start_date=datetime(2020, 2, 1),  # noqa: DTZ001
        end_date=datetime(2025, 12, 1),  # noqa: DTZ001
        work_description="工作职责：负责推荐系统交付。\n代表性成果：准确率提升18%。",
        is_employed=1,
        updated_at=datetime(2026, 1, 3, 4, 5, 6, 111222),  # noqa: DTZ001
        created_at=datetime(2020, 2, 1),  # noqa: DTZ001
        deleted_at=None,
    )


def _project(proj_id: int = 9301, expert_id: int = 880001) -> ProjectSource:
    return ProjectSource(
        id=proj_id, expert_id=expert_id, project_title="推荐引擎重构",
        company_name="京东", start_date=datetime(2021, 3, 1),  # noqa: DTZ001
        end_date=datetime(2024, 6, 1),  # noqa: DTZ001
        work_description="项目目标：提升吞吐。\n关键动作：重写召回层。\n项目结果：QPS提升3倍。",
        updated_at=datetime(2026, 1, 4, 5, 6, 7, 333444),  # noqa: DTZ001
        created_at=datetime(2021, 3, 1),  # noqa: DTZ001
        deleted_at=None,
    )


DV = "dv-test"
BB = 987654


# ===========================================================================
# serial_number：业务标识符规范化
# ===========================================================================


class TestNormalizeSerialNumber:
    def test_numeric_identifier_is_not_removed_as_pii(self):
        raw = " ０１２３４５６７８９０１２３４５６ "

        # 相同数字作为正文会命中银行卡 PII；作为业务编号必须保留且统一为半角。
        assert clean_text(raw) == ""
        assert normalize_serial_number(raw) == "01234567890123456"

    @pytest.mark.parametrize("value", [None, "", " null ", "--", "/"])
    def test_empty_value_returns_none(self, value):
        assert normalize_serial_number(value) is None


# ===========================================================================
# normalize_profile：教育排序与去重、来源时间
# ===========================================================================


class TestNormalizeProfile:
    def test_basic_profile_stage_fields(self):
        stage = normalize_profile(
            _profile(), [_education(1), _education(2, school="清华大学", major="人工智能")],
            data_version=DV, build_batch_id=BB,
        )
        assert stage.expert_id == 880001
        assert stage.company_name_raw is not None
        assert stage.role_name_raw == "首席执行官"
        assert stage.background_clean == "专注知识图谱。"
        # 教育按 id ASC 顺序去重，两所不同学校都保留
        assert stage.school_names_raw == ("清华大学",)  # id=1 和 id=2 默认学校同为清华，去重为 1 条
        assert stage.major_names_raw == ("计算机", "人工智能")
        assert stage.data_version == DV
        assert stage.build_batch_id == BB

    def test_profile_uses_business_identifier_normalization(self):
        profile = replace(_profile(), serial_number="０１２３４５６７８９０１２３４５６")

        stage = normalize_profile(profile, [], data_version=DV, build_batch_id=BB)

        assert stage.serial_number == "01234567890123456"

    def test_education_id_asc_ordering_deterministic(self):
        # 乱序传入，应按 id ASC 重新排序；三所不同学校全部保留
        unordered = [
            _education(30, school="学校C"),
            _education(10, school="学校A"),
            _education(20, school="学校B"),
        ]
        stage = normalize_profile(_profile(), unordered, data_version=DV, build_batch_id=BB)
        # school_names_raw 按排序后顺序：id=10 -> id=20 -> id=30
        assert stage.school_names_raw == ("学校A", "学校B", "学校C")

    def test_education_dedupe_keeps_first_occurrence(self):
        # 两条相同学校/专业，按 id ASC 排序后保留首次出现
        stage = normalize_profile(
            _profile(), [_education(5, school="清华", major="CS"), _education(3, school="清华", major="CS")],
            data_version=DV, build_batch_id=BB,
        )
        assert stage.school_names_raw == ("清华",)
        assert stage.major_names_raw == ("CS",)

    def test_source_updated_at_takes_latest(self):
        latest = datetime(2026, 1, 10, 0, 0, 0, 1)  # noqa: DTZ001
        stage = normalize_profile(
            _profile(),
            [_education(1, updated_at=datetime(2026, 1, 5)),  # noqa: DTZ001
             _education(2, updated_at=latest)],
            data_version=DV, build_batch_id=BB,
        )
        assert stage.source_updated_at == latest

    def test_empty_educations_valid(self):
        stage = normalize_profile(_profile(), [], data_version=DV, build_batch_id=BB)
        assert stage.school_names_raw == ()
        assert stage.major_names_raw == ()

    def test_source_hash_stable_for_same_input(self):
        edu = [_education(1)]
        a = normalize_profile(_profile(), edu, data_version=DV, build_batch_id=BB)
        b = normalize_profile(_profile(), edu, data_version=DV, build_batch_id=BB)
        assert a.source_hash == b.source_hash


# ===========================================================================
# normalize_employment：单条事实
# ===========================================================================


class TestNormalizeEmployment:
    def test_single_row_stage(self):
        stage = normalize_employment(_employment(), data_version=DV, build_batch_id=BB)
        assert stage.expert_id == 880001
        assert stage.source_field_id == "9201"
        assert stage.source_table == "experts_work_experience"
        assert stage.company_name_raw is not None
        assert stage.role_name_raw == "项目经理"
        assert stage.start_month == "202002"
        assert stage.is_current is True  # is_employed=1 且有 end_date 但 is_current 强制 end_month=None
        assert stage.end_month is None  # 在职强制清空
        assert stage.responsibilities_clean == "负责推荐系统交付。"
        assert stage.achievements_clean == "准确率提升18%。"
        assert stage.source_hash  # 非空 sha256
        assert stage.data_version == DV

    def test_former_employment_keeps_end_month(self):
        # is_employed=2 不在职，保留 end_month
        former = EmploymentSource(
            id=9202, expert_id=880001, company_name="X", position_title="Y",
            start_date=datetime(2018, 1, 1),  # noqa: DTZ001
            end_date=datetime(2022, 6, 1),  # noqa: DTZ001
            work_description="工作职责：开发。", is_employed=2,
            updated_at=datetime(2026, 1, 1),  # noqa: DTZ001
            created_at=datetime(2018, 1, 1),  # noqa: DTZ001
            deleted_at=None,
        )
        stage = normalize_employment(former, data_version=DV, build_batch_id=BB)
        assert stage.is_current is False
        assert stage.end_month == "202206"


# ===========================================================================
# normalize_project：单条事实
# ===========================================================================


class TestNormalizeProject:
    def test_single_row_stage(self):
        stage = normalize_project(_project(), data_version=DV, build_batch_id=BB)
        assert stage.expert_id == 880001
        assert stage.source_field_id == "9301"
        assert stage.source_table == "experts_project_experience"
        assert stage.project_name == "推荐引擎重构"
        assert stage.start_month == "202103"
        assert stage.end_month == "202406"
        assert stage.objective_clean == "提升吞吐。"
        assert stage.actions_clean == "重写召回层。"
        assert stage.results_clean == "QPS提升3倍。"
        assert stage.source_hash
        assert stage.data_version == DV


# ===========================================================================
# normalize_aggregate 组合一致性
# ===========================================================================


class TestAggregateComposition:
    def test_aggregate_equals_three_public_functions(self):
        profile = _profile()
        employments = [_employment(), _employment(9202)]
        projects = [_project(), _project(9302)]
        educations = [_education(1), _education(2)]
        aggregate = ExpertAggregate(
            profile=profile,
            employments=tuple(employments),
            projects=tuple(projects),
            educations=tuple(educations),
        )
        agg_profile, agg_emps, agg_projs = normalize_aggregate(
            aggregate, data_version=DV, build_batch_id=BB,
        )

        # 与三个公开入口单独调用结果逐字段一致
        exp_profile = normalize_profile(profile, educations, data_version=DV, build_batch_id=BB)
        assert agg_profile == exp_profile
        assert [normalize_employment(e, data_version=DV, build_batch_id=BB) for e in employments] == agg_emps
        assert [normalize_project(p, data_version=DV, build_batch_id=BB) for p in projects] == agg_projs

    def test_aggregate_empty_children(self):
        aggregate = ExpertAggregate(
            profile=_profile(), employments=(), projects=(), educations=(),
        )
        profile_stage, emps, projs = normalize_aggregate(
            aggregate, data_version=DV, build_batch_id=BB,
        )
        assert profile_stage.expert_id == 880001
        assert emps == []
        assert projs == []


# ===========================================================================
# 结构性断言：无私有函数残留、无 source_record_id
# ===========================================================================


class TestStructure:
    def test_no_private_normalize_definitions(self):
        import expert_index.normalize as mod
        assert not hasattr(mod, "_normalize_employment")
        assert not hasattr(mod, "_normalize_project")
        assert hasattr(mod, "normalize_profile")
        assert hasattr(mod, "normalize_employment")
        assert hasattr(mod, "normalize_project")

    def test_no_source_record_id_in_normalize_module(self):
        import expert_index.normalize as mod
        with open(mod.__file__, encoding="utf-8") as fh:
            source = fh.read()
        assert "source_record_id" not in source
