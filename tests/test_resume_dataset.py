"""DatasetSpec 的冻结映射和 Expert 默认兼容性测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.dataset import EXPERT_SPEC
from expert_index.models import (
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
)
from expert_index.normalize import normalize_aggregate
from resume_index.dataset import RESUME_SPEC


def _settings() -> Settings:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _aggregate(person_id: int = 70001) -> ExpertAggregate:
    """构造不含真实个人信息的公共 Source 快照。"""

    when = datetime(2026, 9, 2, 10, 0, 0)  # noqa: DTZ001
    return ExpertAggregate(
        profile=ExpertSource(
            id=person_id,
            serial_number="０１２３４５６７８９０１２３４５６",
            company="示例科技有限公司",
            job_status=1,
            job_name="项目经理",
            contact_status=3,
            doubtful_flag=1,
            abroad_flag=2,
            background="负责企业知识库建设。",
            work_years=6.0,
            updated_at=when,
            created_at=when,
            deleted_at=None,
        ),
        employments=(
            EmploymentSource(
                id=81001,
                expert_id=person_id,
                company_name="示例科技有限公司",
                position_title="项目经理",
                start_date=when,
                end_date=None,
                work_description="工作职责：负责知识库建设。",
                is_employed=1,
                updated_at=when,
                created_at=when,
                deleted_at=None,
            ),
        ),
        projects=(
            ProjectSource(
                id=82001,
                expert_id=person_id,
                project_title="知识库项目",
                company_name="示例科技有限公司",
                start_date=when,
                end_date=None,
                work_description="项目目标：建设知识库。\n关键动作：设计检索流程。",
                updated_at=when,
                created_at=when,
                deleted_at=None,
            ),
        ),
        educations=(
            EducationSource(
                id=83001,
                expert_id=person_id,
                school_name="示例大学",
                major="信息管理",
                updated_at=when,
                created_at=when,
                deleted_at=None,
            ),
        ),
    )


def test_resume_spec_freezes_identity_lifecycle_and_table_mapping() -> None:
    """Resume 规格必须完整表达三个冻结上游契约。"""

    assert RESUME_SPEC.source_tables == (
        "resume",
        "resume_work_experience",
        "resume_project_experience",
        "resume_education_experience",
    )
    assert RESUME_SPEC.profile_identity_column == "resume_id"
    assert RESUME_SPEC.child_identity_column == "resume_id"
    assert RESUME_SPEC.profile_deleted_at_column == "resume_deleted_at"
    assert RESUME_SPEC.child_deleted_at_column == "deleted_at"
    assert RESUME_SPEC.employment_deleted_at_column == "resume_work_experience_deleted_at"
    assert RESUME_SPEC.project_deleted_at_column == "resume_project_experience_deleted_at"
    assert RESUME_SPEC.education_deleted_at_column == "deleted_at"
    assert RESUME_SPEC.vector_doc_id_prefix == "resume"
    assert EXPERT_SPEC.vector_doc_id_prefix == "expert"
    assert all(
        EXPERT_SPEC.lifecycle_deleted_at_column_for_source_table(table) == "deleted_at"
        for table in (
            "experts_work_experience",
            "experts_project_experience",
            "experts_education_experience",
        )
    )
    assert RESUME_SPEC.target_type_for_source_table("resume") == "profile"
    assert RESUME_SPEC.target_type_for_source_table("resume_education_experience") == "education"
    assert (
        RESUME_SPEC.lifecycle_deleted_at_column_for_source_table("resume_work_experience")
        == "resume_work_experience_deleted_at"
    )
    assert (
        RESUME_SPEC.lifecycle_deleted_at_column_for_source_table("resume_project_experience")
        == "resume_project_experience_deleted_at"
    )
    assert (
        RESUME_SPEC.lifecycle_deleted_at_column_for_source_table("resume_education_experience")
        == "deleted_at"
    )


def test_dataset_spec_rejects_unknown_table_and_sql_fragment() -> None:
    """未知来源和可注入标识符都必须 fail-closed。"""

    with pytest.raises(ValueError, match="不支持来源表"):
        RESUME_SPEC.target_type_for_source_table("experts")
    with pytest.raises(ValueError, match="不支持来源表"):
        RESUME_SPEC.lifecycle_deleted_at_column_for_source_table("experts")
    with pytest.raises(ValueError, match="不是合法 SQL 标识符"):
        replace(RESUME_SPEC, profile_table="resume; DROP TABLE x")


def test_normalize_default_remains_identical_to_explicit_expert_spec() -> None:
    """不传规格的现有 Expert 调用必须逐字段等于显式 EXPERT_SPEC。"""

    aggregate = _aggregate()
    default_result = normalize_aggregate(
        aggregate,
        data_version="dataset-spec-test",
        build_batch_id=11,
    )
    explicit_result = normalize_aggregate(
        aggregate,
        data_version="dataset-spec-test",
        build_batch_id=11,
        dataset_spec=EXPERT_SPEC,
    )

    assert default_result == explicit_result
    assert default_result[1][0].source_table == "experts_work_experience"
    assert default_result[2][0].source_table == "experts_project_experience"


def test_resume_spec_changes_only_dataset_source_labels() -> None:
    """Resume 显式规格保留清洗与 Hash，只替换物理来源标签。"""

    aggregate = _aggregate()
    expert_result = normalize_aggregate(
        aggregate,
        data_version="dataset-spec-test",
        build_batch_id=12,
    )
    resume_result = normalize_aggregate(
        aggregate,
        data_version="dataset-spec-test",
        build_batch_id=12,
        dataset_spec=RESUME_SPEC,
    )

    assert resume_result[0] == expert_result[0]
    assert resume_result[1][0].source_table == "resume_work_experience"
    assert resume_result[2][0].source_table == "resume_project_experience"
    assert resume_result[1][0].source_field_id == "81001"
    assert resume_result[2][0].source_field_id == "82001"
    assert resume_result[1][0].source_hash == expert_result[1][0].source_hash
    assert resume_result[2][0].source_hash == expert_result[2][0].source_hash

    expert_chunks = ChunkFactory(_settings()).build_all(*expert_result)
    resume_chunks = ChunkFactory(_settings(), dataset_spec=RESUME_SPEC).build_all(*resume_result)
    assert expert_chunks[0].source_table == "experts+experts_education_experience"
    assert resume_chunks[0].source_table == ("resume+resume_education_experience")
    assert all(chunk.vector_doc_id.startswith("resume:") for chunk in resume_chunks)
    assert all(chunk.vector_doc_id.startswith("expert:") for chunk in expert_chunks)
    assert {chunk.source_table for chunk in resume_chunks[1:]} == {
        "resume_work_experience",
        "resume_project_experience",
    }
    assert all(chunk.serial_number == "01234567890123456" for chunk in resume_chunks)
