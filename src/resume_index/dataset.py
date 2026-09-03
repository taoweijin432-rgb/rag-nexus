"""Resume 四表的冻结数据集规格。"""

from expert_index.dataset import DatasetSpec

# ``resume_id`` 是全链路人物身份；四张表的 ``id`` 只承担物理行定位和稳定经历身份。
# 主档、工作、项目、教育分别使用各自冻结的生命周期字段。
RESUME_SPEC = DatasetSpec(
    name="resume",
    profile_table="resume",
    employment_table="resume_work_experience",
    project_table="resume_project_experience",
    education_table="resume_education_experience",
    profile_identity_column="resume_id",
    child_identity_column="resume_id",
    profile_source_updated_at_column="resume_updated_at",
    employment_source_updated_at_column="resume_work_experience_updated_at",
    project_source_updated_at_column="resume_project_experience_updated_at",
    education_source_updated_at_column="updated_at",
    profile_deleted_at_column="resume_deleted_at",
    child_deleted_at_column="deleted_at",
    employment_deleted_at_column="resume_work_experience_deleted_at",
    project_deleted_at_column="resume_project_experience_deleted_at",
    education_deleted_at_column="deleted_at",
    profile_source_columns=(
        "company",
        "job_status",
        "job_name",
        "contact_status",
        "doubtful_flag",
        "abroad_flag",
        "background",
        "work_years",
        "school_name",
        "major",
    ),
    employment_source_columns=(
        "company_name",
        "position_title",
        "start_date",
        "end_date",
        "work_description",
        "is_employed",
    ),
    project_source_columns=(
        "project_title",
        "company_name",
        "start_date",
        "end_date",
        "work_description",
    ),
    vector_doc_id_prefix="resume",
)


__all__ = ["RESUME_SPEC"]
