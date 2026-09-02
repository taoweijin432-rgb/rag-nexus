"""来源数据集的只读结构规格。

该模块只描述 Source Adapter、Normalize、Chunk Builder 和 Validator 需要知道的稳定
结构信息，不包含连接配置、SQL 参数或运行状态。Expert 使用 :data:`EXPERT_SPEC` 作为
默认值；其他数据集必须由各自入口显式传入规格，不能通过环境变量或名称猜测切换。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_TARGET_TYPES = frozenset({"profile", "employment", "project", "education"})


@dataclass(frozen=True)
class DatasetSpec:
    """描述一组人物主档及三类经历表的稳定读取契约。

    Parameters
    ----------
    name:
        数据集稳定名称，仅用于日志、诊断和 fail-closed 边界判断。
    profile_table, employment_table, project_table, education_table:
        四张物理来源表。所有名称在构造时校验为单个 SQL 标识符，允许 Source Adapter
        安全地从冻结规格中组装 SQL；不得传入用户输入或任意 SQL 片段。
    profile_identity_column:
        人物主表中的全链路人物身份列。Expert 为 ``id``，Resume 为 ``resume_id``。
    child_identity_column:
        三张经历表关联人物主表的身份列。
    profile_source_updated_at_column, employment_source_updated_at_column,
    project_source_updated_at_column, education_source_updated_at_column:
        四类事实的业务版本时间。它们与用于 Poll 游标的通用 ``updated_at`` 可能具有
        不同语义，因此必须显式声明。
    profile_deleted_at_column, child_deleted_at_column:
        人物和子记录的权威生命周期字段。
    profile_source_columns, employment_source_columns, project_source_columns:
        进入三类 Chunk 的来源字段白名单，同时用于生成稳定 ``source_field`` 标签。
    vector_doc_id_prefix:
        Milvus ``vector_doc_id`` 的命名空间前缀。Expert 固定为 ``expert``，Resume
        固定为 ``resume``；默认值保持既有 Expert 行为，避免未显式传入规格的调用改变。

    Notes
    -----
    下游 Contract 首版仍使用内部字段名 ``expert_id``。规格只决定该字段从哪个来源身份
    列取得，并声明 ID 命名空间；不修改 Contract 字段名、Metadata 结构或 Hash 规则。
    """

    name: str
    profile_table: str
    employment_table: str
    project_table: str
    education_table: str
    profile_identity_column: str
    child_identity_column: str
    profile_source_updated_at_column: str
    employment_source_updated_at_column: str
    project_source_updated_at_column: str
    education_source_updated_at_column: str
    profile_deleted_at_column: str
    child_deleted_at_column: str
    profile_source_columns: tuple[str, ...]
    employment_source_columns: tuple[str, ...]
    project_source_columns: tuple[str, ...]
    vector_doc_id_prefix: str = "expert"

    def __post_init__(self) -> None:
        """拒绝空规格、重复表和可注入的 SQL 标识符。"""

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DatasetSpec.name 必须是非空字符串")
        if not isinstance(self.vector_doc_id_prefix, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", self.vector_doc_id_prefix
        ):
            raise ValueError(
                "DatasetSpec.vector_doc_id_prefix 必须是小写字母/数字/下划线命名空间"
            )

        identifiers = {
            "profile_table": self.profile_table,
            "employment_table": self.employment_table,
            "project_table": self.project_table,
            "education_table": self.education_table,
            "profile_identity_column": self.profile_identity_column,
            "child_identity_column": self.child_identity_column,
            "profile_source_updated_at_column": self.profile_source_updated_at_column,
            "employment_source_updated_at_column": self.employment_source_updated_at_column,
            "project_source_updated_at_column": self.project_source_updated_at_column,
            "education_source_updated_at_column": self.education_source_updated_at_column,
            "profile_deleted_at_column": self.profile_deleted_at_column,
            "child_deleted_at_column": self.child_deleted_at_column,
        }
        identifiers.update(
            {
                f"profile_source_columns[{index}]": column
                for index, column in enumerate(self.profile_source_columns)
            }
        )
        identifiers.update(
            {
                f"employment_source_columns[{index}]": column
                for index, column in enumerate(self.employment_source_columns)
            }
        )
        identifiers.update(
            {
                f"project_source_columns[{index}]": column
                for index, column in enumerate(self.project_source_columns)
            }
        )
        for label, identifier in identifiers.items():
            if not isinstance(identifier, str) or not _SQL_IDENTIFIER.fullmatch(identifier):
                raise ValueError(f"DatasetSpec.{label} 不是合法 SQL 标识符: {identifier!r}")

        if len(set(self.source_tables)) != len(self.source_tables):
            raise ValueError("DatasetSpec 的四张来源表必须互不相同")
        if not self.profile_source_columns:
            raise ValueError("profile_source_columns 不能为空")
        if not self.employment_source_columns:
            raise ValueError("employment_source_columns 不能为空")
        if not self.project_source_columns:
            raise ValueError("project_source_columns 不能为空")

    @property
    def source_tables(self) -> tuple[str, str, str, str]:
        """按主档、工作、项目、教育的固定顺序返回四张来源表。"""

        return (
            self.profile_table,
            self.employment_table,
            self.project_table,
            self.education_table,
        )

    @property
    def checkpoint_source_tables(self) -> frozenset[str]:
        """返回允许建立 Poll Checkpoint 的完整物理表白名单。"""

        return frozenset(self.source_tables)

    @property
    def profile_source_label(self) -> str:
        """返回 Profile 聚合使用的主档与教育组合来源标签。"""

        return f"{self.profile_table}+{self.education_table}"

    @property
    def employment_source_field(self) -> str:
        """返回 Employment Chunk 的稳定来源字段标签。"""

        return "+".join(self.employment_source_columns)

    @property
    def profile_source_field(self) -> str:
        """返回 Profile Summary Chunk 的稳定来源字段标签。"""

        return "+".join(self.profile_source_columns)

    @property
    def project_source_field(self) -> str:
        """返回 Project Chunk 的稳定来源字段标签。"""

        return "+".join(self.project_source_columns)

    def target_type_for_source_table(self, source_table: str) -> str:
        """把一张来源表映射到最小影响目标类型。

        教育记录只参与 Profile 聚合，但保留 ``education`` 目标类型，供后续增量编排
        明确执行“教育变化 -> Profile 重建”。未知表直接拒绝，避免静默扩大处理范围。
        """

        mapping = {
            self.profile_table: "profile",
            self.employment_table: "employment",
            self.project_table: "project",
            self.education_table: "education",
        }
        try:
            target_type = mapping[source_table]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{self.name} 不支持来源表: {source_table}") from exc
        if target_type not in _SUPPORTED_TARGET_TYPES:  # pragma: no cover - 冻结映射自检
            raise RuntimeError(f"DatasetSpec 产生了不支持的目标类型: {target_type}")
        return target_type


# Expert 规格冻结现有生产表名、身份列和来源标签。Normalize 的默认参数始终指向该对象，
# 因而未显式传规格的现有 Expert 调用保持原输出。
EXPERT_SPEC = DatasetSpec(
    name="expert",
    profile_table="experts",
    employment_table="experts_work_experience",
    project_table="experts_project_experience",
    education_table="experts_education_experience",
    profile_identity_column="id",
    child_identity_column="expert_id",
    profile_source_updated_at_column="experts_updated_at",
    employment_source_updated_at_column="updated_at",
    project_source_updated_at_column="updated_at",
    education_source_updated_at_column="updated_at",
    profile_deleted_at_column="experts_deleted_at",
    child_deleted_at_column="deleted_at",
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
)


__all__ = ["EXPERT_SPEC", "DatasetSpec"]
