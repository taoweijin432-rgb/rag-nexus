"""专家来源事实、标准化 Stage 与最终 Chunk 领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExpertCandidate:
    """第一版全量扫描从 ``experts`` 表读取的最小候选记录。

    候选阶段只需要专家主键和逻辑删除时间，不提前读取正文或三类经历。后续批量
    Pipeline 将使用 ``id`` 推进分页游标，并根据映射后的 ``deleted_at`` 决定是读取四表完整
    快照，还是直接停用该专家的历史 Chunk。

    ``id`` 保持真实字段名，``deleted_at`` 是 Source SQL 对
    ``experts_deleted_at`` 的显式别名。
    """

    id: int  # experts.id：全量主键游标，也是三张经历表关联使用的专家 ID。
    deleted_at: datetime | None  # experts_deleted_at 的别名；非空表示正式来源专家已删除。

    @property
    def is_deleted(self) -> bool:
        """返回该候选是否已逻辑删除，供后续批量 Pipeline 选择处理分支。"""

        return self.deleted_at is not None


@dataclass(frozen=True)
class ExpertSource:
    """从 ``experts`` 表白名单读取的专家主档。

    业务字段保持真实列名；``updated_at/deleted_at`` 分别是 Source SQL 对
    ``experts_updated_at/experts_deleted_at`` 的显式别名。本模型不包含姓名和联系方式。

    ``scan_updated_at`` 是中间表扫描时间 ``experts.updated_at``，仅用于 P4 定向读取的
    事件时序校验；不进入 Normalize、三 Hash 或 Contract，不影响 Full Build golden。
    """

    id: int  # experts.id：专家主键，与三张经历表的 expert_id 关联。
    serial_number: str | None  # experts.serial_number：专家业务编号，不是数据库主键。
    company: str | None  # experts.company：专家当前或最近任职公司。
    job_status: int | None  # experts.job_status：岗位状态原始枚举，1=现任、2=曾任。
    job_name: str | None  # experts.job_name：专家当前或最近岗位。
    contact_status: int | None  # experts.contact_status：联系状态原始枚举，Normalize 映射推荐资格。
    doubtful_flag: int | None  # experts.doubtful_flag：1=普通、2=存疑，Normalize 映射核验状态。
    abroad_flag: int | None  # experts.abroad_flag：1=海外、2=非海外，其他值保持未知。
    background: str | None  # experts.background：专家背景和能力概述。
    work_years: float | None  # experts.work_years：工作年限；源表为 INT，领域层兼容历史小数值。
    updated_at: datetime | None  # experts_updated_at 的别名；表达正式来源版本时间。
    created_at: datetime | None  # experts.created_at：源记录创建时间。
    deleted_at: datetime | None  # experts_deleted_at 的别名；有效 Profile 事实读取结果恒为 None。
    scan_updated_at: datetime | None = None  # experts.updated_at 中间表扫描时间；P4 时序校验专用。


@dataclass(frozen=True)
class EmploymentSource:
    """从 ``experts_work_experience`` 读取的单段任职原始记录。
    create_by和update_by字段在产品方案中已经被排除，不考虑使用该字段。
    deleted_at作为逻辑删除标记，deleted_at不为空的经历不允许进入链路。"""

    id: int  # experts_work_experience.id：这一段工作经历的主键。
    expert_id: int  # experts_work_experience.expert_id：关联 experts.id 的外键语义字段。
    company_name: str | None  # experts_work_experience.company_name：任职公司。
    position_title: str | None  # experts_work_experience.position_title：任职岗位。
    start_date: datetime | None  # experts_work_experience.start_date：任职开始日期。
    end_date: datetime | None  # experts_work_experience.end_date：任职结束日期，在职时可为空。
    work_description: str | None  # experts_work_experience.work_description：工作职责和成果。
    is_employed: int | None  # experts_work_experience.is_employed：1=在职，2=不在职，其他=未知。
    updated_at: datetime | None  # experts_work_experience.updated_at：源记录最后更新时间。
    created_at: datetime | None  # experts_work_experience.created_at：源记录创建时间。
    deleted_at: (
        datetime | None
    )  # experts_work_experience.deleted_at：逻辑删除时间；有效快照恒为 None。


@dataclass(frozen=True)
class ProjectSource:
    """从 ``experts_project_experience`` 读取的单个项目原始记录。
    create_by和update_by字段在产品方案中已经被排除，不考虑使用该字段。
    deleted_at作为逻辑删除标记，deleted_at不为空的经历不允许进入链路。"""

    id: int  # experts_project_experience.id：这一个项目经历的主键。
    expert_id: int  # experts_project_experience.expert_id：关联 experts.id 的外键语义字段。
    project_title: str | None  # experts_project_experience.project_title：项目名称。
    company_name: str | None  # experts_project_experience.company_name：项目所属公司或客户。
    start_date: datetime | None  # experts_project_experience.start_date：项目开始日期。
    end_date: datetime | None  # experts_project_experience.end_date：项目结束日期。
    work_description: (
        str | None
    )  # experts_project_experience.work_description：包含角色/目标/动作/结果的项目。
    updated_at: datetime | None  # experts_project_experience.updated_at：源记录最后更新时间。
    created_at: datetime | None  # experts_project_experience.created_at：源记录创建时间。
    deleted_at: (
        datetime | None
    )  # experts_project_experience.deleted_at：逻辑删除时间；有效快照恒为 None。


@dataclass(frozen=True)
class EducationSource:
    """从 ``experts_education_experience`` 读取的单段教育经历。

    教育经历不单独生成 Chunk，只把学校和专业聚合到 profile_summary Metadata。
    ``create_by``、``update_by`` 等审计操作者字段不进入 AI 索引链路。
    """

    id: int  # experts_education_experience.id：该段教育经历的主键。
    expert_id: int  # experts_education_experience.expert_id：关联 experts.id。
    school_name: str | None  # experts_education_experience.school_name：学校原始名称。
    major: str | None  # experts_education_experience.major：专业原始名称。
    updated_at: datetime | None  # experts_education_experience.updated_at：最后更新时间。
    created_at: datetime | None  # experts_education_experience.created_at：创建时间。
    deleted_at: (
        datetime | None
    )  # experts_education_experience.deleted_at：逻辑删除时间；有效快照恒为 None。


@dataclass(frozen=True)
class ExpertAggregate:
    """重建一位专家所需的完整当前快照。"""

    profile: ExpertSource  # experts 主档：每位专家恰好一个。
    employments: tuple[EmploymentSource, ...]  # 该专家的全部有效工作经历快照。
    projects: tuple[ProjectSource, ...]  # 该专家的全部有效项目经历快照。
    educations: tuple[EducationSource, ...]  # 该专家的全部有效教育经历，仅聚合到画像。


@dataclass(frozen=True)
class ProfileStage:
    """已清洗、可落中间表的专家级允许字段。"""

    expert_id: int  # 对应 experts.id，同时是专家中间表主键。
    serial_number: str | None  # 清洗后的 experts.serial_number。
    company_name_raw: str | None  # 清洗后但未标准化的 experts.company。
    company_name_norm: str | None  # 经公司词典映射后的标准公司名。
    company_normalization_status: str  # 公司映射状态，如 verified/unknown。
    role_name_raw: str | None  # 清洗后但未编码的 experts.job_name。
    role_code: str | None  # 按岗位词典映射的标准岗位代码。
    role_level: str | None  # 按岗位词典映射的职级/责任等级。
    job_status_norm: str | None  # 岗位状态规范码：active/former，未知值为 None。
    abroad_flag_norm: bool | None  # 海外背景规范值；源值 1/2 分别映射 True/False。
    background_clean: str | None  # 删除噪声与 PII 后的 experts.background。
    school_names_raw: tuple[str, ...]  # 有效教育经历的学校原始名，清洗、去空并稳定去重。
    major_names_raw: tuple[str, ...]  # 有效教育经历的专业原始名，清洗、去空并稳定去重。
    work_years: float | None  # 标准化后的工作年限，来自 experts.work_years。
    eligible_for_recommendation: bool | None  # contact_status 映射的推荐资格，未知值为 None。
    verification_status: str  # doubtful_flag 映射：普通=source_summary，存疑/未知=rejected。
    source_hash: str  # 允许进入链路的主档字段规范 JSON 的 SHA-256。
    source_updated_at: datetime | None  # 取专家正式来源时间与有效教育 updated_at 的最新值。
    data_version: str  # 运行时指定的业务数据版本。
    build_batch_id: int  # 对应 expert_index_batch.id，用于追溯构建批次。


@dataclass(frozen=True)
class EmploymentStage:
    """一行严格对应 experts_work_experience 中的一段任职。"""

    expert_id: int  # 对应 experts_work_experience.expert_id。
    source_table: str  # 固定为 experts_work_experience，标识物理来源表。
    source_field: str  # 生成任职 Chunk 所使用的源列名，使用 + 拼接。
    source_field_id: str  # 来源行主键，即 experts_work_experience.id 的字符串。
    source_item_key: str  # 稳定业务项键；独立表当前与 source_field_id 相同。
    company_name_raw: str | None  # 清洗后的 company_name 。
    company_name_norm: str | None  # 词典标准化后的公司名。
    company_normalization_status: str  # 公司标准化状态，如 verified/unknown。
    role_name_raw: str | None  # 清洗后的 position_title 。
    role_code: str | None  # 岗位词典映射得到的标准岗位代码。
    role_level: str | None  # 岗位词典映射得到的职级/责任等级。
    start_month: str | None  # 由 start_date 按新规范转换为 YYYYMM。
    end_month: str | None  # 由 end_date 按新规范转换为 YYYYMM；在职时强制为空。
    is_current: bool | None  # is_employed 转换结果：1=True，2=False，其他=None。
    responsibilities_clean: str | None  # 从 work_description 的“工作职责”区段提取并清洗。
    achievements_clean: str | None  # 从 work_description 的“代表性成果”区段提取并清洗。
    industry_codes: tuple[str, ...]  # 任职行业代码预留字段；当前版本固定为空元组。
    topic_codes: tuple[str, ...]  # 任职主题代码预留字段；当前版本固定为空元组。
    source_hash: str  # 当前任职允许字段规范 JSON 的 SHA-256。
    source_updated_at: datetime | None  # 优先取工作经历 updated_at，为空时取 created_at。
    data_version: str  # 运行时指定的业务数据版本。
    build_batch_id: int  # 对应 expert_index_batch.id。


@dataclass(frozen=True)
class ProjectStage:
    """一个项目的标准化中间记录；未知结果保持 None，不做推断。"""

    expert_id: int  # 对应 experts_project_experience.expert_id。
    source_table: str  # 固定为 experts_project_experience，标识物理来源表。
    source_field: str  # 生成项目 Chunk 所使用的源列名，使用 + 拼接。
    source_field_id: str  # 来源行主键，即 experts_project_experience.id 的字符串。
    source_item_key: str  # 稳定项目键；独立表当前与 source_field_id 相同。
    project_name: str | None  # 清洗后的 project_title。
    company_or_client_raw: str | None  # 清洗后的 company_name 。
    company_or_client_norm: str | None  # 词典标准化后的项目公司/客户名。
    company_normalization_status: str  # 公司/客户标准化状态。
    start_month: str | None  # 由 start_date 按新规范转换为 YYYYMM。
    end_month: str | None  # 由 end_date 按新规范转换为 YYYYMM。
    objective_clean: str | None  # 从 work_description 的“项目目标”区段提取并清洗。
    actions_clean: str | None  # 从 work_description 的“关键动作”区段提取并清洗。
    results_clean: str | None  # 从 work_description 的“项目结果”区段提取并清洗，无证据时为空。
    industry_codes: tuple[str, ...]  # 项目行业代码预留字段；当前版本固定为空元组。
    topic_codes: tuple[str, ...]  # 项目主题代码预留字段；当前版本固定为空元组。
    source_hash: str  # 当前项目允许字段规范 JSON 的 SHA-256。
    source_updated_at: datetime | None  # 优先取项目经历 updated_at，为空时取 created_at。
    data_version: str  # 运行时指定的业务数据版本。
    build_batch_id: int  # 对应 expert_index_batch.id。


@dataclass(frozen=True)
class ChunkRecord:
    """上游生成并交给 Contract Bridge 的内存 Chunk 记录。

    三类 Chunk Metadata 的有效字段并集在本模型中显式声明；不适用于当前
    ``chunk_type`` 的字段使用 ``None`` 或空元组。该对象不写入中间 MySQL 表。
    """

    expert_id: int  # Chunk 所属专家 ID，对应 experts.id。
    serial_number: str | None  # 专家业务编号，来自 experts.serial_number。
    chunk_type: str  # Chunk 主类型：profile_summary/employment/project。
    chunk_subtype: str | None  # 超长记录切分后的语义子类型，未切分时可为空。
    chunk_index: int  # 当前子块序号，从 0 开始。
    chunk_total: int  # 同一 source_field_id 切分出的子块总数。
    chunk_text: str  # 最终供 Embedding 使用的标准文本。
    char_count: int  # chunk_text 的 Unicode 字符数。
    vector_doc_id: str  # Milvus 唯一标识，按统一基础公式生成且不追加版本后缀。
    source_table: str  # 来源标识；Profile 使用主档与教育表的聚合标识。
    source_field: str  # 参与生成该 Chunk 的源列名组合，例如 company+job_name+background。
    source_field_id: str  # 来源行主键 ID，用于回查 source_table 中的具体记录。
    source_version: str  # 由 source_updated_at 生成的带时区 ISO 8601 版本串。
    source_updated_at: datetime | None  # 源行更新时间，用于增量判断与审计。
    source_hash: str  # 允许的标准化来源事实 SHA-256。
    content_hash: str  # 最终 chunk_text 的 SHA-256。
    metadata_hash: str  # 扁平 Metadata 规范值的 SHA-256，用于识别仅 Metadata 变化。
    company_name_raw: str | None  # profile/employment/project 的公司或客户原始名称。
    company_name_norm: str | None  # 标准公司名，便于 MySQL/Milvus 过滤。
    role_name_raw: str | None  # profile/employment 的岗位原始名称；project 为 None。
    role_code: str | None  # profile/employment 的标准岗位代码；project 为 None。
    role_level: str | None  # profile/employment 的岗位等级；project 为 None。
    job_status_norm: str | None  # 仅 profile_summary 使用的岗位状态规范码。
    abroad_flag_norm: bool | None  # 仅 profile_summary 使用的海外背景规范值。
    work_years: float | None  # 仅 profile_summary 使用的工作年限。
    project_name: str | None  # 仅 project 使用的项目名称。
    start_month: str | None  # 任职/项目开始月份，格式 YYYYMM。
    end_month: str | None  # 任职/项目结束月份，格式 YYYYMM。
    is_current: bool | None  # 是否当前任职；项目和主档 Chunk 通常为空。
    industry_codes: tuple[str, ...]  # 去重排序后的行业代码；当前版本为预留空元组。
    function_codes: tuple[str, ...]  # profile_summary 的职能代码；本期只保留字段，值为空。
    topic_codes: tuple[str, ...]  # 去重排序后的主题代码；当前版本为预留空元组。
    school_names_raw: tuple[str, ...]  # 仅 profile_summary 使用的学校原始名集合。
    major_names_raw: tuple[str, ...]  # 仅 profile_summary 使用的专业原始名集合。
    eligible_for_recommendation: bool | None  # 仅 profile_summary 使用；其他类型必须为 None。
    origin_type: str  # source_parsed 或 compressed_from_source。
    verification_status: (
        str  # verified 或 source_summary，如果是存疑/未知则为 rejected，受doubtful_flag字段影响。
    )
    weight_tier: str  # 检索权重等级，如 A/B。
    chunk_policy_version: str  # Chunk 模板与切分规则版本。
    metadata_schema_version: str  # Metadata 数据契约版本。
    embedding_model_alias: str | None  # 下游实际向量化时回填的模型别名。
    embedding_model_version: str | None  # 下游实际向量化时回填的模型版本。
    indexed_at: datetime | None  # 下游写入 Milvus 成功后的索引时间；上游初始为空。
    is_active: bool  # 当前记录是否有效；逻辑删除来源对应的旧 Chunk 会被停用。
    build_batch_id: int  # 对应 expert_index_batch.id，追溯产生该 Chunk 的运行批次。
