"""Checkpoint、同步任务、Source、Stage 与最终 Chunk 的显式领域模型。

Source 模型严格使用 MySQL 源表列名，审查时可以直接与 SELECT 字段逐一对照；
Stage 模型使用清洗和标准化后的业务语义名；ChunkRecord 是内存中的下游输入快照。
所有模型都刻意排除姓名、联系方式、报价等禁止进入向量化链路的字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# Checkpoint 只允许跟踪当前进入专家索引链路的四张物理来源表，避免拼写错误创建出
# 永远不会被扫描的孤立游标。新增来源表时必须显式更新本集合和对应数据库约束。
CHECKPOINT_SOURCE_TABLES = frozenset(
    {
        "experts",
        "experts_work_experience",
        "experts_project_experience",
        "experts_education_experience",
    }
)

# 来源表在单个批次中的扫描状态；只描述扫描生命周期，不表示安全 Checkpoint 已推进。
BATCH_SOURCE_SCAN_STATUSES = frozenset({"pending", "running", "success", "failed"})

# 来源同步任务的完整生命周期。failed 表示本次执行失败但尚未进入重试，dead 表示
# 已超过最大重试次数；是否推进 Checkpoint 由后续任务编排根据这些状态统一判断。
SOURCE_SYNC_TASK_STATUSES = frozenset(
    {"pending", "processing", "success", "failed", "retrying", "dead"}
)

# insert/update/delete 同时覆盖 updated_at 扫描与 CDC 事件，避免两条入口使用不同术语。
SOURCE_SYNC_OPERATION_TYPES = frozenset({"insert", "update", "delete"})

# 失败阶段只用于安全定位处理环节，不保存来源正文、Chunk Text 或其他业务数据。
SOURCE_SYNC_ERROR_STAGES = frozenset(
    {
        "source",
        "parse",
        "pii",
        "chunk_builder",
        "generation",
        "validation",
        "embedding",
        "milvus",
        "checkpoint",
    }
)


@dataclass(frozen=True)
class SourceSyncTask:
    """一条来源记录在增量链路中的持久化处理任务。

    同一批次内以 ``source_table + source_record_id`` 定位唯一业务对象。该模型只保存
    调度状态和安全错误摘要，不包含来源行、Chunk Text、Metadata 或 Embedding。
    """

    task_id: int  # expert_index_sync_task.id。
    batch_id: int  # 所属增量批次 ID，对应 expert_index_batch.id。
    source_table: str  # 发生变化的四张物理来源表之一。
    source_record_id: int  # 来源表主键，用于重新读取当前完整业务记录。
    expert_id: int | None  # 受影响专家；暂时无法映射时允许为空并等待处置。
    operation_type: str  # 来源变化类型：insert/update/delete。
    sync_status: str  # 处理状态：pending/processing/success/failed/retrying/dead。
    retry_count: int  # 已执行的自动重试次数，不包含首次处理。
    created_at: datetime  # 任务首次持久化时间。
    error_stage: str | None = None  # 最近失败所在的标准处理阶段。
    error_type: str | None = None  # 安全异常类型，不保存来源内容。
    error_message: str | None = None  # 截断后的安全错误摘要，最长 512 字符。
    result_code: str | None = None  # 机器结果码，如 no_chunkable_content。
    started_at: datetime | None = None  # 最近一次开始处理的时间。
    completed_at: datetime | None = None  # success 或 dead 的最终完成时间。

    def __post_init__(self) -> None:
        """保证任务字段能够安全映射到数据库约束和后续状态机。"""

        for field_name, value in (
            ("task_id", self.task_id),
            ("batch_id", self.batch_id),
            ("source_record_id", self.source_record_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} 必须是正整数")
        if self.expert_id is not None and (
            isinstance(self.expert_id, bool)
            or not isinstance(self.expert_id, int)
            or self.expert_id < 1
        ):
            raise ValueError("expert_id 必须是正整数或 None")
        if self.source_table not in CHECKPOINT_SOURCE_TABLES:
            raise ValueError(f"不支持的同步任务来源表: {self.source_table}")
        if self.operation_type not in SOURCE_SYNC_OPERATION_TYPES:
            raise ValueError(f"不支持的同步任务操作类型: {self.operation_type}")
        if self.sync_status not in SOURCE_SYNC_TASK_STATUSES:
            raise ValueError(f"不支持的同步任务状态: {self.sync_status}")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise ValueError("retry_count 必须是非负整数")
        if self.error_stage is not None and self.error_stage not in SOURCE_SYNC_ERROR_STAGES:
            raise ValueError(f"不支持的同步任务失败阶段: {self.error_stage}")
        for field_name, value, max_length in (
            ("error_type", self.error_type, 128),
            ("error_message", self.error_message, 512),
            ("result_code", self.result_code, 64),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > max_length
            ):
                raise ValueError(
                    f"{field_name} 必须是长度不超过 {max_length} 的非空字符串或 None"
                )
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at 必须是 datetime")
        for field_name, value in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value is not None and not isinstance(value, datetime):
                raise TypeError(f"{field_name} 必须是 datetime 或 None")


@dataclass(frozen=True, order=True)
class SourceScanCursor:
    """来源表增量扫描使用的 ``(updated_at, id)`` 复合位置。

    ``record_id=0`` 只用于首次扫描的起点；来源表真实记录仍从正整数 ID 开始。
    """

    updated_at: datetime  # 来源记录更新时间，是复合排序的第一部分。
    record_id: int  # 来源记录主键，是相同更新时间下的稳定排序依据。

    def __post_init__(self) -> None:
        """保证游标能够安全参与复合范围比较。"""

        if not isinstance(self.updated_at, datetime):
            raise TypeError("updated_at 必须是 datetime")
        if (
            isinstance(self.record_id, bool)
            or not isinstance(self.record_id, int)
            or self.record_id < 0
        ):
            raise ValueError("record_id 必须是非负整数")


@dataclass(frozen=True)
class BatchSourceScan:
    """一个批次对一张来源表的固定扫描范围和临时进度。

    ``current_cursor`` 只用于批次中断恢复；即使到达 ``upper_bound``，也必须等待
    下游任务全部完成后，才能单独推进该来源表的安全 Checkpoint。
    """

    batch_id: int  # 所属构建批次 ID，对应 expert_index_batch.id。
    source_table: str  # 当前批次独立扫描的物理来源表。
    start_cursor: SourceScanCursor  # 批次启动时读取的安全 Checkpoint。
    current_cursor: SourceScanCursor  # 当前临时扫描位置，不代表数据已安全处理。
    upper_bound: SourceScanCursor  # 批次启动时固定的来源表扫描上界。
    checkpoint_lock_version: int  # 启动时的 Checkpoint 版本，供最终并发校验。
    status: str = "pending"  # 扫描状态：pending/running/success/failed。
    started_at: datetime | None = None  # 实际开始扫描该来源表的时间。
    finished_at: datetime | None = None  # 来源表扫描到达终态的时间。

    def __post_init__(self) -> None:
        """保证批次标识、扫描范围和数据库状态能够可靠持久化。"""

        if (
            isinstance(self.batch_id, bool)
            or not isinstance(self.batch_id, int)
            or self.batch_id < 1
        ):
            raise ValueError("batch_id 必须是正整数")
        if self.source_table not in CHECKPOINT_SOURCE_TABLES:
            raise ValueError(f"不支持的 Checkpoint 来源表: {self.source_table}")
        for field_name, cursor in (
            ("start_cursor", self.start_cursor),
            ("current_cursor", self.current_cursor),
            ("upper_bound", self.upper_bound),
        ):
            if not isinstance(cursor, SourceScanCursor):
                raise TypeError(f"{field_name} 必须是 SourceScanCursor")
        if not self.start_cursor <= self.current_cursor <= self.upper_bound:
            raise ValueError("扫描游标必须满足 start_cursor <= current_cursor <= upper_bound")
        if (
            isinstance(self.checkpoint_lock_version, bool)
            or not isinstance(self.checkpoint_lock_version, int)
            or self.checkpoint_lock_version < 0
        ):
            raise ValueError("checkpoint_lock_version 必须是非负整数")
        if self.status not in BATCH_SOURCE_SCAN_STATUSES:
            raise ValueError(f"不支持的批次来源扫描状态: {self.status}")
        if self.started_at is not None and not isinstance(self.started_at, datetime):
            raise TypeError("started_at 必须是 datetime 或 None")
        if self.finished_at is not None and not isinstance(self.finished_at, datetime):
            raise TypeError("finished_at 必须是 datetime 或 None")


@dataclass(frozen=True)
class SourceChangeCandidate:
    """增量扫描发现的一条最小来源变化，不包含任何业务正文。"""

    source_table: str  # 发生变化的物理来源表。
    record_id: int  # 来源表主键，用于形成下一页游标。
    expert_id: int | None  # 受影响专家；经历表脏数据可能暂时无法关联。
    updated_at: datetime  # 来源更新时间，与 record_id 共同确定顺序。
    deleted_at: datetime | None  # 非空表示需要触发旧 Chunk 清理。

    def __post_init__(self) -> None:
        """拒绝无法定位来源记录或无法参与增量排序的候选。"""

        if self.source_table not in CHECKPOINT_SOURCE_TABLES:
            raise ValueError(f"不支持的 Checkpoint 来源表: {self.source_table}")
        if (
            isinstance(self.record_id, bool)
            or not isinstance(self.record_id, int)
            or self.record_id < 1
        ):
            raise ValueError("record_id 必须是正整数")
        if self.expert_id is not None and (
            isinstance(self.expert_id, bool)
            or not isinstance(self.expert_id, int)
            or self.expert_id < 1
        ):
            raise ValueError("expert_id 必须是正整数或 None")
        if not isinstance(self.updated_at, datetime):
            raise TypeError("updated_at 必须是 datetime")
        if self.deleted_at is not None and not isinstance(self.deleted_at, datetime):
            raise TypeError("deleted_at 必须是 datetime 或 None")

    @property
    def cursor(self) -> SourceScanCursor:
        """返回当前来源记录对应的复合位置，供下一页继续扫描。"""

        return SourceScanCursor(updated_at=self.updated_at, record_id=self.record_id)

    @property
    def is_deleted(self) -> bool:
        """返回该来源记录是否已经逻辑删除。"""

        return self.deleted_at is not None


@dataclass(frozen=True)
class SourceCheckpoint:
    """一张物理来源表最后安全提交的增量扫描位置。

    ``last_updated_at`` 和 ``last_id`` 必须共同使用：来源表时间只有秒级精度，
    ``last_id`` 负责区分同一秒内的多条记录。该模型只表达已经提交的数据库状态，
    不表示批次正在处理的临时位置，也不负责读取或推进 Checkpoint。
    """

    source_table: str  # 被独立跟踪的物理来源表名。
    last_updated_at: datetime  # 最后安全提交的来源记录 updated_at。
    last_id: int  # last_updated_at 相同时最后安全提交的来源记录 ID。
    last_batch_id: int | None = None  # 最后成功推进该位置的构建批次 ID。
    last_success_time: datetime | None = None  # 最近一次安全提交的系统时间。
    lock_version: int = 0  # 乐观锁版本，防止旧批次覆盖新位置。

    def __post_init__(self) -> None:
        """拒绝无法形成可靠复合游标或无法映射到 Checkpoint 表的状态。"""

        if self.source_table not in CHECKPOINT_SOURCE_TABLES:
            raise ValueError(f"不支持的 Checkpoint 来源表: {self.source_table}")
        if not isinstance(self.last_updated_at, datetime):
            raise TypeError("last_updated_at 必须是 datetime")
        if (
            isinstance(self.last_id, bool)
            or not isinstance(self.last_id, int)
            or self.last_id < 0
        ):
            raise ValueError("last_id 必须是非负整数")
        if self.last_batch_id is not None and (
            isinstance(self.last_batch_id, bool)
            or not isinstance(self.last_batch_id, int)
            or self.last_batch_id < 1
        ):
            raise ValueError("last_batch_id 必须是正整数或 None")
        if self.last_success_time is not None and not isinstance(
            self.last_success_time,
            datetime,
        ):
            raise TypeError("last_success_time 必须是 datetime 或 None")
        if (
            isinstance(self.lock_version, bool)
            or not isinstance(self.lock_version, int)
            or self.lock_version < 0
        ):
            raise ValueError("lock_version 必须是非负整数")


@dataclass(frozen=True)
class ExpertCandidate:
    """第一版全量扫描从 ``experts`` 表读取的最小候选记录。

    候选阶段只需要专家主键和逻辑删除时间，不提前读取正文或三类经历。后续批量
    Pipeline 将使用 ``id`` 推进分页游标，并根据 ``deleted_at`` 决定是读取四表完整
    快照，还是直接停用该专家的历史 Chunk。

    字段名与 ``experts`` 源表保持一致，便于审查分页 SQL 与模型映射。
    """

    id: int  # experts.id：全量主键游标，也是三张经历表关联使用的专家 ID。
    deleted_at: datetime | None  # experts.deleted_at：非空表示应进入历史 Chunk 停用分支。

    @property
    def is_deleted(self) -> bool:
        """返回该候选是否已逻辑删除，供后续批量 Pipeline 选择处理分支。"""

        return self.deleted_at is not None


@dataclass(frozen=True)
class ExpertSource:
    """从 ``experts`` 表白名单读取的专家主档。
    字段名与源表完全一致；不读取姓名、联系方式和旧经历长文本。
    """

    id: int  # experts.id：专家主键，与三张经历表的 expert_id 关联。
    serial_number: str | None  # experts.serial_number：专家业务编号，不是数据库主键。
    company: str | None  # experts.company：专家当前或最近任职公司。
    job_status: int | None  # experts.job_status：岗位状态原始枚举；业务值映射待产品确认。
    job_name: str | None  # experts.job_name：专家当前或最近岗位。
    contact_status: int | None  # experts.contact_status：联系状态原始枚举；不在 Source 层猜测语义。
    doubtful_flag: int | None  # experts.doubtful_flag：疑似异常标记原值，用于后续推荐资格规则。
    abroad_flag: int | None  # experts.abroad_flag：海外背景原始标量，只有 0/1 可安全转成布尔值。
    background: str | None  # experts.background：专家背景和能力概述。
    work_years: float | None  # experts.work_years：工作年限；源表为 INT，领域层兼容历史小数值。
    updated_at: datetime | None  # experts.updated_at：源记录最后更新时间。
    created_at: datetime | None  # experts.created_at：源记录创建时间。
    deleted_at: datetime | None  # experts.deleted_at：逻辑删除时间；正常读取结果恒为 None。


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
    deleted_at: datetime | None  # experts_work_experience.deleted_at：逻辑删除时间；有效快照恒为 None。


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
    work_description: str | None  # experts_project_experience.work_description：包含角色/目标/动作/结果的项目。
    updated_at: datetime | None  # experts_project_experience.updated_at：源记录最后更新时间。
    created_at: datetime | None  # experts_project_experience.created_at：源记录创建时间。
    deleted_at: datetime | None  # experts_project_experience.deleted_at：逻辑删除时间；有效快照恒为 None。


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
    deleted_at: datetime | None  # experts_education_experience.deleted_at：逻辑删除时间；有效快照恒为 None。


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
    abroad_flag_norm: bool | None  # 海外背景规范值；源值 0/1 分别映射 False/True。
    background_clean: str | None  # 删除噪声与 PII 后的 experts.background。
    school_names_raw: tuple[str, ...]  # 有效教育经历的学校原始名，清洗、去空并稳定去重。
    major_names_raw: tuple[str, ...]  # 有效教育经历的专业原始名，清洗、去空并稳定去重。
    work_years: float | None  # 标准化后的工作年限，来自 experts.work_years。
    eligible_for_recommendation: bool | None  # contact_status 映射的推荐资格，未知值为 None。
    verification_status: str  # doubtful_flag 映射：普通=source_summary，存疑/未知=rejected。
    source_hash: str  # 允许进入链路的主档字段规范 JSON 的 SHA-256。
    source_updated_at: datetime | None  # 优先取 experts.updated_at，为空时取 created_at。
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
    industry_codes: tuple[str, ...]  # 仅从当前任职证据匹配的行业代码，已去重排序。
    topic_codes: tuple[str, ...]  # 仅从当前任职证据匹配的主题代码，已去重排序。
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
    industry_codes: tuple[str, ...]  # 仅从当前项目证据匹配的行业代码，已去重排序。
    topic_codes: tuple[str, ...]  # 仅从当前项目证据匹配的主题代码，已去重排序。
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
    source_table: str  # 物理来源表名。
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
    industry_codes: tuple[str, ...]  # 去重排序后的行业标签代码。模型在当前其他字段中抽取过来的内容，是为了语义拓展用的
    function_codes: tuple[str, ...]  # profile_summary 的职能代码；本期只保留字段，值为空。
    topic_codes: tuple[str, ...]  # 去重排序后的业务主题标签代码。模型在当前其他字段中抽取过来的内容，是为了语义拓展用的
    school_names_raw: tuple[str, ...]  # 仅 profile_summary 使用的学校原始名集合。
    major_names_raw: tuple[str, ...]  # 仅 profile_summary 使用的专业原始名集合。
    eligible_for_recommendation: bool | None  # 仅 profile_summary 使用；其他类型必须为 None。
    origin_type: str  # source_parsed 或 compressed_from_source。
    verification_status: str  # verified 或 source_summary，如果是存疑/未知则为 rejected，受doubtful_flag字段影响。
    weight_tier: str  # 检索权重等级，如 A/B。
    chunk_policy_version: str  # Chunk 模板与切分规则版本。
    metadata_schema_version: str  # Metadata 数据契约版本。
    embedding_model_alias: str | None  # 下游实际向量化时回填的模型别名。
    embedding_model_version: str | None  # 下游实际向量化时回填的模型版本。
    indexed_at: datetime | None  # 下游写入 Milvus 成功后的索引时间；上游初始为空。
    is_active: bool  # 当前记录是否有效；逻辑删除来源对应的旧 Chunk 会被停用。
    build_batch_id: int  # 对应 expert_index_batch.id，追溯产生该 Chunk 的运行批次。
