"""增量影响范围、期望 Chunk 集与当前事实读取领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..dataset import EXPERT_SPEC, DatasetSpec
from .content import (
    ChunkRecord,
    EducationSource,
    EmploymentSource,
    ExpertSource,
    ProjectSource,
)
from .task import PERSISTED_SOURCE_TABLES, SourceSyncTask, _validate_positive_integer

# P4 增量影响范围与内存期望 Chunk 集模型
# ---------------------------------------------------------------------------
# 下游影响范围与任务发现 target_type 故意分离：target_type（SOURCE_SYNC_TARGET_TYPES）
# 保留 education 供 P3 扫描触发，而影响范围只表达下游真正处理的四类范围，
# education 事件在 P4 Handler 中映射到 profile 范围（重建 Profile Summary Metadata），
# 不新增 education Chunk 类型。
INCREMENTAL_IMPACT_SCOPE_TYPES = frozenset({"expert", "profile", "employment", "project"})

# 影响范围允许的 Chunk 主类型；expert 范围跨三类，其余范围绑定单一类型。
INCREMENTAL_IMPACT_CHUNK_TYPES = frozenset({"profile_summary", "employment", "project"})

# Profile 范围的规范化来源标识，与 ChunkFactory.build_profile 实际输出保持一致。
_PROFILE_IMPACT_SOURCE_TABLE = EXPERT_SPEC.profile_source_label
_PROFILE_IMPACT_SOURCE_TABLES = frozenset(
    {
        _PROFILE_IMPACT_SOURCE_TABLE,
        "resume+resume_education_experience",
    }
)

# Employment / Project 范围的固定来源表，与 EmploymentStage/ProjectStage.source_table 一致。
_EMPLOYMENT_IMPACT_SOURCE_TABLE = EXPERT_SPEC.employment_table
_EMPLOYMENT_IMPACT_SOURCE_TABLES = frozenset(
    {_EMPLOYMENT_IMPACT_SOURCE_TABLE, "resume_work_experience"}
)
_PROJECT_IMPACT_SOURCE_TABLE = EXPERT_SPEC.project_table
_PROJECT_IMPACT_SOURCE_TABLES = frozenset(
    {_PROJECT_IMPACT_SOURCE_TABLE, "resume_project_experience"}
)


@dataclass(frozen=True)
class IncrementalImpactScope:
    """增量处理的下游影响范围。

    仅表达“本次 desired set 适用于哪个下游范围”，不读取 previous state、不比较 Hash、
    不决定下游动作。四类范围与字段组合的约束见 ``__post_init__``；``education``
    不作为范围类型，由 Handler 经 ``impact_scope_from_target`` 映射到 ``profile``。
    """

    scope_type: str  # expert/profile/employment/project。
    expert_id: int  # 影响范围所属专家 ID。
    chunk_type: str | None  # expert 范围为 None（跨三类）；其余范围绑定单一类型。
    source_table: str | None  # expert 范围为 None；其余范围固定来源表。
    source_field_id: str | None  # expert 为 None；profile 为 str(expert_id)；
    # employment/project 为来源行主键字符串。统一使用 source_field_id，禁止 source_record_id。

    def __post_init__(self) -> None:
        """校验四类范围的字段组合，非法组合 fail-closed。"""

        _validate_positive_integer("expert_id", self.expert_id)
        if self.scope_type not in INCREMENTAL_IMPACT_SCOPE_TYPES:
            raise ValueError(f"不支持的影响范围类型: {self.scope_type}")
        if self.scope_type == "expert":
            if (
                self.chunk_type is not None
                or self.source_table is not None
                or self.source_field_id is not None
            ):
                raise ValueError(
                    "expert 范围必须跨三类 Chunk，chunk_type/source_table/source_field_id 必须为 None"
                )
            return
        # 非 expert 范围：chunk_type/source_table/source_field_id 必须齐备。
        if self.chunk_type is None or self.source_table is None or self.source_field_id is None:
            raise ValueError(
                f"{self.scope_type} 范围必须提供 chunk_type/source_table/source_field_id"
            )
        if self.chunk_type not in INCREMENTAL_IMPACT_CHUNK_TYPES:
            raise ValueError(f"不支持的影响范围 Chunk 类型: {self.chunk_type}")
        if self.scope_type == "profile":
            if self.chunk_type != "profile_summary":
                raise ValueError("profile 范围的 chunk_type 必须为 profile_summary")
            if self.source_table not in _PROFILE_IMPACT_SOURCE_TABLES:
                raise ValueError(
                    "profile 范围的 source_table 必须为已冻结数据集的主档+教育来源标签"
                )
            if self.source_field_id != str(self.expert_id):
                raise ValueError("profile 范围的 source_field_id 必须等于 str(expert_id)")
        elif self.scope_type == "employment":
            if self.chunk_type != "employment":
                raise ValueError("employment 范围的 chunk_type 必须为 employment")
            if self.source_table not in _EMPLOYMENT_IMPACT_SOURCE_TABLES:
                raise ValueError("employment 范围的 source_table 必须为已冻结的工作来源表")
            if not isinstance(self.source_field_id, str) or not self.source_field_id.strip():
                raise ValueError("employment 范围的 source_field_id 必须是非空字符串")
            if self.source_field_id != self.source_field_id.strip():
                raise ValueError("employment 范围的 source_field_id 不得含首尾空白")
        elif self.scope_type == "project":
            if self.chunk_type != "project":
                raise ValueError("project 范围的 chunk_type 必须为 project")
            if self.source_table not in _PROJECT_IMPACT_SOURCE_TABLES:
                raise ValueError("project 范围的 source_table 必须为已冻结的项目来源表")
            if not isinstance(self.source_field_id, str) or not self.source_field_id.strip():
                raise ValueError("project 范围的 source_field_id 必须是非空字符串")
            if self.source_field_id != self.source_field_id.strip():
                raise ValueError("project 范围的 source_field_id 不得含首尾空白")


def impact_scope_from_target(
    *,
    target_type: str,
    expert_id: int,
    source_field_id: str | None = None,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> IncrementalImpactScope:
    """把任务发现的 target_type 映射为下游影响范围。

    ``education`` 任务（P3 保留独立类型用于触发 Profile 重聚合）映射到 ``profile``
    范围；其余 target_type 与 scope 一一对应。``expert``/``profile``/``education``
    的来源身份由专家 ID 唯一决定，传入 ``source_field_id`` 必须为 ``None``，否则
    fail-closed（避免调用方误以为可以覆盖来源身份）。``employment``/``project``
    必须提供非空白 ``source_field_id``。本函数只做范围映射，不读取来源行、
    不构造 Chunk；调用方需保证 expert_id/source_field_id 已是当前事实。
    """

    if target_type in {"expert", "profile", "education"} and source_field_id is not None:
        raise ValueError(f"{target_type} 范围不接受 source_field_id，来源身份由 expert_id 决定")
    if target_type == "education":
        return IncrementalImpactScope(
            scope_type="profile",
            expert_id=expert_id,
            chunk_type="profile_summary",
            source_table=dataset_spec.profile_source_label,
            source_field_id=str(expert_id),
        )
    if target_type == "expert":
        return IncrementalImpactScope(
            scope_type="expert",
            expert_id=expert_id,
            chunk_type=None,
            source_table=None,
            source_field_id=None,
        )
    if target_type == "profile":
        return IncrementalImpactScope(
            scope_type="profile",
            expert_id=expert_id,
            chunk_type="profile_summary",
            source_table=dataset_spec.profile_source_label,
            source_field_id=str(expert_id),
        )
    if target_type == "employment":
        if source_field_id is None or not source_field_id.strip():
            raise ValueError("employment 范围必须提供非空白 source_field_id")
        if source_field_id != source_field_id.strip():
            raise ValueError("employment 范围的 source_field_id 不得含首尾空白")
        return IncrementalImpactScope(
            scope_type="employment",
            expert_id=expert_id,
            chunk_type="employment",
            source_table=dataset_spec.employment_table,
            source_field_id=source_field_id,
        )
    if target_type == "project":
        if source_field_id is None or not source_field_id.strip():
            raise ValueError("project 范围必须提供非空白 source_field_id")
        if source_field_id != source_field_id.strip():
            raise ValueError("project 范围的 source_field_id 不得含首尾空白")
        return IncrementalImpactScope(
            scope_type="project",
            expert_id=expert_id,
            chunk_type="project",
            source_table=dataset_spec.project_table,
            source_field_id=source_field_id,
        )
    raise ValueError(f"不支持的同步任务目标类型: {target_type}")


@dataclass(frozen=True)
class PreparedChunkSet:
    """单个已领取任务的内存期望 Chunk 集合。

    保存任务/批次定位、影响范围和不可变 desired ChunkRecord 序列。空 desired set
    合法且必须保留范围，表示该范围当前应为空（合法清空语义，不是 Skip）；下游在
    P5 根据范围读取 previous state、比较三 Hash 并执行动作。本模型不保存
    previous state、Vector、Upsert/Metadata/Disable/Skip Action 或下游版本。
    """

    task_id: int  # 对应 expert_index_sync_task.id。
    batch_id: int  # 对应 expert_index_batch.id。
    data_version: str  # 增量审计值；不进入 Chunk/Hash/Contract，仅用于任务定位。
    impact_scope: IncrementalImpactScope  # 本次 desired set 适用的下游范围。
    desired_chunks: tuple[ChunkRecord, ...]  # 不可变、稳定排序去重后的期望 Chunk。

    def __post_init__(self) -> None:
        """校验任务定位、范围一致性和 Chunk 归属，并稳定排序去重。"""

        _validate_positive_integer("task_id", self.task_id)
        _validate_positive_integer("batch_id", self.batch_id)
        if not isinstance(self.data_version, str) or not self.data_version.strip():
            raise ValueError("data_version 必须是非空字符串")
        if not isinstance(self.impact_scope, IncrementalImpactScope):
            raise TypeError("impact_scope 必须是 IncrementalImpactScope")
        chunks = tuple(self.desired_chunks)
        self._validate_chunks(chunks)
        deduped = self._dedupe(chunks)
        sorted_chunks = tuple(
            sorted(
                deduped,
                key=lambda c: (c.chunk_type, c.source_field_id, c.chunk_index, c.vector_doc_id),
            )
        )
        object.__setattr__(self, "desired_chunks", sorted_chunks)

    def _validate_chunks(self, chunks: tuple[ChunkRecord, ...]) -> None:
        """校验每个 Chunk 归属当前范围。"""

        scope = self.impact_scope
        for chunk in chunks:
            if chunk.expert_id != scope.expert_id:
                raise ValueError(
                    f"Chunk expert_id={chunk.expert_id} 不属于范围 expert_id={scope.expert_id}"
                )
            if scope.scope_type == "expert":
                if chunk.chunk_type not in INCREMENTAL_IMPACT_CHUNK_TYPES:
                    raise ValueError(f"expert 范围不允许 chunk_type={chunk.chunk_type}")
                continue
            if chunk.chunk_type != scope.chunk_type:
                raise ValueError(f"{scope.scope_type} 范围不允许 chunk_type={chunk.chunk_type}")
            if chunk.source_table != scope.source_table:
                raise ValueError(
                    f"{scope.scope_type} 范围 Chunk source_table={chunk.source_table} 与范围不一致"
                )
            if chunk.source_field_id != scope.source_field_id:
                raise ValueError(
                    f"{scope.scope_type} 范围 Chunk source_field_id={chunk.source_field_id} 与范围不一致"
                )

    def _dedupe(self, chunks: tuple[ChunkRecord, ...]) -> list[ChunkRecord]:
        """按 vector_doc_id 去重：相同 ID 的相同记录可折叠，不同记录必须拒绝。"""

        seen: dict[str, ChunkRecord] = {}
        for chunk in chunks:
            existing = seen.get(chunk.vector_doc_id)
            if existing is None:
                seen[chunk.vector_doc_id] = chunk
                continue
            if existing != chunk:
                raise ValueError(f"vector_doc_id={chunk.vector_doc_id} 存在不一致的重复 Chunk")
        return list(seen.values())

    @property
    def is_empty(self) -> bool:
        """当前范围应为空（合法清空语义，区别于无结果）。"""

        return len(self.desired_chunks) == 0

    @property
    def desired_vector_doc_ids(self) -> tuple[str, ...]:
        """稳定排序后的期望 vector_doc_id 列表，供下游计算 stale 差集。"""

        return tuple(chunk.vector_doc_id for chunk in self.desired_chunks)


# ---------------------------------------------------------------------------
# P4.3 定向只读事实读取：请求/结果模型与 P3 任务转换。
#
# 设计要点：
# - 两层分离：``CurrentFactReadRequest`` 既携带下游 ``IncrementalImpactScope``，也单独
#   保留触发行的来源定位（``trigger_source_table`` / ``trigger_source_field_id`` /
#   ``trigger_source_updated_at``），用于校验触发行归属与事件时序。
# - education 任务的触发行是教育记录，但下游范围是 profile；触发行主键进入
#   ``trigger_source_field_id``，``impact_scope.source_field_id`` 仍是 str(expert_id)。
#   消费后不丢弃、不拒绝，保留在 trigger 字段用于归属校验。
# - 结果模型 ``ScopedCurrentFacts`` 严格按 scope 形状校验；逻辑删除行正常返回
#   （deleted_at 非空），物理行不存在 fail-closed；子表任务发现父专家已删除时
#   保留 ``parent_deleted`` 供后续 Handler 升级为 expert scope + 空 desired。
# - 不引入 ``source_record_id``：P3 任务转换时把 ``int`` 主键以 ``str`` 形式映射到
#   ``source_field_id``，符合 P4.2 的 source_field_id 契约。
# ---------------------------------------------------------------------------

# 触发行来源表白名单：覆盖共享状态库允许持久化的全部数据集来源表。
_TRIGGER_SOURCE_TABLES = PERSISTED_SOURCE_TABLES

# 来源表到触发 target_type 的映射，用于校验 trigger_source_table 与 P3 任务一致。
_TRIGGER_TARGET_TYPE_BY_TABLE = {
    "experts": "expert",
    "experts_work_experience": "employment",
    "experts_project_experience": "project",
    "experts_education_experience": "education",
    "resume": "expert",
    "resume_work_experience": "employment",
    "resume_project_experience": "project",
    "resume_education_experience": "education",
}

_PROFILE_SOURCE_LABEL_BY_TRIGGER_TABLE = {
    "experts": "experts+experts_education_experience",
    "experts_education_experience": "experts+experts_education_experience",
    "resume": "resume+resume_education_experience",
    "resume_education_experience": "resume+resume_education_experience",
}


@dataclass(frozen=True)
class CurrentFactReadRequest:
    """定向读取当前事实的请求。

    ``impact_scope`` 表达下游真正处理的影响范围（四类）；``trigger_*`` 字段独立保存
    触发行身份，用于在只读快照中校验触发行归属与事件时序。二者分离使得 education
    任务的触发行（教育记录）与下游范围（profile）可以同时表达。
    """

    impact_scope: IncrementalImpactScope
    trigger_source_table: str
    trigger_source_field_id: str
    trigger_source_updated_at: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.impact_scope, IncrementalImpactScope):
            raise TypeError("impact_scope 必须是 IncrementalImpactScope")
        if self.trigger_source_table not in _TRIGGER_SOURCE_TABLES:
            raise ValueError(f"不支持的触发来源表: {self.trigger_source_table}")
        if (
            not isinstance(self.trigger_source_field_id, str)
            or not self.trigger_source_field_id.strip()
            or self.trigger_source_field_id != self.trigger_source_field_id.strip()
        ):
            raise ValueError("trigger_source_field_id 必须是无首尾空白的非空字符串")
        if self.trigger_source_updated_at is not None and not isinstance(
            self.trigger_source_updated_at, datetime
        ):
            raise TypeError("trigger_source_updated_at 必须是 datetime 或 None")
        self._validate_scope_trigger_consistency()

    def _validate_scope_trigger_consistency(self) -> None:
        """校验触发来源表与下游影响范围语义一致。"""

        trigger_target = _TRIGGER_TARGET_TYPE_BY_TABLE[self.trigger_source_table]
        scope_type = self.impact_scope.scope_type
        expected_profile_source = _PROFILE_SOURCE_LABEL_BY_TRIGGER_TABLE.get(
            self.trigger_source_table
        )
        # expert 范围触发来源必须是当前已冻结数据集的人物主档。Resume 主档物理 id
        # 与 resume_id 分离，因此只对现有 Expert 主档保留 id 相等校验。
        if scope_type == "expert":
            if trigger_target != "expert":
                raise ValueError(
                    "expert 范围的触发来源必须是 experts 或 resume，"
                    f"实际为 {self.trigger_source_table}"
                )
            if self.trigger_source_table == "experts" and self.trigger_source_field_id != str(
                self.impact_scope.expert_id
            ):
                raise ValueError("expert 范围的触发行 source_field_id 必须等于 str(expert_id)")
        elif scope_type == "profile":
            # profile 范围允许同一数据集的主档或教育触发；来源聚合标签必须匹配，
            # 防止把 Expert 触发行和 Resume 影响范围交叉组合。
            if expected_profile_source != self.impact_scope.source_table:
                raise ValueError("profile 触发来源与影响范围不属于同一数据集")
            if trigger_target == "expert":
                if self.trigger_source_table == "experts" and self.trigger_source_field_id != str(
                    self.impact_scope.expert_id
                ):
                    raise ValueError("profile 触发行 source_field_id 必须等于 str(expert_id)")
            elif trigger_target == "education":
                # education 触发：trigger 保留教育记录主键，不进 impact_scope。
                # 归属校验在 Adapter 读取教育行后执行。
                pass
            else:
                raise ValueError(
                    f"profile 范围的触发来源必须是 experts 或 experts_education_experience，"
                    f"实际为 {self.trigger_source_table}"
                )
        elif scope_type == "employment":
            if self.trigger_source_table != self.impact_scope.source_table:
                raise ValueError(
                    "employment 范围的触发来源必须是 experts_work_experience "
                    "或 resume_work_experience，且与影响范围一致"
                )
            if self.trigger_source_field_id != self.impact_scope.source_field_id:
                raise ValueError("employment 触发行 source_field_id 与范围不一致")
        elif scope_type == "project":
            if self.trigger_source_table != self.impact_scope.source_table:
                raise ValueError(
                    "project 范围的触发来源必须是 experts_project_experience "
                    "或 resume_project_experience，且与影响范围一致"
                )
            if self.trigger_source_field_id != self.impact_scope.source_field_id:
                raise ValueError("project 触发行 source_field_id 与范围不一致")
        # education 触发来源表（trigger_target == "education"）映射到 profile 范围，
        # trigger_source_field_id 保留教育记录主键用于归属校验，不进 impact_scope。
        if trigger_target == "education" and scope_type != "profile":
            raise ValueError("education 触发必须映射到 profile 范围")


@dataclass(frozen=True)
class ScopedCurrentFacts:
    """定向只读事实读取的结果。

    - ``expert`` 范围：返回完整 ``ExpertAggregate``（有效专家四表全读）或仅生命周期
      （已删除专家，deleted_at 非空，子表为空元组）。
    - ``profile`` 范围：返回专家主档 + 全部当前有效教育；已删除父专家只返回生命周期。
    - ``employment``/``project`` 范围：返回触发行（含 deleted_at）+ 最小父专家身份；
      父专家已删除时 ``parent_deleted=True`` 且 expert 仍带 deleted_at。

    物理行不存在在 ``MySQLSourceAdapter.read_current_facts`` 层 fail-closed 抛出，
    不会构造本对象。逻辑删除行通过 ``deleted_at`` 非空正常返回。
    """

    scope: IncrementalImpactScope
    expert: ExpertSource
    employments: tuple[EmploymentSource, ...] = ()
    projects: tuple[ProjectSource, ...] = ()
    educations: tuple[EducationSource, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, IncrementalImpactScope):
            raise TypeError("scope 必须是 IncrementalImpactScope")
        if not isinstance(self.expert, ExpertSource):
            raise TypeError("expert 必须是 ExpertSource")
        self._validate_scope_shape()

    def _validate_scope_shape(self) -> None:
        scope_type = self.scope.scope_type
        if scope_type == "expert":
            # expert 范围允许三类子表；逻辑删除专家子表应为空。
            if self.expert.deleted_at is not None and (
                self.employments or self.projects or self.educations
            ):
                raise ValueError("已删除专家的子表必须为空")
        elif scope_type == "profile":
            # profile 只读专家 + 教育；不读工作和项目。
            if self.employments or self.projects:
                raise ValueError("profile 范围不读取工作和项目")
            if self.expert.deleted_at is not None and self.educations:
                raise ValueError("已删除专家的有效教育应为空")
        if scope_type == "employment":
            # employment 范围只携带单条触发行；不读项目/教育。
            if len(self.employments) != 1:
                raise ValueError("employment 范围必须恰好一条工作行")
            if self.projects or self.educations:
                raise ValueError("employment 范围不读取项目和教育")
        if scope_type == "project":
            # project 范围只携带单条触发行；不读工作/教育。
            if len(self.projects) != 1:
                raise ValueError("project 范围必须恰好一条项目行")
            if self.employments or self.educations:
                raise ValueError("project 范围不读取工作和教育")

    @property
    def parent_deleted(self) -> bool:
        """子表任务发现父专家已删除（deleted_at 非空）。

        供后续 Handler 在 employment/project 范围发现父专家已删除时，
        升级为 expert scope + 空 desired，触发旧 Chunk 逻辑清理。
        """

        return self.expert.deleted_at is not None

    @property
    def expert_deleted(self) -> bool:
        """expert/profile 范围内专家本身已逻辑删除。"""

        return self.expert.deleted_at is not None


def source_sync_task_to_read_request(
    task: SourceSyncTask,
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> CurrentFactReadRequest:
    """把 P3 的 :class:`SourceSyncTask` 转换为 P4 定向读取请求。

    本函数是 ``source_record_id``（P3 旧边界字段）的唯一消费点，用于兼容转换；
    P4 请求/结果模型不暴露 ``source_record_id``，本转换不重引入该字段。P3 任务、
    scanner、同步任务表均不被修改。转换规则：
    - 把 ``source_record_id`` 以 ``str`` 映射为 ``trigger_source_field_id``；
    - 通过 :func:`impact_scope_from_target` 把 ``target_type``（含 education）映射为
      下游 ``IncrementalImpactScope``；education 触发行主键进入 trigger 字段，
      impact_scope 仍为 profile（source_field_id=str(expert_id)）；
    - education 任务最终映射：``impact_scope.scope_type == 'profile'``、
      ``impact_scope.source_field_id == str(expert_id)``、
      ``trigger_source_field_id == str(education.id)``；
    - ``task.expert_id is None`` 时本函数 fail-closed：P4.3 不负责从教育行反推专家，
      后续 Handler 或上游需先补齐 expert_id。
    """

    if task.expert_id is None:
        raise ValueError("P3 任务缺少 expert_id，无法构造定向读取请求")
    if task.source_table not in dataset_spec.checkpoint_source_tables:
        raise ValueError(f"任务来源表 {task.source_table} 不属于 {dataset_spec.name} 数据集")
    scope = impact_scope_from_target(
        target_type=task.target_type,
        expert_id=task.expert_id,
        source_field_id=(
            None
            if task.target_type in {"expert", "profile", "education"}
            else str(task.source_record_id)
        ),
        dataset_spec=dataset_spec,
    )
    return CurrentFactReadRequest(
        impact_scope=scope,
        trigger_source_table=task.source_table,
        trigger_source_field_id=str(task.source_record_id),
        trigger_source_updated_at=task.source_updated_at,
    )
