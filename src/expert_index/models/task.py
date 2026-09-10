"""增量同步任务、状态枚举与 Dead Letter 领域模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from ..dataset import EXPERT_SPEC

# Checkpoint 只允许跟踪当前进入专家索引链路的四张物理来源表，避免拼写错误创建出
# 永远不会被扫描的孤立游标。新增来源表时必须显式更新本集合和对应数据库约束。
# 领域模型允许表达两套已经冻结的物理表契约；真正可访问的四张表仍由各自
# Source Adapter 和运行时传入的 DatasetSpec 收窄。这样 Resume 独立 Index Schema
# 能复用状态模型，同时 Expert Scanner 默认仍只遍历 EXPERT_SPEC.source_tables。
RESUME_CHECKPOINT_SOURCE_TABLES = frozenset(
    {
        "resume",
        "resume_work_experience",
        "resume_project_experience",
        "resume_education_experience",
    }
)
# 保留现有 Expert 公开常量的精确含义；允许两套独立 Index Schema 复用模型的联合
# 白名单使用新名称，避免 Expert Source/测试误把 Resume 表纳入自身扫描范围。
CHECKPOINT_SOURCE_TABLES = EXPERT_SPEC.checkpoint_source_tables
PERSISTED_SOURCE_TABLES = CHECKPOINT_SOURCE_TABLES | RESUME_CHECKPOINT_SOURCE_TABLES

# 来源表在单个批次中的扫描状态；只描述扫描生命周期，不表示安全 Checkpoint 已推进。
BATCH_SOURCE_SCAN_STATUSES = frozenset({"pending", "running", "success", "failed"})

# 来源同步任务的完整生命周期。failed 表示本次执行失败但尚未进入重试，dead 表示
# 已超过最大重试次数；是否推进 Checkpoint 由后续任务编排根据这些状态统一判断。
SOURCE_SYNC_TASK_STATUSES = frozenset(
    {"pending", "processing", "success", "failed", "retrying", "dead"}
)

# insert/update/delete 同时覆盖 updated_at 扫描与 CDC 事件，避免两条入口使用不同术语。
SOURCE_SYNC_OPERATION_TYPES = frozenset({"insert", "update", "delete"})

# event_source 标识候选的发现通道，不代替来源表定位或 CDC Offset。
SOURCE_SYNC_EVENT_SOURCES = frozenset({"poll", "cdc", "reconcile", "replay"})

# target_type 表达最小影响对象，教育变化保留独立类型供 Profile 重聚合。
SOURCE_SYNC_TARGET_TYPES = frozenset({"expert", "profile", "employment", "project", "education"})

_SOURCE_SYNC_TARGETS_BY_TABLE = {
    "experts": frozenset({"expert", "profile"}),
    "experts_work_experience": frozenset({"employment"}),
    "experts_project_experience": frozenset({"project"}),
    "experts_education_experience": frozenset({"education"}),
    "resume": frozenset({"expert", "profile"}),
    "resume_work_experience": frozenset({"employment"}),
    "resume_project_experience": frozenset({"project"}),
    "resume_education_experience": frozenset({"education"}),
}

# processing_stage 只描述任务已经完成的标准处理边界；元组顺序是状态推进顺序。
SOURCE_SYNC_PROCESSING_STAGE_ORDER = (
    "candidate",
    "source_read",
    "normalized",
    "chunk_built",
    "validated",
    "contract_ready",
    "handed_off",
    "completed",
)
SOURCE_SYNC_PROCESSING_STAGES = frozenset(SOURCE_SYNC_PROCESSING_STAGE_ORDER)

# 失败阶段只用于安全定位处理环节，不保存来源正文、Chunk Text 或其他业务数据。
# P5.3A 在原有 9 个阶段基础上增加 4 个正式交付链编排阶段：
# artifact（受控 Contract Payload 持久化/复用）、handoff（范围级交付请求构造）、
# scope_consumer（下游范围 Consumer 执行）、receipt（上游回执校验）。
# 注意：sql/001 与 sql/012 保留原 9 个阶段的基线约束；部署时必须继续执行
# sql/014 扩展 CHECK 取值集，否则新增阶段会被数据库拒绝。
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
        "artifact",
        "handoff",
        "scope_consumer",
        "receipt",
    }
)

DEAD_LETTER_RESOLUTION_STATUSES = frozenset({"pending", "requeued", "resolved", "ignored"})

# P5.6 任务终态 result_code 冻结枚举。只表示已完成效果，不记录自由文本；
# 由 Receipt 计数确定性推导，供 finalize_source_sync_task_success 落定任务。
# embedded: 新写入或 content_hash 变化触发 Embedding；
# metadata_updated: 仅非内容字段变化，保留原向量；
# disabled: 逻辑下线（eligibility 跳过或 stale 子块），保留实体设 is_active=false；
# mixed_mutation: 单任务内同时存在多种写入效果（embed+metadata-only 或 embed+disable 等）；
# no_action_unchanged: 三 Hash 全同，零写入（计入 skipped_count）；
# recovered_unchanged: 幂等重放确认已收敛（不计 skipped_count）；
# superseded: 已有更晚重叠范围任务，旧请求未调用下游并安全收尾；
# expert_deleted: 专家删除双时间，空 desired 全专家下线（零 Embedding，不计 skipped）；
# no_chunkable_content: 非删除但零 chunk 可构建（不计 skipped）。
TASK_RESULT_CODES = frozenset(
    {
        "embedded",
        "metadata_updated",
        "disabled",
        "mixed_mutation",
        "no_action_unchanged",
        "recovered_unchanged",
        "superseded",
        "expert_deleted",
        "no_chunkable_content",
    }
)

# 真正无写入（no-write）的 result_code 集合，CheckpointManager 据此重算 skipped_count。
# recovered_unchanged 属幂等重放已收敛，语义上无动作但不是“本次零写入”，故不计 skipped。
# expert_deleted/no_chunkable_content 是 P4 阶段零写入的业务终态，不在本集合内。
NO_WRITE_RESULT_CODES = frozenset({"no_action_unchanged"})

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class NewSourceSyncTask:
    """创建一条增量同步任务所需的不可变业务身份。

    调用方必须在进入 Repository 前按冻结公式生成 ``dedupe_key``。
    Repository 只校验键格式并完成幂等持久化，不重新解释来源时间或策略版本。
    """

    dedupe_key: str  # 冻结业务身份的 64 位小写 SHA-256。
    batch_id: int  # 首次发现该任务的 expert_index_batch.id。
    event_source: str  # 候选来源：poll/cdc/reconcile/replay。
    source_table: str  # 发生变化的四张来源表之一。
    source_record_id: int  # 来源表主键，用于重读 MySQL 当前事实。
    source_updated_at: datetime | None  # 扫描候选 updated_at；物理删除可为空。
    expert_id: int | None  # 受影响专家；暂时无法映射时为空。
    operation_type: str  # 来源变化类型：insert/update/delete。
    target_type: str  # 本任务的最小影响对象类型。
    policy_version: str  # 参与幂等键和后续动作计划的策略版本。

    def __post_init__(self) -> None:
        """拒绝无法安全写入 012 任务表的创建参数。"""

        _validate_source_sync_identity(
            dedupe_key=self.dedupe_key,
            batch_id=self.batch_id,
            event_source=self.event_source,
            source_table=self.source_table,
            source_record_id=self.source_record_id,
            source_updated_at=self.source_updated_at,
            expert_id=self.expert_id,
            operation_type=self.operation_type,
            target_type=self.target_type,
        )
        _validate_optional_text(
            "policy_version",
            self.policy_version,
            max_length=32,
            allow_none=False,
        )


@dataclass(frozen=True)
class SourceSyncTask:
    """一条来源记录在增量链路中的持久化处理任务。

    ``dedupe_key`` 在跨批次重复扫描中唯一定位业务任务。该模型只保存
    来源定位、调度状态和安全错误摘要，不包含来源行、Chunk Text、Metadata 或向量。
    """

    task_id: int  # expert_index_sync_task.id。
    dedupe_key: str  # 跨批次重复扫描幂等键。
    batch_id: int  # 所属增量批次 ID，对应 expert_index_batch.id。
    event_source: str  # 候选发现通道。
    source_table: str  # 发生变化的四张物理来源表之一。
    source_record_id: int  # 来源表主键，用于重新读取当前完整业务记录。
    source_updated_at: datetime | None  # 候选的来源更新时间。
    expert_id: int | None  # 受影响专家；暂时无法映射时允许为空并等待处置。
    operation_type: str  # 来源变化类型：insert/update/delete。
    target_type: str  # 任务最小影响对象类型。
    processing_stage: str  # 当前标准处理阶段。
    sync_status: str  # 处理状态：pending/processing/success/failed/retrying/dead。
    retry_count: int  # 已执行的自动重试次数，不包含首次处理。
    policy_version: str | None  # 策略版本；迁移前历史任务可为空。
    created_at: datetime  # 任务首次持久化时间。
    updated_at: datetime  # 任务最近一次持久化更新时间。
    attempt_token: str | None = None  # 当前认领尝试的唯一令牌。
    lease_until: datetime | None = None  # 当前认领租约到期时间。
    next_retry_at: datetime | None = None  # 下一次允许重试的时间。
    error_stage: str | None = None  # 最近失败所在的标准处理阶段。
    error_code: str | None = None  # 稳定机器错误码。
    error_type: str | None = None  # 安全异常类型，不保存来源内容。
    error_message: str | None = None  # 截断后的安全错误摘要，最长 512 字符。
    result_code: str | None = None  # 机器结果码，如 no_chunkable_content。
    started_at: datetime | None = None  # 最近一次开始处理的时间。
    completed_at: datetime | None = None  # success 或 dead 的最终完成时间。

    def __post_init__(self) -> None:
        """保证任务字段能够安全映射到数据库约束和后续状态机。"""

        _validate_positive_integer("task_id", self.task_id)
        _validate_source_sync_identity(
            dedupe_key=self.dedupe_key,
            batch_id=self.batch_id,
            event_source=self.event_source,
            source_table=self.source_table,
            source_record_id=self.source_record_id,
            source_updated_at=self.source_updated_at,
            expert_id=self.expert_id,
            operation_type=self.operation_type,
            target_type=self.target_type,
        )
        if self.processing_stage not in SOURCE_SYNC_PROCESSING_STAGES:
            raise ValueError(f"不支持的同步任务处理阶段: {self.processing_stage}")
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
        if (self.attempt_token is None) != (self.lease_until is None):
            raise ValueError("attempt_token 和 lease_until 必须同时为空或同时有值")
        if self.sync_status == "processing" and self.attempt_token is None:
            raise ValueError("processing 任务必须持有 attempt_token 和 lease_until")
        if self.sync_status != "processing" and self.attempt_token is not None:
            raise ValueError("非 processing 任务不能保留 attempt_token 和 lease_until")
        if self.sync_status == "success" and (
            self.processing_stage != "completed" or self.completed_at is None
        ):
            raise ValueError("success 任务必须处于 completed 阶段并记录 completed_at")
        _validate_optional_text("attempt_token", self.attempt_token, max_length=64)
        _validate_optional_text("policy_version", self.policy_version, max_length=32)
        for field_name, value, max_length in (
            ("error_code", self.error_code, 64),
            ("error_type", self.error_type, 128),
            ("error_message", self.error_message, 512),
            ("result_code", self.result_code, 64),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > max_length
            ):
                raise ValueError(f"{field_name} 必须是长度不超过 {max_length} 的非空字符串或 None")
        for field_name, value in (
            ("created_at", self.created_at),
            ("updated_at", self.updated_at),
        ):
            if not isinstance(value, datetime):
                raise TypeError(f"{field_name} 必须是 datetime")
        for field_name, value in (
            ("lease_until", self.lease_until),
            ("next_retry_at", self.next_retry_at),
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value is not None and not isinstance(value, datetime):
                raise TypeError(f"{field_name} 必须是 datetime 或 None")


@dataclass(frozen=True)
class DeadLetterRecord:
    """一条 DLQ 失败审计记录，不包含来源正文、Chunk、Metadata 或向量。"""

    dead_letter_id: int
    task_id: int
    batch_id: int
    source_table: str
    source_record_id: int
    expert_id: int | None
    failure_stage: str
    reason_code: str
    reason_message: str | None
    retry_count: int
    artifact_id: int | None
    resolution_status: str
    resolution_note: str | None
    resolution_operator: str | None
    resolution_request_id: str | None
    resolution_version: int
    resolved_at: datetime | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        """校验 DLQ 定位、审计状态和乐观锁字段。"""

        for field_name, value in (
            ("dead_letter_id", self.dead_letter_id),
            ("task_id", self.task_id),
            ("batch_id", self.batch_id),
            ("source_record_id", self.source_record_id),
        ):
            _validate_positive_integer(field_name, value)
        if self.expert_id is not None:
            _validate_positive_integer("expert_id", self.expert_id)
        if self.artifact_id is not None:
            _validate_positive_integer("artifact_id", self.artifact_id)
        if self.source_table not in PERSISTED_SOURCE_TABLES:
            raise ValueError(f"不支持的 DLQ 来源表: {self.source_table}")
        if self.failure_stage not in SOURCE_SYNC_ERROR_STAGES:
            raise ValueError(f"不支持的 DLQ 失败阶段: {self.failure_stage}")
        if self.resolution_status not in DEAD_LETTER_RESOLUTION_STATUSES:
            raise ValueError(f"不支持的 DLQ 处置状态: {self.resolution_status}")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise ValueError("retry_count 必须是非负整数")
        if (
            isinstance(self.resolution_version, bool)
            or not isinstance(self.resolution_version, int)
            or self.resolution_version < 0
        ):
            raise ValueError("resolution_version 必须是非负整数")
        for field_name, value, max_length in (
            ("reason_code", self.reason_code, 64),
            ("reason_message", self.reason_message, 512),
            ("resolution_note", self.resolution_note, 512),
            ("resolution_operator", self.resolution_operator, 128),
            ("resolution_request_id", self.resolution_request_id, 128),
        ):
            if field_name == "reason_code":
                _validate_optional_text(field_name, value, max_length=max_length, allow_none=False)
            elif value is not None:
                _validate_optional_text(field_name, value, max_length=max_length)
        for field_name, value in (
            ("created_at", self.created_at),
            ("updated_at", self.updated_at),
            ("resolved_at", self.resolved_at),
        ):
            if value is not None and not isinstance(value, datetime):
                raise TypeError(f"{field_name} 必须是 datetime 或 None")


def _validate_source_sync_identity(
    *,
    dedupe_key: str,
    batch_id: int,
    event_source: str,
    source_table: str,
    source_record_id: int,
    source_updated_at: datetime | None,
    expert_id: int | None,
    operation_type: str,
    target_type: str,
) -> None:
    """校验新任务和持久化任务共用的不可变身份字段。"""

    if not isinstance(dedupe_key, str) or _SHA256_HEX_PATTERN.fullmatch(dedupe_key) is None:
        raise ValueError("dedupe_key 必须是 64 位小写 SHA-256")
    _validate_positive_integer("batch_id", batch_id)
    _validate_positive_integer("source_record_id", source_record_id)
    if expert_id is not None:
        _validate_positive_integer("expert_id", expert_id, optional_label=True)
    if event_source not in SOURCE_SYNC_EVENT_SOURCES:
        raise ValueError(f"不支持的同步任务事件来源: {event_source}")
    if source_table not in PERSISTED_SOURCE_TABLES:
        raise ValueError(f"不支持的同步任务来源表: {source_table}")
    if source_updated_at is not None and not isinstance(source_updated_at, datetime):
        raise TypeError("source_updated_at 必须是 datetime 或 None")
    if operation_type not in SOURCE_SYNC_OPERATION_TYPES:
        raise ValueError(f"不支持的同步任务操作类型: {operation_type}")
    if target_type not in SOURCE_SYNC_TARGET_TYPES:
        raise ValueError(f"不支持的同步任务目标类型: {target_type}")
    if source_table in _SOURCE_SYNC_TARGETS_BY_TABLE and (
        target_type not in _SOURCE_SYNC_TARGETS_BY_TABLE[source_table]
    ):
        raise ValueError(f"同步任务来源表 {source_table} 不能定位到 {target_type}")


def _validate_positive_integer(
    field_name: str,
    value: int,
    *,
    optional_label: bool = False,
) -> None:
    """校验领域标识可安全映射到 MySQL 无符号正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        suffix = "或 None" if optional_label else ""
        raise ValueError(f"{field_name} 必须是正整数{suffix}")


def _validate_optional_text(
    field_name: str,
    value: str | None,
    *,
    max_length: int,
    allow_none: bool = True,
) -> None:
    """校验可选安全文本的类型、空白和数据库字段长度。"""

    if value is None and allow_none:
        return
    if value is None or not isinstance(value, str) or not value.strip() or len(value) > max_length:
        optional_suffix = "或 None" if allow_none else ""
        raise ValueError(
            f"{field_name} 必须是长度不超过 {max_length} 的非空字符串{optional_suffix}"
        )
