"""Checkpoint、同步任务、Source、Stage 与最终 Chunk 的显式领域模型。

Source 模型严格使用 MySQL 源表列名，审查时可以直接与 SELECT 字段逐一对照；
Stage 模型使用清洗和标准化后的业务语义名；ChunkRecord 是内存中的下游输入快照。
所有模型都刻意排除姓名、联系方式、报价等禁止进入向量化链路的字段。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .dataset import EXPERT_SPEC, DatasetSpec

# Checkpoint 只允许跟踪当前进入专家索引链路的四张物理来源表，避免拼写错误创建出
# 永远不会被扫描的孤立游标。新增来源表时必须显式更新本集合和对应数据库约束。
# 领域模型允许表达两套已经冻结的物理表契约；真正可访问的四张表仍由各自
# Source Adapter 和运行时传入的 DatasetSpec 收窄。这样 Resume 独立 Index Schema
# 能复用状态模型，同时 Expert Scanner 默认仍只遍历 EXPERT_SPEC.source_tables。
RESUME_CHECKPOINT_SOURCE_TABLES = frozenset(
    {
        "resume_profile",
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
    "resume_profile": frozenset({"expert", "profile"}),
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
# 失败阶段只用于安全定位处理环节，不保存来源正文、Chunk Text 或其他业务数据。
# P5.3A 在原有 9 个阶段基础上增加 4 个正式交付链编排阶段：
# artifact（受控 Contract Payload 持久化/复用）、handoff（范围级交付请求构造）、
# scope_consumer（下游范围 Consumer 执行）、receipt（上游回执校验）。
# 注意：sql/012 的 chk_expert_sync_task_error_stage 仍只允许原 9 个值；
# 正式部署 012 前同事必须同步该 CHECK 取值集，否则新值会被 DDL 拒绝。
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
    candidate_count: int = 0  # 固定范围内发现的来源候选数。
    success_count: int = 0  # 已完成的来源任务数。
    failed_count: int = 0  # 失败或进入 Dead Letter 的任务数。
    skipped_count: int = 0  # 无需产生下游动作的任务数。
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
        if self.source_table not in PERSISTED_SOURCE_TABLES:
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
        for field_name, value in (
            ("candidate_count", self.candidate_count),
            ("success_count", self.success_count),
            ("failed_count", self.failed_count),
            ("skipped_count", self.skipped_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} 必须是非负整数")
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

        if self.source_table not in PERSISTED_SOURCE_TABLES:
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
    lease_owner: str | None = None  # 当前扫描租约持有者。
    lease_until: datetime | None = None  # 扫描租约到期时间。

    def __post_init__(self) -> None:
        """拒绝无法形成可靠复合游标或无法映射到 Checkpoint 表的状态。"""

        if self.source_table not in PERSISTED_SOURCE_TABLES:
            raise ValueError(f"不支持的 Checkpoint 来源表: {self.source_table}")
        if not isinstance(self.last_updated_at, datetime):
            raise TypeError("last_updated_at 必须是 datetime")
        if isinstance(self.last_id, bool) or not isinstance(self.last_id, int) or self.last_id < 0:
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
        if (self.lease_owner is None) != (self.lease_until is None):
            raise ValueError("lease_owner 和 lease_until 必须同时为空或同时有值")
        _validate_optional_text("lease_owner", self.lease_owner, max_length=128)
        if self.lease_until is not None and not isinstance(self.lease_until, datetime):
            raise TypeError("lease_until 必须是 datetime 或 None")


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


# ---------------------------------------------------------------------------
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
        "resume_profile+resume_education_experience",
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
    "resume_profile": "expert",
    "resume_work_experience": "employment",
    "resume_project_experience": "project",
    "resume_education_experience": "education",
}

_PROFILE_SOURCE_LABEL_BY_TRIGGER_TABLE = {
    "experts": "experts+experts_education_experience",
    "experts_education_experience": "experts+experts_education_experience",
    "resume_profile": "resume_profile+resume_education_experience",
    "resume_education_experience": "resume_profile+resume_education_experience",
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
                    "expert 范围的触发来源必须是 experts 或 resume_profile，"
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
