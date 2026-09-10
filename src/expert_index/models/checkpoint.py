"""来源扫描游标、批次扫描与安全 Checkpoint 领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .task import (
    BATCH_SOURCE_SCAN_STATUSES,
    PERSISTED_SOURCE_TABLES,
    _validate_optional_text,
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
