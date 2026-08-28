"""P7.7 CDC 任务收敛、业务 Checkpoint 和 Canal ACK 编排。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, Protocol

from ..handoff_orchestrator import HandoffOrchestrationResult
from ..models import SourceSyncTask
from .ingest import (
    CdcBatchPersistencePort,
    CdcBatchReceiver,
    CdcBatchReceiveResult,
)
from .protocol import (
    CANAL_DESTINATION,
    CanalBatch,
    CanalClientPort,
    CanalProtocolBaseline,
)

_BINLOG_FILE_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<sequence>\d+)$")
CdcBatchRunStatus = Literal["empty", "rolled_back", "not_ready", "acked"]


class CdcCheckpointConflictError(RuntimeError):
    """表示 CDC 业务位点、Event 或 Task 证据不一致。"""


class CdcBatchNotReadyError(RuntimeError):
    """表示当前或更早 CDC Task 未全部成功，不能提交位点。"""


class CdcAckPendingError(RuntimeError):
    """表示业务 Checkpoint 已提交，但 Canal ACK 仍需重试。"""

    def __init__(self, batch_id: int, checkpoint: CdcCheckpointCommitResult) -> None:
        """保留安全的 batch 和 checkpoint 结果，不保留底层异常正文。"""

        super().__init__("CDC 业务 Checkpoint 已提交，Canal ACK 待重试")
        self.batch_id = batch_id
        self.checkpoint = checkpoint


@dataclass(frozen=True)
class CdcBinlogPosition:
    """可持久、可保守比较的 Canal 业务应用位点。"""

    server_id: int
    logfile_name: str
    logfile_offset: int

    def __post_init__(self) -> None:
        """拒绝缺失 server_id、非标准 Binlog 文件名和负偏移。"""

        if isinstance(self.server_id, bool) or not isinstance(self.server_id, int):
            raise TypeError("server_id 必须是非负整数")
        if self.server_id < 0:
            raise ValueError("server_id 必须是非负整数")
        if (
            not isinstance(self.logfile_name, str)
            or not self.logfile_name
            or len(self.logfile_name) > 128
            or _BINLOG_FILE_PATTERN.fullmatch(self.logfile_name) is None
        ):
            raise ValueError("logfile_name 必须以数字序号结尾")
        if (
            isinstance(self.logfile_offset, bool)
            or not isinstance(self.logfile_offset, int)
            or self.logfile_offset < 0
        ):
            raise ValueError("logfile_offset 必须是非负整数")

    def compare(self, other: CdcBinlogPosition) -> int:
        """比较同 server/同文件系列位点，无法证明单调时停止。"""

        if not isinstance(other, CdcBinlogPosition):
            raise TypeError("other 必须是 CdcBinlogPosition")
        if self.server_id != other.server_id:
            raise CdcCheckpointConflictError("CDC server_id 变化，无法自动比较位点")
        own_match = _BINLOG_FILE_PATTERN.fullmatch(self.logfile_name)
        other_match = _BINLOG_FILE_PATTERN.fullmatch(other.logfile_name)
        if own_match is None or other_match is None:  # __post_init__ 已防御，保留封闭性。
            raise CdcCheckpointConflictError("CDC Binlog 文件名不可比较")
        if own_match.group("prefix") != other_match.group("prefix"):
            raise CdcCheckpointConflictError("CDC Binlog 文件系列变化，需人工校验")
        own_key = (int(own_match.group("sequence")), self.logfile_offset)
        other_key = (int(other_match.group("sequence")), other.logfile_offset)
        return (own_key > other_key) - (own_key < other_key)


@dataclass(frozen=True)
class CdcEventCommitIdentity:
    """提交时必须与 Index Event/Task 一致的最小身份。"""

    cdc_event_id: str
    task_id: int
    position: CdcBinlogPosition
    row_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cdc_event_id, str) or not self.cdc_event_id.strip():
            raise ValueError("cdc_event_id 必须是非空字符串")
        _require_positive_integer("task_id", self.task_id)
        if isinstance(self.row_index, bool) or not isinstance(self.row_index, int):
            raise TypeError("row_index 必须是非负整数")
        if self.row_index < 0:
            raise ValueError("row_index 必须是非负整数")


@dataclass(frozen=True)
class CdcBatchCommitRequest:
    """一个 Canal batch 的 Event 证据和目标业务位点。"""

    destination: str
    local_batch_id: int
    applied_position: CdcBinlogPosition
    events: tuple[CdcEventCommitIdentity, ...]

    def __post_init__(self) -> None:
        if self.destination != CANAL_DESTINATION:
            raise ValueError("destination 与冻结值不一致")
        _require_positive_integer("local_batch_id", self.local_batch_id)
        event_ids = tuple(event.cdc_event_id for event in self.events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("CDC 提交请求包含重复 cdc_event_id")
        for event in self.events:
            if event.position.compare(self.applied_position) > 0:
                raise ValueError("CDC Event 位点不能超过 batch 应用位点")


@dataclass(frozen=True)
class CdcCheckpointCommitResult:
    """Index MySQL 业务 Checkpoint 提交结果。"""

    position: CdcBinlogPosition
    lock_version: int
    already_applied: bool

    def __post_init__(self) -> None:
        """校验持久化版本和幂等标记，防止伪造提交证据。"""

        if (
            isinstance(self.lock_version, bool)
            or not isinstance(self.lock_version, int)
            or self.lock_version < 1
        ):
            raise ValueError("lock_version 必须是正整数")
        if not isinstance(self.already_applied, bool):
            raise TypeError("already_applied 必须是 bool")


@dataclass(frozen=True)
class CdcBatchRunResult:
    """单轮 CDC 编排的无 PII 结果。"""

    status: CdcBatchRunStatus
    canal_batch_id: int
    local_batch_id: int
    task_count: int
    checkpoint: CdcCheckpointCommitResult | None = None


class CdcCheckpointRepositoryPort(Protocol):
    """原子落定 Event success 和 business checkpoint 的 Index 端口。"""

    def commit_applied_batch(
        self,
        request: CdcBatchCommitRequest,
    ) -> CdcCheckpointCommitResult: ...


class CdcTaskRepositoryPort(Protocol):
    """CDC 单任务处理依赖的读取和精确领取端口。"""

    def get_source_sync_task(self, task_id: int) -> SourceSyncTask | None: ...

    def claim_cdc_source_sync_task(
        self,
        task_id: int,
        *,
        lease_seconds: int,
    ) -> SourceSyncTask | None: ...


class CdcClaimedTaskProcessorPort(Protocol):
    """已领取 CDC Task 的 P7.6 处理端口。"""

    def process_claimed(
        self,
        task: SourceSyncTask,
        attempt_token: str,
    ) -> HandoffOrchestrationResult: ...


class CdcBatchCoordinator:
    """严格按接收、收敛、Checkpoint、ACK 顺序处理单个 batch。"""

    def __init__(
        self,
        *,
        protocol_baseline: CanalProtocolBaseline,
        receiver: CdcBatchReceiver,
        persistence: CdcBatchPersistencePort,
        task_repository: CdcTaskRepositoryPort,
        task_processor: CdcClaimedTaskProcessorPort,
        checkpoint_repository: CdcCheckpointRepositoryPort,
        task_lease_seconds: int,
    ) -> None:
        """保存被动依赖；构造过程不连接 Canal、MySQL 或向量库。"""

        if not isinstance(protocol_baseline, CanalProtocolBaseline):
            raise TypeError("protocol_baseline 必须是 CanalProtocolBaseline")
        _require_positive_integer("task_lease_seconds", task_lease_seconds)
        self._protocol_baseline = protocol_baseline
        self._receiver = receiver
        self._persistence = persistence
        self._task_repository = task_repository
        self._task_processor = task_processor
        self._checkpoint_repository = checkpoint_repository
        self._task_lease_seconds = task_lease_seconds

    def consume_once(
        self,
        client: CanalClientPort,
        *,
        batch_size: int,
        local_batch_id: int,
    ) -> CdcBatchRunResult:
        """处理一个未确认 batch；只有业务 Checkpoint 提交后才 ACK。"""

        self._protocol_baseline.require_production_ready()
        _require_positive_integer("batch_size", batch_size)
        _require_positive_integer("local_batch_id", local_batch_id)
        batch = client.fetch_without_ack(batch_size)
        if batch.is_empty:
            return CdcBatchRunResult("empty", batch.batch_id, local_batch_id, 0)

        try:
            received = self._receiver.receive(
                batch,
                local_batch_id=local_batch_id,
                persistence=self._persistence,
            )
        except Exception:
            _rollback_or_raise(client, batch.batch_id)
            raise
        if received.rollback_required:
            _rollback_or_raise(client, batch.batch_id)
            return CdcBatchRunResult(
                "rolled_back",
                batch.batch_id,
                local_batch_id,
                len(received.receipts),
            )
        try:
            tasks_ready = self._process_tasks(received)
        except Exception:
            _rollback_or_raise(client, batch.batch_id)
            raise
        if not tasks_ready:
            _rollback_or_raise(client, batch.batch_id)
            return CdcBatchRunResult(
                "not_ready",
                batch.batch_id,
                local_batch_id,
                len(received.receipts),
            )
        try:
            request = build_cdc_batch_commit_request(batch, received)
            checkpoint = self._checkpoint_repository.commit_applied_batch(request)
        except Exception:
            _rollback_or_raise(client, batch.batch_id)
            raise
        try:
            client.ack(batch.batch_id)
        except Exception as error:
            raise CdcAckPendingError(batch.batch_id, checkpoint) from error
        return CdcBatchRunResult(
            "acked",
            batch.batch_id,
            local_batch_id,
            len(received.receipts),
            checkpoint,
        )

    def _process_tasks(self, received: CdcBatchReceiveResult) -> bool:
        """严格按回执顺序领取 Task，遇到首个非成功结果立即停止。"""

        for receipt in received.receipts:
            current = self._task_repository.get_source_sync_task(receipt.task_id)
            if current is None:
                raise CdcCheckpointConflictError("CDC Event 回执关联的 Task 不存在")
            if current.sync_status == "success":
                continue
            if current.sync_status == "dead":
                return False
            claimed = self._task_repository.claim_cdc_source_sync_task(
                receipt.task_id,
                lease_seconds=self._task_lease_seconds,
            )
            if claimed is None:
                current = self._task_repository.get_source_sync_task(receipt.task_id)
                if current is not None and current.sync_status == "success":
                    continue
                return False
            token = claimed.attempt_token
            if token is None:
                raise CdcCheckpointConflictError("CDC Task 领取成功但缺少 attempt_token")
            outcome = self._task_processor.process_claimed(claimed, token)
            if outcome.final_status != "success" or not outcome.terminal:
                return False
        return True


def build_cdc_batch_commit_request(
    batch: CanalBatch,
    received: CdcBatchReceiveResult,
) -> CdcBatchCommitRequest:
    """从已校验 batch 和持久化回执构建不含正文的提交请求。"""

    if batch.is_empty:
        raise ValueError("空 Canal batch 不能构建提交请求")
    _require_positive_integer("Canal batch_id", batch.batch_id)
    if received.rollback_required:
        raise ValueError("需要 rollback 的 batch 不能构建提交请求")
    if len(received.plans) != len(received.receipts):
        raise CdcCheckpointConflictError("CDC plan/receipt 数量不一致")
    positions = tuple(_entry_position(entry) for entry in batch.entries)
    for previous, current in pairwise(positions):
        if previous.compare(current) > 0:
            raise CdcCheckpointConflictError("Canal batch 内位点发生回退")
    applied_position = positions[-1]
    events: list[CdcEventCommitIdentity] = []
    for plan, receipt in zip(received.plans, received.receipts):
        if plan.cdc_event_id != receipt.cdc_event_id:
            raise CdcCheckpointConflictError("CDC plan/receipt 身份不一致")
        event = plan.event
        events.append(
            CdcEventCommitIdentity(
                cdc_event_id=plan.cdc_event_id,
                task_id=receipt.task_id,
                position=CdcBinlogPosition(
                    event.server_id if event.server_id is not None else -1,
                    event.logfile_name,
                    event.logfile_offset,
                ),
                row_index=event.row_index,
            )
        )
    return CdcBatchCommitRequest(
        destination=batch.destination or "",
        local_batch_id=received.local_batch_id,
        applied_position=applied_position,
        events=tuple(events),
    )


def _entry_position(entry: object) -> CdcBinlogPosition:
    """从已解码 Entry 构造严格位点，不对缺失字段补默认。"""

    server_id = getattr(entry, "server_id", None)
    logfile_name = getattr(entry, "logfile_name", None)
    logfile_offset = getattr(entry, "logfile_offset", None)
    return CdcBinlogPosition(
        server_id if isinstance(server_id, int) and not isinstance(server_id, bool) else -1,
        logfile_name if isinstance(logfile_name, str) else "",
        logfile_offset
        if isinstance(logfile_offset, int) and not isinstance(logfile_offset, bool)
        else -1,
    )


def _rollback_or_raise(client: CanalClientPort, batch_id: int) -> None:
    """在业务位点未提交时回滚传输 batch，回滚失败直接报错。"""

    client.rollback(batch_id)


def _require_positive_integer(field_name: str, value: int) -> None:
    """校验数据库和 Canal 调用使用的正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} 必须是正整数")


__all__ = [
    "CdcAckPendingError",
    "CdcBatchCommitRequest",
    "CdcBatchCoordinator",
    "CdcBatchNotReadyError",
    "CdcBatchRunResult",
    "CdcBinlogPosition",
    "CdcCheckpointCommitResult",
    "CdcCheckpointConflictError",
    "CdcCheckpointRepositoryPort",
    "CdcClaimedTaskProcessorPort",
    "CdcEventCommitIdentity",
    "CdcTaskRepositoryPort",
    "build_cdc_batch_commit_request",
]
