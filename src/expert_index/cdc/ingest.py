"""P7.4 Canal batch 幂等接收与标准 Sync Task 计划。

接收器只负责解析、生成待持久化的 CDC Event/Task 计划，并要求调用方通过一个 Index
MySQL 原子端口一次性落库。真实 ``cdc_event_id`` 未冻结时必须显式注入身份提供器，
本模块不会根据未经确认的 Offset 自行猜测生产身份，也不会调用 ACK。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from ..models import NewSourceSyncTask
from .parser import CanalChangeEvent, CanalParseIssue, parse_canal_batch
from .protocol import CanalBatch, CanalClientPort

CdcTaskReceiptStatus = Literal["created", "reused_pending", "reused_success"]
_TARGET_TYPES = {
    "experts": "expert",
    "experts_work_experience": "employment",
    "experts_project_experience": "project",
    "experts_education_experience": "education",
}


class CdcIngestError(ValueError):
    """表示 CDC 身份或接收输入不满足安全契约。"""


class CdcPersistenceConflictError(RuntimeError):
    """表示相同 CDC 身份对应了不一致的不可变事件内容。"""


CdcEventIdentityProvider = Callable[[CanalChangeEvent], str]


@dataclass(frozen=True)
class CdcTaskPlan:
    """一个 CDC 变更对应的身份、事件定位和标准任务创建计划。"""

    cdc_event_id: str
    event: CanalChangeEvent
    task: NewSourceSyncTask


@dataclass(frozen=True)
class CdcTaskReceipt:
    """Index 原子持久化端口返回的任务结果，不代表下游业务已完成。"""

    cdc_event_id: str
    task_id: int
    status: CdcTaskReceiptStatus


@dataclass(frozen=True)
class CdcBatchReceiveResult:
    """一个 Canal batch 的接收摘要；ACK 由业务成功后的更外层流程负责。"""

    local_batch_id: int
    plans: tuple[CdcTaskPlan, ...]
    receipts: tuple[CdcTaskReceipt, ...]
    ignored_count: int
    issues: tuple[CanalParseIssue, ...]

    @property
    def rollback_required(self) -> bool:
        """解析存在问题时要求整个 batch 不确认。"""

        return bool(self.issues)


class CdcBatchPersistencePort(Protocol):
    """Index MySQL 原子持久化端口，必须同时完成 Event 与 Task 幂等写入。"""

    def persist_batch(self, plans: Sequence[CdcTaskPlan]) -> tuple[CdcTaskReceipt, ...]:
        """在一个 Index 事务中 insert-or-verify Event 并创建或复用 Task。"""


class CdcBatchReceiver:
    """把已解析的 Canal 变更 batch 转成标准 CDC Task 计划。"""

    def __init__(
        self,
        *,
        identity_provider: CdcEventIdentityProvider,
        policy_version: str,
    ) -> None:
        """注入明确的 CDC 身份策略；不提供未经冻结的默认身份算法。"""

        if not callable(identity_provider):
            raise TypeError("identity_provider 必须是可调用对象")
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise ValueError("policy_version 必须是非空字符串")
        if len(policy_version) > 32:
            raise ValueError("policy_version 长度不能超过 32")
        self.identity_provider = identity_provider
        self.policy_version = policy_version

    def receive(
        self,
        batch: CanalBatch,
        *,
        local_batch_id: int,
        persistence: CdcBatchPersistencePort,
    ) -> CdcBatchReceiveResult:
        """解析 batch 并原子提交 CDC Event/Task 计划，绝不在此调用 ACK。"""

        if isinstance(local_batch_id, bool) or not isinstance(local_batch_id, int):
            raise TypeError("local_batch_id 必须是正整数")
        if local_batch_id < 1:
            raise ValueError("local_batch_id 必须是正整数")
        parsed = parse_canal_batch(batch)
        if parsed.rollback_required:
            return CdcBatchReceiveResult(local_batch_id, (), (), parsed.ignored_count, parsed.issues)

        plans = tuple(
            self._build_plan(event, local_batch_id=local_batch_id)
            for event in parsed.events
        )
        receipts = persistence.persist_batch(plans) if plans else ()
        receipts = tuple(receipts)
        _validate_receipts(plans, receipts)
        return CdcBatchReceiveResult(
            local_batch_id=local_batch_id,
            plans=plans,
            receipts=receipts,
            ignored_count=parsed.ignored_count,
            issues=parsed.issues,
        )

    def consume_once(
        self,
        client: CanalClientPort,
        *,
        batch_size: int,
        local_batch_id: int,
        persistence: CdcBatchPersistencePort,
    ) -> CdcBatchReceiveResult:
        """消费一个未确认 batch；失败或畸形输入回滚，成功只落库不 ACK。"""

        batch = client.fetch_without_ack(batch_size)
        if batch.is_empty:
            return self.receive(
                batch,
                local_batch_id=local_batch_id,
                persistence=persistence,
            )
        rollback_started = False
        try:
            result = self.receive(
                batch,
                local_batch_id=local_batch_id,
                persistence=persistence,
            )
            if result.rollback_required:
                rollback_started = True
                client.rollback(batch.batch_id)
            return result
        except Exception:
            # Index 事务失败、身份冲突或解析异常都不能确认 Canal 位点。
            if not rollback_started:
                client.rollback(batch.batch_id)
            raise

    def _build_plan(self, event: CanalChangeEvent, *, local_batch_id: int) -> CdcTaskPlan:
        """把变更定位事件映射为 CDC Task，不读取来源正文或推断时间。"""

        cdc_event_id = self.identity_provider(event)
        if not isinstance(cdc_event_id, str) or not cdc_event_id.strip():
            raise CdcIngestError("identity_provider 必须返回非空 cdc_event_id")
        if len(cdc_event_id) > 256:
            raise CdcIngestError("cdc_event_id 长度不能超过 256")
        target_type = _TARGET_TYPES[event.source_table]
        dedupe_key = _task_dedupe_key(cdc_event_id, event, self.policy_version, target_type)
        task = NewSourceSyncTask(
            dedupe_key=dedupe_key,
            batch_id=local_batch_id,
            event_source="cdc",
            source_table=event.source_table,
            source_record_id=event.source_record_id,
            source_updated_at=None,
            expert_id=event.expert_id,
            operation_type=event.operation_type,
            target_type=target_type,
            policy_version=self.policy_version,
        )
        return CdcTaskPlan(cdc_event_id=cdc_event_id, event=event, task=task)


def _task_dedupe_key(
    cdc_event_id: str,
    event: CanalChangeEvent,
    policy_version: str,
    target_type: str,
) -> str:
    """按 CDC 身份、策略和 scope 生成标准 Sync Task 的小写 SHA-256。"""

    identity = {
        "cdc_event_id": cdc_event_id,
        "operation_type": event.operation_type,
        "policy_version": policy_version,
        "scope": {
            "source_table": event.source_table,
            "source_record_id": event.source_record_id,
            "expert_id": event.expert_id,
            "target_type": target_type,
        },
    }
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_receipts(
    plans: Sequence[CdcTaskPlan], receipts: Sequence[CdcTaskReceipt]
) -> None:
    """验证持久化端口返回的顺序、身份和任务 ID，拒绝静默错配。"""

    if len(plans) != len(receipts):
        raise CdcIngestError("CDC 持久化回执数量与计划不一致")
    expected_ids = tuple(plan.cdc_event_id for plan in plans)
    actual_ids = tuple(receipt.cdc_event_id for receipt in receipts)
    if expected_ids != actual_ids:
        raise CdcIngestError("CDC 持久化回执身份或顺序不一致")
    for receipt in receipts:
        if isinstance(receipt.task_id, bool) or not isinstance(receipt.task_id, int) or receipt.task_id < 1:
            raise CdcIngestError("CDC 持久化回执 task_id 非法")


__all__ = [
    "CdcBatchPersistencePort",
    "CdcBatchReceiveResult",
    "CdcBatchReceiver",
    "CdcEventIdentityProvider",
    "CdcIngestError",
    "CdcPersistenceConflictError",
    "CdcTaskPlan",
    "CdcTaskReceipt",
    "CdcTaskReceiptStatus",
]
