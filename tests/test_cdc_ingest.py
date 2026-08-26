"""验证 P7.4 CDC batch 接收、任务身份和重复投递行为。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from expert_index.cdc import (
    CANAL_DESTINATION,
    CanalBatch,
    CanalColumn,
    CanalEntry,
    CanalRowData,
    CdcBatchReceiver,
    CdcPersistenceConflictError,
    CdcTaskPlan,
    CdcTaskReceipt,
    parse_canal_batch,
)


def _delete_batch() -> CanalBatch:
    """构造一条带事务边界和定位列值的固定 DELETE batch。"""

    begin = CanalEntry("TRANSACTIONBEGIN", "", "", None, "binlog.000012", 10, 1, 1)
    delete = CanalEntry(
        "ROWDATA",
        "expert",
        "experts_work_experience",
        "DELETE",
        "binlog.000012",
        20,
        1,
        1,
        rows=(
            CanalRowData(
                before=(CanalColumn("id", "901"), CanalColumn("expert_id", "101")),
            ),
        ),
    )
    end = CanalEntry("TRANSACTIONEND", "", "", None, "binlog.000012", 30, 1, 1)
    return CanalBatch(7, (begin, delete, end), CANAL_DESTINATION)


class _Persistence:
    """模拟 Index 事务内 Event/Task insert-or-verify 的最小存储。"""

    def __init__(self) -> None:
        self.calls: list[tuple[CdcTaskPlan, ...]] = []
        self.status: dict[str, CdcTaskReceipt] = {}

    def persist_batch(self, plans: Sequence[CdcTaskPlan]) -> tuple[CdcTaskReceipt, ...]:
        """首次创建任务，重复身份返回已有任务回执。"""

        frozen_plans = tuple(plans)
        self.calls.append(frozen_plans)
        receipts: list[CdcTaskReceipt] = []
        for plan in frozen_plans:
            previous = self.status.get(plan.cdc_event_id)
            if previous is None:
                previous = CdcTaskReceipt(plan.cdc_event_id, 501, "created")
                self.status[plan.cdc_event_id] = previous
            elif previous.status == "created":
                previous = CdcTaskReceipt(plan.cdc_event_id, previous.task_id, "reused_pending")
            receipts.append(previous)
        return tuple(receipts)


class _Client:
    """记录单轮消费的 fetch/rollback/ack 行为。"""

    def __init__(self, batch: CanalBatch) -> None:
        self.batch = batch
        self.rollback_ids: list[int] = []
        self.ack_ids: list[int] = []

    def fetch_without_ack(self, _batch_size: int) -> CanalBatch:
        """返回固定 batch。"""

        return self.batch

    def rollback(self, batch_id: int) -> None:
        """记录回滚，不改变测试输入。"""

        self.rollback_ids.append(batch_id)

    def ack(self, batch_id: int) -> None:
        """记录确认调用，断言业务链路不会调用它。"""

        self.ack_ids.append(batch_id)


def test_receiver_builds_cdc_delete_task_plan_without_source_time() -> None:
    """DELETE 计划使用 CDC 身份和空 source_updated_at，不猜 Poll 时间。"""

    persistence = _Persistence()
    receiver = CdcBatchReceiver(
        identity_provider=lambda event: f"{event.logfile_name}:{event.logfile_offset}:{event.row_index}",
        policy_version="p7.4-test",
    )

    result = receiver.receive(_delete_batch(), local_batch_id=88, persistence=persistence)

    assert len(result.plans) == 1
    plan = result.plans[0]
    assert plan.task.event_source == "cdc"
    assert plan.task.operation_type == "delete"
    assert plan.task.target_type == "employment"
    assert plan.task.source_updated_at is None
    assert plan.task.expert_id == 101
    assert result.receipts[0].status == "created"
    assert result.rollback_required is False


def test_receiver_reuses_same_task_for_replayed_event() -> None:
    """同一显式 CDC 身份重复投递时不生成新的任务身份。"""

    persistence = _Persistence()
    receiver = CdcBatchReceiver(identity_provider=lambda _event: "event-1", policy_version="p7.4")

    first = receiver.receive(_delete_batch(), local_batch_id=88, persistence=persistence)
    second = receiver.receive(_delete_batch(), local_batch_id=89, persistence=persistence)

    assert first.receipts[0].task_id == second.receipts[0].task_id
    assert second.receipts[0].status == "reused_pending"
    assert first.plans[0].task.dedupe_key == second.plans[0].task.dedupe_key


def test_receiver_does_not_persist_malformed_batch() -> None:
    """Parser 发现问题时返回 rollback 要求，持久化端口不会被调用。"""

    persistence = _Persistence()
    receiver = CdcBatchReceiver(identity_provider=lambda _event: "event-1", policy_version="p7.4")
    valid = _delete_batch()
    malformed_entry = valid.entries[1]
    malformed = CanalEntry(
        malformed_entry.entry_type,
        malformed_entry.source_schema,
        malformed_entry.source_table,
        malformed_entry.operation_type,
        malformed_entry.logfile_name,
        malformed_entry.logfile_offset,
        malformed_entry.server_id,
        malformed_entry.execute_time,
    )
    batch = CanalBatch(7, (valid.entries[0], malformed, valid.entries[2]), CANAL_DESTINATION)

    result = receiver.receive(batch, local_batch_id=88, persistence=persistence)

    assert result.rollback_required is True
    assert result.plans == ()
    assert persistence.calls == []


def test_consume_once_rolls_back_malformed_batch_without_ack() -> None:
    """单轮消费遇到畸形 batch 时回滚 Canal，成功路径不在 P7.4 ACK。"""

    valid = _delete_batch()
    malformed_entry = valid.entries[1]
    malformed = CanalEntry(
        malformed_entry.entry_type,
        malformed_entry.source_schema,
        malformed_entry.source_table,
        malformed_entry.operation_type,
        malformed_entry.logfile_name,
        malformed_entry.logfile_offset,
        malformed_entry.server_id,
        malformed_entry.execute_time,
    )
    client = _Client(CanalBatch(7, (valid.entries[0], malformed, valid.entries[2]), CANAL_DESTINATION))
    receiver = CdcBatchReceiver(identity_provider=lambda _event: "event-1", policy_version="p7.4")

    result = receiver.consume_once(
        client,
        batch_size=100,
        local_batch_id=88,
        persistence=_Persistence(),
    )

    assert result.rollback_required is True
    assert client.rollback_ids == [7]
    assert client.ack_ids == []


def test_parser_identity_is_not_created_implicitly() -> None:
    """解析层只产出定位事件，身份必须由接收器显式提供。"""

    parsed = parse_canal_batch(_delete_batch())
    assert parsed.delete_events[0].source_record_id == 901
    with pytest.raises(ValueError):
        CdcBatchReceiver(identity_provider=lambda _event: "", policy_version="p7.4").receive(
            _delete_batch(), local_batch_id=88, persistence=_Persistence()
        )


def test_conflict_error_type_remains_exported_for_atomic_store() -> None:
    """持久化冲突类型保留给 P7.4 Index Store 实现使用。"""

    assert issubclass(CdcPersistenceConflictError, RuntimeError)
