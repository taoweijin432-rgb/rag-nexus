"""验证 P7.7 CDC 业务 Checkpoint、顺序收敛和 ACK 恢复语义。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Self

import pytest

from expert_index.cdc import (
    CANAL_DESTINATION,
    CanalBatch,
    CanalColumn,
    CanalEntry,
    CanalProtocolBaseline,
    CanalProtocolNotFrozenError,
    CanalRowData,
    CdcAckPendingError,
    CdcBatchCommitRequest,
    CdcBatchCoordinator,
    CdcBatchNotReadyError,
    CdcBatchReceiver,
    CdcBinlogPosition,
    CdcCheckpointCommitResult,
    CdcTaskReceipt,
)
from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.handoff_orchestrator import HandoffOrchestrationResult
from expert_index.models import SourceSyncTask
from expert_index.repositories.cdc_checkpoint import MySQLCdcCheckpointRepository

NOW = datetime(2026, 8, 26, 14, 0, tzinfo=timezone.utc)


def _batch() -> CanalBatch:
    """构造包含一条工作经历 DELETE 的完整事务 batch。"""

    return CanalBatch(
        7,
        (
            CanalEntry("TRANSACTIONBEGIN", "", "", None, "binlog.000012", 10, 1, 1),
            CanalEntry(
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
                        before=(
                            CanalColumn("id", "901"),
                            CanalColumn("expert_id", "101"),
                        )
                    ),
                ),
            ),
            CanalEntry("TRANSACTIONEND", "", "", None, "binlog.000012", 30, 1, 1),
        ),
        CANAL_DESTINATION,
    )


def _baseline(*, ready: bool = True) -> CanalProtocolBaseline:
    """构造显式的协议门禁，不从环境推测。"""

    return CanalProtocolBaseline(
        server_version="1.1.8",
        server_mode="tcp",
        tcp_port=11111,
        destination=CANAL_DESTINATION,
        subscription="expert\\..*",
        source_connected=True,
        probe_batch_id=1,
        probe_entry_count=3,
        ack_called=False,
        production_ready=ready,
        unresolved=() if ready else ("real_ack_replay",),
    )


def _task(*, status: str = "pending") -> SourceSyncTask:
    """构造 CDC Task 在 pending、dead 或 success 的领域状态。"""

    success = status == "success"
    return SourceSyncTask(
        task_id=501,
        dedupe_key="a" * 64,
        batch_id=88,
        event_source="cdc",
        source_table="experts_work_experience",
        source_record_id=901,
        source_updated_at=None,
        expert_id=101,
        operation_type="delete",
        target_type="employment",
        processing_stage="completed" if success else "candidate",
        sync_status=status,
        retry_count=0,
        policy_version="1.3",
        created_at=NOW,
        updated_at=NOW,
        completed_at=NOW if success or status == "dead" else None,
    )


class _Client:
    """记录 fetch/rollback/ack，可模拟 Checkpoint 提交后 ACK 丢失。"""

    def __init__(self, *, ack_error: bool = False) -> None:
        self.ack_error = ack_error
        self.fetch_count = 0
        self.rollbacks: list[int] = []
        self.acks: list[int] = []

    def fetch_without_ack(self, _batch_size: int) -> CanalBatch:
        self.fetch_count += 1
        return _batch()

    def rollback(self, batch_id: int) -> None:
        self.rollbacks.append(batch_id)

    def ack(self, batch_id: int) -> None:
        self.acks.append(batch_id)
        if self.ack_error:
            raise RuntimeError("transport unavailable")


class _Persistence:
    """返回与计划同序的固定 Task 回执。"""

    def persist_batch(self, plans: object) -> tuple[CdcTaskReceipt, ...]:
        return tuple(CdcTaskReceipt(plan.cdc_event_id, 501, "created") for plan in plans)  # type: ignore[union-attr]


class _TaskRepository:
    """模拟精确领取并保留 Processor 写回后的任务状态。"""

    def __init__(self, task: SourceSyncTask) -> None:
        self.task = task
        self.claim_count = 0

    def get_source_sync_task(self, task_id: int) -> SourceSyncTask | None:
        assert task_id == 501
        return self.task

    def claim_cdc_source_sync_task(
        self, task_id: int, *, lease_seconds: int
    ) -> SourceSyncTask | None:
        assert task_id == 501 and lease_seconds == 180
        self.claim_count += 1
        if self.task.sync_status != "pending":
            return None
        self.task = replace(
            self.task,
            sync_status="processing",
            attempt_token="attempt-501",
            lease_until=NOW + timedelta(minutes=3),
        )
        return self.task


class _Processor:
    """模拟 P7.6 成功收敛并落定 Task success。"""

    def __init__(self, repository: _TaskRepository) -> None:
        self.repository = repository

    def process_claimed(
        self, task: SourceSyncTask, attempt_token: str
    ) -> HandoffOrchestrationResult:
        assert task.attempt_token == attempt_token
        self.repository.task = replace(
            task,
            processing_stage="completed",
            sync_status="success",
            attempt_token=None,
            lease_until=None,
            completed_at=NOW,
        )
        return HandoffOrchestrationResult("success", True, "disabled", None)


class _FailingProcessor:
    """模拟下游处理阶段抛出未分类异常。"""

    def process_claimed(
        self, _task: SourceSyncTask, _attempt_token: str
    ) -> HandoffOrchestrationResult:
        raise AttributeError("downstream method missing")


class _CheckpointRepository:
    """记录原子提交请求并返回固定水位。"""

    def __init__(self, *, already_applied: bool = False) -> None:
        self.already_applied = already_applied
        self.requests: list[CdcBatchCommitRequest] = []

    def commit_applied_batch(self, request: CdcBatchCommitRequest) -> CdcCheckpointCommitResult:
        self.requests.append(request)
        return CdcCheckpointCommitResult(
            request.applied_position,
            2,
            self.already_applied,
        )


def _coordinator(
    task_repository: _TaskRepository,
    checkpoint_repository: _CheckpointRepository,
    *,
    ready: bool = True,
) -> CdcBatchCoordinator:
    """组装不连接外部系统的单轮编排器。"""

    return CdcBatchCoordinator(
        protocol_baseline=_baseline(ready=ready),
        receiver=CdcBatchReceiver(
            identity_provider=lambda event: (
                f"{event.logfile_name}:{event.logfile_offset}:{event.row_index}"
            ),
            policy_version="1.3",
        ),
        persistence=_Persistence(),
        task_repository=task_repository,
        task_processor=_Processor(task_repository),
        checkpoint_repository=checkpoint_repository,
        task_lease_seconds=180,
    )


def test_coordinator_processes_then_commits_then_acks() -> None:
    """首次投递必须完成 Task 后提交终点位置并 ACK。"""

    task_repository = _TaskRepository(_task())
    checkpoint_repository = _CheckpointRepository()
    client = _Client()

    result = _coordinator(task_repository, checkpoint_repository).consume_once(
        client,
        batch_size=100,
        local_batch_id=88,
    )

    assert result.status == "acked"
    assert task_repository.task.sync_status == "success"
    assert checkpoint_repository.requests[0].applied_position == CdcBinlogPosition(
        1, "binlog.000012", 30
    )
    assert client.acks == [7]
    assert client.rollbacks == []


def test_dead_task_rolls_back_without_checkpoint_or_ack() -> None:
    """dead Task 不能被跳过，当前 Canal batch 应 rollback。"""

    task_repository = _TaskRepository(_task(status="dead"))
    checkpoint_repository = _CheckpointRepository()
    client = _Client()

    result = _coordinator(task_repository, checkpoint_repository).consume_once(
        client,
        batch_size=100,
        local_batch_id=88,
    )

    assert result.status == "not_ready"
    assert checkpoint_repository.requests == []
    assert client.acks == []
    assert client.rollbacks == [7]


def test_processor_exception_rolls_back_without_checkpoint_or_ack() -> None:
    """下游异常必须回滚当前 batch，不能留下未确认悬挂。"""

    task_repository = _TaskRepository(_task())
    checkpoint_repository = _CheckpointRepository()
    client = _Client()
    coordinator = _coordinator(task_repository, checkpoint_repository)
    coordinator._task_processor = _FailingProcessor()

    with pytest.raises(AttributeError, match="downstream method missing"):
        coordinator.consume_once(client, batch_size=100, local_batch_id=88)

    assert checkpoint_repository.requests == []
    assert client.acks == []
    assert client.rollbacks == [7]


def test_ack_failure_keeps_committed_checkpoint_for_replay() -> None:
    """ACK 失败只报告待重试，不 rollback 已提交业务位点。"""

    task_repository = _TaskRepository(_task(status="success"))
    checkpoint_repository = _CheckpointRepository(already_applied=True)
    client = _Client(ack_error=True)

    with pytest.raises(CdcAckPendingError) as raised:
        _coordinator(task_repository, checkpoint_repository).consume_once(
            client,
            batch_size=100,
            local_batch_id=89,
        )

    assert raised.value.checkpoint.already_applied is True
    assert task_repository.claim_count == 0
    assert client.acks == [7]
    assert client.rollbacks == []


def test_protocol_gate_rejects_before_fetch() -> None:
    """协议未冻结时禁止读取和 ACK，不使用测试默认绕过。"""

    task_repository = _TaskRepository(_task())
    client = _Client()

    with pytest.raises(CanalProtocolNotFrozenError):
        _coordinator(task_repository, _CheckpointRepository(), ready=False).consume_once(
            client,
            batch_size=100,
            local_batch_id=88,
        )

    assert client.fetch_count == 0


class _Cursor:
    """模拟 CDC Checkpoint Repository 的锁定和写入顺序。"""

    def __init__(
        self,
        *,
        checkpoint: dict[str, Any] | None,
        event_rows: list[dict[str, Any]],
        blockers: list[dict[str, Any]],
    ) -> None:
        self.checkpoint = checkpoint
        self.event_rows = event_rows
        self.blockers = blockers
        self.current_one: dict[str, Any] | None = None
        self.current_all: list[dict[str, Any]] = []
        self.rowcount = 0
        self.executed: list[str] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, _params: tuple[object, ...] = ()) -> None:
        normalized = " ".join(sql.split())
        upper = normalized.upper()
        self.executed.append(normalized)
        self.current_one = None
        self.current_all = []
        self.rowcount = 0
        if "FROM EXPERT_INDEX_CDC_CHECKPOINT" in upper and upper.startswith("SELECT"):
            self.current_one = self.checkpoint
        elif "LEFT JOIN EXPERT_INDEX_SYNC_TASK" in upper:
            self.current_all = self.event_rows
        elif "STATUS <> 'SUCCESS'" in upper:
            self.current_all = self.blockers
        elif upper.startswith(
            ("UPDATE EXPERT_INDEX_CDC_EVENT", "INSERT INTO EXPERT_INDEX_CDC_CHECKPOINT")
        ):
            self.rowcount = 1

    def fetchone(self) -> dict[str, Any] | None:
        return self.current_one

    def fetchall(self) -> list[dict[str, Any]]:
        return self.current_all


class _Connection:
    """记录 CDC Checkpoint 事务提交与回滚。"""

    def __init__(self, cursor: _Cursor) -> None:
        self.cursor_obj = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _Cursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        return None


def _commit_request() -> CdcBatchCommitRequest:
    """通过真实 Receiver 构建 Repository 提交请求。"""

    from expert_index.cdc import build_cdc_batch_commit_request

    receiver = CdcBatchReceiver(identity_provider=lambda _event: "event-1", policy_version="1.3")
    received = receiver.receive(_batch(), local_batch_id=88, persistence=_Persistence())
    return build_cdc_batch_commit_request(_batch(), received)


def _event_row(*, status: str = "pending", task_status: str = "success") -> dict[str, Any]:
    """构造持久化 Event 和关联 Task 的最小锁定结果。"""

    return {
        "id": 11,
        "cdc_event_id": "event-1",
        "task_id": 501,
        "status": status,
        "destination": CANAL_DESTINATION,
        "binlog_server_id": 1,
        "binlog_file": "binlog.000012",
        "binlog_position": 20,
        "row_index": 0,
        "task_event_source": "cdc",
        "task_operation_type": "delete",
        "task_processing_stage": "completed" if task_status == "success" else "handed_off",
        "task_sync_status": task_status,
    }


def _index_config() -> DatabaseConfig:
    return DatabaseConfig(
        "127.0.0.1",
        3306,
        "index",
        "",
        "expert_index",
        DatabaseRole.INDEX,
    )


def test_repository_atomically_marks_event_before_initial_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次提交在同一事务先落 Event success，再创建 Checkpoint。"""

    cursor = _Cursor(checkpoint=None, event_rows=[_event_row()], blockers=[_event_row()])
    connection = _Connection(cursor)
    monkeypatch.setattr(
        "expert_index.repositories.cdc_checkpoint.pymysql.connect",
        lambda **_kwargs: connection,
    )

    result = MySQLCdcCheckpointRepository(_index_config()).commit_applied_batch(_commit_request())

    event_update = next(
        index
        for index, sql in enumerate(cursor.executed)
        if sql.upper().startswith("UPDATE EXPERT_INDEX_CDC_EVENT")
    )
    checkpoint_insert = next(
        index
        for index, sql in enumerate(cursor.executed)
        if sql.upper().startswith("INSERT INTO EXPERT_INDEX_CDC_CHECKPOINT")
    )
    assert event_update < checkpoint_insert
    assert result.lock_version == 1 and result.already_applied is False
    assert connection.commits == 1 and connection.rollbacks == 0


def test_repository_rejects_non_success_task_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关联 Task 未 success 时不得写 Event 或 Checkpoint。"""

    cursor = _Cursor(
        checkpoint=None,
        event_rows=[_event_row(task_status="pending")],
        blockers=[],
    )
    connection = _Connection(cursor)
    monkeypatch.setattr(
        "expert_index.repositories.cdc_checkpoint.pymysql.connect",
        lambda **_kwargs: connection,
    )

    with pytest.raises(CdcBatchNotReadyError):
        MySQLCdcCheckpointRepository(_index_config()).commit_applied_batch(_commit_request())

    assert not any(sql.startswith("UPDATE expert_index_cdc_event") for sql in cursor.executed)
    assert connection.commits == 0 and connection.rollbacks == 1


def test_repository_rejects_earlier_unfinished_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当前 batch 全部成功也不能跳过更早未完成 Event。"""

    blocker = {
        "cdc_event_id": "earlier-event",
        "binlog_server_id": 1,
        "binlog_file": "binlog.000012",
        "binlog_position": 15,
    }
    cursor = _Cursor(
        checkpoint=None,
        event_rows=[_event_row()],
        blockers=[_event_row(), blocker],
    )
    connection = _Connection(cursor)
    monkeypatch.setattr(
        "expert_index.repositories.cdc_checkpoint.pymysql.connect",
        lambda **_kwargs: connection,
    )

    with pytest.raises(CdcBatchNotReadyError, match="更早位点"):
        MySQLCdcCheckpointRepository(_index_config()).commit_applied_batch(_commit_request())

    assert connection.commits == 0 and connection.rollbacks == 1


def test_repository_recognizes_checkpoint_before_ack_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """位点和 Event 已成功时返回 already_applied，不重复写业务状态。"""

    checkpoint = {
        "destination": CANAL_DESTINATION,
        "binlog_server_id": 1,
        "binlog_file": "binlog.000012",
        "binlog_position": 30,
        "last_local_batch_id": 88,
        "lock_version": 2,
    }
    cursor = _Cursor(
        checkpoint=checkpoint,
        event_rows=[_event_row(status="success")],
        blockers=[],
    )
    connection = _Connection(cursor)
    monkeypatch.setattr(
        "expert_index.repositories.cdc_checkpoint.pymysql.connect",
        lambda **_kwargs: connection,
    )

    result = MySQLCdcCheckpointRepository(_index_config()).commit_applied_batch(_commit_request())

    assert result.already_applied is True and result.lock_version == 2
    assert not any(sql.startswith("UPDATE expert_index") for sql in cursor.executed)
    assert connection.commits == 1 and connection.rollbacks == 0
