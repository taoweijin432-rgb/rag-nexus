"""P5.7 单来源批次协调器专项测试。

全部使用 Fake（Fake PreparationCoordinator / Fake Orchestrator / Fake
CheckpointManager / Fake SourceStateRepository / Fake TaskRepository），不连真实
MySQL/Milvus/Embedding，不触发任何物理删除。

覆盖用户冻结的最小测试矩阵：

1. 全 success 提交 Checkpoint；
2. claim 固定 limit=1；
3. 新任务走 P4→P5（candidate 阶段经 prepare → orchestrator）；
4. contract_ready/handed_off 恢复（直接经 orchestrator，不重新 prepare）；
5. retryable 停止（final_status=pending → 立即停止，不调 try_commit）；
6. dead 后 Checkpoint 不提交（可继续处理，但 try_commit 抛 NotReady）；
7. task/source lease conflict 停止；
8. 未知结果 fail-closed 停止；
9. 重复提交幂等（BatchSourceScan status=success 直接返回 already_committed）；
10. 不调用物理删除、不 import Milvus/V15 Work Item（结构断言）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from expert_index.models import (
    BatchSourceScan,
    IncrementalImpactScope,
    PreparedChunkSet,
    SourceScanCursor,
    SourceSyncTask,
)
from expert_index.repositories.sync_task import SourceSyncTaskLeaseConflictError
from expert_index.source_batch_coordinator import (
    PreparationFailureHandler,
    SourceBatchCoordinator,
    SourceBatchRunResult,
)
from expert_index.validators import ChunkValidationError

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_TOKEN = "tok-001"


@dataclass
class FakeOrchestratorResult:
    """记录一次 orchestrator.process 调用的入参与返回。"""

    call_task_id: int | None = None
    call_token: str | None = None
    call_stage: str | None = None  # 入参时 task 的 processing_stage
    return_value: object | None = None  # HandoffOrchestrationResult 或异常


class SourceLeaseConflictError(RuntimeError):
    pass


class SourceBatchNotReadyError(RuntimeError):
    pass


class SourceBatchCommitConflictError(RuntimeError):
    pass


class SourceSyncTaskStageConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class HandoffResultStub:
    final_status: str
    terminal: bool
    result_code: str | None = None
    error_stage: str | None = None


@dataclass
class CommitResultStub:
    skipped_count: int
    already_committed: bool = False


@dataclass
class FakePreparationCoordinator:
    """只记录调用并按脚本返回 PreparedChunkSet 或抛异常。"""

    returns: object = None  # PreparedChunkSet 或 Exception 实例
    calls: list = field(default_factory=list)

    def prepare(self, task, attempt_token):
        self.calls.append((task.task_id, task.processing_stage, attempt_token))
        if isinstance(self.returns, BaseException):
            raise self.returns
        return self.returns


@dataclass
class FakeOrchestrator:
    """按 task_id 脚本返回结果或抛异常。"""

    script: dict = field(default_factory=dict)  # task_id -> result 或异常
    calls: list = field(default_factory=list)

    def process(
        self, task_id, attempt_token, prepared=None, *, prepare_if_missing=None,
    ):
        self.calls.append((task_id, attempt_token, prepared is not None))
        entry = self.script.get(task_id)
        if entry is None:
            return HandoffResultStub(final_status="success", terminal=True)
        if isinstance(entry, BaseException):
            raise entry
        return entry


@dataclass
class FakeCheckpointManager:
    """记录 try_commit 调用，按脚本返回结果或抛异常。"""

    return_value: object = None  # CommitResultStub 或 Exception
    raise_on_call: object = None
    calls: list = field(default_factory=list)

    def try_commit_source_batch(self, source_table, batch_id, lease_owner, *, checkpoint_lock_version):
        self.calls.append((source_table, batch_id, lease_owner, checkpoint_lock_version))
        if self.raise_on_call is not None:
            exc = self.raise_on_call
            self.raise_on_call = None
            raise exc
        if isinstance(self.return_value, BaseException):
            raise self.return_value
        return self.return_value


@dataclass
class FakeSourceStateRepository:
    """记录续租/读取，按脚本控制续租是否抛异常。"""

    scan: BatchSourceScan | None = None
    renew_fail_on_call: int | None = None  # 第 N 次 renew 抛异常（1-based）
    renew_calls: list = field(default_factory=list)
    ensure_calls: list = field(default_factory=list)

    def ensure_source_lease(self, source_table, lease_owner, *, lease_seconds):
        self.ensure_calls.append((source_table, lease_owner, lease_seconds))
        # 在测试替身中把首次确认计入续租序列，保持既有冲突脚本的时序语义。
        self.renew_calls.append((source_table, lease_owner, lease_seconds))
        if self.renew_fail_on_call is not None and len(self.renew_calls) == self.renew_fail_on_call:
            raise SourceLeaseConflictError(f"来源表 {source_table} 的扫描 lease 确认失败")

    def renew_source_lease(self, source_table, lease_owner, *, lease_seconds):
        self.renew_calls.append((source_table, lease_owner, lease_seconds))
        if self.renew_fail_on_call is not None and len(self.renew_calls) == self.renew_fail_on_call:
            raise SourceLeaseConflictError(f"来源表 {source_table} 的扫描 lease 续租失败")

    def get_batch_source_scan(self, batch_id, source_table):
        return self.scan


@dataclass
class FakeTaskRepository:
    """按脚本返回 claim 序列。"""

    claim_queue: list = field(default_factory=list)  # 每次返回 tuple 或 []
    calls: list = field(default_factory=list)
    claim_fail_on_call: int | None = None
    claim_exc: object = None

    def claim_source_sync_tasks(self, batch_id, source_table, *, limit, lease_seconds):
        self.calls.append((batch_id, source_table, limit, lease_seconds))
        if self.claim_fail_on_call is not None and len(self.calls) == self.claim_fail_on_call:
            self.claim_fail_on_call = None
            raise self.claim_exc if self.claim_exc is not None else SourceSyncTaskStageConflictError(
                "claim 失败"
            )
        if not self.claim_queue:
            return ()
        return self.claim_queue.pop(0)


@dataclass
class FakePreparationFailureHandler:
    """记录 Coordinator 的 P4 异常落定调用，并返回预设结果。"""

    result: HandoffResultStub
    calls: list = field(default_factory=list)

    def handle(self, task, attempt_token, error):
        self.calls.append((task.task_id, attempt_token, error))
        return self.result


@dataclass
class FakeFailureRepository:
    """只实现失败落定器需要的最小 Repository 接口。"""

    retry_status: str = "pending"
    rejected_calls: list = field(default_factory=list)
    retryable_calls: list = field(default_factory=list)

    def mark_source_sync_task_rejected(self, task_id, attempt_token, **kwargs):
        self.rejected_calls.append((task_id, attempt_token, kwargs))
        return type("UpdatedTask", (), {"sync_status": "dead"})()

    def mark_source_sync_task_retryable(self, task_id, attempt_token, **kwargs):
        self.retryable_calls.append((task_id, attempt_token, kwargs))
        return type("UpdatedTask", (), {"sync_status": self.retry_status})()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 8, 26, 12, 0, 0)  # noqa: DTZ001 -- 测试固定时刻，naive 用于模型校验


def _cursor(record_id=0):
    return SourceScanCursor(updated_at=_NOW, record_id=record_id)


def _scan(*, status="running", candidate_count=2, source_table="experts", batch_id=100):
    return BatchSourceScan(
        batch_id=batch_id,
        source_table=source_table,
        start_cursor=_cursor(0),
        current_cursor=_cursor(5),
        upper_bound=_cursor(10),
        checkpoint_lock_version=7,
        status=status,
        candidate_count=candidate_count,
    )


def _task(*, task_id=1, stage="candidate", source_table="experts", expert_id=1001, source_record_id=1):
    return SourceSyncTask(
        task_id=task_id,
        dedupe_key="d" * 64,
        batch_id=100,
        event_source="poll",
        source_table=source_table,
        source_record_id=source_record_id,
        source_updated_at=None,
        expert_id=expert_id,
        operation_type="update",
        target_type="expert",
        processing_stage=stage,
        sync_status="processing",
        retry_count=0,
        policy_version="1.3",
        created_at=_NOW,
        updated_at=_NOW,
        attempt_token=_TOKEN,
        lease_until=_NOW,
    )


def _prepared():
    return PreparedChunkSet(
        task_id=1,
        batch_id=100,
        data_version="100",
        impact_scope=IncrementalImpactScope(
            scope_type="expert", expert_id=1001, chunk_type=None,
            source_table=None, source_field_id=None,
        ),
        desired_chunks=(),
    )


def _coordinator(
    prep, orch, checkpoint, source_state, task_repo, *,
    task_lease_seconds=120, source_lease_seconds=120, preparation_failure_handler=None,
):
    return SourceBatchCoordinator(
        preparation_coordinator=prep,
        orchestrator=orch,
        checkpoint_manager=checkpoint,
        source_state_repository=source_state,
        task_repository=task_repo,
        task_lease_seconds=task_lease_seconds,
        source_lease_seconds=source_lease_seconds,
        preparation_failure_handler=preparation_failure_handler,
    )


# ---------------------------------------------------------------------------
# 1. 全 success 提交 Checkpoint
# ---------------------------------------------------------------------------

def test_all_success_commits_checkpoint():
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="success", terminal=True),
        2: HandoffResultStub(final_status="success", terminal=True),
    })
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),), (_task(task_id=2),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert isinstance(result, SourceBatchRunResult)
    assert result.checkpoint_committed is True
    assert result.checkpoint_already_committed is False
    assert result.checkpoint_skipped_count == 0
    assert result.success_count == 2
    assert result.claimed_count == 2
    assert result.candidate_count == 2
    assert result.error is None
    assert len(checkpoint.calls) == 1
    assert checkpoint.calls[0] == ("experts", 100, "owner-1", 7)


# ---------------------------------------------------------------------------
# 2. claim 固定 limit=1
# ---------------------------------------------------------------------------

def test_claim_uses_limit_one():
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="success", terminal=True),
    })
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=1))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert all(call[2] == 1 for call in task_repo.calls), "claim 必须固定 limit=1"


# ---------------------------------------------------------------------------
# 3. 新任务走 P4→P5（candidate → prepare → orchestrator）
# ---------------------------------------------------------------------------

def test_new_task_candidate_stage_goes_through_prepare_then_orchestrator():
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="success", terminal=True),
    })
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=1))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1, stage="candidate"),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.success_count == 1
    assert len(prep.calls) == 1
    assert prep.calls[0][0] == 1  # task_id
    assert prep.calls[0][1] == "candidate"  # stage
    assert len(orch.calls) == 1
    assert orch.calls[0][0] == 1


def test_prepare_validation_failure_is_settled_and_does_not_retry_forever():
    """P4 校验失败必须落 dead；后续由 Checkpoint 闸门阻断，但不再重领。"""

    prep = FakePreparationCoordinator(returns=ChunkValidationError("来源正文不应写入状态库"))
    orch = FakeOrchestrator()
    handler = FakePreparationFailureHandler(
        result=HandoffResultStub(final_status="dead", terminal=True)
    )
    checkpoint = FakeCheckpointManager(return_value=SourceBatchNotReadyError("存在 dead Task"))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=1))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),)])

    result = _coordinator(
        prep,
        orch,
        checkpoint,
        source_state,
        task_repo,
        preparation_failure_handler=handler,
    ).run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.rejected_count == 1
    assert result.retryable_count == 0
    assert result.error == "SourceBatchNotReadyError"
    assert len(handler.calls) == 1
    assert orch.calls == []
    assert len(task_repo.calls) == 2  # 领取坏任务后再确认已无更多待处理任务


def test_prepare_transient_failure_is_settled_pending_and_stops_batch():
    """P4 瞬时异常退回 pending，当前批次立即停止且不推进 Checkpoint。"""

    prep = FakePreparationCoordinator(returns=RuntimeError("短暂的来源读取失败"))
    handler = FakePreparationFailureHandler(
        result=HandoffResultStub(final_status="pending", terminal=False)
    )
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),), (_task(task_id=2),)])

    result = _coordinator(
        prep,
        FakeOrchestrator(),
        checkpoint,
        source_state,
        task_repo,
        preparation_failure_handler=handler,
    ).run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.retryable_count == 1
    assert result.rejected_count == 0
    assert result.error == "retryable"
    assert len(task_repo.calls) == 1
    assert checkpoint.calls == []


def test_preparation_failure_handler_writes_safe_validation_evidence():
    """确定性错误写 dead/DLQ 时只携带稳定摘要，不落原始异常正文。"""

    repository = FakeFailureRepository()
    handler = PreparationFailureHandler(
        repository,
        max_attempts=3,
        retry_delay_seconds=30,
    )

    result = handler.handle(_task(), _TOKEN, ChunkValidationError("原始业务文本"))

    assert result.final_status == "dead"
    assert result.terminal is True
    assert len(repository.rejected_calls) == 1
    _, _, kwargs = repository.rejected_calls[0]
    assert kwargs["error_stage"] == "validation"
    assert kwargs["error_code"] == "chunk_validation_failed"
    assert kwargs["safe_error_message"] == "Chunk 校验失败"
    assert "原始业务文本" not in kwargs["safe_error_message"]


def test_preparation_failure_handler_uses_existing_retry_policy():
    """瞬时 P4 异常复用 max_attempts/retry_delay_seconds，不新增重试规则。"""

    repository = FakeFailureRepository(retry_status="pending")
    handler = PreparationFailureHandler(
        repository,
        max_attempts=4,
        retry_delay_seconds=45,
    )

    result = handler.handle(_task(), _TOKEN, RuntimeError("数据库瞬时不可用"))

    assert result.final_status == "pending"
    assert result.terminal is False
    assert len(repository.retryable_calls) == 1
    _, _, kwargs = repository.retryable_calls[0]
    assert kwargs["max_attempts"] == 4
    assert kwargs["retry_delay_seconds"] == 45
    assert kwargs["error_stage"] == "source"


def test_preparation_failure_handler_does_not_overwrite_lease_conflict():
    """旧 attempt 遇到 lease 冲突时只返回 processing，不写死信或重试状态。"""

    repository = FakeFailureRepository()
    handler = PreparationFailureHandler(
        repository,
        max_attempts=4,
        retry_delay_seconds=45,
    )

    result = handler.handle(_task(), _TOKEN, SourceSyncTaskLeaseConflictError("token 已失效"))

    assert result.final_status == "processing"
    assert result.terminal is False
    assert repository.rejected_calls == []
    assert repository.retryable_calls == []


# ---------------------------------------------------------------------------
# 4. contract_ready/handed_off 恢复（直接经 orchestrator，不重新 prepare）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", ["contract_ready", "handed_off", "validated"])
def test_recovery_stage_skips_prepare(stage):
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="success", terminal=True),
    })
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=1))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1, stage=stage),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.success_count == 1
    # 恢复阶段不调用 prepare
    assert prep.calls == []
    # 但调用了 orchestrator
    assert len(orch.calls) == 1
    # Artifact 恢复不再构造占位 PreparedChunkSet。
    assert orch.calls[0][2] is False


# ---------------------------------------------------------------------------
# 5. retryable 停止（final_status=pending → 立即停止，不调 try_commit）
# ---------------------------------------------------------------------------

def test_retryable_stops_without_commit():
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="pending", terminal=False),
    })
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2))
    # 第二条 claim 不应被消费（立即停止）
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),), (_task(task_id=2),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.checkpoint_committed is False
    assert result.retryable_count == 1
    assert result.error == "retryable"
    assert len(checkpoint.calls) == 0, "retryable 不应调用 try_commit"


# ---------------------------------------------------------------------------
# 6. dead 后 Checkpoint 不提交（可继续处理，但 try_commit 抛 NotReady）
# ---------------------------------------------------------------------------

def test_dead_continues_but_checkpoint_not_committed():
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="dead", terminal=True),
        2: HandoffResultStub(final_status="success", terminal=True),
    })
    # try_commit 因 dead 存在抛 NotReady
    checkpoint = FakeCheckpointManager(return_value=SourceBatchNotReadyError("仍有未成功任务"))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),), (_task(task_id=2),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.rejected_count == 1
    assert result.success_count == 1
    assert result.claimed_count == 2
    assert result.checkpoint_committed is False
    assert result.error == "SourceBatchNotReadyError"
    assert len(checkpoint.calls) == 1, "dead 仍尝试 try_commit，由闸门阻断"


# ---------------------------------------------------------------------------
# 7. task/source lease conflict 停止
# ---------------------------------------------------------------------------

def test_task_lease_conflict_stops():
    """orchestrator 返回 processing+terminal=False（fencing/lease 失配）→ 停止。"""
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="processing", terminal=False),
    })
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),), (_task(task_id=2),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.lease_conflict_count == 1
    assert result.error == "task_lease_conflict"
    assert result.checkpoint_committed is False
    assert len(checkpoint.calls) == 0


def test_source_lease_conflict_stops_before_processing():
    """首次续租 source lease 冲突 → 立即停止，不处理任何 Task。"""
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator()
    checkpoint = FakeCheckpointManager()
    # renew_fail_on_call=1 → 首次续租（循环前的首次续租）抛异常
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2), renew_fail_on_call=1)
    task_repo = FakeTaskRepository()

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.claimed_count == 0
    assert result.error == "SourceLeaseConflictError"
    assert result.checkpoint_committed is False
    assert task_repo.calls == [], "source lease 冲突不应 claim 任何 task"
    assert orch.calls == []


def test_source_lease_conflict_mid_batch_stops():
    """处理中途续租 source lease 冲突 → 停止剩余 Task。"""
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="success", terminal=True),
    })
    checkpoint = FakeCheckpointManager()
    # task1 执行前后均续租；第 5 次（task2 前）抛异常。
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2), renew_fail_on_call=5)
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),), (_task(task_id=2),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.success_count == 1
    assert result.claimed_count == 1
    assert result.error == "SourceLeaseConflictError"


# ---------------------------------------------------------------------------
# 8. 未知结果 fail-closed 停止
# ---------------------------------------------------------------------------

def test_unknown_outcome_fail_closed():
    """final_status 非 success/dead/pending/processing → fail-closed 停止。"""
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="weird", terminal=True),
    })
    checkpoint = FakeCheckpointManager()
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),), (_task(task_id=2),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.error == "unknown_outcome"
    assert result.checkpoint_committed is False
    assert len(checkpoint.calls) == 0


def test_contradictory_terminal_fail_closed():
    """success + terminal=False 矛盾 → fail-closed。"""
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="success", terminal=False),
    })
    checkpoint = FakeCheckpointManager()
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=2))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),), (_task(task_id=2),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.error == "unknown_outcome"


@pytest.mark.parametrize("stage", ["source_read", "normalized", "chunk_built"])
def test_p4_intermediate_stage_rebuilds_current_facts(stage):
    """Artifact 前中间阶段由 P4 重建后继续交付，不再永久卡住。"""

    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator()
    coord = _coordinator(
        prep,
        orch,
        FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0)),
        FakeSourceStateRepository(scan=_scan(candidate_count=1)),
        FakeTaskRepository(claim_queue=[(_task(task_id=1, stage=stage),)]),
    )

    result = coord.run_claimed_source_batch(
        "experts", 100, "owner-1", checkpoint_lock_version=7,
    )

    assert result.success_count == 1
    assert prep.calls == [(1, stage, "tok-001")]


# ---------------------------------------------------------------------------
# 9. 重复提交幂等
# ---------------------------------------------------------------------------

def test_already_committed_batch_source_returns_idempotent():
    """BatchSourceScan status=success → 直接返回 already_committed，不 claim/不 try_commit。"""
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator()
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(status="success", candidate_count=2))
    task_repo = FakeTaskRepository()

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.checkpoint_committed is False
    assert result.checkpoint_already_committed is True
    assert task_repo.calls == [], "已提交批次不应 claim"
    assert checkpoint.calls == [], "已提交批次不应调 try_commit"
    assert orch.calls == []


def test_repeat_run_after_commit_is_idempotent():
    """第一次提交成功后，第二次重入无 pending Task → try_commit 幂等命中 already_committed。"""
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="success", terminal=True),
    })
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0, already_committed=True))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=1))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.checkpoint_committed is False  # already_committed=True → committed=False
    assert result.checkpoint_already_committed is True


# ---------------------------------------------------------------------------
# 10. 结构断言：不调用物理删除、不 import Milvus/V15 Work Item
# ---------------------------------------------------------------------------

def test_source_batch_coordinator_source_has_no_milvus_or_v15_imports():
    """source_batch_coordinator.py 不得 import Milvus/pymilvus/V15 Work Item/物理删除。"""
    import inspect

    from expert_index import source_batch_coordinator

    source = inspect.getsource(source_batch_coordinator)
    forbidden = [
        "pymilvus",
        "MilvusClient",
        "V15Vectorization",
        "v15_vectorization",
        "delete_by_vector_doc_ids",
        "delete_vectors",
        "disable_vectors",
    ]
    for token in forbidden:
        assert token not in source, f"source_batch_coordinator 不得包含 {token!r}"


def test_source_batch_coordinator_does_not_modify_checkpoint_directly():
    """Coordinator 只调用 try_commit_source_batch，不自行写 Checkpoint/Batch/计数。"""
    # 通过检查 FakeCheckpointManager 只被调用一次（全 success 场景）且没有额外写方法
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator(script={
        1: HandoffResultStub(final_status="success", terminal=True),
    })
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=1))
    task_repo = FakeTaskRepository(claim_queue=[(_task(task_id=1),)])

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    # 只调用 try_commit_source_batch 一次，不调用任何其他写方法
    assert len(checkpoint.calls) == 1


# ---------------------------------------------------------------------------
# 边界：无 pending Task（claim 返回空）→ 直接进入闸门提交
# ---------------------------------------------------------------------------

def test_no_pending_tasks_still_attempts_commit():
    """claim 立即返回空 → 全部 Task 已 success → 进入闸门提交。"""
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator()
    checkpoint = FakeCheckpointManager(return_value=CommitResultStub(skipped_count=0))
    source_state = FakeSourceStateRepository(scan=_scan(candidate_count=0))
    task_repo = FakeTaskRepository(claim_queue=[])  # 永远返回空

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    result = coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)

    assert result.claimed_count == 0
    assert result.checkpoint_committed is True
    assert len(checkpoint.calls) == 1


# ---------------------------------------------------------------------------
# 边界：BatchSourceScan 不存在 → ValueError
# ---------------------------------------------------------------------------

def test_missing_batch_source_scan_raises():
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator()
    checkpoint = FakeCheckpointManager()
    source_state = FakeSourceStateRepository(scan=None)
    task_repo = FakeTaskRepository()

    coord = _coordinator(prep, orch, checkpoint, source_state, task_repo)
    with pytest.raises(ValueError, match="Batch Source 不存在"):
        coord.run_claimed_source_batch("experts", 100, "owner-1", checkpoint_lock_version=7)


def test_task_claim_limit_must_remain_one():
    """串行协调器必须在访问 Repository 前拒绝多条领取。"""

    coord = _coordinator(
        FakePreparationCoordinator(returns=_prepared()),
        FakeOrchestrator(),
        FakeCheckpointManager(),
        FakeSourceStateRepository(scan=_scan()),
        FakeTaskRepository(),
    )

    with pytest.raises(ValueError, match="必须固定为 1"):
        coord.run_claimed_source_batch(
            "experts", 100, "owner-1", checkpoint_lock_version=7,
            task_claim_limit=2,
        )


# ---------------------------------------------------------------------------
# 边界：非法 lease_seconds
# ---------------------------------------------------------------------------

def test_invalid_lease_seconds_rejected():
    prep = FakePreparationCoordinator(returns=_prepared())
    orch = FakeOrchestrator()
    checkpoint = FakeCheckpointManager()
    source_state = FakeSourceStateRepository()
    task_repo = FakeTaskRepository()

    with pytest.raises(ValueError, match="task_lease_seconds"):
        SourceBatchCoordinator(
            preparation_coordinator=prep,
            orchestrator=orch,
            checkpoint_manager=checkpoint,
            source_state_repository=source_state,
            task_repository=task_repo,
            task_lease_seconds=0,
            source_lease_seconds=120,
        )
    with pytest.raises(ValueError, match="source_lease_seconds"):
        SourceBatchCoordinator(
            preparation_coordinator=prep,
            orchestrator=orch,
            checkpoint_manager=checkpoint,
            source_state_repository=source_state,
            task_repository=task_repo,
            task_lease_seconds=120,
            source_lease_seconds=-1,
        )
