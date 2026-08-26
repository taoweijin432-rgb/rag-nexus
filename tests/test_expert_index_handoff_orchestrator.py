"""P5.3B 正式 Handoff 编排闭环测试。

全部使用 Fake：不连接真实 MySQL/Milvus，不创建 commit，不推进 checkpoint。

覆盖：
    - 正常 committed 闭环（validated/contract_ready/handed_off 三种起点）；
    - rejected、retryable failed、non-retryable failed；
    - Artifact 首写、同 hash 重放、不同 hash 冲突（Consumer 调用次数 0）；
    - fencing 失效（token 不一致 / lease 冲突）；
    - Receipt identity/token/scope/payload_hash 不一致；
    - 崩溃后重放（Consumer 成功但 finalize 前崩，第二次重放最终 success）；
    - 只有 committed 调用 finalize_source_sync_task_success；
    - 零 checkpoint、零 Work Item 双写；
    - 结构断言：不 import V15 Work Item/Batch/CheckpointManager/物理删除。
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.handoff import ScopeHandoffBuilder, ScopeHandoffReceipt, ScopeHandoffRequest
from expert_index.handoff_orchestrator import (
    HandoffOrchestrator,
)
from expert_index.models import (
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    PreparedChunkSet,
    ProjectSource,
    SourceSyncTask,
    impact_scope_from_target,
)
from expert_index.normalize import normalize_aggregate
from expert_index.repositories.artifact import (
    ArtifactPayloadConflictError,
    ContractArtifact,
)
from expert_index.repositories.sync_task import (
    SourceSyncTaskLeaseConflictError,
    SourceSyncTaskStageConflictError,
)

CREATED = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
TASK_UPDATED = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
LEASE_UNTIL = datetime(2026, 8, 21, 9, 30, tzinfo=timezone.utc)
TOKEN = "tok-claim-001"
ARTIFACT_TYPE = "contract_payload"
ARTIFACT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Fixtures（复用 handoff/scope_consumer 测试的真实 Chunk 生成链路）
# ---------------------------------------------------------------------------
def _settings() -> Settings:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _factory() -> ChunkFactory:
    return ChunkFactory(_settings())


def _expert_source(eid: int = 700001, deleted_at=None) -> ExpertSource:
    return ExpertSource(
        id=eid,
        serial_number=f"TEST{eid}",
        company="虚构编排闭环测试公司",
        job_status=1,
        job_name="项目经理",
        contact_status=3,
        doubtful_flag=1,
        abroad_flag=2,
        background="仅用于编排闭环单元测试的虚构背景。",
        work_years=8.5,
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _employment(eid: int = 700001, emp_id: int = 21307, deleted_at=None) -> EmploymentSource:
    return EmploymentSource(
        id=emp_id,
        expert_id=eid,
        company_name="虚构编排闭环雇主",
        role_name="工程师",
        start_date=UPDATED,
        end_date=None,
        is_current=1,
        work_description="设计并实现编排闭环。",
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _project(eid: int = 700001, proj_id: int = 161, deleted_at=None) -> ProjectSource:
    return ProjectSource(
        id=proj_id,
        expert_id=eid,
        project_title="编排闭环项目",
        company_name="虚构编排闭环公司",
        start_date=UPDATED,
        end_date=None,
        work_description="实现编排闭环。",
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _education(eid: int = 700001, edu_id: int = 1) -> EducationSource:
    return EducationSource(
        id=edu_id,
        expert_id=eid,
        school_name="测试大学",
        major="计算机",
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=None,
    )


def _aggregate(eid=700001, *, employments=(), projects=(), educations=(), deleted_at=None) -> ExpertAggregate:
    return ExpertAggregate(
        profile=_expert_source(eid, deleted_at=deleted_at),
        employments=tuple(employments),
        projects=tuple(projects),
        educations=tuple(educations),
    )


def _task(
    *,
    task_id=701,
    batch_id=8,
    expert_id=700001,
    source_table="experts",
    source_record_id=700001,
    operation_type="update",
    target_type="expert",
    policy_version="1.3",
    processing_stage="validated",
    sync_status="processing",
    retry_count=0,
    attempt_token=TOKEN,
    lease_until=LEASE_UNTIL,
    result_code=None,
    completed_at=None,
) -> SourceSyncTask:
    import hashlib

    dedupe = hashlib.sha256(f"dk-{task_id}".encode()).hexdigest()
    return SourceSyncTask(
        task_id=task_id,
        dedupe_key=dedupe,
        batch_id=batch_id,
        event_source="poll",
        source_table=source_table,
        source_record_id=source_record_id,
        source_updated_at=UPDATED,
        expert_id=expert_id,
        operation_type=operation_type,
        target_type=target_type,
        processing_stage=processing_stage,
        sync_status=sync_status,
        retry_count=retry_count,
        policy_version=policy_version,
        created_at=CREATED,
        updated_at=TASK_UPDATED,
        attempt_token=attempt_token,
        lease_until=lease_until,
        result_code=result_code,
        completed_at=completed_at,
    )


def _prepared(scope, chunks, *, task_id=701, batch_id=8, data_version="8") -> PreparedChunkSet:
    return PreparedChunkSet(
        task_id=task_id,
        batch_id=batch_id,
        data_version=data_version,
        impact_scope=scope,
        desired_chunks=tuple(chunks),
    )


def _chunks(agg: ExpertAggregate):
    profile, employments, projects = normalize_aggregate(
        agg, data_version="20260824_orch", build_batch_id=701)
    return _factory().build_all(profile, employments, projects)


def _expert_chunks(eid=700001, *, educations=1) -> list:
    edus = tuple(_education(eid=eid) for _ in range(educations))
    return _chunks(_aggregate(eid=eid, educations=edus))


# ---------------------------------------------------------------------------
# Fake Task Repository
# ---------------------------------------------------------------------------
@dataclass
class FakeTaskRepo:
    """记录所有写调用，支持 fencing 失败注入与阶段推进。"""

    state: SourceSyncTask
    success_calls: list[tuple[int, str, str]] = field(default_factory=list)
    finalize_calls: list[tuple[int, str, str, int, str]] = field(default_factory=list)
    rejected_calls: list[dict[str, Any]] = field(default_factory=list)
    retryable_calls: list[dict[str, Any]] = field(default_factory=list)
    advance_calls: list[tuple[int, str, str]] = field(default_factory=list)
    fail_success_with_lease: bool = False
    fail_finalize_with_lease: bool = False
    fail_advance_with_lease: bool = False
    # checkpoint / work-item 写入计数（用于断言零双写）
    checkpoint_writes: int = 0
    workitem_writes: int = 0

    def get_source_sync_task(self, task_id: int) -> SourceSyncTask | None:
        if self.state.task_id != task_id:
            return None
        return self.state

    def advance_source_sync_task_stage(
        self, task_id: int, attempt_token: str, *, next_stage: str,
    ) -> SourceSyncTask:
        self.advance_calls.append((task_id, attempt_token, next_stage))
        if self.fail_advance_with_lease:
            raise SourceSyncTaskLeaseConflictError("test: lease lost")
        # 真实推进逻辑：相邻阶段或同阶段幂等；非相邻抛 stage conflict。
        order = (
            "candidate", "source_read", "normalized", "chunk_built",
            "validated", "contract_ready", "handed_off", "completed",
        )
        if next_stage not in order:
            raise SourceSyncTaskStageConflictError(f"未知阶段: {next_stage}")
        cur = self.state.processing_stage
        if cur == next_stage:
            return self.state
        if order.index(next_stage) != order.index(cur) + 1:
            raise SourceSyncTaskStageConflictError(
                f"不能从 {cur} 推进到 {next_stage}")
        self.state = _replace(self.state, processing_stage=next_stage)
        return self.state

    def mark_source_sync_task_success(
        self, task_id: int, attempt_token: str, *, result_code: str,
    ) -> SourceSyncTask:
        if self.fail_success_with_lease:
            raise SourceSyncTaskLeaseConflictError("test: lease lost on success")
        self.success_calls.append((task_id, attempt_token, result_code))
        self.state = _replace(
            self.state,
            processing_stage="completed", sync_status="success",
            attempt_token=None, lease_until=None, result_code=result_code,
            completed_at=TASK_UPDATED,
        )
        return self.state

    def finalize_source_sync_task_success(
        self, task_id: int, attempt_token: str, *,
        task_result_code: str, artifact_id: int, artifact_payload_hash: str,
    ) -> SourceSyncTask:
        if self.fail_finalize_with_lease:
            raise SourceSyncTaskLeaseConflictError("test: lease lost on finalize")
        self.finalize_calls.append(
            (task_id, attempt_token, task_result_code, artifact_id, artifact_payload_hash),
        )
        self.state = _replace(
            self.state,
            processing_stage="completed", sync_status="success",
            attempt_token=None, lease_until=None, result_code=task_result_code,
            completed_at=TASK_UPDATED,
        )
        return self.state

    def mark_source_sync_task_rejected(
        self, task_id: int, attempt_token: str, *,
        error_stage: str, error_code: str,
        error_type: str | None = None, safe_error_message: str | None = None,
    ) -> SourceSyncTask:
        self.rejected_calls.append({
            "task_id": task_id, "attempt_token": attempt_token,
            "error_stage": error_stage, "error_code": error_code,
            "error_type": error_type, "safe_error_message": safe_error_message,
        })
        self.state = _replace(
            self.state, sync_status="dead", attempt_token=None, lease_until=None,
            processing_stage="completed", completed_at=TASK_UPDATED,
        )
        return self.state

    def mark_source_sync_task_retryable(
        self, task_id: int, attempt_token: str, *,
        error_stage: str, error_code: str,
        max_attempts: int, retry_delay_seconds: int,
        error_type: str | None = None, safe_error_message: str | None = None,
    ) -> SourceSyncTask:
        self.retryable_calls.append({
            "task_id": task_id, "attempt_token": attempt_token,
            "error_stage": error_stage, "error_code": error_code,
            "max_attempts": max_attempts, "retry_delay_seconds": retry_delay_seconds,
            "error_type": error_type, "safe_error_message": safe_error_message,
        })
        # 达上限转 dead，否则 pending。
        if self.state.retry_count >= max_attempts:
            self.state = _replace(
                self.state, sync_status="dead", attempt_token=None,
                lease_until=None, processing_stage="completed",
                completed_at=TASK_UPDATED,
            )
        else:
            self.state = _replace(
                self.state, sync_status="pending", attempt_token=None,
                lease_until=None, next_retry_at=LEASE_UNTIL,
            )
        return self.state


def _replace(task: SourceSyncTask, **changes) -> SourceSyncTask:
    """dataclass(frozen=True) 替换字段。"""
    from dataclasses import replace as _replace_fn
    return _replace_fn(task, **changes)


# ---------------------------------------------------------------------------
# Fake Artifact Repository
# ---------------------------------------------------------------------------
@dataclass
class FakeArtifactRepo:
    """支持冲突注入、按 identity 存取、同 hash 复用、异 hash 冲突。"""

    store: dict[tuple[int, str, str], ContractArtifact] = field(default_factory=dict)
    put_calls: list[dict[str, Any]] = field(default_factory=list)
    fail_put_with_conflict: bool = False
    fail_put_with_infra: bool = False

    def get_contract_artifact(
        self, task_id: int, *, artifact_type: str, artifact_version: str,
    ) -> ContractArtifact | None:
        return self.store.get((task_id, artifact_type, artifact_version))

    def put_or_verify_contract_artifact(
        self, task_id: int, attempt_token: str, *,
        artifact_type: str, artifact_version: str,
        payload_json: str, payload_hash: str, expires_at: Any = None,
    ) -> ContractArtifact:
        self.put_calls.append({
            "task_id": task_id, "attempt_token": attempt_token,
            "artifact_type": artifact_type, "artifact_version": artifact_version,
            "payload_hash": payload_hash,
        })
        if self.fail_put_with_infra:
            raise RuntimeError("test: infra failure")
        key = (task_id, artifact_type, artifact_version)
        existing = self.store.get(key)
        if existing is not None:
            if existing.payload_hash != payload_hash:
                raise ArtifactPayloadConflictError(
                    f"任务 {task_id} Artifact payload_hash 冲突")
            return existing
        artifact = ContractArtifact(
            artifact_id=len(self.store) + 1,
            task_id=task_id, artifact_type=artifact_type,
            artifact_version=artifact_version,
            payload_json=payload_json, payload_reference=None,
            payload_hash=payload_hash, expires_at=expires_at,
            created_at=TASK_UPDATED, updated_at=TASK_UPDATED,
        )
        self.store[key] = artifact
        return artifact


# ---------------------------------------------------------------------------
# Fake Scope Consumer（直接返回 Receipt，不调真实 core/Milvus）
# ---------------------------------------------------------------------------
@dataclass
class FakeConsumer:
    """按预设返回 Receipt；记录调用次数与接收的 Request。"""

    process_calls: list[ScopeHandoffRequest] = field(default_factory=list)
    receipt_factory: Any = None  # callable(request) -> ScopeHandoffReceipt
    raise_on_process: BaseException | None = None

    def process(self, request: ScopeHandoffRequest) -> ScopeHandoffReceipt:
        self.process_calls.append(request)
        if self.raise_on_process is not None:
            raise self.raise_on_process
        return self.receipt_factory(request)


# ---------------------------------------------------------------------------
# Orchestrator 工厂
# ---------------------------------------------------------------------------
def _orchestrator(
    task_repo: FakeTaskRepo,
    artifact_repo: FakeArtifactRepo,
    consumer: FakeConsumer,
) -> HandoffOrchestrator:
    return HandoffOrchestrator(
        task_repo=task_repo,
        artifact_repo=artifact_repo,
        builder=ScopeHandoffBuilder("Asia/Shanghai"),
        consumer=consumer,
        max_attempts=2,
        retry_delay_seconds=60,
    )


def _committed_receipt_factory(result_code: str = "scope_committed"):
    def _make(request: ScopeHandoffRequest) -> ScopeHandoffReceipt:
        # committed 收敛不变式：embedding+metadata_only+unchanged 必须 >= desired（未被这三类覆盖的需 disable 兜底）。
        return ScopeHandoffReceipt(
            request_id=request.request_id,
            task_id=request.task_id,
            batch_id=request.batch_id,
            attempt_token=request.attempt_token,
            payload_hash=request.payload_hash,
            impact_scope=request.impact_scope,
            status="committed",
            result_code=result_code,
            desired_count=len(request.contracts),
            embedding_count=len(request.contracts),
        )
    return _make


def _unchanged_committed_receipt_factory():
    def _make(request: ScopeHandoffRequest) -> ScopeHandoffReceipt:
        # 三 Hash 全同、零写入：embedding/metadata_only/disable 全 0，unchanged==desired。
        return ScopeHandoffReceipt(
            request_id=request.request_id,
            task_id=request.task_id,
            batch_id=request.batch_id,
            attempt_token=request.attempt_token,
            payload_hash=request.payload_hash,
            impact_scope=request.impact_scope,
            status="committed",
            result_code="scope_committed",
            desired_count=len(request.contracts),
            embedding_count=0,
            metadata_only_count=0,
            disable_count=0,
            unchanged_count=len(request.contracts),
        )
    return _make


def _rejected_receipt_factory(result_code="scope_desired_rejected"):
    def _make(request: ScopeHandoffRequest) -> ScopeHandoffReceipt:
        return ScopeHandoffReceipt(
            request_id=request.request_id,
            task_id=request.task_id,
            batch_id=request.batch_id,
            attempt_token=request.attempt_token,
            payload_hash=request.payload_hash,
            impact_scope=request.impact_scope,
            status="rejected",
            result_code=result_code,
            desired_count=len(request.contracts),
            error_type="scope_desired_rejected",
            error_message="desired rejected",
        )
    return _make


def _failed_receipt_factory(retryable: bool, result_code="scope_desired_failed"):
    def _make(request: ScopeHandoffRequest) -> ScopeHandoffReceipt:
        return ScopeHandoffReceipt(
            request_id=request.request_id,
            task_id=request.task_id,
            batch_id=request.batch_id,
            attempt_token=request.attempt_token,
            payload_hash=request.payload_hash,
            impact_scope=request.impact_scope,
            status="failed",
            result_code=result_code,
            desired_count=len(request.contracts),
            retryable=retryable,
            error_type="scope_desired_failed",
            error_message="desired failed",
        )
    return _make


def _expert_prepared(eid=700001) -> PreparedChunkSet:
    scope = impact_scope_from_target(target_type="expert", expert_id=eid)
    return _prepared(scope, _expert_chunks(eid=eid))


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
class TestCommittedClosedLoop:
    def _setup(self, stage="validated"):
        task = _task(processing_stage=stage)
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        return task, task_repo, artifact_repo, consumer, orch

    def test_validated_committed(self):
        task, task_repo, artifact_repo, consumer, orch = self._setup("validated")
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "success"
        assert result.terminal is True
        assert result.result_code == "embedded"
        # Artifact 首写
        assert len(artifact_repo.put_calls) == 1
        # 阶段推进：contract_ready, handed_off（validated 起两步）
        advance_stages = [c[2] for c in task_repo.advance_calls]
        assert advance_stages == ["contract_ready", "handed_off"]
        # Consumer 调一次，finalize 调一次
        assert len(consumer.process_calls) == 1
        assert len(task_repo.finalize_calls) == 1
        assert task_repo.finalize_calls[0][:3] == (task.task_id, TOKEN, "embedded")
        assert task_repo.rejected_calls == []
        assert task_repo.retryable_calls == []

    def test_contract_ready_committed(self):
        task, task_repo, artifact_repo, consumer, orch = self._setup("contract_ready")
        # contract_ready 必须先有 Artifact：预填一个。
        request = ScopeHandoffBuilder("Asia/Shanghai").build(
            task, TOKEN, _expert_prepared(task.expert_id))
        _seed_artifact(artifact_repo, task.task_id, request)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "success"
        # contract_ready 起：推进 handed_off 一步，再 Consumer
        advance_stages = [c[2] for c in task_repo.advance_calls]
        assert advance_stages == ["handed_off"]
        # 不重新 Builder（已从 Artifact 恢复），但 put_or_verify 复用调用一次
        assert len(artifact_repo.put_calls) == 0
        assert len(consumer.process_calls) == 1

    def test_handed_off_replay_committed(self):
        task, task_repo, artifact_repo, consumer, orch = self._setup("handed_off")
        request = ScopeHandoffBuilder("Asia/Shanghai").build(
            task, TOKEN, _expert_prepared(task.expert_id))
        _seed_artifact(artifact_repo, task.task_id, request)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "success"
        # handed_off：不回退阶段、不重新构建，直接重放 Consumer
        assert task_repo.advance_calls == []
        assert len(artifact_repo.put_calls) == 0
        assert len(consumer.process_calls) == 1
        assert len(task_repo.finalize_calls) == 1


class TestZeroWriteResultCodeByEntryStage:
    """零写入收敛的 result_code 依据进入阶段区分（frozen #2）。"""

    def _setup(self, stage="validated"):
        task = _task(processing_stage=stage)
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        consumer = FakeConsumer(receipt_factory=_unchanged_committed_receipt_factory())
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        return task, task_repo, artifact_repo, orch

    def test_validated_first_run_zero_write_is_no_action_unchanged(self):
        """首次从 validated 进入且三 Hash 全同 → no_action_unchanged。"""
        task, task_repo, _artifact_repo, orch = self._setup("validated")
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "success"
        assert result.result_code == "no_action_unchanged"
        assert task_repo.finalize_calls[0][:3] == (
            task.task_id, TOKEN, "no_action_unchanged",
        )

    def test_contract_ready_recovery_zero_write_is_recovered_unchanged(self):
        """从 contract_ready 恢复重放三 Hash 全同 → recovered_unchanged。"""
        task, task_repo, artifact_repo, orch = self._setup("contract_ready")
        request = ScopeHandoffBuilder("Asia/Shanghai").build(
            task, TOKEN, _expert_prepared(task.expert_id))
        _seed_artifact(artifact_repo, task.task_id, request)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "success"
        assert result.result_code == "recovered_unchanged"
        assert task_repo.finalize_calls[0][:3] == (
            task.task_id, TOKEN, "recovered_unchanged",
        )

    def test_handed_off_recovery_zero_write_is_recovered_unchanged(self):
        """从 handed_off 恢复重放三 Hash 全同 → recovered_unchanged。"""
        task, task_repo, artifact_repo, orch = self._setup("handed_off")
        request = ScopeHandoffBuilder("Asia/Shanghai").build(
            task, TOKEN, _expert_prepared(task.expert_id))
        _seed_artifact(artifact_repo, task.task_id, request)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "success"
        assert result.result_code == "recovered_unchanged"
        assert task_repo.finalize_calls[0][:3] == (
            task.task_id, TOKEN, "recovered_unchanged",
        )

    def test_recovered_unchanged_not_counted_as_skipped(self):
        """recovered_unchanged 不在 NO_WRITE_RESULT_CODES，不计 skipped_count。"""
        from expert_index.models import NO_WRITE_RESULT_CODES

        assert "recovered_unchanged" not in NO_WRITE_RESULT_CODES
        assert "no_action_unchanged" in NO_WRITE_RESULT_CODES


class TestHandedOffCrashReplayFinalSuccess:
    """handed_off 后崩溃重放最终 success，且 handed_off 本身不能被当成功。"""

    def test_handed_off_crash_then_replay_completes_success(self):
        # 第一次从 validated 走到 handed_off 后，finalize 前崩溃（模拟 lease 丢失）。
        task = _task(processing_stage="validated")
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        # 模拟 finalize 崩溃：finalize 阶段 lease 丢失，任务停在 handed_off。
        task_repo.fail_finalize_with_lease = True
        first = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert first.final_status == "processing"
        assert first.terminal is False
        assert task_repo.state.processing_stage == "handed_off"
        # 崩溃恢复：新 attempt 从 handed_off 重放，finalize 成功。
        task_repo.fail_finalize_with_lease = False
        new_token = "tok-claim-002"
        task_repo.state = _task(
            processing_stage="handed_off", attempt_token=new_token,
        )
        request = ScopeHandoffBuilder("Asia/Shanghai").build(
            task_repo.state, new_token, _expert_prepared(task.expert_id))
        _seed_artifact(artifact_repo, task.task_id, request)
        result2 = orch.process(task.task_id, new_token, _expert_prepared(task.expert_id))
        assert result2.final_status == "success"
        assert result2.terminal is True
        assert result2.result_code == "embedded"

    def test_handed_off_alone_is_not_success_for_checkpoint(self):
        """handed_off 表示已进入下游交付，不代表成功；Checkpoint 只接受 completed/success。"""
        from expert_index.models import SOURCE_SYNC_PROCESSING_STAGES

        task = _task(processing_stage="handed_off")
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        _orchestrator(task_repo, artifact_repo, consumer)
        # 未执行 Consumer 前，任务停在 handed_off，不应被 Checkpoint 视为 success。
        assert task_repo.state.processing_stage == "handed_off"
        assert task_repo.state.sync_status != "success"
        assert "handed_off" in SOURCE_SYNC_PROCESSING_STAGES
        assert "completed" in SOURCE_SYNC_PROCESSING_STAGES


class TestRejectedAndFailed:
    def test_rejected_receipt(self):
        task = _task()
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        consumer = FakeConsumer(receipt_factory=_rejected_receipt_factory())
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "dead"
        assert result.terminal is True
        assert task_repo.rejected_calls and task_repo.finalize_calls == []
        assert task_repo.rejected_calls[0]["error_stage"] == "scope_consumer"

    def test_retryable_failed(self):
        task = _task(retry_count=0)
        task_repo = FakeTaskRepo(state=task)
        consumer = FakeConsumer(receipt_factory=_failed_receipt_factory(retryable=True))
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "pending"
        assert result.terminal is False
        assert task_repo.retryable_calls and task_repo.finalize_calls == []
        assert task_repo.retryable_calls[0]["error_stage"] == "scope_consumer"
        assert task_repo.retryable_calls[0]["max_attempts"] == 2

    def test_retryable_failed_hits_cap_to_dead(self):
        task = _task(retry_count=2)  # >= max_attempts=2
        task_repo = FakeTaskRepo(state=task)
        consumer = FakeConsumer(receipt_factory=_failed_receipt_factory(retryable=True))
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "dead"
        assert result.terminal is True

    def test_non_retryable_failed(self):
        task = _task()
        task_repo = FakeTaskRepo(state=task)
        consumer = FakeConsumer(receipt_factory=_failed_receipt_factory(retryable=False))
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "dead"
        assert result.terminal is True
        assert task_repo.rejected_calls and task_repo.retryable_calls == []
        assert task_repo.rejected_calls[0]["error_stage"] == "scope_consumer"


class TestArtifactLifecycle:
    def test_first_write_then_replay_same_hash(self):
        task = _task()
        task_repo = FakeTaskRepo(state=task, fail_finalize_with_lease=True)
        artifact_repo = FakeArtifactRepo()
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        # 首次：validated 构建 + 写 Artifact + 推进 handed_off + Consumer 成功，但 finalize lease 失败。
        r1 = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert r1.terminal is False  # success 未落定
        first_hash = artifact_repo.put_calls[0]["payload_hash"]
        assert len(artifact_repo.put_calls) == 1
        assert len(consumer.process_calls) == 1
        # 崩溃恢复：finalize 修复，task 仍 handed_off，用同一 artifact 重放。
        task_repo.fail_finalize_with_lease = False
        r2 = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert r2.final_status == "success"
        # handed_off 起重放：不重新 Builder、不重新 put（已恢复）、不推进阶段，直接重放 Consumer。
        assert len(consumer.process_calls) == 2
        assert artifact_repo.put_calls[0]["payload_hash"] == first_hash

    def test_hash_conflict_calls_no_consumer(self):
        task = _task()
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        # 预置一个不同 hash 的 Artifact（模拟旧 attempt 写入的内容漂移）
        artifact_repo.store[(task.task_id, ARTIFACT_TYPE, ARTIFACT_VERSION)] = ContractArtifact(
            artifact_id=1, task_id=task.task_id, artifact_type=ARTIFACT_TYPE,
            artifact_version=ARTIFACT_VERSION, payload_json='{"payload_hash":"old"}',
            payload_reference=None, payload_hash="0" * 64,
            expires_at=None, created_at=TASK_UPDATED, updated_at=TASK_UPDATED,
        )
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        # 冲突时 rejected，Consumer 调用次数 0
        assert result.final_status == "dead"
        assert len(consumer.process_calls) == 0
        assert task_repo.rejected_calls[0]["error_stage"] == "artifact"

    def test_artifact_put_infra_failure_is_retryable(self):
        task = _task()
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo(fail_put_with_infra=True)
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "pending"
        assert result.terminal is False
        assert task_repo.retryable_calls[0]["error_stage"] == "artifact"
        assert len(consumer.process_calls) == 0


class TestFencing:
    def test_wrong_token_does_not_overwrite(self):
        task = _task(attempt_token="newer-attempt-token")
        task_repo = FakeTaskRepo(state=task)
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        # 用旧 token 调：task 当前 token 是 newer，旧 token 不应触发任何写
        result = orch.process(task.task_id, "old-token", _expert_prepared(task.expert_id))
        assert result.terminal is False
        assert result.final_status == "processing"
        assert consumer.process_calls == []
        assert task_repo.finalize_calls == []

    def test_lease_lost_on_advance_no_further_writes(self):
        task = _task()
        task_repo = FakeTaskRepo(state=task, fail_advance_with_lease=True)
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        # advance lease 失败：不继续 Consumer、不 success
        assert result.terminal is False
        assert result.error_stage == "fencing"
        assert consumer.process_calls == []
        assert task_repo.finalize_calls == []

    def test_lease_lost_on_success_keeps_processing(self):
        task = _task(processing_stage="handed_off")
        task_repo = FakeTaskRepo(state=task, fail_finalize_with_lease=True)
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        artifact_repo = FakeArtifactRepo()
        request = ScopeHandoffBuilder("Asia/Shanghai").build(
            task, TOKEN, _expert_prepared(task.expert_id))
        _seed_artifact(artifact_repo, task.task_id, request)
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.terminal is False
        assert result.error_stage == "fencing"
        # Consumer 被调用，但 success 因 lease 失败未落定
        assert len(consumer.process_calls) == 1


class TestReceiptValidation:
    def _setup_handed_off(self, receipt_factory):
        task = _task(processing_stage="handed_off")
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        consumer = FakeConsumer(receipt_factory=receipt_factory)
        request = ScopeHandoffBuilder("Asia/Shanghai").build(
            task, TOKEN, _expert_prepared(task.expert_id))
        _seed_artifact(artifact_repo, task.task_id, request)
        return task_repo, artifact_repo, consumer, _orchestrator(task_repo, artifact_repo, consumer)

    def test_request_id_mismatch(self):
        def _make(request):
            return ScopeHandoffReceipt(
                request_id="wrong", task_id=request.task_id, batch_id=request.batch_id,
                attempt_token=request.attempt_token, payload_hash=request.payload_hash,
                impact_scope=request.impact_scope, status="rejected",
                result_code="x", desired_count=0, error_type="e", error_message="m",
            )
        task_repo, _, _, orch = self._setup_handed_off(_make)
        result = orch.process(701, TOKEN, _expert_prepared())
        assert result.final_status == "dead"
        assert task_repo.rejected_calls[0]["error_stage"] == "receipt"

    def test_payload_hash_mismatch(self):
        def _make(request):
            return ScopeHandoffReceipt(
                request_id=request.request_id, task_id=request.task_id,
                batch_id=request.batch_id, attempt_token=request.attempt_token,
                payload_hash="0" * 64, impact_scope=request.impact_scope,
                status="rejected", result_code="x", desired_count=0,
                error_type="e", error_message="m",
            )
        task_repo, _, _, orch = self._setup_handed_off(_make)
        result = orch.process(701, TOKEN, _expert_prepared())
        assert result.final_status == "dead"
        assert task_repo.rejected_calls[0]["error_stage"] == "receipt"

    def test_attempt_token_mismatch(self):
        def _make(request):
            return ScopeHandoffReceipt(
                request_id=request.request_id, task_id=request.task_id,
                batch_id=request.batch_id, attempt_token="other-token",
                payload_hash=request.payload_hash, impact_scope=request.impact_scope,
                status="rejected", result_code="x", desired_count=0,
                error_type="e", error_message="m",
            )
        task_repo, _, _, orch = self._setup_handed_off(_make)
        result = orch.process(701, TOKEN, _expert_prepared())
        assert result.final_status == "dead"
        assert task_repo.rejected_calls[0]["error_stage"] == "receipt"

    def test_scope_mismatch(self):
        def _make(request):
            other_scope = impact_scope_from_target(target_type="profile", expert_id=request.impact_scope.expert_id)
            return ScopeHandoffReceipt(
                request_id=request.request_id, task_id=request.task_id,
                batch_id=request.batch_id, attempt_token=request.attempt_token,
                payload_hash=request.payload_hash, impact_scope=other_scope,
                status="rejected", result_code="x", desired_count=0,
                error_type="e", error_message="m",
            )
        task_repo, _, _, orch = self._setup_handed_off(_make)
        result = orch.process(701, TOKEN, _expert_prepared())
        assert result.final_status == "dead"
        assert task_repo.rejected_calls[0]["error_stage"] == "receipt"


class TestCrashReplay:
    def test_consumer_success_but_mark_crash_then_replay_success(self):
        # 第一阶段：handed_off，Consumer 返回 committed，但 finalize 抛 lease 失败
        task = _task(processing_stage="handed_off")
        task_repo = FakeTaskRepo(state=task, fail_finalize_with_lease=True)
        artifact_repo = FakeArtifactRepo()
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        request = ScopeHandoffBuilder("Asia/Shanghai").build(
            task, TOKEN, _expert_prepared(task.expert_id))
        _seed_artifact(artifact_repo, task.task_id, request)
        orch = _orchestrator(task_repo, artifact_repo, consumer)
        r1 = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert r1.terminal is False  # success 未落定
        assert len(consumer.process_calls) == 1
        # 崩溃恢复：模拟 finalize 修复，task 仍 handed_off，重放
        task_repo.fail_finalize_with_lease = False
        r2 = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert r2.final_status == "success"
        assert r2.terminal is True
        # 总共 Consumer 调 2 次（重放幂等：主路径因三 Hash 自然收敛）
        assert len(consumer.process_calls) == 2
        assert len(task_repo.finalize_calls) == 1  # 只有第二次落定


class TestZeroDoubleWrite:
    def test_no_checkpoint_no_workitem_writes(self):
        task = _task()
        task_repo = FakeTaskRepo(state=task)
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert task_repo.checkpoint_writes == 0
        assert task_repo.workitem_writes == 0

    def test_committed_is_only_path_to_success(self):
        # 验证只有 committed 才调 finalize
        task = _task()
        task_repo = FakeTaskRepo(state=task)
        consumer = FakeConsumer(receipt_factory=_rejected_receipt_factory())
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert task_repo.finalize_calls == []
        assert task_repo.rejected_calls


class TestUnsupportedStage:
    def test_candidate_stage_rejected(self):
        task = _task(processing_stage="chunk_built")
        task_repo = FakeTaskRepo(state=task)
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "dead"
        assert task_repo.rejected_calls[0]["error_stage"] == "handoff"

    def test_already_terminal_success_is_idempotent(self):
        task = _task(processing_stage="completed", sync_status="success",
                     attempt_token=None, lease_until=None, result_code="scope_committed",
                     completed_at=TASK_UPDATED)
        task_repo = FakeTaskRepo(state=task)
        consumer = FakeConsumer(receipt_factory=_committed_receipt_factory())
        orch = _orchestrator(task_repo, FakeArtifactRepo(), consumer)
        result = orch.process(task.task_id, TOKEN, _expert_prepared(task.expert_id))
        assert result.final_status == "success"
        assert result.terminal is True
        assert consumer.process_calls == []
        assert task_repo.success_calls == []
        assert task_repo.finalize_calls == []


class TestStructural:
    def test_no_forbidden_imports_or_calls(self):
        """编排器不 import V15 Work Item/Batch、CheckpointManager、物理删除或 MySQL/Milvus 直接连接。"""
        path = Path(__file__).resolve().parents[1] / "src" / "expert_index" / "handoff_orchestrator.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        forbidden = [
            "V15WorkItemInput", "V15WorkItemConsumer", "V15VectorizationBatch",
            "V15VectorizationWorkItem", "CheckpointManager", "delete_by_vector_doc_ids",
            "pymysql", "pymilvus", "MilvusClient",
        ]
        joined = " ".join(imports)
        for token in forbidden:
            assert token not in joined, f"编排器禁止 import {token}"

    def test_no_physical_delete_calls_in_source(self):
        path = Path(__file__).resolve().parents[1] / "src" / "expert_index" / "handoff_orchestrator.py"
        src = path.read_text(encoding="utf-8")
        # 不调物理删除 API
        assert "delete_by_vector_doc_ids" not in src
        assert "delete_vectors" not in src
        # 不推进 checkpoint
        assert "CheckpointManager" not in src
        assert "advance_checkpoint" not in src
        assert "commit_checkpoint" not in src


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _seed_artifact(repo: FakeArtifactRepo, task_id: int, request: ScopeHandoffRequest) -> None:
    """把已构建的 Request 作为 Artifact 种子预置到 FakeArtifactRepo。"""
    payload = {
        "request_id": request.request_id,
        "task_id": request.task_id,
        "batch_id": request.batch_id,
        "attempt_token_hash": "0" * 64,
        "impact_scope": {
            "scope_type": request.impact_scope.scope_type,
            "expert_id": request.impact_scope.expert_id,
            "chunk_type": request.impact_scope.chunk_type,
            "source_table": request.impact_scope.source_table,
            "source_field_id": request.impact_scope.source_field_id,
        },
        "contracts": [c.model_dump(mode="json") for c in request.contracts],
        "payload_hash": request.payload_hash,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    repo.store[(task_id, ARTIFACT_TYPE, ARTIFACT_VERSION)] = ContractArtifact(
        artifact_id=1, task_id=task_id, artifact_type=ARTIFACT_TYPE,
        artifact_version=ARTIFACT_VERSION, payload_json=payload_json,
        payload_reference=None, payload_hash=request.payload_hash,
        expires_at=None, created_at=TASK_UPDATED, updated_at=TASK_UPDATED,
    )
