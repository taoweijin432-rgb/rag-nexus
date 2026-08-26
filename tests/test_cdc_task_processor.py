"""验证 P7.6 CDC DELETE Task 复用 P5/P6 编排与失败恢复。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from expert_index.cdc import CdcDeleteTaskProcessor
from expert_index.handoff_orchestrator import HandoffOrchestrationResult
from expert_index.models import PreparedChunkSet, ScopedCurrentFacts, SourceSyncTask

NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)
TOKEN = "cdc-attempt-1"


def _task(
    *,
    stage: str = "candidate",
    source_table: str = "experts_work_experience",
    source_record_id: int = 501,
    target_type: str = "employment",
) -> SourceSyncTask:
    """构造已领取且持有有效 token 的 CDC DELETE Task。"""

    return SourceSyncTask(
        task_id=801,
        dedupe_key="a" * 64,
        batch_id=88,
        event_source="cdc",
        source_table=source_table,
        source_record_id=source_record_id,
        source_updated_at=None,
        expert_id=101,
        operation_type="delete",
        target_type=target_type,
        processing_stage=stage,
        sync_status="processing",
        retry_count=0,
        policy_version="1.3",
        created_at=NOW,
        updated_at=NOW,
        attempt_token=TOKEN,
        lease_until=NOW + timedelta(minutes=3),
    )


class _Repository:
    """模拟阶段、同批专家删除集合和 P6 失败写回。"""

    def __init__(self, task: SourceSyncTask, *, parent_ids: frozenset[int] = frozenset()) -> None:
        self.task = task
        self.parent_ids = parent_ids
        self.advanced: list[str] = []
        self.failures: list[tuple[str, dict[str, object]]] = []

    def renew_source_sync_task_lease(
        self, _task_id: int, _token: str, *, lease_seconds: int
    ) -> SourceSyncTask:
        assert lease_seconds == 120
        return self.task

    def advance_source_sync_task_stage(
        self, _task_id: int, _token: str, *, next_stage: str
    ) -> SourceSyncTask:
        self.advanced.append(next_stage)
        self.task = replace(self.task, processing_stage=next_stage)
        return self.task

    def list_cdc_expert_delete_ids(self, batch_id: int) -> frozenset[int]:
        assert batch_id == 88
        return self.parent_ids

    def mark_source_sync_task_rejected_with_dlq(
        self, _task_id: int, _token: str, **kwargs: object
    ) -> SourceSyncTask:
        self.failures.append(("rejected", kwargs))
        self.task = replace(
            self.task,
            sync_status="dead",
            attempt_token=None,
            lease_until=None,
        )
        return self.task

    def mark_source_sync_task_retryable_with_dlq(
        self, _task_id: int, _token: str, **kwargs: object
    ) -> SourceSyncTask:
        self.failures.append(("retryable", kwargs))
        self.task = replace(
            self.task,
            sync_status="pending",
            attempt_token=None,
            lease_until=None,
        )
        return self.task


class _Source:
    """记录教育删除事实读取；普通工作/项目/专家删除不应调用。"""

    def __init__(self, facts: ScopedCurrentFacts | None = None) -> None:
        self.facts = facts
        self.calls: list[int] = []

    def read_cdc_education_delete_facts(self, expert_id: int) -> ScopedCurrentFacts | None:
        self.calls.append(expert_id)
        return self.facts


class _Orchestrator:
    """记录 P7.6 传入的 prepared/恢复入口并返回固定终态。"""

    def __init__(self, *, invoke_prepare: bool = False, error: Exception | None = None) -> None:
        self.invoke_prepare = invoke_prepare
        self.error = error
        self.calls: list[tuple[PreparedChunkSet | None, object | None]] = []

    def process(
        self,
        _task_id: int,
        _token: str,
        prepared: PreparedChunkSet | None = None,
        *,
        prepare_if_missing=None,
    ) -> HandoffOrchestrationResult:
        self.calls.append((prepared, prepare_if_missing))
        if self.error is not None:
            raise self.error
        if self.invoke_prepare and prepare_if_missing is not None:
            prepared = prepare_if_missing()
            self.calls[-1] = (prepared, prepare_if_missing)
        return HandoffOrchestrationResult("success", True, "disabled", None)


def _processor(
    task: SourceSyncTask,
    *,
    source: _Source | None = None,
    parent_ids: frozenset[int] = frozenset(),
    orchestrator: _Orchestrator | None = None,
) -> tuple[CdcDeleteTaskProcessor, _Repository, _Source, _Orchestrator]:
    """组装不连接 MySQL/Milvus 的 CDC Task Processor。"""

    repository = _Repository(task, parent_ids=parent_ids)
    source = source or _Source()
    orchestrator = orchestrator or _Orchestrator()
    factory = SimpleNamespace(settings=SimpleNamespace(chunk_policy_version="1.3"))
    processor = CdcDeleteTaskProcessor(
        task_repository=repository,
        source=source,
        factory=factory,  # type: ignore[arg-type]
        orchestrator=orchestrator,
        lease_seconds=120,
        max_attempts=3,
        retry_delay_seconds=30,
    )
    return processor, repository, source, orchestrator


def test_candidate_delete_advances_to_validated_and_hands_off_empty_scope() -> None:
    """首次工作删除应单调推进并把空 employment desired 交给既有编排器。"""

    task = _task()
    processor, repository, source, orchestrator = _processor(task)

    result = processor.process_claimed(task, TOKEN)

    prepared = orchestrator.calls[0][0]
    assert prepared is not None and prepared.impact_scope.scope_type == "employment"
    assert prepared.is_empty is True
    assert repository.advanced == ["source_read", "normalized", "chunk_built", "validated"]
    assert source.calls == []
    assert result.final_status == "success"


def test_same_batch_parent_delete_promotes_education_without_source_read() -> None:
    """同批专家删除使教育 Task 直接交付 expert 空范围。"""

    task = _task(
        source_table="experts_education_experience",
        source_record_id=701,
        target_type="education",
    )
    processor, _repository, source, orchestrator = _processor(task, parent_ids=frozenset({101}))

    processor.process_claimed(task, TOKEN)

    prepared = orchestrator.calls[0][0]
    assert prepared is not None and prepared.impact_scope.scope_type == "expert"
    assert prepared.is_empty is True
    assert source.calls == []


def test_validated_recovery_lets_orchestrator_choose_artifact_or_rebuild() -> None:
    """validated 恢复只提供惰性准备入口，Artifact 存在时不重读 Source。"""

    task = _task(stage="validated")
    processor, repository, source, orchestrator = _processor(task)

    processor.process_claimed(task, TOKEN)

    assert orchestrator.calls[0][0] is None
    assert callable(orchestrator.calls[0][1])
    assert repository.advanced == []
    assert source.calls == []


def test_validated_recovery_rebuilds_only_when_artifact_is_missing() -> None:
    """Artifact 缺失时才调用惰性入口重建空 desired。"""

    task = _task(stage="validated")
    processor, repository, source, orchestrator = _processor(
        task,
        orchestrator=_Orchestrator(invoke_prepare=True),
    )

    processor.process_claimed(task, TOKEN)

    prepared = orchestrator.calls[0][0]
    assert prepared is not None and prepared.is_empty is True
    assert repository.advanced == []
    assert source.calls == []


@pytest.mark.parametrize(
    ("error", "expected_status", "failure_kind"),
    ((ValueError("bad facts"), "dead", "rejected"), (RuntimeError("db"), "pending", "retryable")),
)
def test_preparation_failure_uses_p6_dlq_policy(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: str,
    failure_kind: str,
) -> None:
    """确定性失败进入 dead，基础设施失败按 P6 上限退避。"""

    task = _task()
    processor, repository, _source, _orchestrator = _processor(task)

    def fail(*_args: object, **_kwargs: object) -> PreparedChunkSet:
        raise error

    monkeypatch.setattr("expert_index.cdc.task_processor.prepare_cdc_delete_task", fail)
    result = processor.process_claimed(task, TOKEN)

    assert result.final_status == expected_status
    assert repository.failures[0][0] == failure_kind
    assert "bad facts" not in str(repository.failures[0][1])


def test_unexpected_handoff_error_returns_to_p6_retry() -> None:
    """未分类交付异常不丢任务，使用同一 Artifact 身份等待新 attempt 重放。"""

    task = _task(stage="contract_ready")
    processor, repository, source, _orchestrator = _processor(
        task,
        orchestrator=_Orchestrator(error=RuntimeError("response lost")),
    )

    result = processor.process_claimed(task, TOKEN)

    assert result.final_status == "pending"
    assert repository.failures[0][0] == "retryable"
    assert repository.failures[0][1]["error_stage"] == "handoff"
    assert source.calls == []
