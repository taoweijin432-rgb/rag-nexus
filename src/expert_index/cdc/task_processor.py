"""CDC Task 到既有 P4/P5/P6 Handoff 闭环的接线。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..chunk_builder import ChunkFactory
from ..handoff_orchestrator import HandoffOrchestrationResult
from ..models import PreparedChunkSet, SourceSyncTask
from ..preparation_coordinator import PreparationCoordinator
from ..repositories.sync_task import (
    SourceSyncTaskLeaseConflictError,
    SourceSyncTaskStageConflictError,
)
from .delete_scope import CdcDeleteSourcePort, prepare_cdc_delete_task

_PREPARATION_STAGES = ("candidate", "source_read", "normalized", "chunk_built", "validated")
_PREPARATION_STAGE_INDEX = {stage: index for index, stage in enumerate(_PREPARATION_STAGES)}
_ARTIFACT_RECOVERY_STAGES = frozenset({"contract_ready", "handed_off"})


class CdcDeleteTaskRepositoryPort(Protocol):
    """CDC 删除准备和失败恢复依赖的现有 Task Repository 方法。"""

    def renew_source_sync_task_lease(
        self,
        task_id: int,
        attempt_token: str,
        *,
        lease_seconds: int,
    ) -> SourceSyncTask: ...

    def advance_source_sync_task_stage(
        self,
        task_id: int,
        attempt_token: str,
        *,
        next_stage: str,
    ) -> SourceSyncTask: ...

    def list_cdc_expert_delete_ids(self, batch_id: int) -> frozenset[int]: ...

    def mark_source_sync_task_rejected_with_dlq(
        self,
        task_id: int,
        attempt_token: str,
        *,
        error_stage: str,
        error_code: str,
        error_type: str | None = None,
        safe_error_message: str | None = None,
        artifact_id: int | None = None,
    ) -> SourceSyncTask: ...

    def mark_source_sync_task_retryable_with_dlq(
        self,
        task_id: int,
        attempt_token: str,
        *,
        error_stage: str,
        error_code: str,
        max_attempts: int,
        retry_delay_seconds: int,
        error_type: str | None = None,
        safe_error_message: str | None = None,
        artifact_id: int | None = None,
    ) -> SourceSyncTask: ...


class CdcHandoffOrchestratorPort(Protocol):
    """P7.6 复用的 P5 Handoff 编排最小接口。"""

    def process(
        self,
        task_id: int,
        attempt_token: str,
        prepared: PreparedChunkSet | None = None,
        *,
        prepare_if_missing: Callable[[], PreparedChunkSet] | None = None,
    ) -> HandoffOrchestrationResult: ...


class CdcDeleteTaskProcessor:
    """推进一个已领取 CDC Task，并复用 Artifact/Consumer/Receipt/DLQ。

    DELETE 使用专用范围准备器处理物理删除；INSERT/UPDATE 复用 P4
    ``PreparationCoordinator`` 重读当前 MySQL 事实，避免维护第二套 Chunk 链路。
    """

    def __init__(
        self,
        *,
        task_repository: CdcDeleteTaskRepositoryPort,
        source: CdcDeleteSourcePort,
        factory: ChunkFactory,
        orchestrator: CdcHandoffOrchestratorPort,
        lease_seconds: int,
        max_attempts: int,
        retry_delay_seconds: int,
    ) -> None:
        """保存现有 P4/P5/P6 依赖并校验租约、重试参数。"""

        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds < 1
        ):
            raise ValueError("lease_seconds 必须是正整数")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
            raise ValueError("max_attempts 必须是非负整数")
        if (
            isinstance(retry_delay_seconds, bool)
            or not isinstance(retry_delay_seconds, int)
            or retry_delay_seconds < 1
        ):
            raise ValueError("retry_delay_seconds 必须是正整数")
        self._task_repository = task_repository
        self._source = source
        self._factory = factory
        self._orchestrator = orchestrator
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds

    def process_claimed(
        self,
        task: SourceSyncTask,
        attempt_token: str,
    ) -> HandoffOrchestrationResult:
        """从当前阶段处理一个 CDC Task，终点由既有 P5/P6 编排决定。

        Artifact 前阶段始终按当前 CDC 事实重建 ``PreparedChunkSet``；validated 优先
        复用已有 Artifact，仅在缺失时重建；contract_ready/handed_off 只从 Artifact
        恢复。方法不修改 CDC Event、不推进业务 Checkpoint，也不调用 Canal ACK。
        """

        self._validate_claimed_task(task, attempt_token)
        stage = task.processing_stage
        if stage in _PREPARATION_STAGE_INDEX and stage != "validated":
            prepared = self._prepare_or_fail(task, attempt_token)
            if isinstance(prepared, HandoffOrchestrationResult):
                return prepared
            return self._run_handoff(task, attempt_token, prepared=prepared)
        if stage == "validated":
            return self._run_handoff(
                task,
                attempt_token,
                prepare_if_missing=lambda: self._prepare_for_recovery(task, attempt_token),
            )
        if stage in _ARTIFACT_RECOVERY_STAGES:
            return self._run_handoff(task, attempt_token)
        raise SourceSyncTaskStageConflictError(f"CDC Task 不支持从 {stage!r} 阶段执行")

    def _prepare_or_fail(
        self,
        task: SourceSyncTask,
        attempt_token: str,
    ) -> PreparedChunkSet | HandoffOrchestrationResult:
        """准备失败时使用 P6 原子失败写回，避免等待租约自然过期。"""

        try:
            return self._prepare_for_recovery(task, attempt_token)
        except (SourceSyncTaskLeaseConflictError, SourceSyncTaskStageConflictError):
            return HandoffOrchestrationResult(
                final_status="processing",
                terminal=False,
                result_code=None,
                error_stage="fencing",
            )
        except (TypeError, ValueError) as error:
            return self._mark_rejected(task, attempt_token, error)
        except Exception as error:  # noqa: BLE001 - Source/构建基础设施失败统一有限重试
            return self._mark_retryable(task, attempt_token, error)

    def _run_handoff(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        *,
        prepared: PreparedChunkSet | None = None,
        prepare_if_missing: Callable[[], PreparedChunkSet] | None = None,
    ) -> HandoffOrchestrationResult:
        """调用既有编排器；未分类基础设施异常按同一 Artifact 身份有限重试。"""

        try:
            return self._orchestrator.process(
                task.task_id,
                attempt_token,
                prepared,
                prepare_if_missing=prepare_if_missing,
            )
        except (SourceSyncTaskLeaseConflictError, SourceSyncTaskStageConflictError):
            return _fencing_result()
        except Exception as error:  # noqa: BLE001 - 响应丢失/数据库异常交由 P6 重放
            return self._mark_handoff_retryable(task, attempt_token, error)

    def _prepare_for_recovery(
        self,
        task: SourceSyncTask,
        attempt_token: str,
    ) -> PreparedChunkSet:
        """按任务类型重读当前事实，并单调补齐到 validated。"""

        if task.operation_type != "delete":
            return self._prepare_current_change(task, attempt_token)

        current = self._task_repository.renew_source_sync_task_lease(
            task.task_id,
            attempt_token,
            lease_seconds=self._lease_seconds,
        )
        self._validate_claimed_task(current, attempt_token)
        if current.processing_stage not in _PREPARATION_STAGE_INDEX:
            raise SourceSyncTaskStageConflictError("CDC DELETE Task 已越过 preparation 阶段")
        parent_delete_ids = self._task_repository.list_cdc_expert_delete_ids(current.batch_id)
        prepared = prepare_cdc_delete_task(
            current,
            self._source,
            self._factory,
            parent_delete_expert_ids=parent_delete_ids,
        )
        self._task_repository.renew_source_sync_task_lease(
            current.task_id,
            attempt_token,
            lease_seconds=self._lease_seconds,
        )
        current_index = _PREPARATION_STAGE_INDEX[current.processing_stage]
        for next_stage in ("source_read", "normalized", "chunk_built", "validated"):
            if current_index < _PREPARATION_STAGE_INDEX[next_stage]:
                current = self._task_repository.advance_source_sync_task_stage(
                    current.task_id,
                    attempt_token,
                    next_stage=next_stage,
                )
                current_index = _PREPARATION_STAGE_INDEX[current.processing_stage]
        return prepared

    def _prepare_current_change(
        self,
        task: SourceSyncTask,
        attempt_token: str,
    ) -> PreparedChunkSet:
        """对 INSERT/UPDATE 复用 P4 Coordinator 完成事实重读和 Chunk 准备。"""

        coordinator = PreparationCoordinator(
            self._task_repository,  # type: ignore[arg-type]
            self._source,  # type: ignore[arg-type]
            self._factory,
            lease_seconds=self._lease_seconds,
        )
        return coordinator.prepare(task, attempt_token)

    def _mark_rejected(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        error: Exception,
    ) -> HandoffOrchestrationResult:
        """把确定性的 CDC desired 校验失败写入 Task 与追加式 DLQ。"""

        try:
            updated = self._task_repository.mark_source_sync_task_rejected_with_dlq(
                task.task_id,
                attempt_token,
                error_stage="validation",
                error_code="cdc_prepare_validation_failed",
                error_type=type(error).__name__,
                safe_error_message="CDC 变更范围或当前事实校验失败",
            )
        except SourceSyncTaskLeaseConflictError:
            return _fencing_result()
        return _failure_result(updated, "validation")

    def _mark_retryable(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        error: Exception,
    ) -> HandoffOrchestrationResult:
        """把 Source/构建瞬时失败按 P6 上限退避，耗尽后进入追加式 DLQ。"""

        try:
            updated = self._task_repository.mark_source_sync_task_retryable_with_dlq(
                task.task_id,
                attempt_token,
                error_stage="source",
                error_code="cdc_prepare_failed",
                error_type=type(error).__name__,
                safe_error_message="CDC 当前事实准备失败",
                max_attempts=self._max_attempts,
                retry_delay_seconds=self._retry_delay_seconds,
            )
        except SourceSyncTaskLeaseConflictError:
            return _fencing_result()
        return _failure_result(updated, "source")

    def _mark_handoff_retryable(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        error: Exception,
    ) -> HandoffOrchestrationResult:
        """把未被 P5 编排器分类的交付异常安全退回，供 Artifact 幂等重放。"""

        try:
            updated = self._task_repository.mark_source_sync_task_retryable_with_dlq(
                task.task_id,
                attempt_token,
                error_stage="handoff",
                error_code="cdc_handoff_unexpected",
                error_type=type(error).__name__,
                safe_error_message="CDC 交付发生未分类基础设施失败",
                max_attempts=self._max_attempts,
                retry_delay_seconds=self._retry_delay_seconds,
            )
        except SourceSyncTaskLeaseConflictError:
            return _fencing_result()
        return _failure_result(updated, "handoff")

    def _validate_claimed_task(self, task: SourceSyncTask, attempt_token: str) -> None:
        """校验调用方持有 CDC processing Task 的当前 attempt。"""

        if not isinstance(task, SourceSyncTask):
            raise TypeError("task 必须是 SourceSyncTask")
        if task.sync_status != "processing":
            raise ValueError("CDC Task 必须处于 processing")
        if (
            not isinstance(attempt_token, str)
            or not attempt_token.strip()
            or task.attempt_token != attempt_token
        ):
            raise ValueError("attempt_token 与 CDC Task 不一致")
        if task.event_source != "cdc" or task.operation_type not in {"insert", "update", "delete"}:
            raise ValueError("任务必须是 CDC INSERT/UPDATE/DELETE")
        expected_policy = self._factory.settings.chunk_policy_version
        if task.policy_version != expected_policy:
            raise ValueError("CDC Task policy_version 与当前 ChunkFactory 不一致")


def _failure_result(task: SourceSyncTask, error_stage: str) -> HandoffOrchestrationResult:
    """把 P6 失败写回后的 Task 状态转换为统一编排结果。"""

    if task.sync_status == "dead":
        return HandoffOrchestrationResult("dead", True, None, error_stage)
    if task.sync_status == "pending":
        return HandoffOrchestrationResult("pending", False, None, error_stage)
    return _fencing_result()


def _fencing_result() -> HandoffOrchestrationResult:
    """返回不覆盖新 attempt 的统一 fencing 结果。"""

    return HandoffOrchestrationResult("processing", False, None, "fencing")


__all__ = [
    "CdcDeleteTaskProcessor",
    "CdcDeleteTaskRepositoryPort",
    "CdcHandoffOrchestratorPort",
]
