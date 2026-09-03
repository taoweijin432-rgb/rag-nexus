"""P5.7 单来源批次协调器：把 P3/P4/P5 串成单来源、单批次、单任务串行执行链。

职责边界（严格）：

- 只对“一个已经由 P3 scanner 创建好的 Batch Source”做一次受控执行：确认/续租
  source lease → 按 ``limit=1`` 逐条 claim task → P4 prepare 或 P5 orchestrator → 累计
  结果 → 满足闸门后调用现有 :class:`CheckpointManager.try_commit_source_batch`。
- **不创建** Batch Source、不创建 Task、不直接修改 Batch/计数、不推进 Checkpoint；
  正常路径仍由 P3/P5.3A/P5.6 现有实现权威写入。仅当注入的
  :class:`PreparationFailureHandler` 捕获 P4 异常时，才按既有 Task Repository
  接口落定当前失败 Task，避免异常任务无限重领。
- 只持有受控执行期的 source lease（``ensure_source_lease``/续租），提交成功后由
  :class:`CheckpointManager` 释放；未提交不自行释放（留待下次执行/超时）。
- 不解读 ``task.lease_until`` 的时区或与本机时间比较，全部由 Repository 以
  Index MySQL ``CURRENT_TIMESTAMP(6)`` + ``attempt_token`` + rowcount 权威判定。

首期限制（用户冻结）：

- 只支持单进程、一次只运行一个 :class:`SourceBatchCoordinator`；不实现
  ``run_forever``、多 Worker 或跨来源并发。
- 不实现 ``GET_LOCK``、scope fence、generation 或新 Migration；不声称解决
  跨来源旧 Artifact 防回退。
- ``INCREMENTAL_ENABLED`` 保持关闭；跨来源 fencing 留作 P5.8/P9 上线前门禁。
- 头部顺序依赖 P5.7A 已修复的 ``claim_source_sync_tasks`` 严格队头语义
  （pending ``next_retry_at`` 未到时整批等待，不领取更晚 task）。

失败语义（用户冻结规则）：

- Task ``final_status='success'``（terminal）→ 继续处理剩余 Task。
- Task ``final_status='dead'``（rejected，确定性终态）→ 可继续处理剩余 Task；
  只有 Repository 已在同一事务写入 DLQ 的 dead 才能通过 Manager 闸门，Checkpoint
  会推进并在批次统计中记录 failed_count，后续人工处置以 DLQ 为准。
- Task ``final_status='pending'``（retryable 退回）→ 立即停止当前 Source Batch。
- Task lease/token 冲突（``terminal==False``，``final_status=='processing'``）
  → 立即停止当前 Source Batch，计入 ``lease_conflict_count``。
- P4 prepare 异常：注入 Handler 时按确定性/瞬时失败落定；未注入 Handler 时
  保持 fail-closed 停止。
- 未知/非法结果（``final_status`` 非 success/dead/pending/processing，或
  ``terminal`` 与 ``final_status`` 矛盾）→ fail-closed 立即停止。
- source lease 续租冲突 → 立即停止剩余 Task，``error`` 记异常类名。

阶段恢复规则：

- ``candidate``：P3 新建任务固定 ``candidate/pending``，经
  :meth:`PreparationCoordinator.prepare` 推进到 ``validated``。
- ``source_read``/``normalized``/``chunk_built``：重读 MySQL 当前事实并从已完成阶段
  继续补齐到 ``validated``。
- ``validated``：优先恢复已有 Artifact；只有 Artifact 不存在时才调用 P4 重建入口。
- ``contract_ready``/``handed_off``：只从 Artifact 恢复，不重新读取来源事实。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .handoff_orchestrator import HandoffOrchestrationResult
from .models import SourceSyncTask
from .preparation_coordinator import PreparationCoordinator
from .repositories.source_state import SourceLeaseConflictError
from .repositories.sync_task import (
    DEFAULT_SOURCE_SYNC_MAX_ATTEMPTS,
    SourceSyncTaskLeaseConflictError,
)
from .validators import ChunkValidationError

__all__ = [
    "PreparationFailureHandler",
    "PreparationFailureHandlerPort",
    "SourceBatchCoordinator",
    "SourceBatchRunResult",
]


@dataclass(frozen=True)
class SourceBatchRunResult:
    """一次单来源批次受控执行的内存结构化结果（不新增数据库字段）。

    全部字段为内存计数，``error`` 只记稳定异常类名，不记业务正文/PII。
    """

    source_table: str
    batch_id: int
    candidate_count: int  # 来自 BatchSourceScan.candidate_count
    claimed_count: int  # 本次领取并处理的 Task 数
    success_count: int  # final_status=="success" (terminal)
    retryable_count: int  # final_status=="pending" (retryable 退回)
    rejected_count: int  # final_status=="dead" (rejected/non-retryable)
    lease_conflict_count: int  # task lease 失配/fencing 冲突 (terminal=False)
    checkpoint_committed: bool  # try_commit 首次成功
    checkpoint_already_committed: bool  # try_commit 幂等命中
    checkpoint_skipped_count: int | None  # 提交成功时的重算 skipped
    error: str | None  # 稳定异常类名（无业务正文）


# Artifact 前允许重读当前事实并恢复的 P4 阶段。
_P4_PREPARATION_STAGES = frozenset(
    {"candidate", "source_read", "normalized", "chunk_built"}
)
# 只允许从 Artifact 恢复的 P5 阶段。
_HANDOFF_RECOVERY_STAGES = frozenset({"contract_ready", "handed_off"})

# 终态映射（来自 HandoffOrchestrationResult.final_status）。
_TERMINAL_SUCCESS = "success"
_TERMINAL_DEAD = "dead"
_RETRYABLE_PENDING = "pending"
_LEASE_CONFLICT_PROCESSING = "processing"


class PreparationFailureHandlerPort(Protocol):
    """P4 准备阶段异常的任务状态落定端口。

    Coordinator 只依赖这个窄接口，不直接依赖 MySQL Repository，因而不会把
    Expert 的正常 P4/P5 数据路径和任务失败写回耦合在一起；测试和未来 Resume
    运行时也可以注入各自的实现。
    """

    def handle(
        self,
        task: SourceSyncTask,
        attempt_token: str | None,
        error: Exception,
    ) -> HandoffOrchestrationResult: ...


class PreparationFailureHandler:
    """将 P4 prepare 异常按确定性/瞬时性质安全落定到 Task + DLQ。

    ``PreparationCoordinator`` 本身是纯准备阶段，按设计只抛出异常、不写任务
    终态。这个 Handler 由 Poll 运行时在更外层注入，补齐“异常必须结算”的边界：

    * ``ChunkValidationError``、``ValueError``、``TypeError`` 是确定性契约问题，
      直接写 ``dead``，避免同一坏数据反复领取；
    * 其他异常按瞬时失败退回 ``pending``，达到 ``max_attempts`` 后由 Repository
      原子转为 ``dead``；
    * token/lease 冲突只返回 ``processing``，绝不覆盖新 Worker 的有效 attempt。

    失败消息使用固定摘要，不把来源正文、Chunk 文本或异常原文写入 Index DB/DLQ。
    """

    def __init__(
        self,
        task_repository,
        *,
        max_attempts: int,
        retry_delay_seconds: int,
    ) -> None:
        """保存状态仓储和重试策略；参数校验与 HandoffOrchestrator 保持一致。"""

        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 0
        ):
            raise ValueError("max_attempts 必须是非负整数")
        if (
            isinstance(retry_delay_seconds, bool)
            or not isinstance(retry_delay_seconds, int)
            or retry_delay_seconds < 1
        ):
            raise ValueError("retry_delay_seconds 必须是正整数")
        if not callable(getattr(task_repository, "mark_source_sync_task_rejected", None)):
            raise TypeError("task_repository 缺少 mark_source_sync_task_rejected")
        if not callable(getattr(task_repository, "mark_source_sync_task_retryable", None)):
            raise TypeError("task_repository 缺少 mark_source_sync_task_retryable")
        self._task_repo = task_repository
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds

    def handle(
        self,
        task: SourceSyncTask,
        attempt_token: str | None,
        error: Exception,
    ) -> HandoffOrchestrationResult:
        """落定一次 P4 异常，并返回 Coordinator 可复用的统一结果模型。"""

        if not isinstance(task, SourceSyncTask):
            raise TypeError("task 必须是 SourceSyncTask")
        if not isinstance(error, Exception):
            raise TypeError("error 必须是 Exception")
        # 没有 token 就无法安全执行带 fencing 的状态写回，按未终态停止；
        # 这类状态属于 claim/repository 异常，不应猜测或覆盖数据库记录。
        if not isinstance(attempt_token, str) or not attempt_token.strip():
            return _preparation_processing_result()
        if isinstance(error, (SourceSyncTaskLeaseConflictError, SourceLeaseConflictError)):
            return _preparation_processing_result()

        error_stage, error_code, safe_message, deterministic = _preparation_error_details(error)
        error_type = type(error).__name__
        try:
            if deterministic:
                mark_rejected = getattr(
                    self._task_repo,
                    "mark_source_sync_task_rejected_with_dlq",
                    None,
                )
                if not callable(mark_rejected):
                    mark_rejected = self._task_repo.mark_source_sync_task_rejected
                mark_rejected(
                    task.task_id,
                    attempt_token,
                    error_stage=error_stage,
                    error_code=error_code,
                    error_type=error_type,
                    safe_error_message=safe_message,
                )
                return HandoffOrchestrationResult(
                    final_status="dead",
                    terminal=True,
                    result_code=None,
                    error_stage=error_stage,
                )

            mark_retryable = getattr(
                self._task_repo,
                "mark_source_sync_task_retryable_with_dlq",
                None,
            )
            if not callable(mark_retryable):
                mark_retryable = self._task_repo.mark_source_sync_task_retryable
            updated = mark_retryable(
                task.task_id,
                attempt_token,
                error_stage=error_stage,
                error_code=error_code,
                max_attempts=self._max_attempts,
                retry_delay_seconds=self._retry_delay_seconds,
                error_type=error_type,
                safe_error_message=safe_message,
            )
        except SourceSyncTaskLeaseConflictError:
            # 新 attempt 已接管时，旧 Worker 只能停止；旧 token 不能覆盖新状态。
            return _preparation_processing_result()

        final_status = getattr(updated, "sync_status", None)
        if final_status == "dead":
            return HandoffOrchestrationResult(
                final_status="dead", terminal=True, result_code=None, error_stage=error_stage
            )
        if final_status == "pending":
            return HandoffOrchestrationResult(
                final_status="pending", terminal=False, result_code=None, error_stage=error_stage
            )
        # Repository 返回非法状态时 fail-closed，不让 Coordinator 误认为成功。
        return _preparation_processing_result()


def _preparation_error_details(error: Exception) -> tuple[str, str, str, bool]:
    """把异常映射为安全的失败阶段、错误码、摘要和确定性标记。"""

    if isinstance(error, ChunkValidationError):
        return "validation", "chunk_validation_failed", "Chunk 校验失败", True
    if isinstance(error, (ValueError, TypeError)):
        return "chunk_builder", "preparation_contract_failed", "P4 准备阶段契约校验失败", True
    return "source", "preparation_failed", "P4 当前事实准备暂时失败", False


def _preparation_processing_result() -> HandoffOrchestrationResult:
    """返回不可安全落定时的统一未终态结果。"""

    return HandoffOrchestrationResult(
        final_status="processing", terminal=False, result_code=None, error_stage="fencing"
    )


class SourceBatchCoordinator:
    """单来源、单批次、单任务串行执行协调器。

    注入：

    :param preparation_coordinator: P4 外层协调器（续租 + 阶段推进 + 纯事实准备）。
    :param orchestrator: P5.3B Handoff 编排闭环（validated/contract_ready/handed_off
        恢复入口）。
    :param checkpoint_manager: P3 Checkpoint 提交器（只调 ``try_commit_source_batch``）。
    :param source_state_repository: P3 Source lease 确认/续租 + BatchSourceScan 读取。
    :param task_repository: P3 Task 领取（``claim_source_sync_tasks``）；与
        :class:`PreparationCoordinator` 共用同一 Repository 实例。
    :param task_lease_seconds: claim/renew task lease 的建议秒数（真正有效性由
        Repository 以 Index MySQL ``CURRENT_TIMESTAMP(6)`` 权威判定）。
    :param max_attempts: 任务自动重试上限。必须与 Handoff/P4 失败处理器使用同一个
        值，并传入每次 Poll claim；这样 Worker 崩溃导致的 lease 回收也不会绕过上限。
    :param source_lease_seconds: 续租 source lease 的建议秒数。
    :param preparation_failure_handler: P4 prepare 异常的可选落定器。生产 Poll
        runtime 会注入它；未注入时保留原有 fail-closed 行为，兼容直接使用
        Coordinator 的调用方和测试替身。
    """

    def __init__(
        self,
        preparation_coordinator: PreparationCoordinator,
        orchestrator,
        checkpoint_manager,
        source_state_repository,
        task_repository,
        *,
        task_lease_seconds: int,
        source_lease_seconds: int,
        max_attempts: int = DEFAULT_SOURCE_SYNC_MAX_ATTEMPTS,
        preparation_failure_handler: PreparationFailureHandlerPort | None = None,
    ) -> None:
        self._prepare = preparation_coordinator
        self._orchestrator = orchestrator
        self._checkpoint = checkpoint_manager
        self._source_state = source_state_repository
        self._task_repo = task_repository
        if preparation_failure_handler is not None and not callable(
            getattr(preparation_failure_handler, "handle", None)
        ):
            raise TypeError("preparation_failure_handler 缺少 handle")
        self._preparation_failure_handler = preparation_failure_handler
        if (
            isinstance(task_lease_seconds, bool)
            or not isinstance(task_lease_seconds, int)
            or task_lease_seconds < 1
        ):
            raise ValueError("task_lease_seconds 必须是正整数")
        if (
            isinstance(source_lease_seconds, bool)
            or not isinstance(source_lease_seconds, int)
            or source_lease_seconds < 1
        ):
            raise ValueError("source_lease_seconds 必须是正整数")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 0
        ):
            raise ValueError("max_attempts 必须是非负整数")
        self._task_lease_seconds = task_lease_seconds
        self._source_lease_seconds = source_lease_seconds
        self._max_attempts = max_attempts

    def run_claimed_source_batch(
        self,
        source_table: str,
        batch_id: int,
        lease_owner: str,
        *,
        checkpoint_lock_version: int,
        task_claim_limit: int = 1,
    ) -> SourceBatchRunResult:
        """对单个已存在的 Batch Source 做一次受控串行执行。

        :param source_table: 目标来源表（四表之一）。
        :param batch_id: P3 scanner 已创建的 Batch Source 所属批次 ID。
        :param lease_owner: 调用方持有的 source lease owner（与 scanner acquire 时一致）。
        :param checkpoint_lock_version: Batch Source 启动时记录的 Checkpoint 版本。
        :param task_claim_limit: 单次 claim 的 task 数，首期固定 ``1``（串行）。
        :return: :class:`SourceBatchRunResult`（内存计数，无新 DB 字段）。
        """

        if (
            isinstance(task_claim_limit, bool)
            or not isinstance(task_claim_limit, int)
            or task_claim_limit != 1
        ):
            raise ValueError("task_claim_limit 首期必须固定为 1")

        # 1) 读取并校验 BatchSourceScan：Coordinator 不创建 Batch Source，只读。
        scan = self._source_state.get_batch_source_scan(batch_id, source_table)
        if scan is None:
            raise ValueError(
                f"Batch Source 不存在: batch_id={batch_id}, source_table={source_table}"
            )
        if scan.source_table != source_table or scan.batch_id != batch_id:
            raise RuntimeError("BatchSourceScan 身份与请求范围不一致")
        if scan.status == "success":
            # 已提交：幂等返回（不再续租、不再 claim、不再 try_commit）。
            return SourceBatchRunResult(
                source_table=source_table,
                batch_id=batch_id,
                candidate_count=scan.candidate_count,
                claimed_count=0,
                success_count=0,
                retryable_count=0,
                rejected_count=0,
                lease_conflict_count=0,
                checkpoint_committed=False,
                checkpoint_already_committed=True,
                checkpoint_skipped_count=None,
                error=None,
            )
        if scan.status not in {"pending", "running", "failed"}:
            # 未知 BatchSourceScan 状态：fail-closed。
            raise ValueError(
                f"BatchSourceScan 状态不可执行: {scan.status!r}"
            )

        candidate_count = scan.candidate_count
        claimed_count = 0
        success_count = 0
        retryable_count = 0
        rejected_count = 0
        lease_conflict_count = 0
        stop_reason: str | None = None

        # 2) 确保 source lease（首次进入受控执行期）。扫描与串行执行之间
        #    可能跨越租约周期；过期时仅允许原子重新认领，不覆盖其他有效 owner。
        try:
            self._source_state.ensure_source_lease(
                source_table,
                lease_owner,
                lease_seconds=self._source_lease_seconds,
            )
        except Exception as error:  # noqa: BLE001 - 续租失败统一停止
            return SourceBatchRunResult(
                source_table=source_table,
                batch_id=batch_id,
                candidate_count=candidate_count,
                claimed_count=0,
                success_count=0,
                retryable_count=0,
                rejected_count=0,
                lease_conflict_count=0,
                checkpoint_committed=False,
                checkpoint_already_committed=False,
                checkpoint_skipped_count=None,
                error=type(error).__name__,
            )

        # 3) 循环 claim（limit=1 串行）直到返回空或提前停止。
        while stop_reason is None:
            # 每条 Task 前续租 source lease（DB 时间权威）。
            try:
                self._source_state.renew_source_lease(
                    source_table,
                    lease_owner,
                    lease_seconds=self._source_lease_seconds,
                )
            except Exception as error:  # noqa: BLE001
                stop_reason = type(error).__name__
                break

            tasks = self._task_repo.claim_source_sync_tasks(
                batch_id,
                source_table,
                limit=task_claim_limit,
                lease_seconds=self._task_lease_seconds,
                max_attempts=self._max_attempts,
            )
            if not tasks:
                # 无更多 pending/可回收 processing：claim 完成，进入提交闸门。
                break
            task = tasks[0]
            claimed_count += 1

            # 4) 按阶段路由：candidate → P4 prepare；validated/contract_ready/
            #    handed_off → P5 orchestrator 恢复；其余 P4 中间阶段 prepare 会抛
            #    StageConflict → fail-closed 停止。
            try:
                if task.processing_stage in _P4_PREPARATION_STAGES:
                    try:
                        prepared = self._prepare.prepare(task, task.attempt_token)
                    except Exception as error:  # noqa: BLE001 - 由 Handler 统一分类落定
                        result = self._handle_preparation_failure(task, error)
                        if result is None:
                            # 未注入 Handler 时保留原有行为：异常直接停止，不猜测
                            # Task 状态，确保现有直接调用方行为不变。
                            stop_reason = type(error).__name__
                            break
                    else:
                        self._source_state.renew_source_lease(
                            source_table,
                            lease_owner,
                            lease_seconds=self._source_lease_seconds,
                        )
                        result = self._orchestrator.process(
                            task.task_id, task.attempt_token, prepared
                        )
                elif task.processing_stage == "validated":
                    # Orchestrator 先查 Artifact；仅在不存在时调用 P4 当前事实重建。
                    self._source_state.renew_source_lease(
                        source_table,
                        lease_owner,
                        lease_seconds=self._source_lease_seconds,
                    )
                    result = self._orchestrator.process(
                        task.task_id,
                        task.attempt_token,
                        prepare_if_missing=lambda task=task: self._prepare.prepare(
                            task, task.attempt_token
                        ),
                    )
                elif task.processing_stage in _HANDOFF_RECOVERY_STAGES:
                    # contract_ready/handed_off 必须仅使用已持久化 Artifact。
                    self._source_state.renew_source_lease(
                        source_table,
                        lease_owner,
                        lease_seconds=self._source_lease_seconds,
                    )
                    result = self._orchestrator.process(
                        task.task_id, task.attempt_token
                    )
                else:
                    # completed/未知阶段不属于 processing 可执行范围。
                    stop_reason = "SourceSyncTaskStageConflictError"
                    break
                # 外部 Consumer 返回后再次确认 Source lease，失效时停止当前 Worker；
                # Task 自身结果已经由 attempt fencing 决定，新 Worker可幂等接管。
                self._source_state.renew_source_lease(
                    source_table,
                    lease_owner,
                    lease_seconds=self._source_lease_seconds,
                )
            except Exception as error:  # noqa: BLE001
                # prepare/orchestrator 抛异常（StageConflict/LeaseConflict/未知）
                # 统一按 fail-closed 停止，记录稳定异常类名。
                stop_reason = type(error).__name__
                break

            # 5) 按 final_status/terminal 决定是否继续（用户冻结失败语义）。
            outcome = _classify_outcome(result)
            if outcome == "success":
                success_count += 1
                continue
            if outcome == "dead":
                rejected_count += 1
                continue
            if outcome == "retryable":
                retryable_count += 1
                stop_reason = "retryable"
                break
            if outcome == "lease_conflict":
                lease_conflict_count += 1
                stop_reason = "task_lease_conflict"
                break
            # 未知/非法结果：fail-closed 立即停止。
            stop_reason = "unknown_outcome"
            break

        # 6) 终态汇总与 Checkpoint 闸门。
        # 提前停止（retryable/lease 冲突/未知/中间阶段/未落定的 prepare 异常）→
        # 不调 try_commit，水位不动，source lease 不释放（留待下次执行/超时）。
        # stop_reason 为 None 表示全部 Task 处理完（含 dead 可继续场景），进入闸门。
        if stop_reason is not None:
            return SourceBatchRunResult(
                source_table=source_table,
                batch_id=batch_id,
                candidate_count=candidate_count,
                claimed_count=claimed_count,
                success_count=success_count,
                retryable_count=retryable_count,
                rejected_count=rejected_count,
                lease_conflict_count=lease_conflict_count,
                checkpoint_committed=False,
                checkpoint_already_committed=False,
                checkpoint_skipped_count=None,
                error=stop_reason,
            )

        # 全部 Task 处理完（含 dead 但无提前停止）：续租 source lease 后调 try_commit。
        try:
            self._source_state.renew_source_lease(
                source_table,
                lease_owner,
                lease_seconds=self._source_lease_seconds,
            )
        except Exception as error:  # noqa: BLE001
            return SourceBatchRunResult(
                source_table=source_table,
                batch_id=batch_id,
                candidate_count=candidate_count,
                claimed_count=claimed_count,
                success_count=success_count,
                retryable_count=retryable_count,
                rejected_count=rejected_count,
                lease_conflict_count=lease_conflict_count,
                checkpoint_committed=False,
                checkpoint_already_committed=False,
                checkpoint_skipped_count=None,
                error=type(error).__name__,
            )

        # 调用现有 CheckpointManager：success 与已具备 DLQ 证据的 dead 都是终态；
        # 后者会以 failed_count 记录并允许水位继续前进。Coordinator 不自行修改
        # 水位、不释放 source lease，所有收口仍由 Manager 的单事务完成。
        try:
            commit_result = self._checkpoint.try_commit_source_batch(
                source_table,
                batch_id,
                lease_owner,
                checkpoint_lock_version=checkpoint_lock_version,
            )
        except Exception as error:  # noqa: BLE001 - 闸门冲突/未就绪统一记录
            return SourceBatchRunResult(
                source_table=source_table,
                batch_id=batch_id,
                candidate_count=candidate_count,
                claimed_count=claimed_count,
                success_count=success_count,
                retryable_count=retryable_count,
                rejected_count=rejected_count,
                lease_conflict_count=lease_conflict_count,
                checkpoint_committed=False,
                checkpoint_already_committed=False,
                checkpoint_skipped_count=None,
                error=type(error).__name__,
            )

        return SourceBatchRunResult(
            source_table=source_table,
            batch_id=batch_id,
            candidate_count=candidate_count,
            claimed_count=claimed_count,
            success_count=success_count,
            retryable_count=retryable_count,
            rejected_count=rejected_count,
            lease_conflict_count=lease_conflict_count,
            checkpoint_committed=not commit_result.already_committed,
            checkpoint_already_committed=commit_result.already_committed,
            checkpoint_skipped_count=commit_result.skipped_count,
            error=None,
        )

    def _handle_preparation_failure(
        self,
        task: SourceSyncTask,
        error: Exception,
    ) -> HandoffOrchestrationResult | None:
        """调用可选 P4 异常落定器；未配置时返回 None 以维持旧行为。"""

        if self._preparation_failure_handler is None:
            return None
        return self._preparation_failure_handler.handle(task, task.attempt_token, error)


def _classify_outcome(result: HandoffOrchestrationResult) -> str:
    """把 :class:`HandoffOrchestrationResult` 映射为 Coordinator 失败语义类别。

    返回 ``success`` / ``dead`` / ``retryable`` / ``lease_conflict`` / ``unknown``。
    """

    status = result.final_status
    terminal = result.terminal
    if status == _TERMINAL_SUCCESS and terminal:
        return "success"
    if status == _TERMINAL_DEAD and terminal:
        return "dead"
    if status == _RETRYABLE_PENDING and not terminal:
        # retryable 退回：pending + terminal=False（按 orchestrator 语义）。
        return "retryable"
    if status == _LEASE_CONFLICT_PROCESSING and not terminal:
        # fencing/lease 失配：未终态。
        return "lease_conflict"
    # 矛盾组合（如 success + not terminal）或未知 final_status：fail-closed。
    return "unknown"
