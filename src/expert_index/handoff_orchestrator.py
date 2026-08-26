"""P5.3B 正式 Handoff 编排闭环。

把 P5.1 ``ScopeHandoffBuilder``、P5.3A ``MySQLArtifactRepository`` 与 task 状态接口、
P5.2 ``ScopeConsumer`` 接成单一闭环，按 P5.3/P5.4 恢复规则安全推进与原子落定：

    claimed task + PreparedChunkSet
    → ScopeHandoffBuilder（仅 validated 且无 Artifact 时）
    → Artifact put_or_verify（带 attempt_token fencing）
    → advance contract_ready
    → advance handed_off
    → ScopeConsumer.process
    → Receipt 校验（identity/token/scope/payload_hash/status 全一致）
    → success / rejected / retryable 状态流转

设计约束（对齐 V2.2 §P5.3–P5.6 与 P5.3A 边界）：

- **不新增状态表、Repository 或 Migration**，只复用 P5.3A 的
  ``MySQLArtifactRepository`` 与 task repository 的公开方法。
- **使用现有 ``task_id``/``batch_id``/``attempt_token``，所有写操作都带当前 fencing
  token**；捕获 ``SourceSyncTaskLeaseConflictError`` 后立即停止，旧 token 绝不覆盖
  新 attempt。
- **Artifact 同 identity 同 hash 重放复用**；同 identity 不同 hash 冲突立即
  ``mark_source_sync_task_rejected(error_stage="artifact")``，**禁止调用 Consumer**。
- **阶段幂等推进**：
  - ``validated``：Builder 或 Artifact 恢复 → put_or_verify → contract_ready →
    handed_off；
  - ``contract_ready``：只从 Artifact 恢复 Request → handed_off；
  - ``handed_off``：只从 Artifact 恢复，**不回退阶段、不重新构建**，直接重放
    Consumer。
- **Receipt identity/token/scope/payload_hash 必须与当前 Request 完全一致**，
  任何不一致 → 确定性 ``rejected(error_stage="receipt")``。
- **committed → ``mark_source_sync_task_success``**；rejected 或 non-retryable failed
  → ``mark_source_sync_task_rejected``；retryable failed →
  ``mark_source_sync_task_retryable``。Consumer/Artifact 异常按确定性与瞬时分类，
  记录正确 ``error_stage``。
- **不直接推进 checkpoint**；P3 继续以任务成功状态驱动水位。
- **不使用 V15 Work Item/Batch**，不物理删除。
- **不声称单事务原子闭环**：``advance_source_sync_task_stage`` 与
  ``mark_source_sync_task_success`` 分两次提交存在崩溃窗口，本编排器依赖按同一
  Artifact 重放恢复，不绕过 ``advance`` 的相邻阶段守卫，也不把 ``handed_off`` 当
  成功。

Artifact 恢复语义（P5.4 关键）：P5.1 的 ``request_id`` 派生自
``task_id:batch_id:attempt_token``，新 attempt 的 ``request_id`` 会变化。恢复旧
Artifact 时只取稳定的 ``impact_scope``/``contracts``/``payload_hash``，用**当前**
task/batch/token 重建当前 Request identity，并重新校验重建 Request 的
``payload_hash`` 与 Artifact hash 一致；只有 ``validated`` 且无 Artifact 时才用传入
的 ``PreparedChunkSet`` 经 Builder 构建，不重新调用 Builder 生成新 ``updated_at``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .handoff import ScopeHandoffBuilder, ScopeHandoffReceipt, ScopeHandoffRequest
from .models import PreparedChunkSet, SourceSyncTask
from .repositories.artifact import (
    ArtifactPayloadConflictError,
    ArtifactPayloadValidationError,
    ContractArtifact,
)
from .repositories.sync_task import (
    SourceSyncTaskLeaseConflictError,
    SourceSyncTaskStageConflictError,
)

__all__ = [
    "ArtifactRepositoryPort",
    "HandoffOrchestrationResult",
    "HandoffOrchestrator",
    "ReceiptValidationError",
    "TaskRepositoryPort",
]

# Artifact 持久化命名约定（与 P5.3A 表结构与白名单一致）。
_ARTIFACT_TYPE = "contract_payload"
_ARTIFACT_VERSION = "v1"

# 可重试基础设施错误分类使用的稳定标识（不进入 PII，单行截断）。
_ERROR_CODE_ARTIFACT_PUT = "artifact_put_failed"
_ERROR_CODE_ARTIFACT_BUILD = "artifact_build_failed"
_ERROR_CODE_RECEIPT_INVALID = "receipt_invalid"
_ERROR_CODE_CONSUMER = "consumer_failed"
_ERROR_CODE_FENCING = "fencing_lost"
_ERROR_CODE_UNSUPPORTED_STAGE = "unsupported_stage"


class ReceiptValidationError(RuntimeError):
    """回执与当前 Request 的 identity/token/scope/payload_hash 不一致。"""


@runtime_checkable
class TaskRepositoryPort(Protocol):
    """P5.3A task 状态接口的结构化只读协议。

    生产实现为现有 ``MySQLSourceSyncTaskRepository``；测试注入 Fake。本编排器只使用
    这些公开方法，不直接访问其内部 cursor/SQL。
    """

    def get_source_sync_task(self, task_id: int) -> SourceSyncTask | None: ...

    def advance_source_sync_task_stage(
        self,
        task_id: int,
        attempt_token: str,
        *,
        next_stage: str,
    ) -> SourceSyncTask: ...

    def mark_source_sync_task_success(
        self,
        task_id: int,
        attempt_token: str,
        *,
        result_code: str,
    ) -> SourceSyncTask: ...

    def mark_source_sync_task_rejected(
        self,
        task_id: int,
        attempt_token: str,
        *,
        error_stage: str,
        error_code: str,
        error_type: str | None = None,
        safe_error_message: str | None = None,
    ) -> SourceSyncTask: ...

    def mark_source_sync_task_retryable(
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
    ) -> SourceSyncTask: ...


@runtime_checkable
class ArtifactRepositoryPort(Protocol):
    """P5.3A Artifact 接口的结构化只读协议。"""

    def get_contract_artifact(
        self,
        task_id: int,
        *,
        artifact_type: str,
        artifact_version: str,
    ) -> ContractArtifact | None: ...

    def put_or_verify_contract_artifact(
        self,
        task_id: int,
        attempt_token: str,
        *,
        artifact_type: str,
        artifact_version: str,
        payload_json: str,
        payload_hash: str,
        expires_at: Any = None,
    ) -> ContractArtifact: ...


@runtime_checkable
class ScopeConsumerPort(Protocol):
    """P5.2 ``ScopeConsumer.process`` 的结构化只读协议。"""

    def process(self, request: ScopeHandoffRequest) -> ScopeHandoffReceipt: ...


@dataclass(frozen=True)
class HandoffOrchestrationResult:
    """编排一次执行的结构化结果。

    ``final_status`` 取任务最终状态（success/dead/pending），``terminal`` 表示是否
    已落定（不再可重试）；fencing 失败或回执非法导致 ``final_status`` 仍为
    ``processing`` 时 ``terminal=False``，调用方按未终态处理。
    """

    final_status: str
    terminal: bool
    result_code: str | None
    error_stage: str | None


class HandoffOrchestrator:
    """正式 Handoff 编排闭环。

    注入：``task_repo``（P5.3A）、``artifact_repo``（P5.3A）、``builder``（P5.1）、
    ``consumer``（P5.2）、``receipt_factory``（``ScopeHandoffReceipt`` 构造器，用于
    恢复 Request identity 重建时构造真实 Receipt 路径——实际生产由 Consumer 返回，
    此参数为 ``ScopeConsumer`` 使用的 sink 一致类型）。

    ``max_attempts`` 与 ``retry_delay_seconds`` 是显式参数（不猜全局默认），用于
    ``mark_source_sync_task_retryable``。
    """

    def __init__(
        self,
        *,
        task_repo: TaskRepositoryPort,
        artifact_repo: ArtifactRepositoryPort,
        builder: ScopeHandoffBuilder,
        consumer: ScopeConsumerPort,
        max_attempts: int,
        retry_delay_seconds: int,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
            raise ValueError("max_attempts 必须是非负整数")
        if isinstance(retry_delay_seconds, bool) or not isinstance(retry_delay_seconds, int) or retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds 必须是正整数")
        self._task_repo = task_repo
        self._artifact_repo = artifact_repo
        self._builder = builder
        self._consumer = consumer
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds

    def process(self, task_id: int, attempt_token: str, prepared: PreparedChunkSet) -> HandoffOrchestrationResult:
        """对单个已领取任务执行 Handoff 编排闭环。

        ``prepared`` 仅在 ``validated`` 且无 Artifact 时被 Builder 使用；其余阶段
        只从 Artifact 恢复 Request。所有写操作携带 ``attempt_token`` fencing。
        """

        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 1:
            raise ValueError("task_id 必须是正整数")
        if not isinstance(attempt_token, str) or not attempt_token.strip():
            raise ValueError("attempt_token 必须是非空字符串")
        if not isinstance(prepared, PreparedChunkSet):
            raise TypeError("prepared 必须是 PreparedChunkSet")

        current = self._task_repo.get_source_sync_task(task_id)
        if current is None:
            raise ValueError(f"任务 {task_id} 不存在")
        # 已终态（success/dead）幂等返回，不重复处理，也不校验 token（成功/死亡任务已清除 token）。
        if current.sync_status in ("success", "dead"):
            return HandoffOrchestrationResult(
                final_status=current.sync_status,
                terminal=True,
                result_code=current.result_code,
                error_stage=None,
            )
        if current.sync_status != "processing":
            # pending/retrying 等非 processing 状态：不在 P5.3B 编排范围，幂等返回未终态。
            return HandoffOrchestrationResult(
                final_status=current.sync_status,
                terminal=False,
                result_code=current.result_code,
                error_stage=None,
            )
        if current.attempt_token is None or current.attempt_token != attempt_token:
            # 旧 token 晚到 / 任务已被新 attempt 接管：绝不覆盖，直接返回未终态。
            return HandoffOrchestrationResult(
                final_status=current.sync_status,
                terminal=False,
                result_code=None,
                error_stage=None,
            )

        stage = current.processing_stage
        if stage == "validated":
            return self._from_validated(current, attempt_token, prepared)
        if stage == "contract_ready":
            return self._from_contract_ready_or_handed_off(current, attempt_token, is_handed_off=False)
        if stage == "handed_off":
            return self._from_contract_ready_or_handed_off(current, attempt_token, is_handed_off=True)
        # 其他阶段（candidate/source_read/normalized/chunk_built/completed）不在 P5.3B
        # 编排范围，fail-closed deterministic rejected。
        return self._reject(
            current,
            attempt_token,
            error_stage="handoff",
            error_code=_ERROR_CODE_UNSUPPORTED_STAGE,
            safe_error_message=f"P5.3B 不支持从 {stage} 编排",
        )

    # ------------------------------------------------------------------
    # 阶段入口
    # ------------------------------------------------------------------
    def _from_validated(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        prepared: PreparedChunkSet,
    ) -> HandoffOrchestrationResult:
        """validated：优先复用已存 Artifact，无则经 Builder 构建后 put_or_verify。"""

        existing = self._artifact_repo.get_contract_artifact(
            task.task_id,
            artifact_type=_ARTIFACT_TYPE,
            artifact_version=_ARTIFACT_VERSION,
        )
        if existing is not None:
            # validated + 已有 Artifact：验证一致后推进，不重新调用 Builder。
            request, verify_error = self._request_from_artifact(task, attempt_token, existing)
            if verify_error is not None:
                # Artifact 内容损坏或 payload_hash 不一致（确定性）。
                return self._reject(
                    task,
                    attempt_token,
                    error_stage="artifact",
                    error_code=_ERROR_CODE_ARTIFACT_BUILD,
                    safe_error_message=_single_line_truncate(verify_error),
                )
            return self._persist_and_advance_to_contract_ready(task, attempt_token, request, existing)

        # 无 Artifact：经 Builder 构建（首次或新 attempt 重建当前事实后）。
        try:
            request = self._builder.build(task, attempt_token, prepared)
        except (ValueError, TypeError) as error:
            # Builder 契约/身份校验失败：确定性 rejected。
            return self._reject(
                task,
                attempt_token,
                error_stage="handoff",
                error_code=_ERROR_CODE_ARTIFACT_BUILD,
                safe_error_message=_single_line_truncate(str(error)),
            )
        return self._persist_and_advance_to_contract_ready(task, attempt_token, request, None)

    def _from_contract_ready_or_handed_off(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        *,
        is_handed_off: bool,
    ) -> HandoffOrchestrationResult:
        """contract_ready / handed_off：只从 Artifact 恢复，不回退、不重新构建。"""

        artifact = self._artifact_repo.get_contract_artifact(
            task.task_id,
            artifact_type=_ARTIFACT_TYPE,
            artifact_version=_ARTIFACT_VERSION,
        )
        if artifact is None:
            # contract_ready/handed_off 必须存在 Artifact（P5.4 不允许阶段先走而 Artifact
            # 缺失）。确定性 rejected，不重建。
            return self._reject(
                task,
                attempt_token,
                error_stage="artifact",
                error_code=_ERROR_CODE_ARTIFACT_BUILD,
                safe_error_message="contract_ready/handed_off 缺少可恢复 Artifact",
            )
        request, verify_error = self._request_from_artifact(task, attempt_token, artifact)
        if verify_error is not None:
            return self._reject(
                task,
                attempt_token,
                error_stage="artifact",
                error_code=_ERROR_CODE_ARTIFACT_BUILD,
                safe_error_message=_single_line_truncate(verify_error),
            )
        if not is_handed_off:
            # contract_ready：推进到 handed_off 后执行 Consumer。
            advance_result = self._safe_advance(task, attempt_token, "handed_off")
            if advance_result.failure is not None:
                return advance_result.failure
        # handed_off：直接重放 Consumer（不回退阶段、不重建）。
        return self._run_consumer(task, attempt_token, request)

    # ------------------------------------------------------------------
    # Artifact 持久化 + contract_ready 推进
    # ------------------------------------------------------------------
    def _persist_and_advance_to_contract_ready(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        request: ScopeHandoffRequest,
        existing: ContractArtifact | None,
    ) -> HandoffOrchestrationResult:
        """put_or_verify → advance contract_ready → advance handed_off → Consumer。"""

        payload_json = self._serialize_payload(request)
        try:
            if existing is not None:
                # 已存在 Artifact：内容已通过重建校验，fenced put_or_verify 复用/校验。
                self._artifact_repo.put_or_verify_contract_artifact(
                    task.task_id,
                    attempt_token,
                    artifact_type=_ARTIFACT_TYPE,
                    artifact_version=_ARTIFACT_VERSION,
                    payload_json=payload_json,
                    payload_hash=request.payload_hash,
                )
            else:
                self._artifact_repo.put_or_verify_contract_artifact(
                    task.task_id,
                    attempt_token,
                    artifact_type=_ARTIFACT_TYPE,
                    artifact_version=_ARTIFACT_VERSION,
                    payload_json=payload_json,
                    payload_hash=request.payload_hash,
                )
        except ArtifactPayloadConflictError as error:
            # 同 identity 不同 hash：确定性 rejected，禁止调用 Consumer。
            return self._reject(
                task,
                attempt_token,
                error_stage="artifact",
                error_code=_ERROR_CODE_ARTIFACT_PUT,
                safe_error_message=_single_line_truncate(str(error)),
            )
        except ArtifactPayloadValidationError as error:
            # payload 白名单/格式非法：确定性 rejected。
            return self._reject(
                task,
                attempt_token,
                error_stage="artifact",
                error_code=_ERROR_CODE_ARTIFACT_PUT,
                safe_error_message=_single_line_truncate(str(error)),
            )
        except (ValueError, TypeError) as error:
            # 参数校验非法：确定性 rejected。
            return self._reject(
                task,
                attempt_token,
                error_stage="artifact",
                error_code=_ERROR_CODE_ARTIFACT_PUT,
                safe_error_message=_single_line_truncate(str(error)),
            )
        except Exception as error:  # noqa: BLE001 - 基础设施失败统一转可重试
            return self._retryable(
                task,
                attempt_token,
                error_stage="artifact",
                error_code=_ERROR_CODE_ARTIFACT_PUT,
                safe_error_message=_single_line_truncate(f"Artifact 写入基础设施失败：{error}"),
            )

        # 推进 contract_ready（validated 必经阶段）。
        advance = self._safe_advance(task, attempt_token, "contract_ready")
        if advance.failure is not None:
            return advance.failure
        # 再推进 handed_off。
        advance2 = self._safe_advance(task, attempt_token, "handed_off")
        if advance2.failure is not None:
            return advance2.failure
        return self._run_consumer(task, attempt_token, request)

    # ------------------------------------------------------------------
    # Consumer 执行 + Receipt 校验 + 状态落定
    # ------------------------------------------------------------------
    def _run_consumer(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        request: ScopeHandoffRequest,
    ) -> HandoffOrchestrationResult:
        """调用 ScopeConsumer，校验 Receipt，按终态落定任务。"""

        try:
            receipt = self._consumer.process(request)
        except (ValueError, TypeError) as error:
            # 确定性契约/范围/状态非法：rejected。
            return self._reject(
                task,
                attempt_token,
                error_stage="scope_consumer",
                error_code=_ERROR_CODE_CONSUMER,
                safe_error_message=_single_line_truncate(str(error)),
            )
        except Exception as error:  # noqa: BLE001 - 向量/网络/下游瞬时失败统一可重试
            return self._retryable(
                task,
                attempt_token,
                error_stage="scope_consumer",
                error_code=_ERROR_CODE_CONSUMER,
                safe_error_message=_single_line_truncate(f"Consumer 调用瞬时失败：{error}"),
            )

        # 严格校验 Receipt 与当前 Request 一致。
        try:
            self._validate_receipt(request, receipt)
        except ReceiptValidationError as error:
            return self._reject(
                task,
                attempt_token,
                error_stage="receipt",
                error_code=_ERROR_CODE_RECEIPT_INVALID,
                safe_error_message=_single_line_truncate(str(error)),
            )

        if receipt.status == "committed":
            return self._mark_success(task, attempt_token, receipt.result_code)
        if receipt.status == "rejected":
            return self._reject(
                task,
                attempt_token,
                error_stage="scope_consumer",
                error_code=receipt.result_code,
                error_type=receipt.error_type,
                safe_error_message=receipt.error_message,
            )
        # failed：按 retryable 决定 retryable / rejected。
        if receipt.retryable:
            return self._retryable(
                task,
                attempt_token,
                error_stage="scope_consumer",
                error_code=receipt.result_code,
                error_type=receipt.error_type,
                safe_error_message=receipt.error_message,
            )
        return self._reject(
            task,
            attempt_token,
            error_stage="scope_consumer",
            error_code=receipt.result_code,
            error_type=receipt.error_type,
            safe_error_message=receipt.error_message,
        )

    # ------------------------------------------------------------------
    # Artifact → Request 恢复
    # ------------------------------------------------------------------
    def _request_from_artifact(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        artifact: ContractArtifact,
    ) -> tuple[ScopeHandoffRequest | None, str | None]:
        """从 Artifact 恢复当前 Request（用当前 token 重建 identity）。

        返回 ``(request, error_message)``，error_message 非 None 表示恢复失败
        （payload 损坏、payload_hash 不一致等确定性原因）。
        """

        try:
            payload = json.loads(artifact.payload_json or "{}")
        except json.JSONDecodeError as error:
            return None, f"Artifact payload 不是合法 JSON: {error}"
        if not isinstance(payload, dict):
            return None, "Artifact payload 顶层不是 JSON 对象"
        scope_obj = payload.get("impact_scope")
        contracts_list = payload.get("contracts")
        declared_hash = payload.get("payload_hash")
        if not isinstance(declared_hash, str) or len(declared_hash) != 64:
            return None, "Artifact payload 缺少合法 payload_hash"
        if artifact.payload_hash != declared_hash:
            return None, "Artifact payload_hash 与持久化行不一致"
        try:
            scope = _scope_from_payload(scope_obj, task)
            contracts = _contracts_from_payload(contracts_list, scope)
            # 用当前 task/batch/token 重建当前 Request identity。
            request = _rebuild_request(
                task=task,
                attempt_token=attempt_token,
                impact_scope=scope,
                contracts=contracts,
            )
        except (ValueError, TypeError) as error:
            return None, f"Artifact 恢复失败: {error}"
        if request.payload_hash != artifact.payload_hash:
            return None, "重建 Request 的 payload_hash 与 Artifact 不一致"
        return request, None

    # ------------------------------------------------------------------
    # Receipt 校验
    # ------------------------------------------------------------------
    def _validate_receipt(
        self,
        request: ScopeHandoffRequest,
        receipt: ScopeHandoffReceipt,
    ) -> None:
        """严格比较 Receipt 与当前 Request 的全部 identity/scope/payload_hash。"""

        if receipt.request_id != request.request_id:
            raise ReceiptValidationError("request_id 不一致")
        if receipt.task_id != request.task_id:
            raise ReceiptValidationError("task_id 不一致")
        if receipt.batch_id != request.batch_id:
            raise ReceiptValidationError("batch_id 不一致")
        if receipt.attempt_token != request.attempt_token:
            raise ReceiptValidationError("attempt_token 不一致")
        if receipt.payload_hash != request.payload_hash:
            raise ReceiptValidationError("payload_hash 不一致")
        if receipt.impact_scope != request.impact_scope:
            raise ReceiptValidationError("impact_scope 不一致")
        if receipt.status not in ("committed", "rejected", "failed"):
            raise ReceiptValidationError(f"未知回执状态: {receipt.status!r}")
        # committed 不得带错误；rejected/failed 必须带错误。
        if receipt.status == "committed":
            if receipt.error_type is not None or receipt.error_message is not None:
                raise ReceiptValidationError("committed 回执不得携带错误字段")
        else:
            if receipt.error_type is None or receipt.error_message is None:
                raise ReceiptValidationError(f"{receipt.status} 回执必须携带 error_type/error_message")
            if receipt.status == "rejected" and receipt.retryable:
                raise ReceiptValidationError("rejected 回执 retryable 必须为 False")

    # ------------------------------------------------------------------
    # 状态落定辅助
    # ------------------------------------------------------------------
    def _mark_success(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        result_code: str,
    ) -> HandoffOrchestrationResult:
        try:
            self._task_repo.mark_source_sync_task_success(
                task.task_id, attempt_token, result_code=result_code,
            )
        except SourceSyncTaskLeaseConflictError:
            return HandoffOrchestrationResult(
                final_status="processing", terminal=False, result_code=None, error_stage="fencing",
            )
        except SourceSyncTaskStageConflictError:
            return HandoffOrchestrationResult(
                final_status="processing", terminal=False, result_code=None, error_stage="fencing",
            )
        return HandoffOrchestrationResult(
            final_status="success", terminal=True, result_code=result_code, error_stage=None,
        )

    def _reject(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        *,
        error_stage: str,
        error_code: str,
        error_type: str | None = None,
        safe_error_message: str | None,
    ) -> HandoffOrchestrationResult:
        try:
            self._task_repo.mark_source_sync_task_rejected(
                task.task_id,
                attempt_token,
                error_stage=error_stage,
                error_code=error_code,
                error_type=error_type,
                safe_error_message=safe_error_message,
            )
        except SourceSyncTaskLeaseConflictError:
            return HandoffOrchestrationResult(
                final_status="processing", terminal=False, result_code=None, error_stage="fencing",
            )
        return HandoffOrchestrationResult(
            final_status="dead", terminal=True, result_code=None, error_stage=error_stage,
        )

    def _retryable(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        *,
        error_stage: str,
        error_code: str,
        error_type: str | None = None,
        safe_error_message: str | None,
    ) -> HandoffOrchestrationResult:
        try:
            updated = self._task_repo.mark_source_sync_task_retryable(
                task.task_id,
                attempt_token,
                error_stage=error_stage,
                error_code=error_code,
                max_attempts=self._max_attempts,
                retry_delay_seconds=self._retry_delay_seconds,
                error_type=error_type,
                safe_error_message=safe_error_message,
            )
        except SourceSyncTaskLeaseConflictError:
            return HandoffOrchestrationResult(
                final_status="processing", terminal=False, result_code=None, error_stage="fencing",
            )
        # retry_count 达上限则转 dead（终态），否则 pending（未终态）。
        terminal = updated.sync_status == "dead"
        return HandoffOrchestrationResult(
            final_status=updated.sync_status,
            terminal=terminal,
            result_code=None,
            error_stage=error_stage,
        )

    # ------------------------------------------------------------------
    # 阶段推进（带 fencing 守卫）
    # ------------------------------------------------------------------
    def _safe_advance(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        next_stage: str,
    ) -> _AdvanceOutcome:
        """推进阶段，捕获 fencing/stage conflict 转为对应结果。"""

        try:
            self._task_repo.advance_source_sync_task_stage(
                task.task_id, attempt_token, next_stage=next_stage,
            )
        except SourceSyncTaskLeaseConflictError:
            return _AdvanceOutcome(
                failure=HandoffOrchestrationResult(
                    final_status="processing", terminal=False, result_code=None, error_stage="fencing",
                ),
            )
        except SourceSyncTaskStageConflictError as error:
            # 非相邻阶段（如重放时已 contract_ready 再次推进）：按确定性 rejected。
            return _AdvanceOutcome(
                failure=self._reject(
                    task,
                    attempt_token,
                    error_stage="handoff",
                    error_code=_ERROR_CODE_UNSUPPORTED_STAGE,
                    safe_error_message=_single_line_truncate(str(error)),
                ),
            )
        return _AdvanceOutcome(failure=None)

    # ------------------------------------------------------------------
    # payload 序列化
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_payload(request: ScopeHandoffRequest) -> str:
        """序列化为 Artifact payload_json（白名单：identity/scope/contracts/hash）。

        不携带 embedding/previous_state/action/stale_vector_doc_ids 等禁止字段；
        Contract 用 ``model_dump(mode="json")``，保留列表原顺序。``attempt_token``
        不存（fencing 已由 task 行锁完成，P5.3A 明确不保存 token 到 Artifact JSON）。
        """

        import hashlib

        scope_obj = {
            "scope_type": request.impact_scope.scope_type,
            "expert_id": request.impact_scope.expert_id,
            "chunk_type": request.impact_scope.chunk_type,
            "source_table": request.impact_scope.source_table,
            "source_field_id": request.impact_scope.source_field_id,
        }
        contracts = [c.model_dump(mode="json") for c in request.contracts]
        payload = {
            "request_id": request.request_id,
            "task_id": request.task_id,
            "batch_id": request.batch_id,
            "attempt_token_hash": hashlib.sha256(request.attempt_token.encode("utf-8")).hexdigest(),
            "impact_scope": scope_obj,
            "contracts": contracts,
            "payload_hash": request.payload_hash,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 恢复用纯函数（不依赖编排器实例状态，便于测试与复用）
# ---------------------------------------------------------------------------
def _scope_from_payload(scope_obj: object, task: SourceSyncTask) -> Any:
    """从 payload 的 impact_scope 重建 ``IncrementalImpactScope``。"""

    from .models import IncrementalImpactScope

    if not isinstance(scope_obj, dict):
        raise TypeError("impact_scope 不是 JSON 对象")
    return IncrementalImpactScope(
        scope_type=str(scope_obj["scope_type"]),
        expert_id=int(scope_obj["expert_id"]),
        chunk_type=scope_obj.get("chunk_type"),
        source_table=scope_obj.get("source_table"),
        source_field_id=scope_obj.get("source_field_id"),
    )


def _contracts_from_payload(contracts_list: object, scope: Any) -> tuple[Any, ...]:
    """从 payload 的 contracts 列表重建 ``ExpertChunkEmbeddingInput`` 元组。"""

    from expert_contracts import ExpertChunkEmbeddingInput

    if not isinstance(contracts_list, list):
        raise TypeError("contracts 不是 JSON 数组")
    contracts = tuple(
        ExpertChunkEmbeddingInput.model_validate(contract) for contract in contracts_list
    )
    return contracts


def _rebuild_request(
    *,
    task: SourceSyncTask,
    attempt_token: str,
    impact_scope: Any,
    contracts: tuple[Any, ...],
) -> ScopeHandoffRequest:
    """用当前 task/batch/token 重建当前 Request（identity 由当前 attempt 决定）。"""

    import hashlib

    from .handoff import _REQUEST_ID_PREFIX, _sha256_hex

    request_id = (
        f"{_REQUEST_ID_PREFIX}:"
        f"{_sha256_hex(f'{task.task_id}:{task.batch_id}:{attempt_token}')}"
    )
    # ScopeHandoffRequest.__post_init__ 会复算 payload_hash 校验，必须传入与内容一致的 hash。
    scope_obj = {
        "scope_type": impact_scope.scope_type,
        "expert_id": impact_scope.expert_id,
        "chunk_type": impact_scope.chunk_type,
        "source_table": impact_scope.source_table,
        "source_field_id": impact_scope.source_field_id,
    }
    payload = {
        "task_id": task.task_id,
        "batch_id": task.batch_id,
        "impact_scope": scope_obj,
        "contracts": [c.model_dump(mode="json") for c in contracts],
    }
    import json as _json

    serialized = _json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return ScopeHandoffRequest(
        request_id=request_id,
        task_id=task.task_id,
        batch_id=task.batch_id,
        attempt_token=attempt_token,
        impact_scope=impact_scope,
        contracts=contracts,
        payload_hash=payload_hash,
    )


def _single_line_truncate(value: str, limit: int = 512) -> str:
    """归一为单行并截断到限制长度。"""

    flat = " ".join(value.splitlines())
    if len(flat) > limit:
        flat = flat[:limit]
    return flat


@dataclass(frozen=True)
class _AdvanceOutcome:
    failure: HandoffOrchestrationResult | None
