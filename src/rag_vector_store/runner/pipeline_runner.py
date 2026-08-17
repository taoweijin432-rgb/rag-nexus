"""VectorizationPipelineRunner：向量化任务的执行编排层。

职责：
    - 接收一个已创建的 ``task_id``（状态 created）
    - 更新任务状态 → running
    - 读取 MySQL ``expert_chunk``（MySqlExpertChunkReader）
    - 调用 ``ContractBatchIngestionService.ingest_batch()``
    - 按结果更新 ``vectorization_task_item``（success/skipped/failed）
    - 更新任务 → completed / failed

不职责（复用已有组件，不复制逻辑）：
    - Contract 校验 / Eligibility / Embedding：``ContractBatchIngestionService``
    - Milvus 写入：``ContractRepository``
    - MySQL 读取：``MySqlExpertChunkReader``
    - Task State 持久化：``TaskStateStore``
"""

from __future__ import annotations

from typing import Any

from expert_contracts.models import ExpertChunkEmbeddingInput

from ..chunking.contract_ingestion import (
    ContractBatchIngestionResult,
    ContractBatchIngestionService,
)
from ..mysql.expert_chunk_reader import MySqlExpertChunkReader, MySQLReadResult
from ..task_state.models import TaskItemState, TaskState
from ..task_state.store import TaskStateError, TaskStateStore

__all__ = ["PipelineRunnerError", "VectorizationPipelineRunner"]


class PipelineRunnerError(RuntimeError):
    """Pipeline Runner 编排错误（安全摘要，不含 PII / chunk_text / 向量）。"""


class VectorizationPipelineRunner:
    """全量批次向量化任务的执行编排器。

    调用链（成功）：
        run(task_id)
        -> store.get_task(task_id)                  # 前置：created
        -> store.update_task_status(running)        # 状态 → running
        -> reader.read()                            # MySQL 读取
        -> store.create_items(pending)              # 条目 → pending
        -> service.ingest_batch(records)            # 已有批量编排（校验/Embedding/写入）
        -> store.update_item_status(...)            # 每条 → success / skipped / failed
        -> store.update_task_status(completed)      # 状态 → completed

    调用链（失败）：
        run(task_id)
        -> store.update_task_status(running)
        -> ... 任意阶段异常 ...
        -> store.update_task_status(failed)         # 状态 → failed（安全摘要）
        -> 返回 failed 的 TaskState

    前置条件错误（任务不存在 / 状态不是 created）在进入 running 之前抛出
    ``PipelineRunnerError``，任务状态不变。
    """

    def __init__(
        self,
        task_state_store: TaskStateStore,
        reader: MySqlExpertChunkReader | Any,
        ingestion_service: ContractBatchIngestionService | Any,
    ) -> None:
        """注入依赖。

        ``reader`` / ``ingestion_service`` 可以直接传实例，也可以传零参工厂
        （延迟创建，便于 API 层按请求绑定独立连接）。
        """

        self.task_state_store = task_state_store
        self.reader = reader
        self.ingestion_service = ingestion_service

    def run(self, task_id: str) -> TaskState:
        """执行一次向量化任务，返回执行后的任务状态。

        执行期任何异常（读取 / 条目创建 / 批量编排 / 状态更新）都会把任务
        标记为 ``failed`` 后返回；不会向调用方抛业务异常。
        仅前置条件错误（任务不存在或状态不是 created）抛出
        ``PipelineRunnerError``，此时任务状态未被修改。
        """

        task_id = task_id.strip().lower()
        self._load_task(task_id)

        records: list[ExpertChunkEmbeddingInput] = []

        try:
            # 1. 状态 → running
            self._update_task(task_id, status="running", started_at=True)

            # 2. 读取 MySQL（全量）
            read_result = self._read_records()
            records = read_result.records

            # 3. 预创建条目（pending）
            self._create_items(task_id, records)

            # 4. 调用已有批量编排（校验 / Eligibility / Embedding / Milvus）
            result = self._ingest(task_id, records)

            # 5. 按结果更新每条条目
            self._update_items_from_result(task_id, result)

            # 6. 状态 → completed
            self._update_task(
                task_id,
                status="completed",
                success_count=result.embedded_count,
                skipped_count=result.skipped_count,
                failed_count=0,
                finished_at=True,
            )
        except Exception as exc:  # noqa: BLE001
            # 任何执行期异常 → failed（安全摘要，不泄露 PII / chunk_text / 向量）
            failed_count = self._mark_items_failed(task_id, records)
            self._update_task(
                task_id,
                status="failed",
                failed_count=failed_count,
                error_message=_safe_summary(exc)[:512],
                finished_at=True,
            )

        return self._get_task(task_id)

    # ── 内部步骤 ──

    def _load_task(self, task_id: str) -> TaskState:
        """加载任务并校验前置条件：必须存在且状态为 created。"""

        try:
            task = self.task_state_store.get_task(task_id)
        except TaskStateError as exc:
            raise PipelineRunnerError(f"任务不存在: {task_id!r}") from exc

        if task.status != "created":
            raise PipelineRunnerError(
                f"任务状态不允许执行: {task.status!r}（仅 created 可运行）"
            )
        return task

    def _read_records(self) -> MySQLReadResult:
        """调用 Reader；读取错误与单行映射错误统一 fail-closed。"""

        reader = self._resolve_reader()
        try:
            result = reader.read()
        except Exception as exc:
            raise PipelineRunnerError(
                "MySQL 读取失败: " + _safe_summary(exc)
            ) from exc

        if result.errors:
            details = "; ".join(
                f"{err.code}@{err.row_id or '-'}" for err in result.errors[:5]
            )
            raise PipelineRunnerError(f"MySQL 行映射错误: {details}")
        return result

    def _create_items(
        self,
        task_id: str,
        records: list[ExpertChunkEmbeddingInput],
    ) -> None:
        """预创建条目（pending），保证 task_item 与输入记录一一对应。"""

        if not records:
            return
        items = [
            TaskItemState(
                task_id=task_id,
                vector_doc_id=record.vector_doc_id,
                status="pending",
            )
            for record in records
        ]
        self._state_guard(lambda: self.task_state_store.create_items(items))

    def _ingest(
        self,
        task_id: str,
        records: list[ExpertChunkEmbeddingInput],
    ) -> ContractBatchIngestionResult:
        """调用已有批量编排；Task State 异常包装为 PipelineRunnerError。"""

        service = self._resolve_service()
        try:
            return service.ingest_batch(records)
        except Exception as exc:
            raise PipelineRunnerError(
                "向量化批量执行失败: " + _safe_summary(exc)
            ) from exc

    def _mark_items_failed(self, task_id: str, records: list[ExpertChunkEmbeddingInput]) -> int:
        """异常时将已创建的 items 标记为 failed；忽略更新失败的条目。"""

        count = 0
        for record in records:
            try:
                self.task_state_store.update_item_status(
                    task_id,
                    record.vector_doc_id,
                    status="failed",
                    result="failed",
                    error_message="batch ingestion failed",
                )
                count += 1
            except Exception:  # noqa: S110, BLE001
                pass
        return count

    def _update_items_from_result(
        self,
        task_id: str,
        result: ContractBatchIngestionResult,
    ) -> None:
        """按批量结果更新每条 task_item 为 success / skipped / failed。"""

        for item_result in result.item_results:
            if item_result.status == "embedded":
                status, result_value, error = "success", "embedded", None
            elif item_result.status == "skipped":
                status = "skipped"
                result_value = item_result.skip_reason or "eligibility"
                error = None
            else:  # 防御：ContractIngestionResult 只有 embedded/skipped
                status = "failed"
                result_value = "unknown"
                error = "未知结果状态: " + str(item_result.status)

            self._safe_update_item(
                task_id,
                item_result.vector_doc_id,
                status,
                result_value,
                error,
            )

    def _safe_update_item(
        self,
        task_id: str,
        vector_doc_id: str,
        status: str,
        result_value: str,
        error: str | None,
    ) -> None:
        """更新单条 item 状态，TaskStateError 转换为 PipelineRunnerError。"""

        try:
            self.task_state_store.update_item_status(
                task_id,
                vector_doc_id,
                status=status,
                result=result_value,
                error_message=error,
            )
        except TaskStateError as exc:
            raise PipelineRunnerError(
                "Task State 操作失败: " + _safe_summary(exc)
            ) from exc

    def _update_task(self, task_id: str, **kwargs: Any) -> None:
        """更新任务状态；Task State 异常包装为 PipelineRunnerError。"""

        self._state_guard(lambda: self.task_state_store.update_task_status(task_id, **kwargs))

    def _get_task(self, task_id: str) -> TaskState:
        try:
            return self.task_state_store.get_task(task_id)
        except TaskStateError as exc:
            raise PipelineRunnerError(
                "任务状态查询失败: " + _safe_summary(exc)
            ) from exc

    def _state_guard(self, operation: Any) -> None:
        """把 TaskStateError 转换为 PipelineRunnerError。"""

        try:
            operation()
        except TaskStateError as exc:
            raise PipelineRunnerError(
                "Task State 操作失败: " + _safe_summary(exc)
            ) from exc

    def _resolve_reader(self) -> Any:
        """支持实例或零参工厂。"""

        reader = self.reader
        if isinstance(reader, MySqlExpertChunkReader) or not callable(reader):
            return reader
        return reader()

    def _resolve_service(self) -> Any:
        """支持实例或零参工厂。"""

        service = self.ingestion_service
        if isinstance(service, ContractBatchIngestionService) or not callable(
            service
        ):
            return service
        return service()


def _safe_summary(exc: Exception) -> str:
    """返回安全错误摘要（不含 PII / chunk_text / 向量 / 凭据）。"""

    return f"{type(exc).__name__}: {str(exc)[:200]}"