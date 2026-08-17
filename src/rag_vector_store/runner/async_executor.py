"""异步任务执行器。

Task2 Step 3：将 PipelineRunner 放入后台线程执行。

职责：
    - submit(task_id)：提交任务到 ThreadPoolExecutor 后台执行
    - 后台执行 runner.run(task_id)
    - 捕获执行异常（记录到 Future，不传播到线程池外）
    - 管理运行中的任务集合（线程安全查询 is_running / running_task_ids）

不职责：
    - 不负责 Reader / Embedding / Milvus / Task State 业务逻辑
    - 不负责防并发 full-batch（由 API 层或 Runner 前置检查）
    - 不负责重试 / 检查点 / 恢复
    - 不引入 Redis / Celery / MQ

线程安全：
    - running_futures 用 threading.Lock 保护
    - submit / is_running / running_task_ids / get_future 均线程安全
    - 异常在 worker 线程内被 Future 捕获，不会导致 executor 崩溃
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline_runner import VectorizationPipelineRunner

__all__ = ["AsyncTaskExecutor", "TaskSubmission"]

_logger = logging.getLogger(__name__)


class TaskSubmission:
    """提交后返回的轻量句柄。

    封装 task_id 和底层 Future，提供同步访问接口。
    """

    __slots__ = ("_future", "task_id")

    def __init__(self, task_id: str, future: Future[Any]) -> None:
        self.task_id = task_id
        self._future = future

    @property
    def future(self) -> Future[Any]:
        """底层 concurrent.futures.Future（供调用方 add_done_callback 等）。"""

        return self._future

    @property
    def done(self) -> bool:
        """任务是否已完成（含成功和异常）。"""

        return self._future.done()

    @property
    def running(self) -> bool:
        """任务是否仍在执行中。"""

        return not self._future.done()


class AsyncTaskExecutor:
    """基于 ThreadPoolExecutor 的异步向量化任务执行器。

    用法::

        runner = VectorizationPipelineRunner(store, reader, service)
        executor = AsyncTaskExecutor(runner, max_workers=4)
        submission = executor.submit(task_id)
        # submission.future.result() 可阻塞等待（异常会重新抛出）
    """

    def __init__(
        self,
        runner: VectorizationPipelineRunner | Any,
        *,
        max_workers: int = 4,
        thread_name_prefix: str = "vectorization-task",
    ) -> None:
        """初始化执行器。

        Args:
            runner: 已注入依赖的 VectorizationPipelineRunner（或兼容对象）。
            max_workers: 线程池最大并发数，默认 4。
            thread_name_prefix: 线程名前缀，便于日志诊断。
        """

        if max_workers < 1:
            raise ValueError(f"max_workers 必须 >= 1，收到 {max_workers}")

        self._runner = runner
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._lock = threading.Lock()
        self._running: dict[str, Future[Any]] = {}

    # ── 公共接口 ──

    def submit(self, task_id: str) -> TaskSubmission:
        """提交一个任务到后台线程执行。

        线程内调用 ``runner.run(task_id)``；任何异常被 Future 捕获，
        不会传播到执行器外。返回 TaskSubmission 句柄。

        如果 task_id 已在运行中，直接返回已有的 submission（幂等）。
        """

        task_id = task_id.strip().lower()

        with self._lock:
            existing = self._running.get(task_id)
            if existing is not None and not existing.done():
                return TaskSubmission(task_id, existing)

            future = self._executor.submit(self._run_safely, task_id)
            self._running[task_id] = future
            return TaskSubmission(task_id, future)

    def is_running(self, task_id: str) -> bool:
        """查询指定任务是否仍在执行中。"""

        task_id = task_id.strip().lower()
        with self._lock:
            future = self._running.get(task_id)
            return future is not None and not future.done()

    def running_task_ids(self) -> list[str]:
        """返回当前仍在执行中的 task_id 列表（快照）。"""

        with self._lock:
            return [
                tid
                for tid, fut in self._running.items()
                if not fut.done()
            ]

    def get_future(self, task_id: str) -> Future[Any] | None:
        """获取指定任务的 Future（已完成或运行中均返回；不存在返回 None）。"""

        task_id = task_id.strip().lower()
        with self._lock:
            return self._running.get(task_id)

    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池。

        Args:
            wait: 是否等待正在执行的任务完成。
        """

        self._executor.shutdown(wait=wait)

    # ── 内部 ──

    def _run_safely(self, task_id: str) -> Any:
        """worker 线程入口：调用 runner.run，捕获所有异常。

        异常记录到日志后重新设置到 Future（由 ThreadPoolExecutor 自动完成），
        确保 executor 不会因任务异常而崩溃。
        """

        try:
            result = self._runner.run(task_id)
            _logger.info(
                "向量化任务执行完成: task_id=%s status=%s",
                task_id,
                getattr(result, "status", "unknown"),
            )
            return result
        except Exception:
            _logger.exception("向量化任务执行异常: task_id=%s", task_id)
            raise
        finally:
            with self._lock:
                self._cleanup_done(task_id)

    def _cleanup_done(self, task_id: str) -> None:
        """清理已完成的 future（在 lock 内调用）。"""

        future = self._running.get(task_id)
        if future is not None and future.done():
            del self._running[task_id]
