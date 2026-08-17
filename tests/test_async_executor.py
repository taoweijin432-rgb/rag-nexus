"""AsyncTaskExecutor 单元测试。

Task2 Step 3：验证异步执行器的提交、后台调用、异常隔离、多任务并发。

全部使用内存 Fake，不依赖真实 MySQL / Milvus / Embedding。
"""

from __future__ import annotations

import threading
import time

import pytest

from rag_vector_store.runner.async_executor import (
    AsyncTaskExecutor,
    TaskSubmission,
)
from rag_vector_store.task_state.models import TaskState

# ── Fake Runner ──


class FakeRunner:
    """记录调用 + 可控制延迟/异常的 Runner 替身。"""

    def __init__(
        self,
        *,
        delay: float = 0.0,
        raise_error: Exception | None = None,
    ) -> None:
        self.delay = delay
        self.raise_error = raise_error
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def run(self, task_id: str) -> TaskState:
        with self._lock:
            self.calls.append(task_id)

        if self.delay > 0:
            time.sleep(self.delay)

        if self.raise_error is not None:
            raise self.raise_error

        return TaskState(
            task_id=task_id,
            source="full_batch",
            mode="full",
            status="completed",
            total_count=1,
            success_count=1,
            skipped_count=0,
            failed_count=0,
            collection_name="expert_candidates_v14_contract_dev",
        )


def _make_task_id() -> str:
    """生成合法 CHAR(36) UUID 格式 task_id。"""

    import uuid

    return str(uuid.uuid4())


# ── 1. submit 成功 + runner 被调用 ──


def test_submit_success_runner_called() -> None:
    """提交后 runner.run 在后台被调用，返回 completed TaskState。"""

    runner = FakeRunner()
    executor = AsyncTaskExecutor(runner, max_workers=2)
    task_id = _make_task_id()

    try:
        submission = executor.submit(task_id)
        assert isinstance(submission, TaskSubmission)
        assert submission.task_id == task_id

        # 等待完成
        result = submission.future.result(timeout=5)
        assert result.status == "completed"
        assert result.task_id == task_id
        assert submission.done is True
        assert submission.running is False

        # runner 被调用了且参数正确
        assert runner.calls == [task_id]
    finally:
        executor.shutdown()


def test_submit_returns_submission_immediately() -> None:
    """submit 不阻塞，立即返回 TaskSubmission。"""

    runner = FakeRunner(delay=0.3)
    executor = AsyncTaskExecutor(runner, max_workers=1)
    task_id = _make_task_id()

    try:
        submission = executor.submit(task_id)
        # submit 返回时任务还在运行
        assert submission.running is True
        assert submission.done is False

        result = submission.future.result(timeout=5)
        assert result.status == "completed"
    finally:
        executor.shutdown()


# ── 2. 异常不会导致 executor 崩溃 ──


def test_runner_exception_does_not_crash_executor() -> None:
    """runner 抛异常时，异常被 Future 捕获，executor 仍可继续提交。"""

    runner = FakeRunner(raise_error=RuntimeError("embedding down"))
    executor = AsyncTaskExecutor(runner, max_workers=2)
    task_id = _make_task_id()

    try:
        submission = executor.submit(task_id)

        # Future 捕获异常，result() 重新抛出
        with pytest.raises(RuntimeError, match="embedding down"):
            submission.future.result(timeout=5)

        assert submission.done is True

        # executor 没崩溃，可以继续提交
        runner2 = FakeRunner()
        executor._runner = runner2  # type: ignore[attr-defined]
        task_id2 = _make_task_id()
        submission2 = executor.submit(task_id2)
        result2 = submission2.future.result(timeout=5)
        assert result2.status == "completed"
        assert runner2.calls == [task_id2]
    finally:
        executor.shutdown()


def test_runner_exception_isolated_per_task() -> None:
    """一个任务异常不影响另一个并行任务。"""

    fail_runner = FakeRunner(raise_error=ValueError("bad chunk"))
    ok_runner = FakeRunner()
    executor_fail = AsyncTaskExecutor(fail_runner, max_workers=1)
    executor_ok = AsyncTaskExecutor(ok_runner, max_workers=1)

    try:
        tid_fail = _make_task_id()
        tid_ok = _make_task_id()

        sub_fail = executor_fail.submit(tid_fail)
        sub_ok = executor_ok.submit(tid_ok)

        with pytest.raises(ValueError, match="bad chunk"):
            sub_fail.future.result(timeout=5)

        result_ok = sub_ok.future.result(timeout=5)
        assert result_ok.status == "completed"
    finally:
        executor_fail.shutdown()
        executor_ok.shutdown()


# ── 3. 多任务提交 ──


def test_multiple_tasks_submitted_and_executed() -> None:
    """多个任务可以并发提交并全部执行。"""

    runner = FakeRunner(delay=0.05)
    executor = AsyncTaskExecutor(runner, max_workers=4)
    task_ids = [_make_task_id() for _ in range(5)]

    try:
        submissions = [executor.submit(tid) for tid in task_ids]
        results = [sub.future.result(timeout=10) for sub in submissions]

        assert len(results) == 5
        assert {r.task_id for r in results} == set(task_ids)
        assert all(r.status == "completed" for r in results)
        assert set(runner.calls) == set(task_ids)
    finally:
        executor.shutdown()


def test_concurrent_submission_thread_safe() -> None:
    """多线程同时 submit 不会丢任务或状态错乱。"""

    runner = FakeRunner(delay=0.02)
    executor = AsyncTaskExecutor(runner, max_workers=8)

    task_ids = [_make_task_id() for _ in range(10)]
    submissions: list[TaskSubmission] = []
    errors: list[Exception] = []

    def submit_one(tid: str) -> None:
        try:
            submissions.append(executor.submit(tid))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=submit_one, args=(tid,)) for tid in task_ids]

    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        assert len(submissions) == 10

        results = [sub.future.result(timeout=10) for sub in submissions]
        assert {r.task_id for r in results} == set(task_ids)
    finally:
        executor.shutdown()


# ── 4. 管理运行中的任务 ──


def test_is_running_and_running_task_ids() -> None:
    """is_running / running_task_ids 正确反映执行状态。"""

    runner = FakeRunner(delay=0.2)
    executor = AsyncTaskExecutor(runner, max_workers=2)
    tid1 = _make_task_id()
    tid2 = _make_task_id()

    try:
        sub1 = executor.submit(tid1)
        sub2 = executor.submit(tid2)

        # 两个都在运行
        assert executor.is_running(tid1) is True
        assert executor.is_running(tid2) is True
        assert set(executor.running_task_ids()) == {tid1, tid2}

        # 等待完成
        sub1.future.result(timeout=5)
        sub2.future.result(timeout=5)

        # 完成后不再运行
        assert executor.is_running(tid1) is False
        assert executor.is_running(tid2) is False
        assert executor.running_task_ids() == []
    finally:
        executor.shutdown()


def test_duplicate_submit_returns_existing_future() -> None:
    """对同一 task_id 重复 submit（尚未完成）返回已有 submission。"""

    runner = FakeRunner(delay=0.3)
    executor = AsyncTaskExecutor(runner, max_workers=1)
    tid = _make_task_id()

    try:
        sub1 = executor.submit(tid)
        sub2 = executor.submit(tid)

        # 同一 task_id，同一个 future
        assert sub1.future is sub2.future
        assert runner.calls == [tid]  # 只提交一次

        sub1.future.result(timeout=5)
    finally:
        executor.shutdown()


def test_resubmit_after_completion() -> None:
    """任务完成后可以重新提交同一 task_id。"""

    runner = FakeRunner()
    executor = AsyncTaskExecutor(runner, max_workers=1)
    tid = _make_task_id()

    try:
        sub1 = executor.submit(tid)
        sub1.future.result(timeout=5)

        # 任务完成后清理，可以重新提交
        sub2 = executor.submit(tid)
        assert sub1.future is not sub2.future
        sub2.future.result(timeout=5)

        assert runner.calls == [tid, tid]
    finally:
        executor.shutdown()


def test_get_future() -> None:
    """get_future 返回已注册的 Future。"""

    runner = FakeRunner(delay=0.1)
    executor = AsyncTaskExecutor(runner, max_workers=1)
    tid = _make_task_id()

    try:
        submission = executor.submit(tid)
        future = executor.get_future(tid)
        assert future is submission.future

        # 不存在的 task_id
        assert executor.get_future("fake-" + "a" * 31) is None

        submission.future.result(timeout=5)
    finally:
        executor.shutdown()


# ── 5. 参数校验 ──


def test_max_workers_zero_raises() -> None:
    """max_workers < 1 时抛 ValueError。"""

    with pytest.raises(ValueError, match="max_workers"):
        AsyncTaskExecutor(FakeRunner(), max_workers=0)


# ── 6. shutdown ──


def test_shutdown_waits_for_running_tasks() -> None:
    """shutdown(wait=True) 等待正在执行的任务完成。"""

    runner = FakeRunner(delay=0.1)
    executor = AsyncTaskExecutor(runner, max_workers=2)
    tid = _make_task_id()

    submission = executor.submit(tid)
    executor.shutdown(wait=True)

    # shutdown 后任务已完成
    assert submission.done is True
    assert runner.calls == [tid]
