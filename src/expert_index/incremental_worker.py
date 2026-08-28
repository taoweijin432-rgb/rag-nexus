"""普通 Poll 增量常驻 Worker。

Worker 只重复调用现有 :class:`IncrementalRuntime` 单轮入口，不重新实现扫描、Hash、
任务、下游交付或 Checkpoint 逻辑，适合测试服务器由 systemd 等进程管理器常驻运行。
"""

from __future__ import annotations

import math
import signal
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .incremental_runtime import IncrementalOnceResult


class IncrementalRuntimePort(Protocol):
    """常驻 Worker 调用的单轮 Poll 端口。"""

    def run_once(self, worker_id: str, *, cycle_id: str) -> IncrementalOnceResult:
        """执行一轮四表扫描和增量任务处理。"""


@dataclass(frozen=True)
class IncrementalWorkerConfig:
    """保存 Poll 间隔、失败退避和测试循环上限。"""

    interval_seconds: float = 60.0
    error_sleep_seconds: float = 30.0
    max_cycles: int | None = None

    def __post_init__(self) -> None:
        """在启动数据库连接前校验等待参数。"""

        for name, value in (
            ("interval_seconds", self.interval_seconds),
            ("error_sleep_seconds", self.error_sleep_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} 必须是数字")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数字")
        if self.max_cycles is not None:
            if isinstance(self.max_cycles, bool) or not isinstance(self.max_cycles, int):
                raise TypeError("max_cycles 必须是整数或 None")
            if self.max_cycles < 1:
                raise ValueError("max_cycles 必须是正整数")


@dataclass(frozen=True)
class IncrementalWorkerSummary:
    """返回常驻 Poll 循环的安全统计，不包含来源正文。"""

    cycles: int
    successful_cycles: int
    failed_cycles: int
    error_count: int
    stop_reason: str
    last_error_type: str | None = None


class IncrementalWorker:
    """按固定间隔执行普通 Poll 增量的一进程 Worker。"""

    def __init__(
        self,
        runtime: IncrementalRuntimePort,
        worker_id: str,
        *,
        config: IncrementalWorkerConfig | None = None,
        stop_event: threading.Event | None = None,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        """注入已组装的运行时和稳定 Worker 身份。"""

        if not callable(getattr(runtime, "run_once", None)):
            raise TypeError("runtime 缺少 run_once")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id 必须是非空字符串")
        self.runtime = runtime
        self.worker_id = worker_id
        self.config = config or IncrementalWorkerConfig()
        self.stop_event = stop_event or threading.Event()
        self._wait = wait or self.stop_event.wait

    def request_stop(self) -> None:
        """请求 Worker 在当前 Poll 周期完成后停止。"""

        self.stop_event.set()

    def run(self) -> IncrementalWorkerSummary:
        """持续执行 Poll；失败周期退避，收到停止信号后安全退出。"""

        cycles = successful_cycles = failed_cycles = error_count = 0
        last_error_type: str | None = None
        stop_reason = "stop_requested"

        while not self.stop_event.is_set():
            if self.config.max_cycles is not None and cycles >= self.config.max_cycles:
                stop_reason = "max_cycles"
                break
            cycles += 1
            try:
                result = self.runtime.run_once(
                    self.worker_id,
                    cycle_id=f"{self.worker_id}:{uuid.uuid4().hex}",
                )
            except Exception as error:  # noqa: BLE001 - 常驻服务需保护进程并退避。
                error_count += 1
                last_error_type = type(error).__name__
                if self._wait(self.config.error_sleep_seconds):
                    stop_reason = "stop_requested"
                    break
                continue

            if (
                result.scan_cycle.failed_source_count
                or result.execution.failed_source_count
            ):
                failed_cycles += 1
                if self._wait(self.config.error_sleep_seconds):
                    stop_reason = "stop_requested"
                    break
            else:
                successful_cycles += 1
                if self._wait(self.config.interval_seconds):
                    stop_reason = "stop_requested"
                    break

        return IncrementalWorkerSummary(
            cycles=cycles,
            successful_cycles=successful_cycles,
            failed_cycles=failed_cycles,
            error_count=error_count,
            stop_reason=stop_reason,
            last_error_type=last_error_type,
        )


def install_signal_handlers(stop_event: threading.Event) -> None:
    """将 SIGINT/SIGTERM 转换为可安全检查的停止事件。"""

    if not isinstance(stop_event, threading.Event):
        raise TypeError("stop_event 必须是 threading.Event")

    def _request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


def run_incremental_worker(
    runtime: IncrementalRuntimePort,
    worker_id: str,
    *,
    config: IncrementalWorkerConfig | None = None,
    stop_event: threading.Event | None = None,
    install_signals: bool = True,
) -> IncrementalWorkerSummary:
    """作为测试/生产进程入口运行 Poll Worker。"""

    event = stop_event or threading.Event()
    if install_signals:
        install_signal_handlers(event)
    return IncrementalWorker(
        runtime,
        worker_id,
        config=config,
        stop_event=event,
    ).run()


__all__ = [
    "IncrementalRuntimePort",
    "IncrementalWorker",
    "IncrementalWorkerConfig",
    "IncrementalWorkerSummary",
    "install_signal_handlers",
    "run_incremental_worker",
]
