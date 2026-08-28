"""CDC 常驻 Worker 服务入口。

Worker 只负责调度已有的 :class:`CdcRuntime`，不复制 Parser、Task 或 Checkpoint
逻辑。每个非空 Canal batch 都必须由调用方提供一个独立的 Index 本地批次号；
Canal batch_id 只作为 ACK/rollback 句柄，不能直接复用。
"""

from __future__ import annotations

import math
import re
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .finalize import (
    CdcBatchRunResult,
    CdcCheckpointConflictError,
)
from .protocol import CanalProtocolNotFrozenError


class CdcRuntimePort(Protocol):
    """常驻 Worker 调用的单批次运行时端口。"""

    def consume_once(self, *, batch_size: int, local_batch_id: int) -> CdcBatchRunResult:
        """消费一个未确认 batch，并在安全完成后 ACK。"""


class LocalBatchIdProvider(Protocol):
    """为每轮消费提供 Index MySQL 中独立的本地批次号。"""

    def __call__(self) -> int:
        """返回新的正整数本地批次号。"""


@dataclass(frozen=True)
class CdcWorkerConfig:
    """保存 Worker 循环参数；不包含 Canal 身份或数据库凭据。"""

    batch_size: int = 100
    idle_sleep_seconds: float = 5.0
    error_sleep_seconds: float = 10.0
    max_cycles: int | None = None

    def __post_init__(self) -> None:
        """在建立连接前校验批量、等待和测试循环上限。"""

        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size 必须是整数")
        if self.batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        for name, value in (
            ("idle_sleep_seconds", self.idle_sleep_seconds),
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
class CdcWorkerSummary:
    """返回常驻循环的安全统计，不包含事件正文或凭据。"""

    cycles: int
    empty_count: int
    acked_count: int
    rolled_back_count: int
    not_ready_count: int
    error_count: int
    stop_reason: str
    last_error_type: str | None = None
    last_error_message: str | None = None


class CdcWorker:
    """驱动 CDC 单 batch 编排的可停止常驻循环。"""

    _TERMINAL_ERRORS = (
        CdcCheckpointConflictError,
        CanalProtocolNotFrozenError,
    )

    def __init__(
        self,
        runtime: CdcRuntimePort,
        local_batch_id_provider: LocalBatchIdProvider,
        *,
        config: CdcWorkerConfig | None = None,
        stop_event: threading.Event | None = None,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        """注入既有运行时、本地批次号分配器和可中断等待器。"""

        if not callable(getattr(runtime, "consume_once", None)):
            raise TypeError("runtime 缺少 consume_once")
        if not callable(local_batch_id_provider):
            raise TypeError("local_batch_id_provider 必须可调用")
        self.runtime = runtime
        self.local_batch_id_provider = local_batch_id_provider
        self.config = config or CdcWorkerConfig()
        self.stop_event = stop_event or threading.Event()
        self._wait = wait or self.stop_event.wait

    def request_stop(self) -> None:
        """请求 Worker 在当前 batch 结束后停止，不中断已进入的业务事务。"""

        self.stop_event.set()

    def run(self) -> CdcWorkerSummary:
        """持续消费 batch；空批次等待，瞬时错误退避，安全错误立即停机。"""

        cycles = empty = acked = rolled_back = not_ready = errors = 0
        last_error_type: str | None = None
        last_error_message: str | None = None
        stop_reason = "stop_requested"

        while not self.stop_event.is_set():
            if self.config.max_cycles is not None and cycles >= self.config.max_cycles:
                stop_reason = "max_cycles"
                break
            cycles += 1
            try:
                local_batch_id = self.local_batch_id_provider()
            except Exception as error:  # noqa: BLE001 - 配置错误必须安全停机。
                errors += 1
                last_error_type = type(error).__name__
                last_error_message = _safe_error_message(error)
                stop_reason = f"terminal_error:{last_error_type}"
                break
            if isinstance(local_batch_id, bool) or not isinstance(local_batch_id, int):
                errors += 1
                last_error_type = "TypeError"
                last_error_message = "local_batch_id 必须是正整数"
                stop_reason = "terminal_error:TypeError"
                break
            if local_batch_id < 1:
                errors += 1
                last_error_type = "ValueError"
                last_error_message = "local_batch_id 必须是正整数"
                stop_reason = "terminal_error:ValueError"
                break
            try:
                result = self.runtime.consume_once(
                    batch_size=self.config.batch_size,
                    local_batch_id=local_batch_id,
                )
            except self._TERMINAL_ERRORS as error:
                last_error_type = type(error).__name__
                last_error_message = _safe_error_message(error)
                stop_reason = f"terminal_error:{last_error_type}"
                break
            except Exception as error:  # noqa: BLE001 - Worker 需保护服务进程并退避。
                errors += 1
                last_error_type = type(error).__name__
                last_error_message = _safe_error_message(error)
                if self._wait(self.config.error_sleep_seconds):
                    stop_reason = "stop_requested"
                    break
                continue

            if result.status == "empty":
                empty += 1
                if self._wait(self.config.idle_sleep_seconds):
                    stop_reason = "stop_requested"
                    break
            elif result.status == "acked":
                acked += 1
            elif result.status == "rolled_back":
                # 畸形/不支持 batch 重投后仍会重复失败；停机交由人工或发布修复。
                rolled_back += 1
                stop_reason = "rolled_back_batch"
                break
            elif result.status == "not_ready":
                not_ready += 1
                if self._wait(self.config.error_sleep_seconds):
                    stop_reason = "stop_requested"
                    break
            else:  # 防御未来新增状态，避免无等待的忙循环。
                errors += 1
                last_error_type = "UnknownCdcBatchStatus"
                last_error_message = "Worker 收到未支持的 batch 状态"
                if self._wait(self.config.error_sleep_seconds):
                    stop_reason = "stop_requested"
                    break

        return CdcWorkerSummary(
            cycles=cycles,
            empty_count=empty,
            acked_count=acked,
            rolled_back_count=rolled_back,
            not_ready_count=not_ready,
            error_count=errors,
            stop_reason=stop_reason,
            last_error_type=last_error_type,
            last_error_message=last_error_message,
        )


def _safe_error_message(error: Exception) -> str | None:
    """返回截断脱敏的错误摘要，不保留 Payload、正文或凭据。"""

    message = " ".join(str(error).split())
    if not message:
        return None
    message = re.sub(
        r"(?i)(password|passwd|token|api[_-]?key|secret)=\S+",
        r"\1=<redacted>",
        message,
    )
    return message[:300]


def install_signal_handlers(stop_event: threading.Event) -> None:
    """将 SIGINT/SIGTERM 转换为可安全检查的停止事件。"""

    if not isinstance(stop_event, threading.Event):
        raise TypeError("stop_event 必须是 threading.Event")

    def _request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


def run_cdc_worker(
    runtime: CdcRuntimePort,
    local_batch_id_provider: LocalBatchIdProvider,
    *,
    config: CdcWorkerConfig | None = None,
    stop_event: threading.Event | None = None,
    install_signals: bool = True,
) -> CdcWorkerSummary:
    """作为服务进程入口运行 Worker，并管理可选运行时会话生命周期。"""

    event = stop_event or threading.Event()
    if install_signals:
        install_signal_handlers(event)
    start = getattr(runtime, "start", None)
    stop = getattr(runtime, "stop", None)
    if callable(start):
        start()
    try:
        return CdcWorker(
            runtime,
            local_batch_id_provider,
            config=config,
            stop_event=event,
        ).run()
    finally:
        if callable(stop):
            stop()


__all__ = [
    "CdcRuntimePort",
    "CdcWorker",
    "CdcWorkerConfig",
    "CdcWorkerSummary",
    "LocalBatchIdProvider",
    "install_signal_handlers",
    "run_cdc_worker",
]
