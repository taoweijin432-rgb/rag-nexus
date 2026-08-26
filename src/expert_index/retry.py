"""P6.1 失败分类契约。

分类器只处理已脱敏的机器字段和下游终态，不读取异常正文；调用方必须在捕获异常时
先决定是否可重试，再把稳定错误码交给本模块。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

FailureDisposition = Literal["retryable", "dead"]
DownstreamStatus = Literal["committed", "rejected", "failed"]

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class FailureClassification:
    """表示一次失败的有限处置结果和安全定位字段。"""

    disposition: FailureDisposition
    reason_code: str
    error_stage: str
    exception_type: str | None


def classify_failure(
    *,
    error_stage: str,
    error_code: str,
    downstream_status: DownstreamStatus,
    exception_type: str | None,
    retryable: bool,
) -> FailureClassification:
    """根据显式终态和调用方分类结果确定重试或终止。

    ``retryable`` 由异常类型/下游协议映射产生，分类器不匹配异常消息，避免把业务正文
    当作控制信号。``rejected`` 永远进入 dead；``failed`` 仅在显式允许时重试。
    """

    _validate_identifier("error_stage", error_stage, 32)
    _validate_identifier("error_code", error_code, 64)
    if downstream_status not in ("committed", "rejected", "failed"):
        raise ValueError(f"不支持的下游终态: {downstream_status!r}")
    if exception_type is not None:
        _validate_identifier("exception_type", exception_type, 128)
    if downstream_status == "committed":
        raise ValueError("committed 不是失败，不能创建失败分类")
    disposition: FailureDisposition = (
        "dead" if downstream_status == "rejected" or not retryable else "retryable"
    )
    return FailureClassification(
        disposition=disposition,
        reason_code=error_code,
        error_stage=error_stage,
        exception_type=exception_type,
    )


def _validate_identifier(field_name: str, value: str, max_length: int) -> None:
    """限制错误定位字段为稳定、无正文的机器标识。"""

    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= max_length
        or _SAFE_IDENTIFIER.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} 必须是长度不超过 {max_length} 的安全机器标识")


__all__ = ["DownstreamStatus", "FailureClassification", "FailureDisposition", "classify_failure"]
