"""领域核心共享的输入安全门。"""

from __future__ import annotations

from typing import Any

from .adapters import VectorContractAdapter


def require_adapter_input(
    adapter: VectorContractAdapter[Any, Any, Any],
    value: object,
    *,
    operation: str,
) -> None:
    """确认输入属于当前 Adapter 绑定的领域。

    Worker 与 Ingestion 都可能被独立调用，因此两处仍保留边界检查，但具体
    判定和错误格式集中在这里，避免不同入口逐渐产生不一致的规则。
    """

    if not adapter.is_input(value):
        raise TypeError(f"{operation} 只接受当前领域 Contract")


__all__ = ["require_adapter_input"]
