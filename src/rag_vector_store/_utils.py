"""rag_vector_store 内部通用小工具。

这里只收敛语义完全一致、且不属于具体领域的数据处理。模块是私有实现，不作为
公共 API 导出。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

ErrorT = TypeVar("ErrorT", bound=Exception)
T = TypeVar("T")


def safe_error_message(error: Exception, *, limit: int = 200) -> str:
    """返回不包含堆栈和内部对象的安全错误摘要。"""

    return f"{type(error).__name__}: {str(error)[:limit]}"


def single_line_truncate(value: str, *, limit: int = 512) -> str:
    """把文本归一为单行并截断到指定字符数。"""

    flat = " ".join(value.splitlines())
    return flat[:limit]


def dedupe_nonempty_strings(
    values: Sequence[str],
    *,
    prefix: str | None = None,
    error_factory: Callable[[str], ErrorT] = ValueError,
    field_name: str = "vector_doc_id",
) -> list[str]:
    """校验非空字符串并按输入顺序去重。"""

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise error_factory(f"{field_name} 必须是非空字符串")
        normalized = value.strip()
        if prefix is not None and not normalized.startswith(prefix):
            raise error_factory(
                f"{field_name} {normalized!r} 不符合当前领域前缀 {prefix!r}"
            )
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def batched(values: Sequence[T], *, size: int) -> list[Sequence[T]]:
    """按固定大小切分序列；size 非正数时 fail-closed。"""

    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("batch size 必须是正整数")
    return [values[start : start + size] for start in range(0, len(values), size)]


__all__ = [
    "batched",
    "dedupe_nonempty_strings",
    "safe_error_message",
    "single_line_truncate",
]
