"""旧调用方式的兼容层。

历史代码可能直接调用 ``person_to_text`` 或 ``format_persons``。保留这两个函数，
但内部统一委托给新的标准 chunk 构建器，避免出现两套格式化逻辑。
"""

from __future__ import annotations

from typing import Any

from .chunk import build_expert_chunk


def person_to_text(person: dict[str, Any]) -> str:
    """返回单条专家的最终 chunk 文本。"""

    return build_expert_chunk(person)["text"]


def format_persons(persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量转换专家，并为每条记录附带稳定 ID 和 metadata。"""

    return [build_expert_chunk(person) for person in persons]
