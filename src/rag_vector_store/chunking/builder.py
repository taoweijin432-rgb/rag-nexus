"""Chunk Builder 的接入协议。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from .contracts import PreparedChunk


class ChunkBuilder(Protocol):
    """同事的 Chunk 模块需要实现的最小构建接口。"""

    def build(self, source: Any) -> Iterable[PreparedChunk]:
        """将一个来源对象转换为可向量化的 PreparedChunk 序列。"""

        ...
