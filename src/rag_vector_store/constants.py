"""向量化内核和 V15 持久化共同使用的稳定常量。"""

from __future__ import annotations

from typing import Final

# 当前正式模型空间的维度；领域实现仍可通过 Worker/Repository 显式注入其它维度。
FORMAL_EMBEDDING_DIMENSION: Final = 1024

# V15 物理 Schema 与正式 Contract 消费链必须共享的 UTF-8 字节上限。
V15_VECTOR_DOC_ID_MAX_BYTES: Final = 2048
V15_CHUNK_TEXT_MAX_BYTES: Final = 8192

__all__ = [
    "FORMAL_EMBEDDING_DIMENSION",
    "V15_CHUNK_TEXT_MAX_BYTES",
    "V15_VECTOR_DOC_ID_MAX_BYTES",
]
