"""专家 Chunk 领域契约与构建接口。"""

from .builder import ChunkBuilder
from .contracts import EmbeddedPreparedChunk, PreparedChunk

__all__ = [
    "ChunkBuilder",
    "EmbeddedPreparedChunk",
    "PreparedChunk",
]
