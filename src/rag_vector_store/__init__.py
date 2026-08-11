"""用于 chunk 向量化并写入 Milvus 的可组合基础模块。"""

from .chunking import ChunkBuilder, EmbeddedPreparedChunk, PreparedChunk
from .config import MilvusSettings
from .embedding import EmbeddingModel
from .embedding_adapters import (
    CompanyCloudEmbeddingError,
    CompanyCloudEmbeddingModel,
    CompanyCloudEmbeddingSettings,
)
from .ingestion import ChunkIngestionService, IngestionResult
from .models import Chunk, EmbeddedChunk, MutationResult
from .ports import ChunkRepository
from .repository import MilvusVectorRepository

__all__ = [
    "Chunk",
    "ChunkBuilder",
    "ChunkIngestionService",
    "ChunkRepository",
    "CompanyCloudEmbeddingError",
    "CompanyCloudEmbeddingModel",
    "CompanyCloudEmbeddingSettings",
    "EmbeddedChunk",
    "EmbeddedPreparedChunk",
    "EmbeddingModel",
    "IngestionResult",
    "MilvusSettings",
    "MilvusVectorRepository",
    "MutationResult",
    "PreparedChunk",
]
