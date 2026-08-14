"""专家 Chunk 的共享 Contract 与 JSON-compatible Mapping Adapter。"""

from .adapter import (
    ExpertChunkMappingAdapter,
    ExpertChunkMappingError,
    adapt_expert_chunk_mapping,
)
from .models import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    ChunkSubtype,
    ChunkType,
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
    ExpertChunkMetadata,
    ExpertChunkSource,
    ExpertChunkVersions,
    OriginType,
    VerificationStatus,
    WeightTier,
)

__all__ = [
    "EXPERT_CHUNK_CONTRACT_VERSION",
    "ChunkSubtype",
    "ChunkType",
    "ContractPreparedChunk",
    "EmbeddedContractPreparedChunk",
    "ExpertChunkEmbeddingInput",
    "ExpertChunkMappingAdapter",
    "ExpertChunkMappingError",
    "ExpertChunkMetadata",
    "ExpertChunkSource",
    "ExpertChunkVersions",
    "OriginType",
    "VerificationStatus",
    "WeightTier",
    "adapt_expert_chunk_mapping",
]
