"""专家 Chunk 的共享 Contract 与 JSON-compatible Mapping Adapter。"""

from .adapter import (
    ExpertChunkMappingAdapter,
    ExpertChunkMappingError,
    adapt_expert_chunk_mapping,
)
from .models import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    EXPERT_CHUNK_METADATA_BY_TYPE,
    ChunkSubtype,
    ChunkType,
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    EmploymentMetadata,
    ExpertChunkEmbeddingInput,
    ExpertChunkSource,
    ExpertChunkVersions,
    OriginType,
    ProfileSummaryMetadata,
    ProjectMetadata,
    VerificationStatus,
    WeightTier,
    metadata_model_for_chunk_type,
    vector_doc_id_prefix_for_source_table,
)

__all__ = [
    "EXPERT_CHUNK_CONTRACT_VERSION",
    "EXPERT_CHUNK_METADATA_BY_TYPE",
    "ChunkSubtype",
    "ChunkType",
    "ContractPreparedChunk",
    "EmbeddedContractPreparedChunk",
    "EmploymentMetadata",
    "ExpertChunkEmbeddingInput",
    "ExpertChunkMappingAdapter",
    "ExpertChunkMappingError",
    "ExpertChunkSource",
    "ExpertChunkVersions",
    "OriginType",
    "ProfileSummaryMetadata",
    "ProjectMetadata",
    "VerificationStatus",
    "WeightTier",
    "adapt_expert_chunk_mapping",
    "metadata_model_for_chunk_type",
    "vector_doc_id_prefix_for_source_table",
]
