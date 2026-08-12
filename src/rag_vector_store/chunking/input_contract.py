"""``expert_contracts`` 的 rag_vector_store 兼容导出入口。

正式模型的唯一实现位于 ``expert_contracts.models``。保留本模块及原有名称，
避免既有 rag_vector_store 调用方因共享 Contract 拆包而失效。
"""

from expert_contracts.models import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    ChunkSubtype,
    ChunkType,
    CompanyNormalizationStatus,
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
    "CompanyNormalizationStatus",
    "ContractPreparedChunk",
    "EmbeddedContractPreparedChunk",
    "ExpertChunkEmbeddingInput",
    "ExpertChunkMetadata",
    "ExpertChunkSource",
    "ExpertChunkVersions",
    "OriginType",
    "VerificationStatus",
    "WeightTier",
]
