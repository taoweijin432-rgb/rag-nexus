"""专家 Chunk 向量化输入契约 v2.0.0 的共享强类型定义。

本模块只表达 Contract 的字段、枚举和校验规则，不读取数据库、不读取文件，
也不依赖任何业务 Builder 或向量存储实现。上游生成的 ``vector_doc_id`` 必须
同时出现在输入顶层和 ``metadata`` 快照中。
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EXPERT_CHUNK_CONTRACT_VERSION = "2.0.0"

_VECTOR_DOC_ID_MAX_BYTES = 2048

ChunkType = Literal["profile_summary", "employment", "project"]
ChunkSubtype = Literal[
    "responsibility",
    "achievement",
    "context",
    "action",
    "result",
    "segment",
]
CompanyNormalizationStatus = Literal[
    "verified",
    "auto_high_confidence",
    "unknown",
    "conflict",
]
OriginType = Literal[
    "source_template",
    "source_parsed",
    "compressed_from_source",
    "fallback_template",
]
VerificationStatus = Literal[
    "source_evidence",
    "validated",
    "partial",
    "conflict",
    "rejected",
]
WeightTier = Literal["A", "B", "C"]

_CHUNK_ID_PATTERN = r"^expert:[1-9][0-9]*:(profile_summary|employment|project):[^:]+:[0-9]+$"
_MONTH_PATTERN = r"^[0-9]{4}-(0[1-9]|1[0-2])$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CODE_PATTERN = r"^[a-z][a-z0-9_]*$"

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


class _ContractModel(BaseModel):
    """所有共享 Contract 模型共用的严格对象配置。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class ExpertChunkMetadata(_ContractModel):
    """契约中的业务 Metadata，保留上游生成的向量主键快照。"""

    vector_doc_id: str = Field(..., min_length=1, max_length=_VECTOR_DOC_ID_MAX_BYTES)
    project_name: str | None = Field(default=None, max_length=512)
    project_role: str | None = Field(default=None, max_length=256)
    company_name_raw: str | None = Field(..., max_length=512)
    company_name_norm: str | None = Field(..., max_length=512)
    company_normalization_status: CompanyNormalizationStatus | None = Field(...)
    role_name_raw: str | None = Field(..., max_length=256)
    role_code: str | None = Field(..., max_length=128)
    role_level: str | None = Field(..., max_length=64)
    start_month: str | None = Field(..., pattern=_MONTH_PATTERN)
    end_month: str | None = Field(..., pattern=_MONTH_PATTERN)
    is_current: bool | None = Field(...)
    industry_codes: list[str] = Field(...)
    topic_codes: list[str] = Field(...)
    origin_type: OriginType = Field(...)
    verification_status: VerificationStatus = Field(...)
    weight_tier: WeightTier = Field(...)

    @field_validator("industry_codes", "topic_codes")
    @classmethod
    def validate_code_arrays(cls, value: list[str]) -> list[str]:
        """校验标签是合法的小写代码，并且已经去重、排序。"""

        if any(not re.fullmatch(_CODE_PATTERN, item) for item in value):
            raise ValueError("industry_codes/topic_codes 必须是小写 snake_case 代码")
        if value != sorted(set(value)):
            raise ValueError("industry_codes/topic_codes 必须去重并按字典序升序排列")
        return value

    @field_validator("vector_doc_id")
    @classmethod
    def validate_vector_doc_id_capacity(cls, value: str) -> str:
        """校验上游主键不超过正式 Milvus 的 UTF-8 字节容量。"""

        return _validate_vector_doc_id_capacity(value)


class ExpertChunkSource(_ContractModel):
    """契约中的来源追溯字段。"""

    source_table: str = Field(..., min_length=1, max_length=128)
    source_item_key: str = Field(..., min_length=1, max_length=128)
    source_field: str = Field(..., min_length=1, max_length=128)
    source_field_id: str = Field(..., min_length=1, max_length=256)
    source_version: str = Field(..., min_length=1, max_length=64)
    source_updated_at: str | None = Field(...)
    source_hash: str = Field(..., pattern=_SHA256_PATTERN)

    @field_validator("source_updated_at")
    @classmethod
    def validate_source_updated_at(cls, value: str | None) -> str | None:
        """校验来源更新时间是带时区的 ISO 8601 时间。"""

        if value is not None:
            _validate_timezone_datetime(value, "source_updated_at")
        return value


class ExpertChunkVersions(_ContractModel):
    """契约中的数据、Chunk、Metadata 和词典版本字段。"""

    data_version: str = Field(..., min_length=1, max_length=64)
    chunk_policy_version: str = Field(..., min_length=1, max_length=32)
    metadata_schema_version: str = Field(..., min_length=1, max_length=32)
    dictionary_version: str = Field(..., min_length=1, max_length=32)


class ExpertChunkEmbeddingInput(_ContractModel):
    """一条待向量化的正式 ``expert_chunk`` 输入记录。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        title="Expert Chunk Embedding Input",
        json_schema_extra={
            "$id": "https://internal.example/schemas/"
            "expert_chunk_embedding_input-2.0.0.json"
        },
    )

    contract_version: Literal["2.0.0"]
    chunk_id: str = Field(..., min_length=1, max_length=256, pattern=_CHUNK_ID_PATTERN)
    vector_doc_id: str = Field(..., min_length=1, max_length=_VECTOR_DOC_ID_MAX_BYTES)
    expert_id: int = Field(..., ge=1)
    serial_number: str | None = Field(..., max_length=64)
    chunk_type: ChunkType
    chunk_subtype: ChunkSubtype | None = Field(...)
    chunk_index: int = Field(..., ge=0)
    chunk_total: int = Field(..., ge=1)
    chunk_text: str = Field(..., min_length=1, max_length=1000)
    char_count: int = Field(..., ge=1, le=1000)
    content_hash: str = Field(..., pattern=_SHA256_PATTERN)
    metadata: ExpertChunkMetadata
    source: ExpertChunkSource
    versions: ExpertChunkVersions
    is_active: bool
    updated_at: str = Field(..., min_length=1)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: str) -> str:
        """校验 Chunk 更新时间是带时区的 ISO 8601 时间。"""

        _validate_timezone_datetime(value, "updated_at")
        return value

    @field_validator("vector_doc_id")
    @classmethod
    def validate_vector_doc_id_capacity(cls, value: str) -> str:
        """校验上游主键不超过正式 Milvus 的 UTF-8 字节容量。"""

        return _validate_vector_doc_id_capacity(value)

    @model_validator(mode="after")
    def validate_contract_version(self) -> ExpertChunkEmbeddingInput:
        """校验正式契约版本及顶层/Metadata 主键一致性。"""

        if self.contract_version != EXPERT_CHUNK_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version 必须是 {EXPERT_CHUNK_CONTRACT_VERSION}"
            )
        if self.metadata.vector_doc_id != self.vector_doc_id:
            raise ValueError("metadata.vector_doc_id 必须与顶层 vector_doc_id 一致")
        return self


class ContractPreparedChunk(ExpertChunkEmbeddingInput):
    """通过契约校验、尚未生成向量的 Chunk。"""


class EmbeddedContractPreparedChunk(ContractPreparedChunk):
    """保留完整输入契约并附加向量化输出字段的记录。"""

    embedding: list[float] = Field(..., min_length=1)
    embedding_dim: int = Field(..., ge=1)
    embedding_model_alias: str = Field(..., min_length=1, max_length=128)
    embedding_model_version: str = Field(..., min_length=1, max_length=128)
    embedded_at: str = Field(..., min_length=1)

    @field_validator("embedding")
    @classmethod
    def validate_embedding_values(cls, value: list[float]) -> list[float]:
        """确保输出向量非空，并且每个元素都是有限数值。"""

        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding 必须只包含有限数值")
        return value

    @field_validator("embedded_at")
    @classmethod
    def validate_embedded_at(cls, value: str) -> str:
        """校验向量生成时间是带时区的 ISO 8601 时间。"""

        _validate_timezone_datetime(value, "embedded_at")
        return value

    @model_validator(mode="after")
    def validate_embedding_dimension(self) -> EmbeddedContractPreparedChunk:
        """确保声明的维度与实际向量长度一致。"""

        if self.embedding_dim != len(self.embedding):
            raise ValueError("embedding_dim 必须等于 embedding 的实际长度")
        return self


def _validate_timezone_datetime(value: str, field_name: str) -> None:
    """校验时间字符串可解析且包含时区信息。"""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} 必须是合法的 ISO 8601 时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} 必须包含时区")


def _validate_vector_doc_id_capacity(value: str) -> str:
    """统一校验上游主键的 Milvus UTF-8 字节容量。"""

    if len(value.encode("utf-8")) > _VECTOR_DOC_ID_MAX_BYTES:
        raise ValueError(
            f"vector_doc_id 超过 Milvus VARCHAR 实现容量 {_VECTOR_DOC_ID_MAX_BYTES} bytes"
        )
    return value
