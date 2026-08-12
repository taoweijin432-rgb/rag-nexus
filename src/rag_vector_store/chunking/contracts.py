"""专家 Chunk 在向量化前后的稳定数据契约。"""

from __future__ import annotations

import json
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_EXPLICIT_METADATA_FIELDS = frozenset(
    {
        "expert_id",
        "chunk_type",
        "company_name_norm",
        "role_code",
        "start_month",
        "end_month",
        "is_current",
        "is_active",
        "eligible_for_recommendation",
    }
)


class PreparedChunk(BaseModel):
    """已经完成业务分块、可交给 embedding 模块的专家 Chunk。

    这个模型不包含 embedding。它负责保存 Chunk Builder 产出的完整业务事实、
    来源追溯信息和版本信息，避免在向量化过程中退化成无结构的 metadata 字典。
    """

    model_config = ConfigDict(extra="forbid")

    vector_doc_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    expert_id: int = Field(ge=1)
    chunk_type: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)

    chunk_subtype: str | None = None
    parent_record_id: str | None = None
    chunk_index: int = Field(ge=0)
    chunk_total: int = Field(ge=1)

    source_table: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    source_item_key: str | None = None

    source_version: int = Field(ge=1)
    source_hash: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    source_updated_at: int | None = None

    company_name_norm: str | None = None
    role_code: str | None = None
    start_month: int | None = None
    end_month: int | None = None
    is_current: bool | None = None

    origin_type: str | None = None
    verification_status: str | None = None
    weight_tier: str | None = None

    embedding_model_alias: str = Field(min_length=1)
    embedding_model_version: str = Field(min_length=1)

    chunk_policy_version: str = Field(min_length=1)
    metadata_schema_version: str = Field(min_length=1)

    eligible_for_recommendation: bool
    is_active: bool

    metadata_ext: dict[str, object] = Field(default_factory=dict)

    @field_validator(
        "vector_doc_id",
        "chunk_id",
        "chunk_text",
        "source_table",
        "source_record_id",
        "source_field",
    )
    @classmethod
    def reject_whitespace_only_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("文本和标识字段不能只包含空白字符")
        return value

    @field_validator("metadata_ext")
    @classmethod
    def validate_metadata_ext(cls, value: dict[str, object]) -> dict[str, object]:
        """校验扩展字段可写入 JSON，并阻止重复保存高频标量字段。"""

        duplicated_fields = _EXPLICIT_METADATA_FIELDS.intersection(value)
        if duplicated_fields:
            raise ValueError(
                "metadata_ext 不应重复保存显式字段: "
                f"{sorted(duplicated_fields)}"
            )
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata_ext 必须可以 JSON 序列化") from exc
        return value

    @model_validator(mode="after")
    def validate_chunk_position(self) -> PreparedChunk:
        if self.chunk_index >= self.chunk_total:
            raise ValueError("chunk_index 必须小于 chunk_total")
        return self


class EmbeddedPreparedChunk(PreparedChunk):
    """保留 PreparedChunk 全部业务字段并附加 embedding 的记录。"""

    embedding: list[float] = Field(min_length=1)

    @field_validator("embedding")
    @classmethod
    def validate_embedding_values(cls, value: list[float]) -> list[float]:
        normalized = [float(item) for item in value]
        if not all(math.isfinite(item) for item in normalized):
            raise ValueError("embedding 必须只包含有限浮点数")
        return normalized

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .input_contract import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
    ExpertChunkMetadata,
    ExpertChunkSource,
    ExpertChunkVersions,
)

__all__ = [
    "EXPERT_CHUNK_CONTRACT_VERSION",
    "ContractPreparedChunk",
    "EmbeddedContractPreparedChunk",
    "EmbeddedPreparedChunk",
    "ExpertChunkEmbeddingInput",
    "ExpertChunkMetadata",
    "ExpertChunkSource",
    "ExpertChunkVersions",
    "PreparedChunk",
    "SearchHit",
]


class SearchHit(BaseModel):
    """Milvus 基础向量检索返回的稳定结果契约。

    Repository 将 SDK 的不稳定返回结构转换成该模型后，业务层只依赖这些字段；
    结果正文只用于返回命中内容，异常摘要不会回显完整输入或凭据。
    """

    model_config = ConfigDict(extra="forbid")

    vector_doc_id: str = Field(min_length=1)
    expert_id: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    chunk_type: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)
    score: float

    @field_validator(
        "vector_doc_id",
        "chunk_id",
        "chunk_type",
        "chunk_text",
    )
    @classmethod
    def reject_whitespace_only_values(cls, value: str) -> str:
        """拒绝检索结果中的空白标识和正文。"""

        if not value.strip():
            raise ValueError("检索结果字段不能只包含空白字符")
        return value

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        """确保 Milvus distance/score 是有限数值。"""

        if not math.isfinite(value):
            raise ValueError("检索结果 score 必须是有限数值")
        return value
