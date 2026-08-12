"""正式 Contract ingestion 和 Milvus 查询使用的稳定结果模型。"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MutationResult(BaseModel):
    """与具体数据库无关的 repository 写操作结果。"""

    affected_count: int = Field(ge=0)
    primary_keys: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    """Milvus 基础向量检索返回的稳定结果契约。"""

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


__all__ = ["MutationResult", "SearchHit"]
