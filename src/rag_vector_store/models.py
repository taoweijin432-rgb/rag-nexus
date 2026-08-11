"""chunk 生成方、向量模型和 repository 之间共享的数据协议。"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Chunk(BaseModel):
    """chunk 生成模块使用的稳定接口协议。

    chunk 生成逻辑只需要提供应用内唯一的 ``chunk_id`` 和待向量化文本。
    解析器或切分器的专有信息应放入 ``metadata``，这样协议就不会绑定某个
    具体的解析器或切分策略。
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, description="chunk 的稳定唯一标识")
    text: str = Field(min_length=1, description="传给 embedding 模型的文本")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "text")
    @classmethod
    def reject_whitespace_only_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id and text must not contain only whitespace")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """提前拦截无法写入 Milvus JSON 字段的值。"""

        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        return value


class EmbeddedChunk(Chunk):
    """文本已经转换为稠密浮点向量后的 chunk。"""

    embedding: list[float] = Field(min_length=1)

    @field_validator("embedding")
    @classmethod
    def validate_embedding_values(cls, value: list[float]) -> list[float]:
        normalized = [float(item) for item in value]
        if not all(math.isfinite(item) for item in normalized):
            raise ValueError("embedding must contain only finite numbers")
        return normalized


class MutationResult(BaseModel):
    """与具体数据库无关的 repository 写操作结果。"""

    affected_count: int = Field(ge=0)
    primary_keys: list[str] = Field(default_factory=list)
