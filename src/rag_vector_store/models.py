"""正式 Contract ingestion 和 Milvus 查询使用的稳定结果模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MutationResult(BaseModel):
    """与具体数据库无关的 repository 写操作结果。"""

    affected_count: int = Field(ge=0)
    primary_keys: list[str] = Field(default_factory=list)


__all__ = ["MutationResult"]
