"""VectorIngestionService 的公共返回结果模型。

这些模型是 ``upsert_chunks`` / ``update_metadata`` / ``disable_vectors`` /
``get_vectors`` / ``list_vectors_by_owner`` 方法的稳定返回契约。

设计约束：
    - 不暴露 pymilvus / Embedding 内部对象。
    - 不返回 Milvus raw row 作为公共契约（查询返回普通字段字典）。
    - 查询模型（``VectorQueryResult``）不携带 embedding。
    - 模型保持精简，不提前加入 Task / 增量 / 业务更新相关字段。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VectorItemStatus = Literal[
    "embedded",
    "metadata_updated",
    "skipped",
    "failed",
]


class VectorItemResult(BaseModel):
    """单条向量操作结果，以 ``vector_doc_id`` 定位。

    status 语义：
        - ``embedded``：upsert_chunks 成功写入新向量。
        - ``metadata_updated``：update_metadata 复用旧向量更新 metadata。
        - ``skipped``：Eligibility 跳过（is_active=false / rejected）。
        - ``failed``：update_metadata 安全门失败或未知失败。
    """

    model_config = ConfigDict(extra="forbid")

    vector_doc_id: str = Field(min_length=1)
    status: VectorItemStatus
    error_message: str | None = None
    recommendation: str | None = None

    @field_validator("vector_doc_id")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        """拒绝空白标识。"""

        if not value.strip():
            raise ValueError("vector_doc_id 不能只包含空白字符")
        return value


class VectorIngestionResult(BaseModel):
    """``upsert_chunks`` / ``update_metadata`` 的统一聚合结果。"""

    model_config = ConfigDict(extra="forbid")

    results: list[VectorItemResult] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
        """本批输入总数。"""

        return len(self.results)

    @property
    def embedded_count(self) -> int:
        return sum(item.status == "embedded" for item in self.results)

    @property
    def metadata_updated_count(self) -> int:
        return sum(item.status == "metadata_updated" for item in self.results)

    @property
    def skipped_count(self) -> int:
        return sum(item.status == "skipped" for item in self.results)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.results)


class VectorDeletionResult(BaseModel):
    """逻辑下线结果；保留历史字段名以兼容现有序列化契约。"""

    model_config = ConfigDict(extra="forbid")

    requested_count: int = Field(default=0, ge=0)
    deleted_count: int = Field(default=0, ge=0)
    not_found_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> VectorDeletionResult:
        """确保删除计数与请求/未命中关系一致。"""

        if self.deleted_count + self.not_found_count != self.requested_count:
            raise ValueError(
                "deleted_count 与 not_found_count 之和必须等于 requested_count"
            )
        return self


class VectorQueryResult(BaseModel):
    """``get_vectors`` 的查询结果，且不携带 embedding。"""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, object]] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_records(self) -> VectorQueryResult:
        """公共查询结果不得包含 embedding，且每条是普通字段字典。"""

        for record in self.records:
            if not isinstance(record, dict):
                raise TypeError("查询结果每条记录必须是普通字段字典")
            if not all(isinstance(key, str) for key in record):
                raise TypeError("查询结果字段名必须是字符串")
            if "embedding" in record:
                raise ValueError("公共查询结果不允许包含 embedding 字段")
        return self


class VectorOwnerList(BaseModel):
    """按当前领域 owner 列出全部 ``vector_doc_id`` 的通用结果。"""

    model_config = ConfigDict(extra="forbid")

    owner_id: int = Field(ge=1)
    vector_doc_ids: list[str] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> VectorOwnerList:
        """``count`` 必须等于 ``vector_doc_ids`` 长度。"""

        if self.count != len(self.vector_doc_ids):
            raise ValueError("count 必须等于 vector_doc_ids 的长度")
        return self


class VectorExpertList(BaseModel):
    """已弃用的专家专用列表结果；请使用 ``VectorOwnerList``。"""

    model_config = ConfigDict(extra="forbid")

    expert_id: int = Field(ge=1)
    vector_doc_ids: list[str] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> VectorExpertList:
        """``count`` 必须等于 ``vector_doc_ids`` 长度。"""

        if self.count != len(self.vector_doc_ids):
            raise ValueError("count 必须等于 vector_doc_ids 的长度")
        return self


class VectorResumeList(BaseModel):
    """已弃用的 Resume 专用列表结果；请使用 ``VectorOwnerList``。"""

    model_config = ConfigDict(extra="forbid")

    resume_id: int = Field(ge=1)
    vector_doc_ids: list[str] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> VectorResumeList:
        """``count`` 必须等于 ``vector_doc_ids`` 的长度。"""

        if self.count != len(self.vector_doc_ids):
            raise ValueError("count 必须等于 vector_doc_ids 的长度")
        return self


__all__ = [
    "VectorDeletionResult",
    "VectorExpertList",
    "VectorIngestionResult",
    "VectorItemResult",
    "VectorItemStatus",
    "VectorOwnerList",
    "VectorQueryResult",
    "VectorResumeList",
]
