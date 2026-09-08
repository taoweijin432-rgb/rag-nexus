"""VectorIngestionService 的公共返回结果模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

VectorItemStatus = Literal[
    "embedded",
    "metadata_updated",
    "skipped",
    "failed",
]


class VectorItemResult(BaseModel):
    """单条向量操作结果，以 ``vector_doc_id`` 定位。"""

    model_config = ConfigDict(extra="forbid")

    vector_doc_id: StrictStr = Field(min_length=1)
    status: VectorItemStatus
    error_message: str | None = None
    recommendation: str | None = None

    @field_validator("vector_doc_id")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("vector_doc_id 不能只包含空白字符")
        return value


class VectorIngestionResult(BaseModel):
    """``upsert_chunks`` / ``update_metadata`` 的统一聚合结果。"""

    model_config = ConfigDict(extra="forbid")

    results: list[VectorItemResult] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
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

    requested_count: StrictInt = Field(default=0, ge=0)
    deleted_count: StrictInt = Field(default=0, ge=0)
    not_found_count: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> VectorDeletionResult:
        if self.deleted_count + self.not_found_count != self.requested_count:
            raise ValueError(
                "deleted_count 与 not_found_count 之和必须等于 requested_count"
            )
        return self


class VectorQueryResult(BaseModel):
    """``get_vectors`` 的查询结果，且不携带 embedding。"""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, object]] = Field(default_factory=list)
    missing: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_records(self) -> VectorQueryResult:
        record_ids: list[str] = []
        for record in self.records:
            if not isinstance(record, dict):
                raise TypeError("查询结果每条记录必须是普通字段字典")
            if not all(isinstance(key, str) for key in record):
                raise TypeError("查询结果字段名必须是字符串")
            if "embedding" in record:
                raise ValueError("公共查询结果不允许包含 embedding 字段")
            vector_doc_id = record.get("vector_doc_id")
            if not isinstance(vector_doc_id, str) or not vector_doc_id.strip():
                raise ValueError("查询结果记录必须包含非空 vector_doc_id")
            record_ids.append(vector_doc_id)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("查询结果 records 不能包含重复 vector_doc_id")
        missing_ids = list(self.missing)
        if len(missing_ids) != len(set(missing_ids)):
            raise ValueError("查询结果 missing 不能包含重复 vector_doc_id")
        if set(record_ids).intersection(missing_ids):
            raise ValueError("查询结果 records 与 missing 不能包含相同 vector_doc_id")
        return self


class VectorOwnerList(BaseModel):
    """按当前领域 owner 列出全部 ``vector_doc_id`` 的通用结果。"""

    model_config = ConfigDict(extra="forbid")

    owner_id: StrictInt = Field(ge=1)
    vector_doc_ids: list[StrictStr] = Field(default_factory=list)
    count: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> VectorOwnerList:
        if self.count != len(self.vector_doc_ids):
            raise ValueError("count 必须等于 vector_doc_ids 的长度")
        if any(not value.strip() for value in self.vector_doc_ids):
            raise ValueError("vector_doc_ids 不能包含空白标识")
        if len(self.vector_doc_ids) != len(set(self.vector_doc_ids)):
            raise ValueError("vector_doc_ids 不能包含重复标识")
        return self


__all__ = [
    "VectorDeletionResult",
    "VectorIngestionResult",
    "VectorItemResult",
    "VectorItemStatus",
    "VectorOwnerList",
    "VectorQueryResult",
]
