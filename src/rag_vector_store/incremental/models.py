"""V15 增量向量化的 Batch / Work Item 持久化模型。

本模块定义增量向量化持久化层的公共契约，不暴露 pymysql 或连接细节：

- ``V15WorkItemOperation``：只允许 upsert / metadata_update / disable；
  不存在物理 delete 语义，disable 只把 is_active 置为 false，
  向量与实体始终保留在 Milvus。
- ``V15WorkItemStatus`` / ``V15BatchStatus``：状态机取值。
- ``V15WorkItemInput``：创建 Work Item 的输入（fail-closed 校验）。
- ``V15VectorizationWorkItem`` / ``V15VectorizationBatch``：读取返回的完整行。

字段与 ``sql/011_add_v15_vectorization_persistence.sql`` 的物理表一一对应；
来源定位统一使用 ``source_field_id``，本模块不存在 ``source_record_id``。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..constants import V15_CHUNK_TEXT_MAX_BYTES, V15_VECTOR_DOC_ID_MAX_BYTES

# 与 Milvus V15 Schema 一致的字节上限。
# Contract 侧 hash 为 SHA-256 十六进制（64 位小写）。
_SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"

# 四张专家来源表；与上游 expert_index 的 Checkpoint 约定一致。
ALLOWED_SOURCE_TABLES: tuple[str, ...] = (
    "experts",
    "experts_work_experience",
    "experts_project_experience",
    "experts_education_experience",
)

# 与 Contract 的 ``ChunkType`` 对齐的三值枚举；独立声明以保持本模块
# 不依赖 expert_contracts 内部符号。
V15ChunkType = Literal["profile_summary", "employment", "project"]

V15WorkItemOperation = Literal["upsert", "metadata_update", "disable"]
V15WorkItemStatus = Literal[
    "created",
    "running",
    "committed",
    "failed",
    "rejected",
]
V15BatchStatus = Literal[
    "created",
    "running",
    "committed",
    "partial",
    "failed",
]

# 允许接收新 Work Item 的批次状态；终态批次拒绝追加。
APPENDABLE_BATCH_STATUSES: tuple[str, ...] = ("created", "running")

# 批次可到达的终态。
BATCH_FINAL_STATUSES: tuple[str, ...] = ("committed", "partial", "failed")


def _utf8_byte_length(value: str) -> int:
    """返回字符串的 UTF-8 字节长度，用于对齐 Milvus VARCHAR 上限。"""

    return len(value.encode("utf-8"))


class V15WorkItemInput(BaseModel):
    """创建 Work Item 的输入。

    按 operation 做 fail-closed 校验：

    - ``upsert``：必须携带 chunk_text 与 content_hash；
    - ``metadata_update``：必须携带 metadata_hash，且不得携带 chunk_text
      （metadata-only 更新不得重新 Embedding）；
    - ``disable``：不得携带 chunk_text / content_hash / metadata_hash，
      is_active 强制为 False（保留向量与实体，仅下线）。
    """

    model_config = ConfigDict(extra="forbid")

    operation: V15WorkItemOperation
    vector_doc_id: str = Field(min_length=1)
    chunk_text: str | None = None
    metadata: dict[str, Any] | None = None
    content_hash: str | None = Field(default=None, pattern=_SHA256_HEX_PATTERN)
    metadata_hash: str | None = Field(default=None, pattern=_SHA256_HEX_PATTERN)
    chunk_policy_version: str | None = Field(default=None, max_length=32)
    embedding_model_alias: str | None = Field(default=None, max_length=128)
    embedding_model_version: str | None = Field(default=None, max_length=64)
    is_active: bool = True
    expert_id: int | None = Field(default=None, ge=0)
    chunk_type: V15ChunkType | None = None
    source_table: str | None = Field(default=None, max_length=256)
    source_field_id: str | None = Field(default=None, max_length=128)
    source_event_id: int | None = Field(
        default=None,
        ge=0,
        description="上游 CDC 单调递增事件 id；新旧事件比较的主依据",
    )
    source_updated_at: datetime | None = None
    stale_vector_doc_ids: tuple[str, ...] = ()

    @field_validator("vector_doc_id")
    @classmethod
    def validate_vector_doc_id_bytes(cls, value: str) -> str:
        """对齐 V15 向量主键的字节上限。"""

        if _utf8_byte_length(value) > V15_VECTOR_DOC_ID_MAX_BYTES:
            raise ValueError(
                "vector_doc_id 超过 V15 上限 "
                f"{V15_VECTOR_DOC_ID_MAX_BYTES} 字节"
            )
        return value

    @field_validator("chunk_text")
    @classmethod
    def validate_chunk_text_bytes(cls, value: str | None) -> str | None:
        """对齐 V15 chunk_text 的字节上限。"""

        if value is not None and _utf8_byte_length(value) > V15_CHUNK_TEXT_MAX_BYTES:
            raise ValueError(
                f"chunk_text 超过 V15 上限 {V15_CHUNK_TEXT_MAX_BYTES} 字节"
            )
        return value

    @field_validator("source_table")
    @classmethod
    def validate_source_table(cls, value: str | None) -> str | None:
        """来源表必须是四张专家来源表之一。"""

        if value is not None and value not in ALLOWED_SOURCE_TABLES:
            allowed = "、".join(ALLOWED_SOURCE_TABLES)
            raise ValueError(f"source_table 必须是以下之一: {allowed}")
        return value

    @field_validator("stale_vector_doc_ids")
    @classmethod
    def validate_stale_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """待失效 id 必须是非空字符串。"""

        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("stale_vector_doc_ids 不能包含空白项")
        return value

    @model_validator(mode="after")
    def validate_operation_semantics(self) -> V15WorkItemInput:
        """按 operation 收紧字段语义，禁止物理 delete 语义进入持久层。"""

        if self.operation == "upsert":
            if not self.chunk_text or not self.chunk_text.strip():
                raise ValueError("upsert 操作必须携带 chunk_text")
            if self.content_hash is None:
                raise ValueError("upsert 操作必须携带 content_hash")
        elif self.operation == "metadata_update":
            if self.chunk_text is not None:
                raise ValueError("metadata_update 是 metadata-only 更新，不得携带 chunk_text")
            if self.metadata_hash is None:
                raise ValueError("metadata_update 操作必须携带 metadata_hash")
        else:  # disable
            if self.chunk_text is not None:
                raise ValueError("disable 操作不得携带 chunk_text")
            if self.content_hash is not None or self.metadata_hash is not None:
                raise ValueError("disable 操作不得携带 content_hash / metadata_hash")
            # disable 只保留向量与实体并下线，强制 is_active=False。
            self.is_active = False
        return self


class V15WorkItemCreateResult(BaseModel):
    """单条 Work Item 的创建结果。

    ``created=False`` 表示命中幂等约束，返回已存在条目的 work_item_id。
    """

    model_config = ConfigDict(extra="forbid")

    work_item_id: int
    created: bool


class V15VectorizationWorkItem(BaseModel):
    """Work Item 完整行，与物理表字段一一对应。"""

    model_config = ConfigDict(extra="forbid")

    work_item_id: int
    batch_id: int
    operation: V15WorkItemOperation
    vector_doc_id: str
    chunk_text: str | None = None
    metadata: dict[str, Any] | None = None
    content_hash: str | None = None
    metadata_hash: str | None = None
    chunk_policy_version: str | None = None
    embedding_model_alias: str | None = None
    embedding_model_version: str | None = None
    is_active: bool = True
    expert_id: int | None = None
    chunk_type: V15ChunkType | None = None
    source_table: str | None = None
    source_field_id: str | None = None
    source_event_id: int | None = None
    source_updated_at: datetime | None = None
    stale_vector_doc_ids: tuple[str, ...] = ()
    status: V15WorkItemStatus
    attempts: int = 0
    error_type: str | None = None
    error_message: str | None = None
    lease_until: datetime | None = None
    processed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class V15VectorizationBatch(BaseModel):
    """Batch 完整行，与物理表字段一一对应。"""

    model_config = ConfigDict(extra="forbid")

    batch_id: int
    upstream_batch_id: int | None = None
    status: V15BatchStatus
    total_count: int = 0
    committed_count: int = 0
    failed_count: int = 0
    rejected_count: int = 0
    error_summary: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None
