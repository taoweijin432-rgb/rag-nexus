"""Task State 数据模型与状态常量。

与 ``vectorization_task`` / ``vectorization_task_item`` 两张表一一对应
（DDL 见 ``sql/006_create_vectorization_task_state.sql``）。

命名以 ``docs/TASK2_SERVICEIZATION_PLAN.md`` 为准：
- 任务状态：created → running → completed / failed
- 条目状态：pending → success / skipped / failed
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── 状态取值 ──
TaskStatus = Literal["created", "running", "completed", "failed"]
TaskItemStatus = Literal["pending", "success", "skipped", "failed"]

# ── 来源 ──
TaskSource = Literal["full_batch", "api_submit"]

# ── 条目结果 ──
# result 是 VARCHAR(128)，存 embedded / 跳过原因 / NULL；
# Literal 只用于类型注释，不限制字段值。
TaskItemResult = str | None

# 时间字段使用 DATETIME(6) 微秒精度（与 DDL 一致）。
# naive datetime 统一按 Asia/Shanghai 解释（与 Reader database_timezone 约定一致）。


def new_task_id() -> str:
    """生成 UUID v4 任务 ID（与 DDL ``CHAR(36)`` 的字符串形式一致）。"""

    return str(uuid.uuid4())


class TaskState(BaseModel):
    """``vectorization_task`` 任务级记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=36, max_length=36)
    source: TaskSource
    mode: str = Field(default="full", min_length=1)
    status: TaskStatus = "created"
    total_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    request_id: str | None = Field(default=None, max_length=64)
    error_message: str | None = Field(default=None, max_length=512)
    collection_name: str = Field(min_length=1, max_length=256)
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("task_id")
    @classmethod
    def _normalize_task_id(cls, value: str) -> str:
        return value.strip().lower()


class TaskItemState(BaseModel):
    """``vectorization_task_item`` 记录级条目。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=36, max_length=36)
    vector_doc_id: str = Field(min_length=1, max_length=256)
    status: TaskItemStatus = "pending"
    result: TaskItemResult | None = None
    error_message: str | None = Field(default=None, max_length=512)
    processed_at: datetime | None = None
    created_at: datetime | None = None

    @field_validator("task_id")
    @classmethod
    def _normalize_task_id(cls, value: str) -> str:
        return value.strip().lower()


class TaskSummary(BaseModel):
    """任务汇总信息（查询响应用）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    status: TaskStatus
    source: TaskSource
    total_count: int = 0
    success_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None


class TaskItemSummary(BaseModel):
    """任务条目汇总信息（查询响应用）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    vector_doc_id: str
    status: TaskItemStatus
    result: TaskItemResult | None = None
    error_message: str | None = None
    processed_at: datetime | None = None