"""API 请求/响应 Pydantic 模型。

这些模型是 HTTP 层的 DTO，与 TaskState/TaskItemState（内部模型）分离。
HTTP 层只做字段投影和序列化，不暴露内部实现细节。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ── 请求模型 ──


class CreateTaskRequest(BaseModel):
    """POST /api/v1/vectorization/tasks 请求体。"""

    model_config = ConfigDict(extra="forbid")

    source: Literal["full_batch", "api_submit"] = "full_batch"
    collection_name: str = Field(
        default="expert_candidates_v14_contract_dev",
        min_length=1,
        max_length=256,
    )
    total_count: int = Field(default=0, ge=0)
    request_id: str | None = Field(default=None, max_length=64)


# ── 响应模型 ──


class CreateTaskResponse(BaseModel):
    """POST /api/v1/vectorization/tasks 响应体（202 Accepted）。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str = "created"
    accepted: bool = True


class TaskDetailResponse(BaseModel):
    """GET /api/v1/vectorization/tasks/{task_id} 响应体。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: str
    mode: str
    status: str
    total_count: int
    success_count: int
    skipped_count: int
    failed_count: int
    request_id: str | None = None
    error_message: str | None = None
    collection_name: str
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TaskItemResponse(BaseModel):
    """单条 task_item 响应体。"""

    model_config = ConfigDict(extra="forbid")

    vector_doc_id: str
    status: str
    result: str | None = None
    error_message: str | None = None
    processed_at: datetime | None = None


class TaskItemListResponse(BaseModel):
    """GET /api/v1/vectorization/tasks/{task_id}/items 响应体。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    items: list[TaskItemResponse]
    total: int
