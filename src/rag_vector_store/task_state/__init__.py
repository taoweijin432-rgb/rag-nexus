"""Task State 模块。

本模块为 Task2 服务化 Step 1 提供独立的 Task State 持久化能力。

- 与 expert_chunk 读取配置完全隔离（使用 ``TASK_STATE_MYSQL_*`` 环境变量，
  不读取 ``MYSQL_*`` / ``SOURCE_DB_*`` / ``DB_*``）。
- 不依赖现有 ``MySQLConnection`` 业务链路；通过 ``TaskStateStore`` 直接读写
  独立的 ``vectorization_task`` / ``vectorization_task_item`` 两张表。
- 所有异常均通过 ``TaskStateError`` 安全暴露（不包含凭据、chunk_text 或向量）。
- Task State 独立于 Milvus：Milvus 只保存向量，Task State 保存处理状态。
"""

from .config import TaskStateMySQLSettings
from .models import (
    TaskItemResult,
    TaskItemState,
    TaskItemStatus,
    TaskItemSummary,
    TaskSource,
    TaskState,
    TaskStatus,
    TaskSummary,
    new_task_id,
)
from .store import TaskStateError, TaskStateStore

__all__ = [
    "TaskItemResult",
    "TaskItemState",
    "TaskItemStatus",
    "TaskItemSummary",
    "TaskSource",
    "TaskState",
    "TaskStateError",
    "TaskStateMySQLSettings",
    "TaskStateStore",
    "TaskStatus",
    "TaskSummary",
    "new_task_id",
]