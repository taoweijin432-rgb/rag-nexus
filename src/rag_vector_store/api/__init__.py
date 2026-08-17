"""FastAPI HTTP 服务入口。

Task2 Step 4：向量化任务 HTTP 接口。

职责：
    - HTTP 请求/响应转换
    - 调用 TaskStateStore 创建/查询任务
    - 调用 AsyncTaskExecutor 提交后台执行
    - 立即返回 202 Accepted

不职责：
    - 不实现 Reader / Contract / Embedding / Milvus
    - 不实现 Runner / AsyncExecutor 内部逻辑
    - 不管理任务执行流程
"""

from .app import ServiceContainer, create_app
from .factory import create_production_app
from .models import (
    CreateTaskRequest,
    CreateTaskResponse,
    TaskDetailResponse,
    TaskItemListResponse,
    TaskItemResponse,
)

__all__ = [
    "CreateTaskRequest",
    "CreateTaskResponse",
    "ServiceContainer",
    "TaskDetailResponse",
    "TaskItemListResponse",
    "TaskItemResponse",
    "create_app",
    "create_production_app",
]
