"""FastAPI 应用工厂与路由定义。

Task2 Step 4：HTTP 服务入口。

三个端点：
    POST   /api/v1/vectorization/tasks                     — 创建任务 + 后台提交
    GET    /api/v1/vectorization/tasks/{task_id}            — 查询任务详情
    GET    /api/v1/vectorization/tasks/{task_id}/items      — 查询条目列表

设计要点：
    - 依赖注入通过 ``ServiceContainer`` 传入 TaskStateStore + AsyncTaskExecutor
    - API 只做 HTTP 转换 + 调用服务，不内嵌业务逻辑
    - POST 创建任务后立即 submit 到 AsyncExecutor，返回 202
    - GET 查询 TaskStateStore；task 不存在返回 404
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status

from ..task_state.store import TaskStateError, TaskStateStore
from .models import (
    CreateTaskRequest,
    CreateTaskResponse,
    TaskDetailResponse,
    TaskItemListResponse,
    TaskItemResponse,
)

if TYPE_CHECKING:
    from ..runner.async_executor import AsyncTaskExecutor

__all__ = ["ServiceContainer", "create_app"]


class ServiceContainer:
    """服务依赖容器，持有 TaskStateStore 和 AsyncTaskExecutor。

    由外部（main.py 或测试）构造后注入到 app.state。
    """

    def __init__(
        self,
        task_state_store: TaskStateStore,
        executor: AsyncTaskExecutor | Any,
    ) -> None:
        self.task_state_store = task_state_store
        self.executor = executor


# ── 依赖注入 ──


def _get_services(request: Request) -> ServiceContainer:
    """FastAPI 依赖：从 app.state 获取 ServiceContainer。"""

    container: ServiceContainer | None = getattr(request.app.state, "services", None)
    if container is None:
        raise RuntimeError("ServiceContainer 未初始化")
    return container


# ── 路由 ──


def _build_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/vectorization/tasks", tags=["vectorization"])

    # ── POST: 创建任务 + 后台提交 ──

    @router.post(
        "",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=CreateTaskResponse,
        summary="创建向量化任务",
    )
    def create_task(
        request: CreateTaskRequest,
        services: ServiceContainer = Depends(_get_services),  # noqa: B008
    ) -> CreateTaskResponse:
        """创建 vectorization_task 并提交到 AsyncExecutor 后台执行。

        返回 202 Accepted，表示任务已接受、正在后台执行。
        """

        try:
            task = services.task_state_store.create_task(
                source=request.source,
                collection_name=request.collection_name,
                total_count=request.total_count,
                request_id=request.request_id,
            )
        except TaskStateError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"任务创建失败: {_safe_msg(exc)}",
            ) from exc

        # 提交到 AsyncExecutor 后台执行
        services.executor.submit(task.task_id)

        return CreateTaskResponse(
            task_id=task.task_id,
            status=task.status,
            accepted=True,
        )

    # ── GET: 查询任务详情 ──

    @router.get(
        "/{task_id}",
        response_model=TaskDetailResponse,
        summary="查询任务详情",
    )
    def get_task(
        task_id: str,
        services: ServiceContainer = Depends(_get_services),  # noqa: B008
    ) -> TaskDetailResponse:
        """查询指定任务的当前状态。"""

        try:
            task = services.task_state_store.get_task(task_id)
        except TaskStateError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务不存在: {_safe_msg(exc)}",
            ) from exc

        return TaskDetailResponse(
            task_id=task.task_id,
            source=task.source,
            mode=task.mode,
            status=task.status,
            total_count=task.total_count,
            success_count=task.success_count,
            skipped_count=task.skipped_count,
            failed_count=task.failed_count,
            request_id=task.request_id,
            error_message=task.error_message,
            collection_name=task.collection_name,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
        )

    # ── GET: 查询条目列表 ──

    @router.get(
        "/{task_id}/items",
        response_model=TaskItemListResponse,
        summary="查询任务条目列表",
    )
    def get_task_items(
        task_id: str,
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        services: ServiceContainer = Depends(_get_services),  # noqa: B008
    ) -> TaskItemListResponse:
        """查询指定任务的条目列表，支持 status 过滤与分页。"""

        # 先验证任务存在
        try:
            services.task_state_store.get_task(task_id)
        except TaskStateError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务不存在: {_safe_msg(exc)}",
            ) from exc

        items = services.task_state_store.list_items(
            task_id,
            status=status_filter,
            limit=limit,
            offset=offset,
        )

        return TaskItemListResponse(
            task_id=task_id.strip().lower(),
            items=[
                TaskItemResponse(
                    vector_doc_id=item.vector_doc_id,
                    status=item.status,
                    result=item.result,
                    error_message=item.error_message,
                    processed_at=item.processed_at,
                )
                for item in items
            ],
            total=len(items),
        )

    return router


# ── 应用工厂 ──


def create_app(services: ServiceContainer) -> FastAPI:
    """构建 FastAPI 应用实例。

    Args:
        services: 已初始化的 ServiceContainer（持有 store + executor）。
    """

    app = FastAPI(
        title="Vectorization Task API",
        description="向量化任务 HTTP 服务（Task2 Step 4）",
        version="0.1.0",
    )
    app.state.services = services
    app.include_router(_build_router())
    return app


def _safe_msg(exc: Exception) -> str:
    """返回安全错误摘要（不含 PII / 凭据）。"""

    return str(exc)[:200]
