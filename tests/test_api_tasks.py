"""FastAPI API 端点测试。

Task2 Step 4：验证 HTTP 接口的创建、查询、404 等场景。

全部使用内存 Fake（FakeTaskStateStore + FakeExecutor），不依赖真实
MySQL / Milvus / Embedding / Reader。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_vector_store.api.app import ServiceContainer, create_app
from rag_vector_store.task_state.models import (
    TaskItemSummary,
    TaskState,
)
from rag_vector_store.task_state.store import TaskStateError

# ── Fake TaskStateStore ──


class FakeTaskStateStore:
    """内存 TaskStateStore，模拟真实 store 语义。"""

    def __init__(self) -> None:
        self.tasks: dict[str, TaskState] = {}
        self.items: dict[str, list[TaskItemSummary]] = {}

    def find_task_by_request_id(self, request_id: str) -> TaskState | None:
        for task in self.tasks.values():
            if task.request_id == request_id:
                return task
        return None

    def create_task(
        self,
        *,
        source: str,
        mode: str = "full",
        collection_name: str,
        total_count: int,
        request_id: str | None = None,
    ) -> TaskState:
        # 幂等：同一 request_id 已存在任务时直接返回
        if request_id is not None:
            existing = self.find_task_by_request_id(request_id)
            if existing is not None:
                return existing

        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        task = TaskState(
            task_id=task_id,
            source=source,
            mode=mode,
            status="created",
            total_count=total_count,
            success_count=0,
            skipped_count=0,
            failed_count=0,
            request_id=request_id,
            collection_name=collection_name,
            created_at=now,
        )
        self.tasks[task_id] = task
        self.items[task_id] = []
        return task

    def get_task(self, task_id: str) -> TaskState:
        tid = task_id.strip().lower()
        task = self.tasks.get(tid)
        if task is None:
            raise TaskStateError(f"任务不存在: {tid}")
        return task

    def list_items(
        self,
        task_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TaskItemSummary]:
        tid = task_id.strip().lower()
        if tid not in self.items:
            raise TaskStateError(f"任务不存在: {tid}")
        items = self.items[tid]
        if status is not None:
            items = [i for i in items if i.status == status]
        return items[offset : offset + limit]

    def add_items(self, task_id: str, items: list[TaskItemSummary]) -> None:
        tid = task_id.strip().lower()
        self.items[tid] = items


# ── Fake Executor ──


class FakeExecutor:
    """记录 submit 调用的假异步执行器。"""

    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, task_id: str) -> Any:
        self.submitted.append(task_id)


# ── Fixtures ──


@pytest.fixture()
def fake_store() -> FakeTaskStateStore:
    return FakeTaskStateStore()


@pytest.fixture()
def fake_executor() -> FakeExecutor:
    return FakeExecutor()


@pytest.fixture()
def client(
    fake_store: FakeTaskStateStore,
    fake_executor: FakeExecutor,
) -> TestClient:
    services = ServiceContainer(fake_store, fake_executor)
    app = create_app(services)
    return TestClient(app)


# ── 1. 创建任务返回 202 ──


def test_create_task_returns_202(client: TestClient, fake_executor: FakeExecutor) -> None:
    """POST 创建任务返回 202 Accepted。"""

    response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "full_batch",
            "collection_name": "expert_candidates_v14_contract_dev",
            "total_count": 100,
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["status"] == "created"
    assert "task_id" in body
    assert len(body["task_id"]) == 36

    # executor 被调用
    assert len(fake_executor.submitted) == 1
    assert fake_executor.submitted[0] == body["task_id"]


def test_create_task_with_request_id(client: TestClient) -> None:
    """带 request_id 创建任务。"""

    response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "full_batch",
            "collection_name": "expert_candidates_v14_contract_dev",
            "total_count": 10,
            "request_id": "req-001",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert len(body["task_id"]) == 36


def test_create_task_default_collection(client: TestClient) -> None:
    """不传 collection_name 时使用默认值。"""

    response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "full_batch",
            "total_count": 0,
        },
    )

    assert response.status_code == 202


def test_create_task_rejects_invalid_source(client: TestClient) -> None:
    """非法 source 返回 422。"""

    response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "invalid_source",
            "collection_name": "expert_candidates_v14_contract_dev",
            "total_count": 1,
        },
    )

    assert response.status_code == 422


# ── 2. 查询任务 ──


def test_get_task_success(client: TestClient, fake_store: FakeTaskStateStore) -> None:
    """GET 查询已存在的任务返回详情。"""

    # 先创建
    create_response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "full_batch",
            "collection_name": "expert_candidates_v14_contract_dev",
            "total_count": 50,
        },
    )
    task_id = create_response.json()["task_id"]

    # 查询
    response = client.get(f"/api/v1/vectorization/tasks/{task_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "created"
    assert body["source"] == "full_batch"
    assert body["total_count"] == 50
    assert body["collection_name"] == "expert_candidates_v14_contract_dev"


def test_get_task_not_found_returns_404(client: TestClient) -> None:
    """查询不存在的任务返回 404。"""

    fake_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/vectorization/tasks/{fake_id}")

    assert response.status_code == 404
    body = response.json()
    assert "不存在" in body["detail"]


# ── 3. 查询 items ──


def test_get_task_items_success(
    client: TestClient,
    fake_store: FakeTaskStateStore,
) -> None:
    """GET items 返回条目列表。"""

    # 创建任务
    create_response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "full_batch",
            "collection_name": "expert_candidates_v14_contract_dev",
            "total_count": 2,
        },
    )
    task_id = create_response.json()["task_id"]

    # 手动注入 items
    fake_store.add_items(
        task_id,
        [
            TaskItemSummary(
                task_id=task_id,
                vector_doc_id="expert:1:profile_summary:1:0",
                status="success",
                result="embedded",
            ),
            TaskItemSummary(
                task_id=task_id,
                vector_doc_id="expert:2:profile_summary:1:0",
                status="skipped",
                result="skipped",
            ),
        ],
    )

    response = client.get(f"/api/v1/vectorization/tasks/{task_id}/items")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["vector_doc_id"] == "expert:1:profile_summary:1:0"
    assert body["items"][0]["status"] == "success"
    assert body["items"][1]["status"] == "skipped"


def test_get_task_items_with_status_filter(
    client: TestClient,
    fake_store: FakeTaskStateStore,
) -> None:
    """items 支持 status 过滤。"""

    create_response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "full_batch",
            "collection_name": "expert_candidates_v14_contract_dev",
            "total_count": 3,
        },
    )
    task_id = create_response.json()["task_id"]

    fake_store.add_items(
        task_id,
        [
            TaskItemSummary(
                task_id=task_id,
                vector_doc_id="expert:1:profile_summary:1:0",
                status="success",
                result="embedded",
            ),
            TaskItemSummary(
                task_id=task_id,
                vector_doc_id="expert:2:profile_summary:1:0",
                status="success",
                result="embedded",
            ),
            TaskItemSummary(
                task_id=task_id,
                vector_doc_id="expert:3:profile_summary:1:0",
                status="failed",
            ),
        ],
    )

    # 只查 success
    response = client.get(
        f"/api/v1/vectorization/tasks/{task_id}/items?status=success"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert all(item["status"] == "success" for item in body["items"])


def test_get_task_items_empty(
    client: TestClient,
    fake_store: FakeTaskStateStore,
) -> None:
    """任务存在但没有 items 时返回空列表。"""

    create_response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "full_batch",
            "collection_name": "expert_candidates_v14_contract_dev",
            "total_count": 0,
        },
    )
    task_id = create_response.json()["task_id"]

    response = client.get(f"/api/v1/vectorization/tasks/{task_id}/items")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_get_task_items_task_not_found_404(client: TestClient) -> None:
    """查询不存在任务的 items 返回 404。"""

    fake_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/vectorization/tasks/{fake_id}/items")

    assert response.status_code == 404


# ── 4. 幂等创建（request_id） ──


def test_create_task_idempotent_request_id(
    client: TestClient,
    fake_executor: FakeExecutor,
) -> None:
    """同一 request_id 重复创建返回同一 task_id。"""

    payload = {
        "source": "full_batch",
        "collection_name": "expert_candidates_v14_contract_dev",
        "total_count": 5,
        "request_id": "idempotent-001",
    }

    response1 = client.post("/api/v1/vectorization/tasks", json=payload)
    response2 = client.post("/api/v1/vectorization/tasks", json=payload)

    assert response1.status_code == 202
    assert response2.status_code == 202
    assert response1.json()["task_id"] == response2.json()["task_id"]


# ── 5. 端到端流程（Fake） ──


def test_full_flow_create_then_query(client: TestClient) -> None:
    """创建后立即查询验证字段完整性。"""

    create_response = client.post(
        "/api/v1/vectorization/tasks",
        json={
            "source": "full_batch",
            "collection_name": "expert_candidates_v14_contract_dev",
            "total_count": 100,
            "request_id": "flow-001",
        },
    )

    assert create_response.status_code == 202
    task_id = create_response.json()["task_id"]

    query_response = client.get(f"/api/v1/vectorization/tasks/{task_id}")

    assert query_response.status_code == 200
    body = query_response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "created"
    assert body["total_count"] == 100
    assert body["request_id"] == "flow-001"
    assert body["success_count"] == 0
    assert body["skipped_count"] == 0
    assert body["failed_count"] == 0
    assert body["created_at"] is not None
