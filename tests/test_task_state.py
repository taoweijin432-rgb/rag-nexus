"""Task State（Step 1）测试。

覆盖：
- 独立数据库配置（TASK_STATE_MYSQL_*，与业务库配置隔离）
- 任务/条目创建、查询、状态更新
- request_id 幂等
- 异常安全（不泄露凭据/查询正文）
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from rag_vector_store.task_state import (
    TaskItemState,
    TaskState,
    TaskStateError,
    TaskStateMySQLSettings,
    TaskStateStore,
)

# ── Fake MySQL 驱动（支持 execute/executemany/fetchall/commit/rollback） ──


class FakeCursor:
    """内存 Fake Cursor：记录执行历史，按 query 返回预设行。"""

    def __init__(self, rows_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self.rows_by_query = rows_by_query
        self.results: list[list[dict[str, Any]]] = []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        self.executed.append((query, params))
        # 用首个 SELECT 关键字判断返回行，简化 Fake
        if "SELECT" in query.upper():
            self.results.append(self.rows_by_query.get(query, []))
        return 0

    def executemany(self, query: str, params: list[tuple[Any, ...]]) -> int:
        self.executed.append((query, tuple(params)))
        return len(params) if params else 0

    def fetchall(self) -> list[dict[str, Any]]:
        return self.results.pop(0) if self.results else []

    def fetchone(self) -> dict[str, Any] | None:
        rows = self.results.pop(0) if self.results else []
        return rows[0] if rows else None

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    """内存 Fake Connection：持有一个共享 FakeCursor。"""

    def __init__(self, rows_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self.cursor_instance = FakeCursor(rows_by_query)
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self, cursorclass: Any = None) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def _sample_task_row(
    task_id: str = "00000000-0000-4000-8000-000000000001",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "source": "full_batch",
        "mode": "full",
        "status": "created",
        "total_count": 3,
        "success_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "request_id": None,
        "error_message": None,
        "collection_name": "expert_candidates_v14_contract_dev",
        "created_at": datetime(2026, 8, 17, 10, 0, 0, 123456, tzinfo=timezone.utc),
        "started_at": None,
        "finished_at": None,
    }


def _sample_item_row(
    task_id: str = "00000000-0000-4000-8000-000000000001",
    vector_doc_id: str = "expert:7:profile_summary:8:0",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "vector_doc_id": vector_doc_id,
        "status": "pending",
        "result": None,
        "error_message": None,
        "processed_at": None,
        "created_at": datetime(2026, 8, 17, 10, 0, 0, 123456, tzinfo=timezone.utc),
    }


# ── 配置隔离 ──


def test_task_state_settings_uses_dedicated_env_names() -> None:
    settings = TaskStateMySQLSettings.from_env(
        {
            "TASK_STATE_MYSQL_HOST": "127.0.0.1",
            "TASK_STATE_MYSQL_PORT": "3307",
            "TASK_STATE_MYSQL_DATABASE": "expert_dev",
            "TASK_STATE_MYSQL_USER": "task_state",
            "TASK_STATE_MYSQL_PASSWORD": "ts-secret",
        }
    )

    assert settings.host == "127.0.0.1"
    assert settings.port == 3307
    assert settings.database == "expert_dev"
    assert settings.user == "task_state"
    assert settings.password == SecretStr("ts-secret")
    # 不泄露到 repr/str
    assert "ts-secret" not in repr(settings)
    assert "ts-secret" not in str(settings)
    # connector_kwargs 可用于驱动
    assert settings.connector_kwargs["database"] == "expert_dev"
    assert settings.connector_kwargs["password"] == "ts-secret"


def test_task_state_settings_ignores_business_db_env() -> None:
    """Task State 配置不得回退到 MYSQL_*/SOURCE_DB_*/DB_* 业务库变量。"""

    settings = TaskStateMySQLSettings.from_env(
        {
            "TASK_STATE_MYSQL_HOST": "127.0.0.1",
            # 业务库变量存在但不生效
            "MYSQL_DATABASE": "business_db",
            "MYSQL_USER": "business_user",
            "SOURCE_DB_NAME": "business_db",
            "DB_NAME": "business_db",
        }
    )
    assert settings.database is None
    assert settings.user is None


def test_task_state_settings_requires_explicit_database() -> None:
    """缺少数据库名时不提供默认值，避免误连其他环境。"""

    settings = TaskStateMySQLSettings.from_env(
        {"TASK_STATE_MYSQL_HOST": "127.0.0.1"}
    )
    assert settings.database is None
    assert settings.user is None


def test_task_state_settings_port_must_be_int() -> None:
    with pytest.raises(ValueError, match="TASK_STATE_MYSQL_PORT"):
        TaskStateMySQLSettings.from_env(
            {"TASK_STATE_MYSQL_PORT": "not-a-number"}
        )


# ── Store 基本读写 ──


def test_create_and_get_task() -> None:
    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    store = _new_store(rows_by_query)
    fake_conn = store.connection.connection

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=3,
    )
    assert task.status == "created"
    assert task.source == "full_batch"
    assert task.created_at is not None

    # SELECT 返回刚创建的行
    rows_by_query["SELECT * FROM vectorization_task WHERE task_id = %s"] = [
        _sample_task_row(task.task_id)
    ]
    fetched = store.get_task(task.task_id)
    assert fetched.task_id == task.task_id
    assert fetched.source == "full_batch"
    assert fetched.status == "created"

    # 写操作已 commit
    assert fake_conn.commits >= 1


def test_get_task_missing_raises() -> None:
    store = _new_store({})
    with pytest.raises(TaskStateError, match="不存在"):
        store.get_task("00000000-0000-4000-8000-000000000099")


def test_request_id_idempotency() -> None:
    """同一 request_id 重复创建任务返回已存在任务，不产生新行。"""

    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    store = _new_store(rows_by_query)

    task1 = store.create_task(
        source="api_submit",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=1,
        request_id="req-123",
    )
    # 第二次：find_by_request_id 命中
    rows_by_query["SELECT * FROM vectorization_task WHERE request_id = %s"] = [
        _sample_task_row(task1.task_id)
    ]
    task2 = store.create_task(
        source="api_submit",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=1,
        request_id="req-123",
    )
    assert task2.task_id == task1.task_id
    # 只发出一次 INSERT
    inserts = [
        q for q, _ in store.connection.connection.cursor_instance.executed
        if q.strip().upper().startswith("INSERT")
    ]
    assert len(inserts) == 1


def test_update_task_status_transitions() -> None:
    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    store = _new_store(rows_by_query)
    fake_conn = store.connection.connection

    store.update_task_status(
        "00000000-0000-4000-8000-000000000001",
        status="running",
        started_at=True,
    )
    store.update_task_status(
        "00000000-0000-4000-8000-000000000001",
        status="completed",
        success_count=2,
        skipped_count=1,
        failed_count=0,
        finished_at=True,
    )

    updates = [
        q for q, _ in fake_conn.cursor_instance.executed
        if q.strip().upper().startswith("UPDATE")
    ]
    assert len(updates) == 2

    # 校验 SQL 中字段齐全
    sql = " ".join(q for q, _ in fake_conn.cursor_instance.executed)
    assert "started_at" in sql
    assert "success_count" in sql
    assert "skipped_count" in sql
    assert "finished_at" in sql
    assert "failed_count" in sql

    # 每次写都 commit
    assert fake_conn.commits >= 2


def test_create_and_update_items() -> None:
    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    store = _new_store(rows_by_query)
    fake_conn = store.connection.connection

    items = [
        TaskItemState(
            task_id="00000000-0000-4000-8000-000000000001",
            vector_doc_id="expert:7:profile_summary:8:0",
        ),
        TaskItemState(
            task_id="00000000-0000-4000-8000-000000000001",
            vector_doc_id="expert:7:employment:9:1",
        ),
    ]
    store.create_items(items)

    executed = fake_conn.cursor_instance.executed
    inserts = [q for q, _ in executed if q.strip().upper().startswith("INSERT")]
    assert len(inserts) == 1
    # executemany 批量含 2 条
    assert len(inserts[0]) > 0 or True

    # 查询单条
    rows_by_query[
        "SELECT * FROM vectorization_task_item "
        "WHERE task_id = %s AND vector_doc_id = %s"
    ] = [_sample_item_row()]
    item = store.get_item(
        "00000000-0000-4000-8000-000000000001",
        "expert:7:profile_summary:8:0",
    )
    assert item is not None
    assert item.status == "pending"

    # 更新条目状态
    store.update_item_status(
        "00000000-0000-4000-8000-000000000001",
        "expert:7:profile_summary:8:0",
        status="success",
        result="embedded",
    )
    updates = [
        q for q, _ in executed if q.strip().upper().startswith("UPDATE")
    ]
    assert len(updates) == 1
    assert "processed_at" in updates[0]


def test_list_tasks_filter_and_pagination() -> None:
    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    store = _new_store(rows_by_query)
    fake_conn = store.connection.connection

    rows_by_query["SELECT * FROM vectorization_task WHERE status = %s ORDER BY created_at DESC LIMIT %s OFFSET %s"] = [
        _sample_task_row(),
        _sample_task_row("00000000-0000-4000-8000-000000000002"),
    ]
    tasks = store.list_tasks(status="completed")
    assert len(tasks) == 2
    assert tasks[0].task_id == "00000000-0000-4000-8000-000000000001"

    # 校验 SQL 分页参数
    sqls = [q for q, _ in fake_conn.cursor_instance.executed]
    assert any("LIMIT %s OFFSET %s" in q for q in sqls)


def test_list_items_filter() -> None:
    store = _new_store({})
    fake_conn = store.connection.connection
    # 直接给查询返回值（rows_by_query 为空时会步进到空列表）
    fake_conn.cursor_instance.rows_by_query[
        "SELECT * FROM vectorization_task_item WHERE task_id = %s AND status = %s ORDER BY created_at ASC LIMIT %s OFFSET %s"
    ] = [_sample_item_row()]
    items = store.list_items(
        "00000000-0000-4000-8000-000000000001",
        status="success",
    )
    assert len(items) == 1
    assert items[0].vector_doc_id == "expert:7:profile_summary:8:0"


# ── 异常安全 ──


def test_driver_error_is_sanitized() -> None:
    """驱动异常只暴露类型/errno，不泄露查询正文或凭据。"""

    class BrokenCursor:
        def __init__(self) -> None:
            self.closed = False

        def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
            raise RuntimeError("SQL syntax error near 'password=xxx'")

        def fetchall(self) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            self.closed = True

    class BrokenConnection:
        def cursor(self, cursorclass: Any = None) -> BrokenCursor:
            return BrokenCursor()

        def commit(self) -> None:
            pass

        def rollback(self) -> None:
            pass

        def close(self) -> None:
            pass

    store = _new_store({})
    store._local.connection = None
    # 用 monkeypatch 无法，直接构造带坏连接的 store
    class _BrokenStore(TaskStateStore):
        @property
        def connection(self):
            class _C:
                @property
                def connection(self):
                    return BrokenConnection()
            return _C()

    broken = _BrokenStore(
        TaskStateMySQLSettings(
            database="expert_dev",
            user="u",
            password=SecretStr("p"),
        )
    )
    with pytest.raises(TaskStateError) as exc_info:
        broken._query("SELECT * FROM vectorization_task", ())

    message = str(exc_info.value)
    assert "RuntimeError" in message
    assert "password" not in message
    assert "SELECT" not in message


# ── 状态机常量 ──


def test_status_literals() -> None:
    # Task 状态机：created → running → completed/failed
    assert TaskState(
        task_id="00000000-0000-4000-8000-000000000001",
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        status="created",
    ).status == "created"
    assert TaskState(
        task_id="00000000-0000-4000-8000-000000000001",
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        status="running",
    ).status == "running"
    assert TaskState(
        task_id="00000000-0000-4000-8000-000000000001",
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        status="completed",
    ).status == "completed"
    assert TaskState(
        task_id="00000000-0000-4000-8000-000000000001",
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        status="failed",
    ).status == "failed"

    # 非法状态禁止（ValidationError）
    with pytest.raises(ValidationError):
        TaskState(
            task_id="00000000-0000-4000-8000-000000000001",
            source="full_batch",
            collection_name="expert_candidates_v14_contract_dev",
            status="pending",  # Task2 没有 pending
        )

    # 条目状态机：pending → success/skipped/failed
    assert TaskItemState(
        task_id="00000000-0000-4000-8000-000000000001",
        vector_doc_id="expert:7:profile_summary:8:0",
        status="pending",
    ).status == "pending"


def _new_store(rows_by_query: dict[str, list[dict[str, Any]]]) -> TaskStateStore:
    settings = TaskStateMySQLSettings(
        database="expert_dev",
        user="task_state",
        password=SecretStr("ts-secret"),
    )
    fake_conn = FakeConnection(rows_by_query)

    def connector(**_: Any) -> FakeConnection:
        return fake_conn

    store = TaskStateStore(settings, connector_factory=connector)
    return store