"""Task State Store 与 SQL 查询边界。

职责：
    - 复用独立连接配置（``TaskStateMySQLSettings``），不读取业务库；
    - 封装 ``vectorization_task`` / ``vectorization_task_item`` 两张表的
      创建、查询、状态更新、批量插入；
    - 所有写操作在同一连接/事务内执行，避免部分成功；
    - 异常以安全的 ``TaskStateError`` 暴露（不含凭据、不含查询正文）。

不职责：
    - 不执行 Embedding / Milvus / Contract 校验；
    - 不管理连接生命周期（由共享 ``MySQLConnection`` 负责）；
    - 不实现 checkpoint / retry / 增量 Reader（属于 Task3）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from threading import local as thread_local
from typing import Any

from pydantic import ValidationError

from ..config import MySQLSettings
from ..mysql.connection import MySQLConnection
from .config import TaskStateMySQLSettings
from .models import (
    TaskItemState,
    TaskItemSummary,
    TaskState,
    TaskSummary,
    new_task_id,
)


class TaskStateError(RuntimeError):
    """Task State 操作失败（查询/写入/校验错误，安全摘要）。"""


class TaskStateStore:
    """Task State 的持久化访问层。

    约定：
        - 所有写操作使用 ``INSERT ... ON DUPLICATE KEY UPDATE`` 或带
          ``WHERE status=...`` 的条件更新，保证幂等；
        - 时间字段使用 ``DATETIME(6)`` 微秒精度，应用层按 Asia/Shanghai 解释；
        - 不将凭据、chunk_text 或向量写入错误消息。
    """

    def __init__(
        self,
        settings: TaskStateMySQLSettings | MySQLSettings,
        *,
        connector_factory: Callable[..., Any] | None = None,
        database_timezone: Any = None,
    ) -> None:
        self.settings = settings
        self.connector_factory = connector_factory
        self.database_timezone = database_timezone
        self._local = thread_local()

    @property
    def connection(self) -> MySQLConnection:
        """延迟创建线程局部连接（每个线程拥有独立连接）。"""

        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = MySQLConnection(
                self.settings,
                connector_factory=self.connector_factory,
            )
            self._local.connection = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    # ── 任务级 ──

    def create_task(
        self,
        *,
        source: str,
        mode: str = "full",
        collection_name: str,
        total_count: int,
        request_id: str | None = None,
    ) -> TaskState:
        """创建任务（状态 created）。

        使用 ``INSERT ... ON DUPLICATE KEY UPDATE`` 保证 request_id 幂等：
        重复提交同一 request_id 时返回已存在任务，不重复创建。
        """

        # 幂等：同一 request_id 已存在任务时直接返回，不重复创建。
        # （先查后插 + 唯一键兜底；并发重复提交时由 uk_vtask_request_id 拦截。）
        if request_id is not None:
            existing = self.find_task_by_request_id(request_id)
            if existing is not None:
                return existing

        task_id = new_task_id()
        now = _now()
        row: dict[str, Any] = {
            "task_id": task_id,
            "source": source,
            "mode": mode,
            "status": "created",
            "total_count": total_count,
            "success_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "request_id": request_id,
            "collection_name": collection_name,
            "created_at": now,
        }
        sql = """
            INSERT INTO vectorization_task (
                task_id, source, mode, status, total_count,
                success_count, skipped_count, failed_count,
                request_id, collection_name, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._execute(sql, tuple(row.values()))
        return TaskState(
            task_id=task_id,
            source=source,
            mode=mode,
            status="created",
            total_count=total_count,
            request_id=request_id,
            collection_name=collection_name,
            created_at=now,
        )

    def get_task(self, task_id: str) -> TaskState:
        """按 task_id 查询任务，不存在时返回 None 语义由调用方决定（抛错）。"""

        rows = self._query(
            "SELECT * FROM vectorization_task WHERE task_id = %s",
            (task_id.strip().lower(),),
        )
        if not rows:
            raise TaskStateError(f"task_id={task_id!r} 不存在")
        return _task_from_row(rows[0])

    def find_task_by_request_id(self, request_id: str) -> TaskState | None:
        """按 request_id 查询任务（幂等键），不存在返回 None。"""

        rows = self._query(
            "SELECT * FROM vectorization_task WHERE request_id = %s",
            (request_id,),
        )
        return _task_from_row(rows[0]) if rows else None

    def list_tasks(
        self,
        *,
        status: str | None = None,
        source: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TaskSummary]:
        """分页查询任务列表，按 created_at 倒序。"""

        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if source is not None:
            clauses.append("source = %s")
            params.append(source)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self._query(
            f"SELECT * FROM vectorization_task {where} "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            tuple(params),
        )
        return [_task_summary_from_row(row) for row in rows]

    def update_task_status(
        self,
        task_id: str,
        *,
        status: str,
        success_count: int | None = None,
        skipped_count: int | None = None,
        failed_count: int | None = None,
        error_message: str | None = None,
        started_at: bool = False,
        finished_at: bool = False,
    ) -> None:
        """更新任务状态和计数。

        ``started_at`` / ``finished_at`` 为 True 时填充当前时间。
        """

        now = _now()
        sets: list[str] = ["status = %s"]
        params: list[Any] = [status]
        if success_count is not None:
            sets.append("success_count = %s")
            params.append(success_count)
        if skipped_count is not None:
            sets.append("skipped_count = %s")
            params.append(skipped_count)
        if failed_count is not None:
            sets.append("failed_count = %s")
            params.append(failed_count)
        if error_message is not None:
            sets.append("error_message = %s")
            params.append(error_message[:512])
        if started_at:
            sets.append("started_at = %s")
            params.append(now)
        if finished_at:
            sets.append("finished_at = %s")
            params.append(now)
        params.append(task_id.strip().lower())
        sql = f"UPDATE vectorization_task SET {', '.join(sets)} WHERE task_id = %s"
        self._execute(sql, tuple(params))

    # ── 条目级 ──

    def create_items(
        self,
        items: Sequence[TaskItemState],
    ) -> None:
        """批量插入条目（pending 状态）。"""

        if not items:
            return
        sql = """
            INSERT INTO vectorization_task_item (
                task_id, vector_doc_id, status, result,
                error_message, processed_at, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        now = _now()
        params = [
            (
                item.task_id.strip().lower(),
                item.vector_doc_id,
                item.status,
                item.result,
                item.error_message,
                item.processed_at,
                item.created_at or now,
            )
            for item in items
        ]
        self._executemany(sql, params)

    def get_item(
        self,
        task_id: str,
        vector_doc_id: str,
    ) -> TaskItemState | None:
        rows = self._query(
            "SELECT * FROM vectorization_task_item "
            "WHERE task_id = %s AND vector_doc_id = %s",
            (task_id.strip().lower(), vector_doc_id),
        )
        return _item_from_row(rows[0]) if rows else None

    def list_items(
        self,
        task_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TaskItemSummary]:
        clauses = ["task_id = %s"]
        params: list[Any] = [task_id.strip().lower()]
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        params.extend([limit, offset])
        rows = self._query(
            f"SELECT * FROM vectorization_task_item "
            f"WHERE {' AND '.join(clauses)} ORDER BY created_at ASC "
            "LIMIT %s OFFSET %s",
            tuple(params),
        )
        return [_item_summary_from_row(row) for row in rows]

    def update_item_status(
        self,
        task_id: str,
        vector_doc_id: str,
        *,
        status: str,
        result: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """更新单条条目状态，终态时自动填充 processed_at。"""

        now = _now()
        sets = ["status = %s"]
        params: list[Any] = [status]
        if result is not None:
            sets.append("result = %s")
            params.append(result)
        if error_message is not None:
            sets.append("error_message = %s")
            params.append(error_message[:512])
        sets.append("processed_at = %s")
        params.append(now)
        params.extend([task_id.strip().lower(), vector_doc_id])
        sql = f"UPDATE vectorization_task_item SET {', '.join(sets)} " \
              "WHERE task_id = %s AND vector_doc_id = %s"
        self._execute(sql, tuple(params))

    # ── 低层执行 ──

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        conn = self.connection.connection
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params)
            conn.commit()
        except Exception as exc:
            _safe_rollback(conn)
            raise _safe_error(exc) from exc
        finally:
            cursor.close()

    def _executemany(
        self,
        sql: str,
        params: Sequence[tuple[Any, ...]],
    ) -> None:
        conn = self.connection.connection
        cursor = conn.cursor()
        try:
            cursor.executemany(sql, params)
            conn.commit()
        except Exception as exc:
            _safe_rollback(conn)
            raise _safe_error(exc) from exc
        finally:
            cursor.close()

    def _query(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        try:
            from pymysql.cursors import DictCursor
        except ImportError:
            DictCursor = None  # type: ignore[assignment, misc]

        conn = self.connection.connection
        cursor = (
            conn.cursor(DictCursor)  # type: ignore[call-arg]
            if DictCursor is not None
            else conn.cursor()
        )
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except Exception as exc:
            raise _safe_error(exc) from exc
        finally:
            cursor.close()
        return [dict(row) for row in rows]


# ── 行转换 ──


def _task_from_row(row: dict[str, Any]) -> TaskState:
    try:
        return TaskState.model_validate(row)
    except ValidationError as exc:
        raise TaskStateError(
            "vectorization_task 行数据不符合模型: "
            + ", ".join(str(e) for e in exc.errors())
        ) from exc


def _task_summary_from_row(row: dict[str, Any]) -> TaskSummary:
    """从完整行投影出汇总模型字段（行可能包含 request_id/mode 等额外列）。"""

    projected = {
        key: row[key]
        for key in (
            "task_id",
            "status",
            "source",
            "total_count",
            "success_count",
            "skipped_count",
            "failed_count",
            "created_at",
            "started_at",
            "finished_at",
            "error_message",
        )
        if key in row
    }
    try:
        return TaskSummary.model_validate(projected)
    except ValidationError as exc:
        raise TaskStateError(
            "vectorization_task 行数据不符合汇总模型: "
            + ", ".join(str(e) for e in exc.errors())
        ) from exc


def _item_from_row(row: dict[str, Any]) -> TaskItemState:
    try:
        return TaskItemState.model_validate(row)
    except ValidationError as exc:
        raise TaskStateError(
            "vectorization_task_item 行数据不符合模型: "
            + ", ".join(str(e) for e in exc.errors())
        ) from exc


def _item_summary_from_row(row: dict[str, Any]) -> TaskItemSummary:
    """从完整行投影出条目汇总模型字段（行可能包含 created_at 等额外列）。"""

    projected = {
        key: row[key]
        for key in (
            "task_id",
            "vector_doc_id",
            "status",
            "result",
            "error_message",
            "processed_at",
        )
        if key in row
    }
    try:
        return TaskItemSummary.model_validate(projected)
    except ValidationError as exc:
        raise TaskStateError(
            "vectorization_task_item 行数据不符合汇总模型: "
            + ", ".join(str(e) for e in exc.errors())
        ) from exc


def _now() -> datetime:
    """当前时间（UTC，存库前由驱动按 Asia/Shanghai 解释，DATETIME(6)）。"""

    return datetime.now(timezone.utc)


def _safe_error(exc: Exception) -> TaskStateError:
    """把驱动异常转换为安全摘要，不回显查询正文或凭据。"""

    details = [type(exc).__name__]
    errno = getattr(exc, "errno", None)
    sqlstate = getattr(exc, "sqlstate", None)
    if errno is not None:
        details.append(f"errno={errno}")
    if sqlstate is not None:
        details.append(f"sqlstate={sqlstate}")
    return TaskStateError("Task State 操作失败: " + " ".join(details))


def _safe_rollback(conn: Any) -> None:
    """失败时尝试回滚当前事务；回滚本身失败也忽略，保证连接可继续复用。"""

    try:
        rollback = getattr(conn, "rollback", None)
        if callable(rollback):
            rollback()
    except Exception:  # noqa: S110, BLE001 - 回滚失败不影响主异常传播
        pass