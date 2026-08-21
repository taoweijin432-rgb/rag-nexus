"""V15 增量向量化的 MySQL 持久化层与状态机。

职责（Phase 0）：

- 管理 ``v15_vectorization_batch`` 与 ``v15_vectorization_work_item``
  两张状态表的创建、流转与查询；
- 用原子守卫 UPDATE / SELECT ... FOR UPDATE 保证状态机不被并发破坏；
- 用 ``lease_until`` 租约防止 Worker 崩溃后任务永久停留在 running；
- 用 ``uk_v15_work_item_idem`` 幂等约束避免同一 Work Item 重复创建。

明确不负责（后续任务）：

- 不实现 Worker、不调用 Embedding、不写 Milvus；
- 不扫描业务源表、不生成 Chunk；
- 不提供任何物理 delete 语义（disable 只把 is_active 置为 false）。

连接约定：构造函数接收零参 ``connector``，每个公开方法创建短连接并在
结束时关闭（与 ``expert_index.repository.MySQLIndexRepository`` 的约定一致）；
连接需使用 DictCursor（``DatabaseConfig.as_kwargs`` 默认满足）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

from pymysql.err import IntegrityError

from .models import (
    APPENDABLE_BATCH_STATUSES,
    BATCH_FINAL_STATUSES,
    V15BatchStatus,
    V15VectorizationBatch,
    V15VectorizationWorkItem,
    V15WorkItemCreateResult,
    V15WorkItemInput,
    V15WorkItemStatus,
)

_BATCH_TABLE = "v15_vectorization_batch"
_ITEM_TABLE = "v15_vectorization_work_item"

_ERROR_TYPE_MAX = 128
_ERROR_MESSAGE_MAX = 512

_ITEM_COLUMNS = (
    "work_item_id, batch_id, operation, vector_doc_id, chunk_text, metadata_json, "
    "content_hash, metadata_hash, chunk_policy_version, embedding_model_alias, "
    "embedding_model_version, is_active, expert_id, chunk_type, source_table, "
    "source_field_id, source_event_id, source_updated_at, stale_vector_doc_ids, "
    "status, attempts, error_type, error_message, "
    "lease_until, processed_at, created_at, updated_at"
)


class V15PersistenceError(RuntimeError):
    """V15 持久层的公共错误基类。"""


class V15BatchNotFoundError(V15PersistenceError):
    """指定批次不存在。"""


class V15WorkItemNotFoundError(V15PersistenceError):
    """指定 Work Item 不存在。"""


class V15TransitionError(V15PersistenceError):
    """状态机流转非法。

    ``target`` 是期望进入的状态，``current_status`` 是数据库中的实际状态。
    """

    def __init__(self, kind: str, target: str, current_status: str | None) -> None:
        self.kind = kind
        self.target = target
        self.current_status = current_status
        where = f"当前状态 {current_status}" if current_status else "记录不存在"
        super().__init__(
            f"{kind} 不能流转到 {target}（{where}），操作已回滚"
        )


def vector_doc_id_hash(vector_doc_id: str) -> str:
    """计算向量主键的 SHA-256 十六进制哈希，作为幂等键与查找键。"""

    return hashlib.sha256(vector_doc_id.encode("utf-8")).hexdigest()


def _truncate(value: str, limit: int) -> str:
    """截断安全错误摘要，避免超长报错写入状态表。"""

    return value if len(value) <= limit else value[: limit - 1] + "…"


def _json_param(value: Any) -> str | None:
    """把 dict / 序列列序列化为 MySQL JSON 参数；空值返回 None。"""

    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


@contextmanager
def _transaction(connection: Any):
    """显式短事务：异常回滚，正常提交。"""

    connection.begin()
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


class MySQLV15VectorizationRepository:
    """V15 增量向量化 Batch / Work Item 的 MySQL 状态持久化。"""

    def __init__(self, connector: Callable[[], Any]) -> None:
        # 只保存连接工厂，不立即访问数据库；连接由各公开方法按需创建并关闭。
        self._connector = connector

    # ------------------------------------------------------------------
    # Batch：创建与生命周期
    # ------------------------------------------------------------------

    def create_batch(
        self,
        total_count: int,
        upstream_batch_id: int | None = None,
    ) -> int:
        """创建 status=created 的批次，返回本地自增 batch_id。

        ``upstream_batch_id`` 只读记录上游触发批次（如 expert_index_batch
        的 batch_id），用于审计与回溯；不建立跨模块外键，也不替代本地 id。
        """

        if total_count < 0:
            raise ValueError("total_count 不能为负数")
        connection = self._connector()
        try:
            # 显式事务提交：连接工厂不保证 autocommit，裸 INSERT 会随
            # 短连接关闭而回滚。
            with _transaction(connection), connection.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO {_BATCH_TABLE} "
                    "(status, total_count, upstream_batch_id) VALUES ('created', %s, %s)",
                    (total_count, upstream_batch_id),
                )
                return int(cursor.lastrowid)
        finally:
            connection.close()

    def start_batch(self, batch_id: int) -> None:
        """created -> running，记录 started_at。"""

        connection = self._connector()
        try:
            with _transaction(connection), connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {_BATCH_TABLE} "
                    "SET status='running', started_at=CURRENT_TIMESTAMP(6), "
                    "updated_at=CURRENT_TIMESTAMP(6) "
                    "WHERE batch_id=%s AND status='created'",
                    (batch_id,),
                )
                if cursor.rowcount != 1:
                    current = self._batch_status_for_error(cursor, batch_id)
                    raise V15TransitionError("Batch", "running", current)
        finally:
            connection.close()

    def finish_batch(
        self,
        batch_id: int,
        final_status: V15BatchStatus,
        error_summary: str | None = None,
    ) -> None:
        """running -> committed / partial / failed，记录 finished_at。"""

        if final_status not in BATCH_FINAL_STATUSES:
            allowed = "、".join(BATCH_FINAL_STATUSES)
            raise ValueError(f"批次终态只允许: {allowed}")
        connection = self._connector()
        try:
            summary = (
                _truncate(error_summary, _ERROR_MESSAGE_MAX)
                if error_summary
                else None
            )
            with _transaction(connection), connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {_BATCH_TABLE} "
                    "SET status=%s, error_summary=%s, finished_at=CURRENT_TIMESTAMP(6), "
                    "updated_at=CURRENT_TIMESTAMP(6) "
                    "WHERE batch_id=%s AND status='running'",
                    (final_status, summary, batch_id),
                )
                if cursor.rowcount != 1:
                    current = self._batch_status_for_error(cursor, batch_id)
                    raise V15TransitionError("Batch", final_status, current)
        finally:
            connection.close()

    def get_batch(self, batch_id: int) -> V15VectorizationBatch:
        """读取批次完整行；不存在时抛出 V15BatchNotFoundError。"""

        connection = self._connector()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT batch_id, upstream_batch_id, status, total_count, "
                    "committed_count, failed_count, rejected_count, error_summary, "
                    "created_at, started_at, finished_at, updated_at "
                    f"FROM {_BATCH_TABLE} WHERE batch_id=%s",
                    (batch_id,),
                )
                row = cursor.fetchone()
        finally:
            connection.close()
        if row is None:
            raise V15BatchNotFoundError(f"批次 {batch_id} 不存在")
        return _batch_from_row(row)

    def refresh_batch_counters(self, batch_id: int) -> V15VectorizationBatch:
        """从 Work Item 明细重算批次计数（崩溃恢复 / 对账路径）。"""

        connection = self._connector()
        try:
            with _transaction(connection), connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {_BATCH_TABLE} b "
                    "LEFT JOIN ("
                    f"SELECT batch_id, SUM(status='committed') AS committed_count, "
                    f"SUM(status='failed') AS failed_count, "
                    f"SUM(status='rejected') AS rejected_count "
                    f"FROM {_ITEM_TABLE} WHERE batch_id=%s GROUP BY batch_id"
                    ") c ON c.batch_id = b.batch_id "
                    "SET b.committed_count=COALESCE(c.committed_count, 0), "
                    "b.failed_count=COALESCE(c.failed_count, 0), "
                    "b.rejected_count=COALESCE(c.rejected_count, 0), "
                    "b.updated_at=CURRENT_TIMESTAMP(6) "
                    "WHERE b.batch_id=%s",
                    (batch_id, batch_id),
                )
                if cursor.rowcount != 1:
                    current = self._batch_status_for_error(cursor, batch_id)
                    if current is None:
                        raise V15BatchNotFoundError(f"批次 {batch_id} 不存在")
                    raise V15PersistenceError(
                        f"批次 {batch_id} 计数重算失败（当前状态 {current}）"
                    )
                cursor.execute(
                    f"SELECT batch_id, upstream_batch_id, status, total_count, "
                    "committed_count, failed_count, rejected_count, error_summary, "
                    "created_at, started_at, finished_at, updated_at "
                    f"FROM {_BATCH_TABLE} WHERE batch_id=%s",
                    (batch_id,),
                )
                row = cursor.fetchone()
        finally:
            connection.close()
        if row is None:
            raise V15BatchNotFoundError(f"批次 {batch_id} 不存在")
        return _batch_from_row(row)

    # ------------------------------------------------------------------
    # Work Item：幂等创建
    # ------------------------------------------------------------------

    def create_work_items(
        self,
        batch_id: int,
        items: Sequence[V15WorkItemInput],
    ) -> tuple[V15WorkItemCreateResult, ...]:
        """在批次下幂等创建 Work Item。

        幂等键为 (batch_id, vector_doc_id_hash, operation)；重复提交不报错，
        返回已存在条目的 work_item_id 并标记 created=False。
        整批提交在一个事务内完成，任一条目发生非幂等错误则整体回滚。
        """

        connection = self._connector()
        results: list[V15WorkItemCreateResult] = []
        try:
            with _transaction(connection), connection.cursor() as cursor:
                self._ensure_batch_appendable(cursor, batch_id)
                for item in items:
                    results.append(
                        self._insert_work_item(cursor, batch_id, item)
                    )
        finally:
            connection.close()
        return tuple(results)

    def _ensure_batch_appendable(self, cursor: Any, batch_id: int) -> None:
        """批次必须存在且处于 created / running 状态才能追加 Work Item。"""

        cursor.execute(
            f"SELECT status FROM {_BATCH_TABLE} WHERE batch_id=%s",
            (batch_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise V15BatchNotFoundError(f"批次 {batch_id} 不存在")
        if row["status"] not in APPENDABLE_BATCH_STATUSES:
            raise V15PersistenceError(
                f"批次 {batch_id} 状态为 {row['status']}，不允许追加 Work Item"
            )

    def _insert_work_item(
        self,
        cursor: Any,
        batch_id: int,
        item: V15WorkItemInput,
    ) -> V15WorkItemCreateResult:
        """插入单条 Work Item；命中幂等约束时返回已存在条目。"""

        idem_hash = vector_doc_id_hash(item.vector_doc_id)
        try:
            cursor.execute(
                f"INSERT INTO {_ITEM_TABLE} ("
                "batch_id, operation, vector_doc_id, vector_doc_id_hash, chunk_text, "
                "metadata_json, content_hash, metadata_hash, chunk_policy_version, "
                "embedding_model_alias, embedding_model_version, is_active, expert_id, "
                "source_table, source_field_id, stale_vector_doc_ids, status, "
                "chunk_type, source_event_id, source_updated_at"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "'created', %s, %s, %s)",
                (
                    batch_id,
                    item.operation,
                    item.vector_doc_id,
                    idem_hash,
                    item.chunk_text,
                    _json_param(item.metadata),
                    item.content_hash,
                    item.metadata_hash,
                    item.chunk_policy_version,
                    item.embedding_model_alias,
                    item.embedding_model_version,
                    1 if item.is_active else 0,
                    item.expert_id,
                    item.source_table,
                    item.source_field_id,
                    _json_param(list(item.stale_vector_doc_ids))
                    if item.stale_vector_doc_ids
                    else None,
                    item.chunk_type,
                    item.source_event_id,
                    item.source_updated_at,
                ),
            )
        except IntegrityError as exc:
            # 幂等约束命中：同一批次内同一向量主键的同一操作已存在。
            if not exc.args or exc.args[0] != 1062:
                raise
            cursor.execute(
                f"SELECT work_item_id FROM {_ITEM_TABLE} "
                "WHERE batch_id=%s AND vector_doc_id_hash=%s AND operation=%s",
                (batch_id, idem_hash, item.operation),
            )
            row = cursor.fetchone()
            if row is None:
                raise V15PersistenceError(
                    f"幂等冲突后未找到已存在的 Work Item: {item.vector_doc_id}"
                ) from exc
            return V15WorkItemCreateResult(
                work_item_id=int(row["work_item_id"]), created=False
            )
        return V15WorkItemCreateResult(
            work_item_id=int(cursor.lastrowid), created=True
        )

    # ------------------------------------------------------------------
    # Work Item：认领（created/过期 running -> running）
    # ------------------------------------------------------------------

    def claim_work_items(
        self,
        batch_id: int,
        limit: int,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> tuple[V15VectorizationWorkItem, ...]:
        """认领可处理的 Work Item，并写入租约。

        认领范围：
        - status=created 的条目；
        - status=running 且租约缺失或已过期（lease_until IS NULL OR < now）
          的条目（Worker 崩溃恢复路径，attempts 会递增）。

        使用 FOR UPDATE SKIP LOCKED，多个 Worker 并发认领互不阻塞。
        """

        if limit <= 0:
            raise ValueError("limit 必须为正数")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds 必须为正数")
        current_time = now or datetime.now().astimezone()
        lease_until = current_time + timedelta(seconds=lease_seconds)

        connection = self._connector()
        claimed: list[V15VectorizationWorkItem] = []
        try:
            with _transaction(connection), connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_ITEM_COLUMNS} FROM {_ITEM_TABLE} "
                    "WHERE batch_id=%s AND ("
                    "status='created' OR (status='running' AND "
                    "(lease_until IS NULL OR lease_until < %s))) "
                    "ORDER BY work_item_id LIMIT %s "
                    "FOR UPDATE SKIP LOCKED",
                    (batch_id, current_time, limit),
                )
                rows = cursor.fetchall()
                for row in rows:
                    cursor.execute(
                        f"UPDATE {_ITEM_TABLE} "
                        "SET status='running', attempts=attempts+1, "
                        "lease_until=%s, updated_at=CURRENT_TIMESTAMP(6) "
                        "WHERE work_item_id=%s",
                        (lease_until, row["work_item_id"]),
                    )
                if rows:
                    cursor.execute(
                        f"SELECT {_ITEM_COLUMNS} FROM {_ITEM_TABLE} "
                        f"WHERE batch_id=%s AND status='running' "
                        "AND lease_until=%s ORDER BY work_item_id",
                        (batch_id, lease_until),
                    )
                    claimed = [
                        _work_item_from_row(r) for r in cursor.fetchall()
                    ]
        finally:
            connection.close()
        return tuple(claimed)

    # ------------------------------------------------------------------
    # Work Item：failed 显式重排（可重试错误的人工/受控恢复路径）
    # ------------------------------------------------------------------

    def requeue_failed_work_items(
        self,
        batch_id: int,
        max_attempts: int,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> tuple[V15VectorizationWorkItem, ...]:
        """显式把可重试的 failed Work Item 重新置为 running 并写入新租约。

        重试范围：status='failed' 且 attempts < max_attempts 的条目；
        attempts 已达上限的条目保持 failed，需人工介入。

        这是显式触发路径：claim_work_items 不会自动捞取 failed 条目，
        避免失败任务被无限自动重试。上一次失败的 error_type / error_message
        保留在行上供追溯，成功提交后由 mark_work_item_committed 清空。
        """

        if max_attempts <= 0:
            raise ValueError("max_attempts 必须为正数")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds 必须为正数")
        current_time = now or datetime.now().astimezone()
        lease_until = current_time + timedelta(seconds=lease_seconds)

        connection = self._connector()
        result_rows: list[dict[str, Any]] = []
        try:
            with _transaction(connection), connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_ITEM_COLUMNS} FROM {_ITEM_TABLE} "
                    "WHERE batch_id=%s AND status='failed' AND attempts < %s "
                    "ORDER BY work_item_id FOR UPDATE SKIP LOCKED",
                    (batch_id, max_attempts),
                )
                rows = cursor.fetchall()
                for row in rows:
                    cursor.execute(
                        f"UPDATE {_ITEM_TABLE} "
                        "SET status='running', attempts=attempts+1, "
                        "lease_until=%s, updated_at=CURRENT_TIMESTAMP(6) "
                        "WHERE work_item_id=%s",
                        (lease_until, row["work_item_id"]),
                    )
                if rows:
                    ids = [int(r["work_item_id"]) for r in rows]
                    placeholders = ", ".join(["%s"] * len(ids))
                    cursor.execute(
                        f"SELECT {_ITEM_COLUMNS} FROM {_ITEM_TABLE} "
                        f"WHERE work_item_id IN ({placeholders}) "
                        "ORDER BY work_item_id",
                        ids,
                    )
                    result_rows = list(cursor.fetchall())
        finally:
            connection.close()
        return tuple(_work_item_from_row(r) for r in result_rows)

    # ------------------------------------------------------------------
    # Work Item：终态流转
    # ------------------------------------------------------------------

    def mark_work_item_committed(self, work_item_id: int) -> None:
        """running -> committed（Milvus 写入成功，不等待 HNSW 索引完成）。"""

        self._finish_work_item(work_item_id, "committed")

    def mark_work_item_failed(
        self,
        work_item_id: int,
        error_type: str,
        error_message: str,
    ) -> None:
        """running -> failed（可重试的 Embedding / Milvus / 网络错误）。"""

        self._finish_work_item(
            work_item_id,
            "failed",
            error_type=error_type,
            error_message=error_message,
        )

    def mark_work_item_rejected(
        self,
        work_item_id: int,
        error_type: str,
        error_message: str,
    ) -> None:
        """created / running -> rejected（字段、模型空间或策略版本不符合）。"""

        connection = self._connector()
        try:
            with _transaction(connection), connection.cursor() as cursor:
                current, batch_id = self._locked_item_status(cursor, work_item_id)
                if current not in ("created", "running"):
                    raise V15TransitionError("WorkItem", "rejected", current)
                cursor.execute(
                    f"UPDATE {_ITEM_TABLE} "
                    "SET status='rejected', processed_at=CURRENT_TIMESTAMP(6), "
                    "lease_until=NULL, error_type=%s, error_message=%s, "
                    "updated_at=CURRENT_TIMESTAMP(6) WHERE work_item_id=%s",
                    (
                        _truncate(error_type, _ERROR_TYPE_MAX),
                        _truncate(error_message, _ERROR_MESSAGE_MAX),
                        work_item_id,
                    ),
                )
                cursor.execute(
                    f"UPDATE {_BATCH_TABLE} "
                    "SET rejected_count=rejected_count+1, "
                    "updated_at=CURRENT_TIMESTAMP(6) WHERE batch_id=%s",
                    (batch_id,),
                )
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Work Item：查询
    # ------------------------------------------------------------------

    def get_work_item(self, work_item_id: int) -> V15VectorizationWorkItem:
        """读取 Work Item 完整行；不存在时抛出 V15WorkItemNotFoundError。"""

        connection = self._connector()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_ITEM_COLUMNS} FROM {_ITEM_TABLE} "
                    "WHERE work_item_id=%s",
                    (work_item_id,),
                )
                row = cursor.fetchone()
        finally:
            connection.close()
        if row is None:
            raise V15WorkItemNotFoundError(f"Work Item {work_item_id} 不存在")
        return _work_item_from_row(row)

    def list_work_items(
        self,
        batch_id: int,
        status: V15WorkItemStatus | None = None,
    ) -> tuple[V15VectorizationWorkItem, ...]:
        """按批次（可选按状态）列出 Work Item，按 work_item_id 排序。"""

        connection = self._connector()
        try:
            with connection.cursor() as cursor:
                if status is None:
                    cursor.execute(
                        f"SELECT {_ITEM_COLUMNS} FROM {_ITEM_TABLE} "
                        "WHERE batch_id=%s ORDER BY work_item_id",
                        (batch_id,),
                    )
                else:
                    cursor.execute(
                        f"SELECT {_ITEM_COLUMNS} FROM {_ITEM_TABLE} "
                        "WHERE batch_id=%s AND status=%s ORDER BY work_item_id",
                        (batch_id, status),
                    )
                rows = cursor.fetchall()
        finally:
            connection.close()
        return tuple(_work_item_from_row(r) for r in rows)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _finish_work_item(
        self,
        work_item_id: int,
        target: V15WorkItemStatus,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """running -> committed / failed 的公共实现，并同步批次计数。"""

        connection = self._connector()
        try:
            with _transaction(connection), connection.cursor() as cursor:
                current, batch_id = self._locked_item_status(cursor, work_item_id)
                if current != "running":
                    raise V15TransitionError("WorkItem", target, current)
                if target == "committed":
                    cursor.execute(
                        f"UPDATE {_ITEM_TABLE} "
                        "SET status='committed', processed_at=CURRENT_TIMESTAMP(6), "
                        "lease_until=NULL, error_type=NULL, error_message=NULL, "
                        "updated_at=CURRENT_TIMESTAMP(6) WHERE work_item_id=%s",
                        (work_item_id,),
                    )
                    counter_column = "committed_count"
                else:
                    cursor.execute(
                        f"UPDATE {_ITEM_TABLE} "
                        "SET status='failed', processed_at=CURRENT_TIMESTAMP(6), "
                        "lease_until=NULL, error_type=%s, error_message=%s, "
                        "updated_at=CURRENT_TIMESTAMP(6) WHERE work_item_id=%s",
                        (
                            _truncate(error_type or "UnknownError", _ERROR_TYPE_MAX),
                            _truncate(
                                error_message or "未提供错误信息",
                                _ERROR_MESSAGE_MAX,
                            ),
                            work_item_id,
                        ),
                    )
                    counter_column = "failed_count"
                cursor.execute(
                    f"UPDATE {_BATCH_TABLE} "
                    f"SET {counter_column}={counter_column}+1, "
                    "updated_at=CURRENT_TIMESTAMP(6) WHERE batch_id=%s",
                    (batch_id,),
                )
        finally:
            connection.close()

    def _locked_item_status(
        self, cursor: Any, work_item_id: int
    ) -> tuple[str | None, int | None]:
        """锁定并读取 Work Item 当前状态，返回 (当前状态, 所属批次 id)。"""

        cursor.execute(
            f"SELECT status, batch_id FROM {_ITEM_TABLE} "
            "WHERE work_item_id=%s FOR UPDATE",
            (work_item_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None, None
        return str(row["status"]), int(row["batch_id"])

    def _batch_status_for_error(self, cursor: Any, batch_id: int) -> str | None:
        """读取批次当前状态，仅用于构造流转错误信息。"""

        cursor.execute(
            f"SELECT status FROM {_BATCH_TABLE} WHERE batch_id=%s",
            (batch_id,),
        )
        row = cursor.fetchone()
        return str(row["status"]) if row else None


def _work_item_from_row(row: dict[str, Any]) -> V15VectorizationWorkItem:
    """把 DictCursor 行转换为 Work Item 模型（解析 JSON 与布尔列）。"""

    metadata = row.get("metadata_json")
    if isinstance(metadata, (str, bytes)):
        metadata = json.loads(metadata)
    stale = row.get("stale_vector_doc_ids")
    if isinstance(stale, (str, bytes)):
        stale = tuple(json.loads(stale))
    elif stale is None:
        stale = ()
    return V15VectorizationWorkItem(
        work_item_id=int(row["work_item_id"]),
        batch_id=int(row["batch_id"]),
        operation=row["operation"],
        vector_doc_id=row["vector_doc_id"],
        chunk_text=row.get("chunk_text"),
        metadata=metadata,
        content_hash=row.get("content_hash"),
        metadata_hash=row.get("metadata_hash"),
        chunk_policy_version=row.get("chunk_policy_version"),
        embedding_model_alias=row.get("embedding_model_alias"),
        embedding_model_version=row.get("embedding_model_version"),
        is_active=bool(row.get("is_active")),
        expert_id=(
            int(row["expert_id"]) if row.get("expert_id") is not None else None
        ),
        chunk_type=row.get("chunk_type"),
        source_table=row.get("source_table"),
        source_field_id=row.get("source_field_id"),
        source_event_id=(
            int(row["source_event_id"])
            if row.get("source_event_id") is not None
            else None
        ),
        source_updated_at=row.get("source_updated_at"),
        stale_vector_doc_ids=stale,
        status=row["status"],
        attempts=int(row.get("attempts", 0)),
        error_type=row.get("error_type"),
        error_message=row.get("error_message"),
        lease_until=row.get("lease_until"),
        processed_at=row.get("processed_at"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _batch_from_row(row: dict[str, Any]) -> V15VectorizationBatch:
    """把 DictCursor 行转换为 Batch 模型。"""

    return V15VectorizationBatch(
        batch_id=int(row["batch_id"]),
        upstream_batch_id=(
            int(row["upstream_batch_id"])
            if row.get("upstream_batch_id") is not None
            else None
        ),
        status=row["status"],
        total_count=int(row.get("total_count", 0)),
        committed_count=int(row.get("committed_count", 0)),
        failed_count=int(row.get("failed_count", 0)),
        rejected_count=int(row.get("rejected_count", 0)),
        error_summary=row.get("error_summary"),
        created_at=row.get("created_at"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        updated_at=row.get("updated_at"),
    )
