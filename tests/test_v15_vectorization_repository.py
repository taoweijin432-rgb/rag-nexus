"""V15 增量向量化持久化层测试。

使用 Scripted Cursor/Connection，不连接或修改真实 MySQL。覆盖任务要求的
全部验收场景：批次与 Work Item 创建、状态机全路径、attempts 与 lease 恢复、
幂等重复提交、source_record_id 禁用、disable 无物理删除语义。
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import pytest
from pymysql.err import IntegrityError

from rag_vector_store.incremental import (
    MySQLV15VectorizationRepository,
    V15BatchNotFoundError,
    V15PersistenceError,
    V15TransitionError,
    V15WorkItemInput,
    V15WorkItemNotFoundError,
    vector_doc_id_hash,
)

_DDL_PATH = (
    Path(__file__).resolve().parents[1] / "sql" / "011_add_v15_vectorization_persistence.sql"
)


# ----------------------------------------------------------------------
# Scripted 测试基础设施
# ----------------------------------------------------------------------


class ScriptedCursor:
    """按脚本顺序应答的 Cursor，记录全部执行语句。"""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...] | None]] = []
        self._steps: deque[dict[str, Any]] = deque()
        self._current: dict[str, Any] = {}
        self.lastrowid = 0

    def expect(
        self,
        *,
        rowcount: int = 0,
        one: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
        lastrowid: int | None = None,
        raises: Exception | None = None,
    ) -> ScriptedCursor:
        self._steps.append(
            {
                "rowcount": rowcount,
                "one": one,
                "rows": rows or [],
                "lastrowid": lastrowid,
                "raises": raises,
            }
        )
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self.executed.append((" ".join(sql.split()), params))
        self._current = self._steps.popleft() if self._steps else {}
        if self._current.get("raises") is not None:
            raise self._current["raises"]
        if self._current.get("lastrowid") is not None:
            self.lastrowid = int(self._current["lastrowid"])

    @property
    def rowcount(self) -> int:
        return int(self._current.get("rowcount", 0))

    def fetchone(self) -> dict[str, Any] | None:
        return self._current.get("one")

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._current.get("rows", []))


class ScriptedConnection:
    """提供 Repository 需要的最小事务接口。"""

    def __init__(self, cursor: ScriptedCursor) -> None:
        self.cursor_instance = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> ScriptedCursor:
        return self.cursor_instance

    def begin(self) -> None:
        return None

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def _repository(
    cursor: ScriptedCursor,
) -> tuple[MySQLV15VectorizationRepository, ScriptedConnection]:
    connection = ScriptedConnection(cursor)
    return MySQLV15VectorizationRepository(lambda: connection), connection


def _item_row(**overrides: Any) -> dict[str, Any]:
    """构造一条完整的 Work Item 数据库行。"""

    row: dict[str, Any] = {
        "work_item_id": 7,
        "batch_id": 3,
        "operation": "upsert",
        "vector_doc_id": "expert:10000:profile_summary",
        "chunk_text": "专家简介文本",
        "metadata_json": '{"contract_version": "4.0.0"}',
        "content_hash": "a" * 64,
        "metadata_hash": "b" * 64,
        "chunk_policy_version": "v4",
        "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
        "embedding_model_version": "v1",
        "is_active": 1,
        "expert_id": 10000,
        "source_table": "experts",
        "source_field_id": "10000",
        "stale_vector_doc_ids": None,
        "status": "created",
        "attempts": 0,
        "error_type": None,
        "error_message": None,
        "lease_until": None,
        "processed_at": None,
        "created_at": datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


def _upsert_input(**overrides: Any) -> V15WorkItemInput:
    payload: dict[str, Any] = {
        "operation": "upsert",
        "vector_doc_id": "expert:10000:profile_summary",
        "chunk_text": "专家简介文本",
        "metadata": {"contract_version": "4.0.0"},
        "content_hash": "a" * 64,
        "chunk_policy_version": "v4",
        "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
        "embedding_model_version": "v1",
        "expert_id": 10000,
        "source_table": "experts",
        "source_field_id": "10000",
    }
    payload.update(overrides)
    return V15WorkItemInput(**payload)


def _executed_sql(cursor: ScriptedCursor) -> list[str]:
    return [sql for sql, _params in cursor.executed]


# ----------------------------------------------------------------------
# 批次创建与生命周期
# ----------------------------------------------------------------------


def test_create_batch_inserts_created_row() -> None:
    cursor = ScriptedCursor().expect(lastrowid=42)
    repo, connection = _repository(cursor)

    batch_id = repo.create_batch(total_count=5)

    assert batch_id == 42
    assert connection.closed is True
    sql, params = cursor.executed[0]
    assert sql.startswith("INSERT INTO v15_vectorization_batch")
    assert "status" in sql and "'created'" in sql
    assert params == (5, None)


def test_create_batch_records_upstream_batch_id() -> None:
    cursor = ScriptedCursor().expect(lastrowid=43)
    repo, _connection = _repository(cursor)

    batch_id = repo.create_batch(total_count=3, upstream_batch_id=8)

    assert batch_id == 43
    sql, params = cursor.executed[0]
    assert "upstream_batch_id" in sql
    assert params == (3, 8)


def test_get_batch_returns_upstream_batch_id() -> None:
    row = {
        "batch_id": 3,
        "upstream_batch_id": 8,
        "status": "running",
        "total_count": 3,
        "committed_count": 1,
        "failed_count": 0,
        "rejected_count": 0,
        "error_summary": None,
        "created_at": datetime(2026, 8, 21, tzinfo=timezone.utc),
        "started_at": datetime(2026, 8, 21, tzinfo=timezone.utc),
        "finished_at": None,
        "updated_at": datetime(2026, 8, 21, tzinfo=timezone.utc),
    }
    cursor = ScriptedCursor().expect(one=row)
    repo, _connection = _repository(cursor)

    batch = repo.get_batch(batch_id=3)

    assert batch.upstream_batch_id == 8
    sql, _params = cursor.executed[0]
    assert "upstream_batch_id" in sql


def test_start_batch_transitions_created_to_running() -> None:
    cursor = ScriptedCursor().expect(rowcount=1)
    repo, _connection = _repository(cursor)

    repo.start_batch(batch_id=3)

    sql, params = cursor.executed[0]
    assert "SET status='running'" in sql and "started_at" in sql
    assert "AND status='created'" in sql
    assert params == (3,)


def test_start_batch_from_committed_raises_and_rolls_back() -> None:
    cursor = ScriptedCursor().expect(rowcount=0).expect(one={"status": "committed"})
    repo, connection = _repository(cursor)

    with pytest.raises(V15TransitionError, match="running"):
        repo.start_batch(batch_id=3)

    assert connection.rollback_count == 1
    assert connection.commit_count == 0


def test_finish_batch_records_final_status() -> None:
    cursor = ScriptedCursor().expect(rowcount=1)
    repo, _connection = _repository(cursor)

    repo.finish_batch(batch_id=3, final_status="partial", error_summary="2 项失败")

    sql, params = cursor.executed[0]
    assert "finished_at" in sql and "AND status='running'" in sql
    assert params == ("partial", "2 项失败", 3)


def test_finish_batch_rejects_non_final_status() -> None:
    cursor = ScriptedCursor()
    repo, _connection = _repository(cursor)

    with pytest.raises(ValueError, match="终态"):
        repo.finish_batch(batch_id=3, final_status="running")
    assert cursor.executed == []


def test_refresh_batch_counters_recomputes_from_items() -> None:
    cursor = ScriptedCursor().expect(rowcount=1).expect(one={"batch_id": 3, "status": "running", "total_count": 3, "committed_count": 2, "failed_count": 1, "rejected_count": 0, "error_summary": None, "created_at": datetime(2026, 8, 21, tzinfo=timezone.utc), "started_at": datetime(2026, 8, 21, tzinfo=timezone.utc), "finished_at": None, "updated_at": datetime(2026, 8, 21, tzinfo=timezone.utc)})
    repo, _connection = _repository(cursor)

    batch = repo.refresh_batch_counters(batch_id=3)

    assert batch.committed_count == 2
    assert batch.failed_count == 1
    assert "SUM(status='committed')" in cursor.executed[0][0]


# ----------------------------------------------------------------------
# Work Item 创建与幂等
# ----------------------------------------------------------------------


def test_create_work_items_inserts_with_hash_and_json() -> None:
    cursor = (
        ScriptedCursor()
        .expect(one={"status": "created"})
        .expect(lastrowid=101)
    )
    repo, connection = _repository(cursor)

    results = repo.create_work_items(batch_id=3, items=[_upsert_input()])

    assert len(results) == 1
    assert results[0].work_item_id == 101
    assert results[0].created is True
    assert connection.commit_count == 1

    insert_sql, params = cursor.executed[1]
    assert insert_sql.startswith("INSERT INTO v15_vectorization_work_item")
    assert params is not None
    # vector_doc_id_hash 是确定性 SHA-256，metadata/stale 序列化为 JSON 文本。
    assert params[3] == vector_doc_id_hash("expert:10000:profile_summary")
    assert params[1] == "upsert"
    assert params[5] == '{"contract_version": "4.0.0"}'
    assert params[11] == 1  # is_active


def test_duplicate_submission_is_idempotent() -> None:
    cursor = (
        ScriptedCursor()
        .expect(one={"status": "created"})
        .expect(raises=IntegrityError(1062, "Duplicate entry '3-xxx'"))
        .expect(one={"work_item_id": 77})
    )
    repo, connection = _repository(cursor)

    results = repo.create_work_items(batch_id=3, items=[_upsert_input()])

    assert results[0].work_item_id == 77
    assert results[0].created is False
    assert connection.commit_count == 1
    select_sql, select_params = cursor.executed[2]
    assert "SELECT work_item_id FROM v15_vectorization_work_item" in select_sql
    assert select_params == (
        3,
        vector_doc_id_hash("expert:10000:profile_summary"),
        "upsert",
    )


def test_create_work_items_rejects_missing_batch() -> None:
    cursor = ScriptedCursor().expect(one=None)
    repo, connection = _repository(cursor)

    with pytest.raises(V15BatchNotFoundError):
        repo.create_work_items(batch_id=999, items=[_upsert_input()])

    assert connection.rollback_count == 1


def test_create_work_items_rejects_terminal_batch() -> None:
    cursor = ScriptedCursor().expect(one={"status": "committed"})
    repo, connection = _repository(cursor)

    with pytest.raises(V15PersistenceError, match="不允许追加"):
        repo.create_work_items(batch_id=3, items=[_upsert_input()])

    assert connection.rollback_count == 1


# ----------------------------------------------------------------------
# 状态机：认领与终态
# ----------------------------------------------------------------------


def test_claim_transitions_created_to_running_with_lease() -> None:
    now = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
    cursor = (
        ScriptedCursor()
        .expect(rows=[_item_row()])
        .expect(rowcount=1)
        .expect(rows=[_item_row(status="running", attempts=1)])
    )
    repo, connection = _repository(cursor)

    claimed = repo.claim_work_items(
        batch_id=3, limit=10, lease_seconds=300, now=now
    )

    assert len(claimed) == 1
    assert claimed[0].status == "running"
    assert claimed[0].attempts == 1
    assert connection.commit_count == 1

    claim_sql, claim_params = cursor.executed[0]
    assert "FOR UPDATE SKIP LOCKED" in claim_sql
    assert "lease_until IS NULL OR lease_until < %s" in claim_sql
    assert claim_params == (3, now, 10)

    update_sql, update_params = cursor.executed[1]
    assert "attempts=attempts+1" in update_sql
    assert update_params == (datetime(2026, 8, 21, 12, 5, tzinfo=timezone.utc), 7)


def test_claim_recovers_expired_lease_running_item() -> None:
    now = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
    expired = _item_row(status="running", attempts=1, lease_until=datetime(2026, 8, 21, 11, 0, 0, tzinfo=timezone.utc))
    cursor = (
        ScriptedCursor()
        .expect(rows=[expired])
        .expect(rowcount=1)
        .expect(rows=[_item_row(status="running", attempts=2)])
    )
    repo, _connection = _repository(cursor)

    claimed = repo.claim_work_items(
        batch_id=3, limit=10, lease_seconds=300, now=now
    )

    # 崩溃恢复路径：running + 过期租约可被重新认领，attempts 递增到 2。
    assert claimed[0].attempts == 2


def test_mark_committed_from_running_updates_batch_counter() -> None:
    cursor = (
        ScriptedCursor()
        .expect(one={"status": "running", "batch_id": 3})
        .expect(rowcount=1)
        .expect(rowcount=1)
    )
    repo, connection = _repository(cursor)

    repo.mark_work_item_committed(work_item_id=7)

    assert connection.commit_count == 1
    item_sql, item_params = cursor.executed[1]
    assert "SET status='committed'" in item_sql
    assert "processed_at=CURRENT_TIMESTAMP(6)" in item_sql
    assert item_params == (7,)
    batch_sql, batch_params = cursor.executed[2]
    assert "committed_count=committed_count+1" in batch_sql
    assert batch_params == (3,)


def test_mark_committed_from_created_raises() -> None:
    cursor = ScriptedCursor().expect(one={"status": "created", "batch_id": 3})
    repo, connection = _repository(cursor)

    with pytest.raises(V15TransitionError, match="committed"):
        repo.mark_work_item_committed(work_item_id=7)

    assert connection.rollback_count == 1
    # 只有锁定读取，没有发生任何 UPDATE。
    assert len(cursor.executed) == 1


def test_mark_failed_records_error_and_counter() -> None:
    cursor = (
        ScriptedCursor()
        .expect(one={"status": "running", "batch_id": 3})
        .expect(rowcount=1)
        .expect(rowcount=1)
    )
    repo, _connection = _repository(cursor)

    repo.mark_work_item_failed(
        work_item_id=7,
        error_type="EmbeddingTimeout",
        error_message="Embedding API 连接或请求超时",
    )

    item_sql, item_params = cursor.executed[1]
    assert "SET status='failed'" in item_sql
    assert item_params == ("EmbeddingTimeout", "Embedding API 连接或请求超时", 7)
    batch_sql, _params = cursor.executed[2]
    assert "failed_count=failed_count+1" in batch_sql


def test_mark_rejected_from_created_is_allowed() -> None:
    cursor = (
        ScriptedCursor()
        .expect(one={"status": "created", "batch_id": 3})
        .expect(rowcount=1)
        .expect(rowcount=1)
    )
    repo, _connection = _repository(cursor)

    repo.mark_work_item_rejected(
        work_item_id=7,
        error_type="PolicyVersionMismatch",
        error_message="chunk_policy_version 不符合正式值",
    )

    item_sql, _params = cursor.executed[1]
    assert "SET status='rejected'" in item_sql
    batch_sql, _batch_params = cursor.executed[2]
    assert "rejected_count=rejected_count+1" in batch_sql


def test_mark_rejected_from_committed_raises() -> None:
    cursor = ScriptedCursor().expect(one={"status": "committed", "batch_id": 3})
    repo, connection = _repository(cursor)

    with pytest.raises(V15TransitionError, match="rejected"):
        repo.mark_work_item_rejected(
            work_item_id=7, error_type="x", error_message="y"
        )

    assert connection.rollback_count == 1


def test_requeue_failed_items_resets_to_running_with_lease() -> None:
    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    cursor = (
        ScriptedCursor()
        .expect(rows=[_item_row(status="failed", attempts=1)])
        .expect(rowcount=1)
        .expect(rows=[_item_row(status="running", attempts=2)])
    )
    repo, connection = _repository(cursor)

    requeued = repo.requeue_failed_work_items(
        batch_id=3, max_attempts=3, lease_seconds=300, now=now
    )

    assert len(requeued) == 1
    assert requeued[0].status == "running"
    assert requeued[0].attempts == 2
    assert connection.commit_count == 1

    select_sql, select_params = cursor.executed[0]
    assert "status='failed' AND attempts < %s" in select_sql
    assert "FOR UPDATE SKIP LOCKED" in select_sql
    assert select_params == (3, 3)

    update_sql, update_params = cursor.executed[1]
    assert "SET status='running'" in update_sql
    assert "attempts=attempts+1" in update_sql
    assert update_params == (
        datetime(2026, 8, 21, 12, 5, tzinfo=timezone.utc),
        7,
    )


def test_requeue_returns_empty_when_attempts_exhausted() -> None:
    # attempts 已达上限：DB 过滤后无行，返回空且不产生任何 UPDATE。
    cursor = ScriptedCursor().expect(rows=[])
    repo, connection = _repository(cursor)

    requeued = repo.requeue_failed_work_items(
        batch_id=3, max_attempts=3, lease_seconds=300
    )

    assert requeued == ()
    assert connection.commit_count == 1
    assert len(cursor.executed) == 1  # 只有锁定 SELECT，没有 UPDATE


def test_requeue_rejects_non_positive_limits() -> None:
    cursor = ScriptedCursor()
    repo, _connection = _repository(cursor)

    with pytest.raises(ValueError, match="max_attempts"):
        repo.requeue_failed_work_items(batch_id=3, max_attempts=0, lease_seconds=300)
    with pytest.raises(ValueError, match="lease_seconds"):
        repo.requeue_failed_work_items(batch_id=3, max_attempts=3, lease_seconds=0)
    assert cursor.executed == []


def test_claim_does_not_pick_failed_items() -> None:
    # 防回归：failed 只能经显式 requeue 恢复，claim 不得自动捞取。
    cursor = ScriptedCursor().expect(rows=[])
    repo, _connection = _repository(cursor)

    repo.claim_work_items(batch_id=3, limit=10, lease_seconds=300)

    claim_sql, _params = cursor.executed[0]
    assert "status='failed'" not in claim_sql


# ----------------------------------------------------------------------
# 查询
# ----------------------------------------------------------------------


def test_get_work_item_parses_json_and_bool() -> None:
    cursor = ScriptedCursor().expect(
        one=_item_row(
            is_active=0,
            stale_vector_doc_ids='["expert:1:employment", "expert:1:profile_summary"]',
        )
    )
    repo, _connection = _repository(cursor)

    item = repo.get_work_item(work_item_id=7)

    assert item.is_active is False
    assert item.metadata == {"contract_version": "4.0.0"}
    assert item.stale_vector_doc_ids == (
        "expert:1:employment",
        "expert:1:profile_summary",
    )


def test_get_work_item_not_found() -> None:
    cursor = ScriptedCursor().expect(one=None)
    repo, _connection = _repository(cursor)

    with pytest.raises(V15WorkItemNotFoundError):
        repo.get_work_item(work_item_id=999)


def test_list_work_items_filters_by_status() -> None:
    cursor = ScriptedCursor().expect(rows=[_item_row()])
    repo, _connection = _repository(cursor)

    items = repo.list_work_items(batch_id=3, status="created")

    assert len(items) == 1
    sql, params = cursor.executed[0]
    assert "WHERE batch_id=%s AND status=%s" in sql
    assert params == (3, "created")


# ----------------------------------------------------------------------
# source_record_id 禁用与 disable 语义
# ----------------------------------------------------------------------


def test_source_record_id_is_rejected_by_input_model() -> None:
    with pytest.raises(ValueError, match="source_record_id"):
        V15WorkItemInput(
            operation="upsert",
            vector_doc_id="expert:1:profile_summary",
            chunk_text="文本",
            content_hash="a" * 64,
            source_record_id="123",  # type: ignore[call-arg]
        )


def test_source_record_id_never_appears_in_sql_or_ddl() -> None:
    cursor = (
        ScriptedCursor()
        .expect(one={"status": "created"})
        .expect(lastrowid=101)
    )
    repo, _connection = _repository(cursor)
    repo.create_work_items(batch_id=3, items=[_upsert_input()])

    for sql, _params in cursor.executed:
        assert "source_record_id" not in sql

    ddl = _DDL_PATH.read_text(encoding="utf-8")
    assert "source_record_id" not in ddl
    assert "source_field_id" in ddl


def test_disable_input_forces_inactive_and_forbids_payload() -> None:
    # disable 强制 is_active=False。
    item = V15WorkItemInput(
        operation="disable", vector_doc_id="expert:1:profile_summary", is_active=True
    )
    assert item.is_active is False
    # disable 不得携带 chunk_text / content_hash / metadata_hash。
    with pytest.raises(ValueError, match="disable"):
        V15WorkItemInput(
            operation="disable",
            vector_doc_id="expert:1:profile_summary",
            chunk_text="文本",
        )


def test_physical_delete_operation_is_rejected() -> None:
    with pytest.raises(ValueError):
        V15WorkItemInput(
            operation="delete",  # type: ignore[dict-item]
            vector_doc_id="expert:1:profile_summary",
        )


def test_disable_item_creates_no_physical_delete_sql() -> None:
    cursor = (
        ScriptedCursor()
        .expect(one={"status": "created"})
        .expect(lastrowid=101)
    )
    repo, _connection = _repository(cursor)

    results = repo.create_work_items(
        batch_id=3,
        items=[
            V15WorkItemInput(
                operation="disable",
                vector_doc_id="expert:1:profile_summary",
                expert_id=1,
                source_table="experts",
                source_field_id="1",
            )
        ],
    )

    assert results[0].created is True
    _sql, params = cursor.executed[1]
    assert params is not None and params[11] == 0  # is_active=False
    for sql, _p in cursor.executed:
        assert not sql.lstrip().upper().startswith(("DELETE", "DROP", "TRUNCATE"))

    # 持久层不提供任何 delete / drop 公开方法。
    forbidden = [
        name
        for name in dir(MySQLV15VectorizationRepository)
        if "delete" in name.lower() or "drop" in name.lower()
    ]
    assert forbidden == []


# ----------------------------------------------------------------------
# 输入校验（fail-closed）
# ----------------------------------------------------------------------


def test_upsert_requires_chunk_text_and_content_hash() -> None:
    with pytest.raises(ValueError, match="chunk_text"):
        V15WorkItemInput(
            operation="upsert",
            vector_doc_id="expert:1:profile_summary",
            content_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="content_hash"):
        V15WorkItemInput(
            operation="upsert",
            vector_doc_id="expert:1:profile_summary",
            chunk_text="文本",
        )


def test_metadata_update_requires_metadata_hash_and_forbids_text() -> None:
    with pytest.raises(ValueError, match="chunk_text"):
        V15WorkItemInput(
            operation="metadata_update",
            vector_doc_id="expert:1:profile_summary",
            chunk_text="文本",
            metadata_hash="b" * 64,
        )
    with pytest.raises(ValueError, match="metadata_hash"):
        V15WorkItemInput(
            operation="metadata_update",
            vector_doc_id="expert:1:profile_summary",
        )


def test_v15_byte_limits_are_enforced() -> None:
    with pytest.raises(ValueError, match="vector_doc_id"):
        V15WorkItemInput(
            operation="disable",
            vector_doc_id="a" * 2049,
        )
    with pytest.raises(ValueError, match="chunk_text"):
        V15WorkItemInput(
            operation="upsert",
            vector_doc_id="expert:1:profile_summary",
            chunk_text="a" * 8193,
            content_hash="a" * 64,
        )


def test_hash_format_and_source_table_are_validated() -> None:
    with pytest.raises(ValueError):
        _upsert_input(content_hash="not-a-hash")
    with pytest.raises(ValueError, match="source_table"):
        _upsert_input(source_table="not_a_source_table")
