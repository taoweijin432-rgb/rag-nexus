"""Repository 全量批次记录、互斥锁和生命周期事务测试。

测试使用 Recording Connection/Cursor，不连接或修改真实 MySQL。它只验证 SQL 结构、
参数顺序、事务收口和安全终态校验；Chunk 内容生成仍由领域测试负责。
"""

from __future__ import annotations

from typing import Any, Self

import pytest

from expert_index.config import DatabaseConfig
from expert_index.repository import ConcurrentBuildError, MySQLIndexRepository


class RecordingCursor:
    """记录执行语句，并返回测试指定的单行或多行结果。"""

    def __init__(
        self,
        *,
        one: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.one = one
        self.rows = rows or []
        self.lastrowid = 801
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self) -> dict[str, Any] | None:
        return self.one

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self.rows)


class RecordingConnection:
    """提供 Repository 公开方法需要的最小事务接口。"""

    def __init__(self, cursor: RecordingCursor) -> None:
        self.cursor_instance = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> RecordingCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def _repository() -> MySQLIndexRepository:
    """构造使用虚构连接参数的 Repository；测试会替换 connect。"""

    config = DatabaseConfig("127.0.0.1", 3306, "test", "test", "index_test")
    return MySQLIndexRepository(config)


def _patch_connection(
    monkeypatch: pytest.MonkeyPatch,
    connection: RecordingConnection,
) -> None:
    """让一次 Repository 操作使用指定 Recording Connection。"""

    monkeypatch.setattr(
        "expert_index.repository.pymysql.connect",
        lambda **_kwargs: connection,
    )


def test_full_batch_start_persists_version_cursor_and_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """full 批次必须从 cursor=0 开始，并固定版本和专家 ID 上界。"""

    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    batch_id = _repository().start_batch(
        "full_repository_test",
        "full",
        data_version="20260814_01",
        upper_bound_id=999,
    )

    sql, params = cursor.executed[0]
    assert batch_id == 801
    assert "data_version, status, cursor_id, upper_bound_id" in sql
    assert "VALUES (%s, 'full', %s, 'running', 0, %s)" in sql
    assert params == ("full_repository_test", "20260814_01", 999)
    assert connection.commit_count == 1
    assert connection.closed is True


def test_batch_item_lifecycle_and_page_progress_use_safe_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pending→running→success 及页末累计必须写入对应批次表。"""

    cursors: list[RecordingCursor] = []
    connections: list[RecordingConnection] = []

    def connect(**_kwargs: Any) -> RecordingConnection:
        cursor = RecordingCursor()
        connection = RecordingConnection(cursor)
        cursors.append(cursor)
        connections.append(connection)
        return connection

    monkeypatch.setattr("expert_index.repository.pymysql.connect", connect)
    repository = _repository()
    repository.prepare_batch_items(801, [10, 20])
    repository.mark_batch_item_running(801, 10)
    repository.finish_batch_item(
        801,
        10,
        status="success",
        extracted_count=4,
        stage_count=3,
        chunk_count=5,
    )
    repository.update_batch_progress(
        801,
        cursor_id=20,
        candidate_count=2,
        success_count=1,
        failed_count=1,
        rejected_count=0,
        deactivated_count=0,
        extracted_count=7,
        stage_count=3,
        chunk_count=5,
    )

    assert len(cursors[0].executed) == 2
    assert all("status) VALUES (%s, %s, 'pending')" in call[0] for call in cursors[0].executed)
    assert "attempt_count=attempt_count+1" in cursors[1].executed[0][0]
    assert "error_message=NULL" in cursors[2].executed[0][0]
    assert cursors[2].executed[0][1] == (
        "success", 4, 3, 5, None, None, 801, 10
    )
    assert "WHERE id=%s AND status='running'" in cursors[3].executed[0][0]
    assert all(connection.commit_count == 1 for connection in connections)


def test_full_batch_finish_persists_final_cursor_and_all_expert_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """full 最终收口不能退回只记录来源行数的旧 SQL。"""

    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    _repository().finish_batch(
        801,
        status="partial",
        cursor_id=999,
        candidate_count=10,
        success_count=6,
        failed_count=1,
        rejected_count=2,
        deactivated_count=1,
        extracted_count=24,
        stage_count=18,
        chunk_count=30,
        error_summary={"failed_expert_count": 1},
    )

    sql, params = cursor.executed[0]
    assert "cursor_id=%s, candidate_count=%s" in sql
    assert "deactivated_count=%s" in sql
    assert params == (
        "partial", 999, 10, 6, 1, 1, 24, 18, 30, 2,
        '{"failed_expert_count":1}', 801,
    )
    assert connection.commit_count == 1


def test_deleted_expert_stage_cleanup_is_one_idempotent_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """逻辑删除分支只清理并标记 Stage，不再读写 expert_chunk。"""

    cursor = RecordingCursor(rows=[])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    _repository().mark_deleted_stage(24, build_batch_id=801)

    sql_calls = [sql for sql, _params in cursor.executed]
    assert "DELETE FROM expert_index_employment_stage" in sql_calls[0]
    assert "DELETE FROM expert_index_project_stage" in sql_calls[1]
    assert "reject_reason='expert_deleted'" in sql_calls[2]
    assert len(sql_calls) == 3
    assert all("expert_chunk" not in sql for sql in sql_calls)
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


def test_batch_item_rejects_message_like_result_code_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批次明细机器码不能误收异常消息或专家正文。"""

    def unexpected_connect(**_kwargs: Any) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr("expert_index.repository.pymysql.connect", unexpected_connect)
    with pytest.raises(ValueError, match="结果标识"):
        _repository().finish_batch_item(
            801,
            10,
            status="failed",
            result_code="包含 空格 的消息",
        )


def test_build_lock_releases_named_lock_after_pipeline_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上下文内部异常也必须执行 RELEASE_LOCK 并关闭持锁连接。"""

    cursor = RecordingCursor(one={"acquired": 1})
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(
        RuntimeError,
        match="虚构流水线失败",
    ), _repository().build_lock("full"):
        raise RuntimeError("虚构流水线失败")

    assert "GET_LOCK" in cursor.executed[0][0]
    assert "RELEASE_LOCK" in cursor.executed[1][0]
    assert connection.closed is True


def test_build_lock_fails_fast_when_another_task_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未取得命名锁时不得进入 Pipeline 任务体。"""

    cursor = RecordingCursor(one={"acquired": 0})
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    entered = False
    with pytest.raises(ConcurrentBuildError), _repository().build_lock("full"):
        entered = True

    assert entered is False
    assert all("RELEASE_LOCK" not in sql for sql, _params in cursor.executed)
    assert connection.closed is True
