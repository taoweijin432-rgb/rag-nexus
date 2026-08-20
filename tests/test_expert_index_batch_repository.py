"""Repository Checkpoint、全量批次记录、互斥锁和生命周期事务测试。

测试使用 Recording Connection/Cursor，不连接或修改真实 MySQL。它只验证 SQL 结构、
参数顺序、事务收口和安全终态校验；Chunk 内容生成仍由领域测试负责。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.config import DatabaseConfig
from expert_index.models import BatchSourceScan, SourceCheckpoint, SourceScanCursor
from expert_index.repository import (
    BatchNotResumableError,
    BatchNotRetryableError,
    ConcurrentBuildError,
    MySQLIndexRepository,
)


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


def test_get_source_checkpoint_returns_committed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repository 应把已提交行完整转换为只读 Checkpoint 模型。"""

    checkpoint_time = datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc)
    success_time = datetime(2026, 8, 20, 9, 35, tzinfo=timezone.utc)
    cursor = RecordingCursor(
        one={
            "source_table": "experts_work_experience",
            "last_updated_at": checkpoint_time,
            "last_id": 2801,
            "last_batch_id": 901,
            "last_success_time": success_time,
            "lock_version": 3,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    checkpoint = _repository().get_source_checkpoint("experts_work_experience")

    assert checkpoint == SourceCheckpoint(
        source_table="experts_work_experience",
        last_updated_at=checkpoint_time,
        last_id=2801,
        last_batch_id=901,
        last_success_time=success_time,
        lock_version=3,
    )
    assert "FROM expert_index_source_checkpoint" in cursor.executed[0][0]
    assert cursor.executed[0][1] == ("experts_work_experience",)
    assert connection.commit_count == 0
    assert connection.closed is True


def test_get_source_checkpoint_returns_none_when_not_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """表中没有对应行时返回 None，不在读取方法中擅自创建起始位置。"""

    cursor = RecordingCursor(one=None)
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    assert _repository().get_source_checkpoint("experts") is None
    assert connection.commit_count == 0
    assert connection.closed is True


def test_get_source_checkpoint_rejects_unknown_table_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法表名必须在连接数据库前失败，避免动态表名扩大读取范围。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("非法来源表不应连接数据库")

    monkeypatch.setattr("expert_index.repository.pymysql.connect", unexpected_connect)

    with pytest.raises(ValueError, match="不支持的 Checkpoint 来源表"):
        _repository().get_source_checkpoint("expert_work_experience")


def test_initialize_source_checkpoint_inserts_explicit_safe_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始化应原样写入调用方确认的位置，并以版本 0 开始。"""

    checkpoint_time = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)
    success_time = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    _repository().initialize_source_checkpoint(
        SourceCheckpoint(
            source_table="experts_project_experience",
            last_updated_at=checkpoint_time,
            last_id=1800,
            last_batch_id=902,
            last_success_time=success_time,
        )
    )

    sql, params = cursor.executed[0]
    assert "INSERT INTO expert_index_source_checkpoint" in sql
    assert "ON DUPLICATE KEY UPDATE" not in sql
    assert params == (
        "experts_project_experience",
        checkpoint_time,
        1800,
        902,
        success_time,
    )
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True


def test_initialize_source_checkpoint_rejects_nonzero_lock_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始化入口不能伪装成已有 Checkpoint 的更新操作。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("非法初始版本不应连接数据库")

    monkeypatch.setattr("expert_index.repository.pymysql.connect", unexpected_connect)
    checkpoint = SourceCheckpoint(
        source_table="experts",
        last_updated_at=datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc),
        last_id=0,
        lock_version=1,
    )

    with pytest.raises(ValueError, match="lock_version 必须为 0"):
        _repository().initialize_source_checkpoint(checkpoint)


def test_create_batch_source_scan_inserts_fixed_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """创建方法应完整保存来源表扫描范围，且不能覆盖已有记录。"""

    start_time = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)
    upper_bound_time = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    _repository().create_batch_source_scan(
        BatchSourceScan(
            batch_id=903,
            source_table="experts_education_experience",
            start_cursor=SourceScanCursor(start_time, 100),
            current_cursor=SourceScanCursor(start_time, 100),
            upper_bound=SourceScanCursor(upper_bound_time, 300),
            checkpoint_lock_version=4,
        )
    )

    sql, params = cursor.executed[0]
    assert "INSERT INTO expert_index_batch_source" in sql
    assert "ON DUPLICATE KEY UPDATE" not in sql
    assert params == (
        903,
        "experts_education_experience",
        start_time,
        100,
        start_time,
        100,
        upper_bound_time,
        300,
        4,
        "pending",
        None,
        None,
    )
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True


def test_create_batch_source_scan_rejects_already_started_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """创建入口不能被用来写入已经推进的临时游标。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("已推进的扫描状态不应连接数据库")

    monkeypatch.setattr("expert_index.repository.pymysql.connect", unexpected_connect)
    start_time = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="current_cursor 必须等于 start_cursor"):
        _repository().create_batch_source_scan(
            BatchSourceScan(
                batch_id=903,
                source_table="experts",
                start_cursor=SourceScanCursor(start_time, 0),
                current_cursor=SourceScanCursor(start_time, 1),
                upper_bound=SourceScanCursor(start_time, 10),
                checkpoint_lock_version=0,
                status="running",
            )
        )


def test_get_batch_source_scan_returns_persisted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查询方法应把数据库行还原为带三个复合游标的领域模型。"""

    start_time = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)
    cursor_time = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
    upper_bound_time = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
    started_at = datetime(2026, 8, 20, 10, 1, tzinfo=timezone.utc)
    cursor = RecordingCursor(
        one={
            "batch_id": 903,
            "source_table": "experts_education_experience",
            "start_updated_at": start_time,
            "start_id": 100,
            "cursor_updated_at": cursor_time,
            "cursor_id": 200,
            "upper_bound_updated_at": upper_bound_time,
            "upper_bound_id": 300,
            "checkpoint_lock_version": 4,
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    scan = _repository().get_batch_source_scan(
        903,
        "experts_education_experience",
    )

    assert scan == BatchSourceScan(
        batch_id=903,
        source_table="experts_education_experience",
        start_cursor=SourceScanCursor(start_time, 100),
        current_cursor=SourceScanCursor(cursor_time, 200),
        upper_bound=SourceScanCursor(upper_bound_time, 300),
        checkpoint_lock_version=4,
        status="running",
        started_at=started_at,
    )
    sql, params = cursor.executed[0]
    assert "FROM expert_index_batch_source" in sql
    assert params == (903, "experts_education_experience")
    assert connection.commit_count == 0
    assert connection.closed is True


def test_get_batch_source_scan_returns_none_when_not_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批次尚未创建来源表扫描范围时应返回 None，且不产生写操作。"""

    cursor = RecordingCursor(one=None)
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    assert _repository().get_batch_source_scan(903, "experts") is None
    assert connection.commit_count == 0
    assert connection.closed is True


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


def test_resume_batch_reuses_checkpoint_and_reactivates_original_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """failed full 批次应在同一事务中恢复原版本、范围、游标和累计计数。"""

    cursor = RecordingCursor(
        one={
            "id": 801,
            "batch_no": "interrupted_full_batch",
            "run_mode": "full",
            "data_version": "20260818_04",
            "status": "failed",
            "cursor_id": 20,
            "upper_bound_id": 40,
            "candidate_count": 2,
            "success_count": 1,
            "failed_count": 0,
            "rejected_count": 1,
            "deactivated_count": 0,
            "extracted_count": 2,
            "stage_count": 2,
            "chunk_count": 1,
            "vector_sync_expert_count": 2,
            "vector_current_count": 3,
            "vector_eligible_count": 2,
            "vector_embedded_count": 1,
            "vector_metadata_updated_count": 1,
            "vector_unchanged_count": 0,
            "vector_deleted_count": 1,
            "vector_delete_not_found_count": 0,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    checkpoint = _repository().resume_batch(801)

    assert checkpoint.batch_id == 801
    assert checkpoint.data_version == "20260818_04"
    assert checkpoint.cursor_id == 20
    assert checkpoint.upper_bound_id == 40
    assert checkpoint.candidate_count == 2
    assert checkpoint.success_count == 1
    assert checkpoint.rejected_count == 1
    assert checkpoint.vector_sync_expert_count == 2
    assert checkpoint.vector_current_count == 3
    assert checkpoint.vector_metadata_updated_count == 1
    assert checkpoint.vector_deleted_count == 1
    assert "FOR UPDATE" in cursor.executed[0][0]
    assert cursor.executed[0][1] == (801,)
    assert "SET status='running', error_summary=NULL, finished_at=NULL" in cursor.executed[1][0]
    assert cursor.executed[1][1] == (801,)
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


def test_resume_batch_rejects_completed_batch_without_reactivating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """success/partial 已完成扫描，不属于断点续跑，不能改回 running。"""

    cursor = RecordingCursor(
        one={
            "id": 801,
            "batch_no": "completed_full_batch",
            "run_mode": "full",
            "data_version": "20260818_04",
            "status": "partial",
            "cursor_id": 40,
            "upper_bound_id": 40,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(BatchNotResumableError, match="不能断点续跑"):
        _repository().resume_batch(801)

    assert len(cursor.executed) == 1
    assert connection.commit_count == 0
    assert connection.rollback_count == 1


def test_partial_batch_retry_context_requires_completed_full_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失败重试必须复用已经扫描到固定上界的 partial full 批次。"""

    cursor = RecordingCursor(
        one={
            "id": 801,
            "batch_no": "partial_retry_batch",
            "run_mode": "full",
            "data_version": "20260818_06",
            "status": "partial",
            "cursor_id": 40,
            "upper_bound_id": 40,
            "candidate_count": 4,
            "success_count": 3,
            "failed_count": 1,
            "rejected_count": 0,
            "deactivated_count": 0,
            "extracted_count": 3,
            "stage_count": 3,
            "chunk_count": 3,
            "vector_sync_expert_count": 3,
            "vector_current_count": 3,
            "vector_eligible_count": 3,
            "vector_embedded_count": 2,
            "vector_metadata_updated_count": 1,
            "vector_unchanged_count": 0,
            "vector_deleted_count": 0,
            "vector_delete_not_found_count": 0,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    checkpoint = _repository().load_partial_batch_for_retry(801)

    assert checkpoint.batch_no == "partial_retry_batch"
    assert checkpoint.data_version == "20260818_06"
    assert checkpoint.cursor_id == checkpoint.upper_bound_id == 40
    assert checkpoint.vector_sync_expert_count == 3
    assert checkpoint.vector_metadata_updated_count == 1
    assert len(cursor.executed) == 1
    assert "FROM expert_index_batch WHERE id=%s" in cursor.executed[0][0]
    assert all("UPDATE" not in sql for sql, _params in cursor.executed)


def test_partial_batch_retry_rejects_non_partial_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """success 或未完成批次不能进入失败专家重试。"""

    cursor = RecordingCursor(
        one={
            "id": 801,
            "batch_no": "success_batch",
            "run_mode": "full",
            "data_version": "20260818_06",
            "status": "success",
            "cursor_id": 40,
            "upper_bound_id": 40,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(BatchNotRetryableError, match="partial full"):
        _repository().load_partial_batch_for_retry(801)


def test_retryable_items_include_failed_and_interrupted_running_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重试分页应只返回游标后的 failed/running 专家 ID。"""

    cursor = RecordingCursor(rows=[{"expert_id": 20}, {"expert_id": 30}])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    expert_ids = _repository().load_retryable_batch_item_ids(
        801,
        after_expert_id=10,
        limit=50,
    )

    assert expert_ids == [20, 30]
    sql, params = cursor.executed[0]
    assert "status IN ('failed', 'running')" in sql
    assert "ORDER BY expert_id ASC LIMIT %s" in sql
    assert params == (801, 10, 50)


def test_batch_item_summary_recalculates_all_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重试收口必须以明细聚合结果覆盖旧批次累计值。"""

    cursor = RecordingCursor(
        one={
            "candidate_count": 4,
            "success_count": 2,
            "failed_count": 1,
            "rejected_count": 1,
            "deactivated_count": 0,
            "pending_count": 0,
            "running_count": 0,
            "extracted_count": 4,
            "stage_count": 4,
            "chunk_count": 5,
        }
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    summary = _repository().summarize_batch_items(801)

    assert summary.candidate_count == 4
    assert summary.success_count == 2
    assert summary.failed_count == 1
    assert summary.rejected_count == 1
    assert summary.extracted_count == 4
    assert summary.stage_count == 4
    assert summary.chunk_count == 5
    assert "SUM(status='failed')" in cursor.executed[0][0]
    assert cursor.executed[0][1] == (801,)


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
        vector_sync_expert_count=1,
        vector_current_count=5,
        vector_eligible_count=4,
        vector_embedded_count=3,
        vector_metadata_updated_count=1,
        vector_unchanged_count=0,
        vector_deleted_count=1,
        vector_delete_not_found_count=0,
    )

    assert len(cursors[0].executed) == 2
    assert all("status) VALUES (%s, %s, 'pending')" in call[0] for call in cursors[0].executed)
    assert "attempt_count=attempt_count+1" in cursors[1].executed[0][0]
    assert "error_message=NULL" in cursors[2].executed[0][0]
    assert cursors[2].executed[0][1] == (
        "success", 4, 3, 5, None, None, 801, 10
    )
    assert "WHERE id=%s AND status='running'" in cursors[3].executed[0][0]
    assert "vector_sync_expert_count=%s" in cursors[3].executed[0][0]
    assert cursors[3].executed[0][1][-9:] == (1, 5, 4, 3, 1, 0, 1, 0, 801)
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
        vector_sync_expert_count=9,
        vector_current_count=30,
        vector_eligible_count=27,
        vector_embedded_count=20,
        vector_metadata_updated_count=3,
        vector_unchanged_count=4,
        vector_deleted_count=2,
        vector_delete_not_found_count=1,
        error_summary={"failed_expert_count": 1},
    )

    sql, params = cursor.executed[0]
    assert "cursor_id=%s, candidate_count=%s" in sql
    assert "deactivated_count=%s" in sql
    assert "vector_metadata_updated_count=%s" in sql
    assert params == (
        "partial", 999, 10, 6, 1, 1, 24, 18, 30, 2,
        9, 30, 27, 20, 3, 4, 2, 1, '{"failed_expert_count":1}', 801,
    )
    assert connection.commit_count == 1


def test_single_batch_finish_persists_vector_sync_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """single 批次收口也应写入本次下游同步结果。"""

    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    _repository().finish_batch(
        801,
        status="success",
        extracted_count=4,
        chunk_count=3,
        vector_sync_expert_count=1,
        vector_current_count=3,
        vector_eligible_count=3,
        vector_embedded_count=2,
        vector_metadata_updated_count=1,
    )

    sql, params = cursor.executed[0]
    assert "cursor_id" not in sql
    assert "vector_sync_expert_count=%s" in sql
    assert params == (
        "success", 4, 0, 3, 0, 1, 3, 3, 2, 1, 0, 0, 0, None, 801,
    )


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
