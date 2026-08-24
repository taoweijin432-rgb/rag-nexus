"""第二阶段全量候选分页与四表一致性读取的 Source 层测试。

测试使用轻量 Fake Connection/Cursor，不访问真实 MySQL，也不触发 Pipeline 或任何写库
行为。重点锁定分页 SQL、逻辑删除候选保留、参数边界和单专家只读事务顺序。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self
from zoneinfo import ZoneInfo

import pytest

from expert_index.config import DatabaseConfig
from expert_index.models import CHECKPOINT_SOURCE_TABLES, SourceScanCursor
from expert_index.source import (
    MAX_FULL_SCAN_PAGE_SIZE,
    MAX_SOURCE_SCAN_PAGE_SIZE,
    MySQLSourceAdapter,
)


class FakeCursor:
    """记录 Source 执行的 SQL，并按测试场景返回预设行。"""

    def __init__(
        self,
        *,
        one: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.one = one
        self.rows = rows or []
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        """保存压缩后的 SQL，便于测试精确检查查询边界和事务顺序。"""

        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self) -> dict[str, Any] | None:
        return self.one

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self.rows)


class FakeConnection:
    """提供 Source 本阶段需要的 cursor/rollback/close 最小连接接口。"""

    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def _adapter() -> MySQLSourceAdapter:
    """构造不会主动连接数据库的 Source Adapter。"""

    return MySQLSourceAdapter(DatabaseConfig("127.0.0.1", 3306, "test", "test", "test"))


def _patch_connection(
    monkeypatch: pytest.MonkeyPatch,
    connection: FakeConnection,
) -> list[dict[str, Any]]:
    """替换 PyMySQL connect，并返回用于检查 autocommit 等参数的记录列表。"""

    calls: list[dict[str, Any]] = []

    def connect(**kwargs: Any) -> FakeConnection:
        calls.append(kwargs)
        return connection

    monkeypatch.setattr("expert_index.source.pymysql.connect", connect)
    return calls


@pytest.mark.parametrize("source_table", sorted(CHECKPOINT_SOURCE_TABLES))
def test_source_scan_upper_bound_uses_updated_at_and_id(
    monkeypatch: pytest.MonkeyPatch,
    source_table: str,
) -> None:
    """每张来源表都应按复合游标倒序读取当前固定上界。"""

    updated_at = datetime(2026, 8, 20, 11, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    cursor = FakeCursor(one={"updated_at": updated_at, "id": 981})
    connection = FakeConnection(cursor)
    connect_calls = _patch_connection(monkeypatch, connection)

    result = _adapter().get_source_scan_upper_bound(source_table)

    assert result == SourceScanCursor(updated_at=updated_at, record_id=981)
    sql, params = cursor.executed[0]
    assert sql == (
        f"SELECT updated_at, id FROM {source_table} "
        "ORDER BY updated_at DESC, id DESC LIMIT 1"
    )
    assert params is None
    assert "deleted_at" not in sql
    assert connect_calls[0]["autocommit"] is True
    assert connection.closed is True


def test_source_scan_upper_bound_returns_none_for_empty_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空来源表没有真实上界，不能伪造时间或 ID。"""

    cursor = FakeCursor(one=None)
    connection = FakeConnection(cursor)
    _patch_connection(monkeypatch, connection)

    assert _adapter().get_source_scan_upper_bound("experts") is None
    assert connection.closed is True


def test_source_scan_upper_bound_rejects_unknown_table_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法表名必须在连接前失败，不能进入动态 SQL。"""

    def unexpected_connect(**_kwargs: Any) -> None:
        raise AssertionError("非法来源表不应连接 MySQL")

    monkeypatch.setattr("expert_index.source.pymysql.connect", unexpected_connect)

    with pytest.raises(ValueError, match="不支持的 Checkpoint 来源表"):
        _adapter().get_source_scan_upper_bound("expert_work_experience")


@pytest.mark.parametrize("source_table", sorted(CHECKPOINT_SOURCE_TABLES))
def test_source_change_candidates_use_fixed_composite_range(
    monkeypatch: pytest.MonkeyPatch,
    source_table: str,
) -> None:
    """四张表都应在起点开区间、固定上界闭区间内按复合游标分页。"""

    start_time = datetime(2026, 8, 20, 11, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    end_time = datetime(2026, 8, 20, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    deleted_at = datetime(2026, 8, 20, 11, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    cursor = FakeCursor(
        rows=[
            {
                "id": 102,
                "expert_id": 71,
                "updated_at": start_time,
                "deleted_at": None,
            },
            {
                "id": 120,
                "expert_id": 72,
                "updated_at": end_time,
                "deleted_at": deleted_at,
            },
        ]
    )
    connection = FakeConnection(cursor)
    connect_calls = _patch_connection(monkeypatch, connection)

    candidates = _adapter().load_source_change_candidates(
        source_table,
        after_cursor=SourceScanCursor(updated_at=start_time, record_id=100),
        upper_bound=SourceScanCursor(updated_at=end_time, record_id=120),
        limit=50,
    )

    assert [item.record_id for item in candidates] == [102, 120]
    assert [item.expert_id for item in candidates] == [71, 72]
    assert candidates[0].cursor == SourceScanCursor(
        updated_at=start_time,
        record_id=102,
    )
    assert candidates[1].is_deleted is True
    sql, params = cursor.executed[0]
    expected_expert_field = "id AS expert_id" if source_table == "experts" else "expert_id"
    expected_deleted_field = (
        "experts_deleted_at AS deleted_at"
        if source_table == "experts"
        else "deleted_at"
    )
    assert f"SELECT id, {expected_expert_field}, updated_at, {expected_deleted_field}" in sql
    assert "updated_at > %s OR (updated_at = %s AND id > %s)" in sql
    assert "updated_at < %s OR (updated_at = %s AND id <= %s)" in sql
    assert "ORDER BY updated_at ASC, id ASC" in sql
    assert "deleted_at IS NULL" not in sql
    if source_table == "experts":
        assert "experts_updated_at" not in sql
    assert params == (start_time, start_time, 100, end_time, end_time, 120, 50)
    assert connect_calls[0]["autocommit"] is True
    assert connection.closed is True


def test_source_change_candidates_keep_missing_expert_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """经历表无法关联专家的脏记录必须保留，后续才能进入隔离流程。"""

    updated_at = datetime(2026, 8, 20, 11, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    cursor = FakeCursor(
        rows=[
            {
                "id": 102,
                "expert_id": None,
                "updated_at": updated_at,
                "deleted_at": None,
            }
        ]
    )
    connection = FakeConnection(cursor)
    _patch_connection(monkeypatch, connection)

    candidates = _adapter().load_source_change_candidates(
        "experts_work_experience",
        after_cursor=SourceScanCursor(updated_at=updated_at, record_id=0),
        upper_bound=SourceScanCursor(updated_at=updated_at, record_id=102),
        limit=10,
    )

    assert candidates[0].expert_id is None


@pytest.mark.parametrize("limit", [0, True, MAX_SOURCE_SCAN_PAGE_SIZE + 1])
def test_source_change_candidates_reject_invalid_limit_before_connect(
    monkeypatch: pytest.MonkeyPatch,
    limit: object,
) -> None:
    """非法页大小不能触发源库连接。"""

    def unexpected_connect(**_kwargs: Any) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr("expert_index.source.pymysql.connect", unexpected_connect)
    updated_at = datetime(2026, 8, 20, 11, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    with pytest.raises(ValueError, match="limit"):
        _adapter().load_source_change_candidates(
            "experts",
            after_cursor=SourceScanCursor(updated_at=updated_at, record_id=0),
            upper_bound=SourceScanCursor(updated_at=updated_at, record_id=1),
            limit=limit,  # type: ignore[arg-type]
        )


def test_source_change_candidates_skip_empty_range_without_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """起点已经到达上界时直接返回空页，不执行无意义查询。"""

    def unexpected_connect(**_kwargs: Any) -> None:
        raise AssertionError("空范围不应连接 MySQL")

    monkeypatch.setattr("expert_index.source.pymysql.connect", unexpected_connect)
    cursor = SourceScanCursor(
        updated_at=datetime(2026, 8, 20, 11, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        record_id=10,
    )

    assert _adapter().load_source_change_candidates(
        "experts",
        after_cursor=cursor,
        upper_bound=cursor,
        limit=10,
    ) == []


def test_full_scan_upper_bound_uses_all_expert_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最大 ID 查询不能过滤 deleted_at，否则逻辑删除专家无法进入停用分支。"""

    cursor = FakeCursor(one={"upper_bound_id": 981})
    connection = FakeConnection(cursor)
    connect_calls = _patch_connection(monkeypatch, connection)

    assert _adapter().get_full_scan_upper_bound() == 981

    sql, params = cursor.executed[0]
    assert sql == "SELECT COALESCE(MAX(id), 0) AS upper_bound_id FROM experts"
    assert params is None
    assert "deleted_at" not in sql
    assert connect_calls[0]["autocommit"] is True
    assert connection.closed is True


def test_full_scan_upper_bound_limits_to_first_n_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """指定 max_experts 时应以按 ID 排序的前 N 位候选确定固定上界。"""

    cursor = FakeCursor(one={"upper_bound_id": 505})
    connection = FakeConnection(cursor)
    connect_calls = _patch_connection(monkeypatch, connection)

    assert _adapter().get_full_scan_upper_bound(max_experts=500) == 505

    sql, params = cursor.executed[0]
    assert "SELECT id FROM experts ORDER BY id ASC LIMIT %s" in sql
    assert "SELECT COALESCE(MAX(id), 0) AS upper_bound_id" in sql
    assert "deleted_at" not in sql
    assert params == (500,)
    assert connect_calls[0]["autocommit"] is True
    assert connection.closed is True


@pytest.mark.parametrize("max_experts", [0, -1, True, 1.5])
def test_full_scan_upper_bound_rejects_invalid_max_experts_before_connect(
    monkeypatch: pytest.MonkeyPatch,
    max_experts: object,
) -> None:
    """无效候选上限不能触发源库连接。"""

    def unexpected_connect(**_kwargs: Any) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr("expert_index.source.pymysql.connect", unexpected_connect)

    with pytest.raises(ValueError, match="max_experts"):
        _adapter().get_full_scan_upper_bound(
            max_experts=max_experts,  # type: ignore[arg-type]
        )


def test_full_scan_candidates_use_keyset_and_keep_deleted_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候选页按固定主键区间读取，同时把逻辑删除标记映射到候选模型。"""

    deleted_at = datetime(2026, 8, 14, 11, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    cursor = FakeCursor(
        rows=[
            {"id": 101, "deleted_at": None},
            {"id": 105, "deleted_at": deleted_at},
        ]
    )
    connection = FakeConnection(cursor)
    connect_calls = _patch_connection(monkeypatch, connection)

    candidates = _adapter().load_full_scan_candidates(
        after_id=100,
        upper_bound_id=500,
        limit=2,
    )

    assert [candidate.id for candidate in candidates] == [101, 105]
    assert candidates[0].is_deleted is False
    assert candidates[1].is_deleted is True
    assert candidates[1].deleted_at == deleted_at
    sql, params = cursor.executed[0]
    assert "SELECT id, experts_deleted_at AS deleted_at FROM experts" in sql
    assert "WHERE id > %s AND id <= %s" in sql
    assert "ORDER BY id ASC LIMIT %s" in sql
    assert "experts_deleted_at IS NULL" not in sql
    assert params == (100, 500, 2)
    assert connect_calls[0]["autocommit"] is True
    assert connection.closed is True


def test_load_expert_candidate_reads_lifecycle_without_source_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失败重试只查询主键和删除标记，不提前读取专家正文。"""

    deleted_at = datetime(2026, 8, 18, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    cursor = FakeCursor(one={"id": 105, "deleted_at": deleted_at})
    connection = FakeConnection(cursor)
    connect_calls = _patch_connection(monkeypatch, connection)

    candidate = _adapter().load_expert_candidate(105)

    assert candidate.id == 105
    assert candidate.is_deleted is True
    assert cursor.executed == [
        (
            "SELECT id, experts_deleted_at AS deleted_at FROM experts WHERE id=%s",
            (105,),
        )
    ]
    assert connect_calls[0]["autocommit"] is True
    assert connection.closed is True


@pytest.mark.parametrize(
    ("after_id", "upper_bound_id", "limit"),
    [
        (-1, 100, 10),
        (0, -1, 10),
        (0, 100, 0),
        (0, 100, MAX_FULL_SCAN_PAGE_SIZE + 1),
    ],
)
def test_full_scan_candidates_reject_invalid_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    after_id: int,
    upper_bound_id: int,
    limit: int,
) -> None:
    """无效分页参数必须在建立数据库连接前失败。"""

    def unexpected_connect(**_kwargs: Any) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr("expert_index.source.pymysql.connect", unexpected_connect)

    with pytest.raises(ValueError):
        _adapter().load_full_scan_candidates(
            after_id=after_id,
            upper_bound_id=upper_bound_id,
            limit=limit,
        )


def test_full_scan_candidates_stop_without_query_at_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """游标到达固定上界后直接返回空页，供后续 Pipeline 结束循环。"""

    def unexpected_connect(**_kwargs: Any) -> None:
        raise AssertionError("到达上界时不应连接 MySQL")

    monkeypatch.setattr("expert_index.source.pymysql.connect", unexpected_connect)

    assert _adapter().load_full_scan_candidates(
        after_id=500,
        upper_bound_id=500,
        limit=100,
    ) == []


def test_load_expert_reads_four_tables_in_one_consistent_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单专家读取必须先开启只读快照，并允许三张经历表全部为空。"""

    profile = {
        "id": 24,
        "serial_number": "EXP000024",
        "company": "虚构测试公司",
        "job_status": 1,
        "job_name": "测试岗位",
        "contact_status": 3,
        "doubtful_flag": 1,
        "abroad_flag": 2,
        "background": "虚构测试背景",
        "work_years": 8,
        "updated_at": None,
        "created_at": None,
        "deleted_at": None,
    }
    cursor = FakeCursor(one=profile, rows=[])
    connection = FakeConnection(cursor)
    connect_calls = _patch_connection(monkeypatch, connection)

    aggregate = _adapter().load_expert(24)

    assert aggregate.profile.id == 24
    assert aggregate.employments == ()
    assert aggregate.projects == ()
    assert aggregate.educations == ()
    executed_sql = [sql for sql, _params in cursor.executed]
    assert executed_sql[:2] == [
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ",
        "START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY",
    ]
    assert "experts_updated_at AS updated_at" in executed_sql[2]
    assert "experts_deleted_at AS deleted_at" in executed_sql[2]
    assert "FROM experts WHERE id = %s AND experts_deleted_at IS NULL" in executed_sql[2]
    assert "FROM experts_work_experience" in executed_sql[3]
    assert "FROM experts_project_experience" in executed_sql[4]
    assert "FROM experts_education_experience" in executed_sql[5]
    # 教育中间表没有 start_date/end_date；只按两套环境共有的主键稳定排序。
    assert "ORDER BY id ASC" in executed_sql[5]
    assert "start_date" not in executed_sql[5]
    assert "end_date" not in executed_sql[5]
    assert connect_calls[0]["autocommit"] is False
    assert connection.rollback_count == 1
    assert connection.closed is True
