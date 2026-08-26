"""P4.3 定向只读事实读取接口的最小契约测试。

全部使用 Fake/Scripted 连接，不连接真实 MySQL。测试覆盖四类 scope 查询形状、
物理 missing fail-closed、逻辑删除正常返回、子表-父专家归属校验、一致性事务边界
（一次 SET TRANSACTION + 一次 START TRANSACTION + 一次 rollback + 零 commit +
一次 close）、以及 education 任务字段到 profile scope 的真实接线。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from typing_extensions import Self

from expert_index.config import DatabaseConfig
from expert_index.models import (
    CurrentFactReadRequest,
    EmploymentSource,
    ExpertSource,
    ProjectSource,
    ScopedCurrentFacts,
    SourceSyncTask,
    impact_scope_from_target,
    source_sync_task_to_read_request,
)
from expert_index.source import ExpertNotFoundError, MySQLSourceAdapter

CREATED = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fake 连接：ScriptedCursor 按 SQL 关键字路由多次 SELECT。
# ---------------------------------------------------------------------------


class ScriptedCursor:
    """按 SQL 中出现的表名/关键字分发预设行，并记录执行序列。"""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []
        # 每个 key 对应一个 (one, rows) 队列，支持同一表多次查询返回不同结果。
        self._routes: dict[str, list[tuple[dict | None, list[dict] | None]]] = {}

    def add(
        self,
        *,
        key: str,
        one: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> Self:
        self._routes.setdefault(key, []).append((one, rows))
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        compact = " ".join(sql.split())
        self.executed.append((compact, params))

    def fetchone(self) -> dict[str, Any] | None:
        key = self._dispatch_key()
        route = self._routes[key]
        return route.pop(0)[0] if route else None

    def fetchall(self) -> list[dict[str, Any]]:
        key = self._dispatch_key()
        route = self._routes.get(key, [])
        if not route:
            return []
        one, rows = route.pop(0)
        return rows if rows is not None else ([] if one is None else [one])

    def _dispatch_key(self) -> str:
        last = self.executed[-1][0]
        for candidate in (
            "FROM experts_education_experience",
            "FROM experts_work_experience",
            "FROM experts_project_experience",
            "FROM experts",
        ):
            if candidate in last:
                return candidate
        if "SET TRANSACTION" in last:
            return "SET TRANSACTION"
        if "START TRANSACTION" in last:
            return "START TRANSACTION"
        raise KeyError(f"无法路由 SQL: {last}")


class ScriptedConnection:
    def __init__(self, cursor: ScriptedCursor) -> None:
        self.cursor_instance = cursor
        self.rollback_count = 0
        self.commit_count = 0
        self.closed = False

    def cursor(self) -> ScriptedCursor:
        return self.cursor_instance

    def rollback(self) -> None:
        self.rollback_count += 1

    def commit(self) -> None:  # 测试断言永远不应被调用
        self.commit_count += 1

    def close(self) -> None:
        self.closed = True


def _patch_connect(
    monkeypatch: pytest.MonkeyPatch,
    connection: ScriptedConnection,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def connect(**kwargs: Any) -> ScriptedConnection:
        calls.append(kwargs)
        return connection

    monkeypatch.setattr("expert_index.source.pymysql.connect", connect)
    return calls


def _adapter() -> MySQLSourceAdapter:
    return MySQLSourceAdapter(DatabaseConfig(host="127.0.0.1", port=3306, user="u",
                                             password="p", database="expert_dev"))


# ---------------------------------------------------------------------------
# 行工厂：构造四类来源行。
# ---------------------------------------------------------------------------


def _expert_row(
    expert_id: int = 100, *, deleted_at: datetime | None = None,
    updated_at: datetime | None = UPDATED,
) -> dict[str, Any]:
    # _load_expert_row_by_id 现在读取显式列：updated_at（中间表扫描时间）/
    # experts_updated_at（正式来源时间）/ experts_deleted_at（删除标记）。
    return {
        "id": expert_id, "serial_number": f"S{expert_id}", "company": "ACME",
        "job_status": "active", "job_name": "Engineer", "contact_status": "normal",
        "doubtful_flag": "normal", "abroad_flag": 1, "background": "bio",
        "work_years": 5.0, "updated_at": updated_at,
        "experts_updated_at": updated_at, "created_at": CREATED,
        "experts_deleted_at": deleted_at,
    }


def _employment_row(
    emp_id: int = 5000, expert_id: int = 100, *, deleted_at: datetime | None = None,
    updated_at: datetime | None = UPDATED,
) -> dict[str, Any]:
    return {
        "id": emp_id, "expert_id": expert_id, "company_name": "ACME",
        "position_title": "Eng", "start_date": None, "end_date": None,
        "work_description": "job", "is_employed": 1, "updated_at": updated_at,
        "created_at": CREATED, "deleted_at": deleted_at,
    }


def _project_row(
    proj_id: int = 6000, expert_id: int = 100, *, deleted_at: datetime | None = None,
    updated_at: datetime | None = UPDATED,
) -> dict[str, Any]:
    return {
        "id": proj_id, "expert_id": expert_id, "project_title": "P",
        "company_name": "ACME", "start_date": None, "end_date": None,
        "work_description": "desc", "updated_at": updated_at,
        "created_at": CREATED, "deleted_at": deleted_at,
    }


def _education_row(
    edu_id: int = 7000, expert_id: int = 100, *, deleted_at: datetime | None = None,
    updated_at: datetime | None = UPDATED,
) -> dict[str, Any]:
    return {
        "id": edu_id, "expert_id": expert_id, "school_name": "MIT",
        "major": "CS", "updated_at": updated_at, "created_at": CREATED,
        "deleted_at": deleted_at,
    }


# ---------------------------------------------------------------------------
# scope 工厂
# ---------------------------------------------------------------------------


def _expert_scope(expert_id: int = 100):
    return impact_scope_from_target(target_type="expert", expert_id=expert_id)


def _profile_scope(expert_id: int = 100):
    return impact_scope_from_target(target_type="profile", expert_id=expert_id)


def _employment_scope(expert_id: int = 100, source_field_id: str = "5000"):
    return impact_scope_from_target(
        target_type="employment", expert_id=expert_id, source_field_id=source_field_id,
    )


def _project_scope(expert_id: int = 100, source_field_id: str = "6000"):
    return impact_scope_from_target(
        target_type="project", expert_id=expert_id, source_field_id=source_field_id,
    )


# ---------------------------------------------------------------------------
# P3 SourceSyncTask 工厂
# ---------------------------------------------------------------------------


def _minimal_expert(expert_id: int = 100, *, deleted_at: datetime | None = None) -> ExpertSource:
    return ExpertSource(
        id=expert_id, serial_number=None, company=None, job_status=None,
        job_name=None, contact_status=None, doubtful_flag=None, abroad_flag=None,
        background=None, work_years=None, updated_at=UPDATED, created_at=CREATED,
        deleted_at=deleted_at,
    )


def _minimal_employment(emp_id: int = 5000, expert_id: int = 100) -> EmploymentSource:
    return EmploymentSource(
        id=emp_id, expert_id=expert_id, company_name=None, position_title=None,
        start_date=None, end_date=None, work_description=None, is_employed=None,
        updated_at=UPDATED, created_at=CREATED, deleted_at=None,
    )


def _minimal_project(proj_id: int = 6000, expert_id: int = 100) -> ProjectSource:
    return ProjectSource(
        id=proj_id, expert_id=expert_id, project_title=None, company_name=None,
        start_date=None, end_date=None, work_description=None, updated_at=UPDATED,
        created_at=CREATED, deleted_at=None,
    )


DEDUPE = "a" * 64
TASK_TS = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)


def _task(**changes: object) -> SourceSyncTask:
    values: dict[str, object] = {
        "task_id": 1, "dedupe_key": DEDUPE, "batch_id": 8, "event_source": "poll",
        "source_table": "experts_work_experience", "source_record_id": 5000,
        "source_updated_at": TASK_TS, "expert_id": 100, "operation_type": "update",
        "target_type": "employment", "processing_stage": "candidate",
        "sync_status": "pending", "retry_count": 0, "policy_version": "1.3",
        "created_at": TASK_TS, "updated_at": TASK_TS,
    }
    values.update(changes)
    return SourceSyncTask(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 事务边界断言
# ---------------------------------------------------------------------------


def _assert_readonly_txn(cursor: ScriptedCursor, connection: ScriptedConnection) -> None:
    sqls = [s for s, _ in cursor.executed]
    set_txn = [s for s in sqls if s.startswith("SET TRANSACTION ISOLATION LEVEL")]
    start_txn = [s for s in sqls if "START TRANSACTION WITH CONSISTENT SNAPSHOT" in s]
    assert len(set_txn) == 1, f"应恰好一次 SET TRANSACTION，实际 {len(set_txn)}"
    assert len(start_txn) == 1, f"应恰好一次 START TRANSACTION，实际 {len(start_txn)}"
    assert connection.rollback_count == 1, "成功路径应 rollback 一次结束快照"
    assert connection.commit_count == 0, "只读事务不应 commit"
    assert connection.closed is True, "连接应关闭"


# ===========================================================================
# 四类 scope 查询形状
# ===========================================================================


class TestExpertScopeRead:
    def test_active_expert_reads_four_tables(self, monkeypatch):
        cursor = ScriptedCursor().add(key="FROM experts", one=_expert_row(100))
        cursor.add(key="FROM experts_work_experience",
                   rows=[_employment_row(5001), _employment_row(5002)])
        cursor.add(key="FROM experts_project_experience", rows=[_project_row(6001)])
        cursor.add(key="FROM experts_education_experience", rows=[_education_row(7001)])
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_expert_scope(100), trigger_source_table="experts",
            trigger_source_field_id="100", trigger_source_updated_at=UPDATED,
        ))
        assert isinstance(facts, ScopedCurrentFacts)
        assert facts.scope.scope_type == "expert"
        assert facts.expert.id == 100
        assert facts.expert_deleted is False
        assert len(facts.employments) == 2
        assert len(facts.projects) == 1
        assert len(facts.educations) == 1
        _assert_readonly_txn(cursor, connection)

    def test_deleted_expert_returns_lifecycle_only(self, monkeypatch):
        cursor = ScriptedCursor().add(
            key="FROM experts",
            one=_expert_row(100, deleted_at=datetime(2026, 2, 4, tzinfo=timezone.utc)),
        )
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_expert_scope(100), trigger_source_table="experts",
            trigger_source_field_id="100", trigger_source_updated_at=UPDATED,
        ))
        assert facts.expert_deleted is True
        assert facts.parent_deleted is True
        assert facts.employments == () and facts.projects == () and facts.educations == ()
        # 删除专家不应读取子表
        select_sqls = [s for s, _ in cursor.executed if s.startswith("SELECT")]
        assert len(select_sqls) == 1  # 只读 experts 主档


class TestProfileScopeRead:
    def test_profile_trigger_reads_expert_and_education_only(self, monkeypatch):
        cursor = ScriptedCursor().add(key="FROM experts", one=_expert_row(100))
        cursor.add(key="FROM experts_education_experience",
                   rows=[_education_row(7001), _education_row(7002)])
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_profile_scope(100), trigger_source_table="experts",
            trigger_source_field_id="100", trigger_source_updated_at=UPDATED,
        ))
        assert facts.scope.scope_type == "profile"
        assert len(facts.educations) == 2
        assert facts.employments == () and facts.projects == ()
        _assert_readonly_txn(cursor, connection)

    def test_profile_deleted_parent_no_education_read(self, monkeypatch):
        cursor = ScriptedCursor().add(
            key="FROM experts",
            one=_expert_row(100, deleted_at=datetime(2026, 2, 4, tzinfo=timezone.utc)),
        )
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_profile_scope(100), trigger_source_table="experts",
            trigger_source_field_id="100", trigger_source_updated_at=UPDATED,
        ))
        assert facts.expert_deleted is True
        assert facts.educations == ()
        select_sqls = [s for s, _ in cursor.executed if s.startswith("SELECT")]
        assert len(select_sqls) == 1


class TestEmploymentScopeRead:
    def test_employment_trigger_reads_row_and_parent(self, monkeypatch):
        cursor = ScriptedCursor().add(
            key="FROM experts_work_experience", one=_employment_row(5000, 100),
        )
        cursor.add(key="FROM experts", one=_expert_row(100))
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_employment_scope(100, "5000"),
            trigger_source_table="experts_work_experience",
            trigger_source_field_id="5000", trigger_source_updated_at=UPDATED,
        ))
        assert facts.scope.scope_type == "employment"
        assert len(facts.employments) == 1
        assert facts.employments[0].id == 5000
        assert facts.expert.id == 100
        assert facts.educations == () and facts.projects == ()
        _assert_readonly_txn(cursor, connection)

    def test_employment_trigger_parent_deleted_flag(self, monkeypatch):
        cursor = ScriptedCursor().add(
            key="FROM experts_work_experience", one=_employment_row(5000, 100),
        )
        cursor.add(key="FROM experts",
                   one=_expert_row(100, deleted_at=datetime(2026, 2, 4, tzinfo=timezone.utc)))
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_employment_scope(100, "5000"),
            trigger_source_table="experts_work_experience",
            trigger_source_field_id="5000", trigger_source_updated_at=UPDATED,
        ))
        assert facts.parent_deleted is True
        # 触发行仍可见（employment 行本身未删除）
        assert len(facts.employments) == 1


class TestProjectScopeRead:
    def test_project_trigger_reads_row_and_parent(self, monkeypatch):
        cursor = ScriptedCursor().add(
            key="FROM experts_project_experience", one=_project_row(6000, 100),
        )
        cursor.add(key="FROM experts", one=_expert_row(100))
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_project_scope(100, "6000"),
            trigger_source_table="experts_project_experience",
            trigger_source_field_id="6000", trigger_source_updated_at=UPDATED,
        ))
        assert facts.scope.scope_type == "project"
        assert len(facts.projects) == 1
        assert facts.projects[0].id == 6000
        assert facts.employments == () and facts.educations == ()
        _assert_readonly_txn(cursor, connection)


# ===========================================================================
# 物理缺失 fail-closed / 逻辑删除正常返回
# ===========================================================================


class TestPhysicalMissingAndLogicalDeletion:
    def test_expert_physical_missing_raises(self, monkeypatch):
        cursor = ScriptedCursor().add(key="FROM experts", one=None)
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        with pytest.raises(ExpertNotFoundError, match="物理不存在"):
            _adapter().read_current_facts(CurrentFactReadRequest(
                impact_scope=_expert_scope(999), trigger_source_table="experts",
                trigger_source_field_id="999", trigger_source_updated_at=UPDATED,
            ))
        # 失败路径也 rollback + close
        assert connection.rollback_count == 1
        assert connection.closed is True

    def test_employment_physical_missing_raises(self, monkeypatch):
        cursor = ScriptedCursor().add(key="FROM experts_work_experience", one=None)
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        with pytest.raises(ExpertNotFoundError, match="工作记录"):
            _adapter().read_current_facts(CurrentFactReadRequest(
                impact_scope=_employment_scope(100, "8888"),
                trigger_source_table="experts_work_experience",
                trigger_source_field_id="8888", trigger_source_updated_at=UPDATED,
            ))

    def test_logical_deletion_returned_normally(self, monkeypatch):
        del_ts = datetime(2026, 2, 4, tzinfo=timezone.utc)
        cursor = ScriptedCursor().add(
            key="FROM experts_work_experience",
            one=_employment_row(5000, 100, deleted_at=del_ts),
        )
        cursor.add(key="FROM experts", one=_expert_row(100))
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_employment_scope(100, "5000"),
            trigger_source_table="experts_work_experience",
            trigger_source_field_id="5000", trigger_source_updated_at=UPDATED,
        ))
        # 触发行逻辑删除仍可读取，deleted_at 非空
        assert facts.employments[0].deleted_at == del_ts


# ===========================================================================
# 子表-父专家归属校验
# ===========================================================================


class TestChildParentOwnership:
    def test_employment_expert_id_mismatch_rejected(self, monkeypatch):
        cursor = ScriptedCursor().add(
            key="FROM experts_work_experience", one=_employment_row(5000, 999),
        )
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        with pytest.raises(ValueError, match="employment 触发行 expert_id=999"):
            _adapter().read_current_facts(CurrentFactReadRequest(
                impact_scope=_employment_scope(100, "5000"),
                trigger_source_table="experts_work_experience",
                trigger_source_field_id="5000", trigger_source_updated_at=UPDATED,
            ))

    def test_project_expert_id_mismatch_rejected(self, monkeypatch):
        cursor = ScriptedCursor().add(
            key="FROM experts_project_experience", one=_project_row(6000, 999),
        )
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        with pytest.raises(ValueError, match="project 触发行 expert_id=999"):
            _adapter().read_current_facts(CurrentFactReadRequest(
                impact_scope=_project_scope(100, "6000"),
                trigger_source_table="experts_project_experience",
                trigger_source_field_id="6000", trigger_source_updated_at=UPDATED,
            ))

    def test_education_expert_id_mismatch_rejected(self, monkeypatch):
        cursor = ScriptedCursor().add(key="FROM experts", one=_expert_row(100))
        cursor.add(key="FROM experts_education_experience", one=_education_row(7000, 999))
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        with pytest.raises(ValueError, match="education 触发行 expert_id=999"):
            _adapter().read_current_facts(CurrentFactReadRequest(
                impact_scope=_profile_scope(100),
                trigger_source_table="experts_education_experience",
                trigger_source_field_id="7000", trigger_source_updated_at=UPDATED,
            ))


# ===========================================================================
# 时序校验：当前 updated_at 早于候选 fail-closed
# ===========================================================================


class TestEventOrdering:
    def test_current_older_than_candidate_rejected(self, monkeypatch):
        older = datetime(2026, 8, 19, tzinfo=timezone.utc)
        newer_candidate = datetime(2026, 8, 20, tzinfo=timezone.utc)
        cursor = ScriptedCursor().add(
            key="FROM experts_work_experience",
            one=_employment_row(5000, 100, updated_at=older),
        )
        cursor.add(key="FROM experts", one=_expert_row(100))
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        with pytest.raises(ValueError, match="早于候选"):
            _adapter().read_current_facts(CurrentFactReadRequest(
                impact_scope=_employment_scope(100, "5000"),
                trigger_source_table="experts_work_experience",
                trigger_source_field_id="5000", trigger_source_updated_at=newer_candidate,
            ))

    def test_candidate_none_skips_time_check(self, monkeypatch):
        cursor = ScriptedCursor().add(
            key="FROM experts_work_experience", one=_employment_row(5000, 100),
        )
        cursor.add(key="FROM experts", one=_expert_row(100))
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_employment_scope(100, "5000"),
            trigger_source_table="experts_work_experience",
            trigger_source_field_id="5000", trigger_source_updated_at=None,
        ))
        assert facts.employments[0].id == 5000

    def test_expert_scope_current_older_than_candidate_rejected(self, monkeypatch):
        older = datetime(2026, 8, 19, tzinfo=timezone.utc)
        newer_candidate = datetime(2026, 8, 20, tzinfo=timezone.utc)
        cursor = ScriptedCursor().add(
            key="FROM experts",
            one=_expert_row(100, updated_at=older),
        )
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        with pytest.raises(ValueError, match="早于候选"):
            _adapter().read_current_facts(CurrentFactReadRequest(
                impact_scope=_expert_scope(100), trigger_source_table="experts",
                trigger_source_field_id="100", trigger_source_updated_at=newer_candidate,
            ))

    def test_expert_scope_candidate_none_passes(self, monkeypatch):
        cursor = ScriptedCursor().add(key="FROM experts", one=_expert_row(100))
        cursor.add(key="FROM experts_work_experience", rows=[])
        cursor.add(key="FROM experts_project_experience", rows=[])
        cursor.add(key="FROM experts_education_experience", rows=[])
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_expert_scope(100), trigger_source_table="experts",
            trigger_source_field_id="100", trigger_source_updated_at=None,
        ))
        assert facts.expert.id == 100


# ===========================================================================
# education 任务字段到 profile scope 真实接线
# ===========================================================================


class TestEducationTaskWiring:
    def test_source_sync_task_to_read_request_education_maps_to_profile(self):
        task = _task(
            source_table="experts_education_experience", source_record_id=7000,
            target_type="education", expert_id=100,
        )
        req = source_sync_task_to_read_request(task)
        # impact_scope 是 profile，source_field_id=expert_id（非教育记录主键）
        assert req.impact_scope.scope_type == "profile"
        assert req.impact_scope.source_field_id == "100"
        # trigger 保留教育记录主键用于归属校验
        assert req.trigger_source_table == "experts_education_experience"
        assert req.trigger_source_field_id == "7000"

    def test_source_sync_task_to_read_request_employment(self):
        task = _task(target_type="employment", expert_id=100, source_record_id=5000)
        req = source_sync_task_to_read_request(task)
        assert req.impact_scope.scope_type == "employment"
        assert req.impact_scope.source_field_id == "5000"
        assert req.trigger_source_field_id == "5000"

    def test_task_without_expert_id_rejected(self):
        task = _task(expert_id=None)
        with pytest.raises(ValueError, match="缺少 expert_id"):
            source_sync_task_to_read_request(task)

    def test_education_read_consumes_trigger_then_reads_parent_education(self, monkeypatch):
        cursor = ScriptedCursor().add(key="FROM experts", one=_expert_row(100))
        cursor.add(key="FROM experts_education_experience",
                   one=_education_row(7000, 100))  # 触发行校验
        cursor.add(key="FROM experts_education_experience",
                   rows=[_education_row(7000, 100), _education_row(7001, 100)])  # 全部有效教育
        connection = ScriptedConnection(cursor)
        _patch_connect(monkeypatch, connection)

        facts = _adapter().read_current_facts(CurrentFactReadRequest(
            impact_scope=_profile_scope(100),
            trigger_source_table="experts_education_experience",
            trigger_source_field_id="7000", trigger_source_updated_at=UPDATED,
        ))
        assert facts.scope.scope_type == "profile"
        assert len(facts.educations) == 2
        # 一次 SET TRANSACTION + 一次 START TRANSACTION + rollback 一次 + close
        _assert_readonly_txn(cursor, connection)


# ===========================================================================
# 请求模型校验与 source_record_id 禁止
# ===========================================================================


class TestRequestModelValidation:
    def test_expert_scope_wrong_trigger_table_rejected(self):
        with pytest.raises(ValueError, match="触发来源必须是 experts"):
            CurrentFactReadRequest(
                impact_scope=_expert_scope(100),
                trigger_source_table="experts_work_experience",
                trigger_source_field_id="100", trigger_source_updated_at=None,
            )

    def test_employment_scope_wrong_trigger_table_rejected(self):
        with pytest.raises(ValueError, match="触发来源必须是 experts_work_experience"):
            CurrentFactReadRequest(
                impact_scope=_employment_scope(100, "5000"),
                trigger_source_table="experts_project_experience",
                trigger_source_field_id="5000", trigger_source_updated_at=None,
            )

    def test_expert_scope_trigger_id_must_equal_expert_id(self):
        with pytest.raises(ValueError, match="触发行 source_field_id 必须等于"):
            CurrentFactReadRequest(
                impact_scope=_expert_scope(100), trigger_source_table="experts",
                trigger_source_field_id="999", trigger_source_updated_at=None,
            )

    def test_whitespace_trigger_id_rejected(self):
        with pytest.raises(ValueError, match="无首尾空白的非空字符串"):
            CurrentFactReadRequest(
                impact_scope=_employment_scope(100, "5000"),
                trigger_source_table="experts_work_experience",
                trigger_source_field_id=" 5000 ", trigger_source_updated_at=None,
            )

    def test_source_record_id_absent_from_new_models(self):
        import dataclasses
        for model in (CurrentFactReadRequest, ScopedCurrentFacts):
            names = {f.name for f in dataclasses.fields(model)}
            assert "source_record_id" not in names

    def test_scoped_facts_shape_enforcement(self):
        # profile scope 不允许携带 employments
        with pytest.raises(ValueError, match="profile 范围不读取工作和项目"):
            ScopedCurrentFacts(
                scope=_profile_scope(100), expert=_minimal_expert(100),
                employments=(_minimal_employment(),),
            )

    def test_project_scope_rejects_employment_rows(self):
        # project scope 不能携带工作行
        with pytest.raises(ValueError, match="project 范围不读取工作和教育"):
            ScopedCurrentFacts(
                scope=_project_scope(100, "6000"), expert=_minimal_expert(100),
                projects=(_minimal_project(),),
                employments=(_minimal_employment(),),
            )

    def test_employment_scope_requires_exactly_one_row(self):
        # employment scope 必须恰好一条工作行
        with pytest.raises(ValueError, match="employment 范围必须恰好一条工作行"):
            ScopedCurrentFacts(
                scope=_employment_scope(100, "5000"), expert=_minimal_expert(100),
                employments=(),
            )
        with pytest.raises(ValueError, match="employment 范围必须恰好一条工作行"):
            ScopedCurrentFacts(
                scope=_employment_scope(100, "5000"), expert=_minimal_expert(100),
                employments=(_minimal_employment(5000), _minimal_employment(5001)),
            )

    def test_project_scope_requires_exactly_one_row(self):
        with pytest.raises(ValueError, match="project 范围必须恰好一条项目行"):
            ScopedCurrentFacts(
                scope=_project_scope(100, "6000"), expert=_minimal_expert(100),
                projects=(),
            )
