"""Resume Source Adapter 的只读 SQL、身份映射和事务边界测试。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pytest

from expert_index.config import DatabaseBoundaryError, DatabaseConfig, DatabaseRole
from expert_index.models import CurrentFactReadRequest, SourceScanCursor, impact_scope_from_target
from expert_index.normalize import normalize_aggregate
from resume_index.dataset import RESUME_SPEC
from resume_index.source import (
    MySQLResumeSourceAdapter,
    ResumeNotFoundError,
)


class ScriptedCursor:
    """按 SQL 中的物理表名返回测试数据，不解释或执行 SQL。"""

    def __init__(self, responses: Mapping[str, Sequence[object]]) -> None:
        self._responses = {key: list(values) for key, values in responses.items()}
        self._current: object = None
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(sql.split())
        self.executed.append((normalized, params))
        if normalized.startswith(("SET TRANSACTION", "START TRANSACTION")):
            self._current = None
            return
        for marker, values in self._responses.items():
            if marker in normalized:
                if not values:
                    raise AssertionError(f"SQL {marker} 没有剩余测试响应")
                self._current = values.pop(0)
                return
        raise AssertionError(f"没有为 SQL 配置响应: {normalized}")

    def fetchone(self) -> dict[str, Any] | None:
        if self._current is None or isinstance(self._current, list):
            return None
        return dict(self._current)  # type: ignore[arg-type]

    def fetchall(self) -> list[dict[str, Any]]:
        if not isinstance(self._current, list):
            return []
        return [dict(row) for row in self._current]


class FakeConnection:
    """记录只读事务是否以 rollback 收口。"""

    def __init__(self, cursor: ScriptedCursor) -> None:
        self._cursor = cursor
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> ScriptedCursor:
        return self._cursor

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def _source_config() -> DatabaseConfig:
    return DatabaseConfig(
        host="127.0.0.1",
        port=3306,
        user="resume_reader",
        password="test-only",
        database="source_test",
        role=DatabaseRole.SOURCE,
    )


def _profile_row(resume_id: int = 70001) -> dict[str, object]:
    when = datetime(2026, 9, 2, 10, 0, 0)  # noqa: DTZ001
    return {
        "internal_person_id": resume_id,
        "serial_number": "RESUME-TEST-001",
        "company": "示例科技有限公司",
        "job_status": 1,
        "job_name": "项目经理",
        "contact_status": 3,
        "doubtful_flag": 1,
        "abroad_flag": 2,
        "background": "负责企业知识库建设。",
        "work_years": 6,
        "business_updated_at": when,
        "scan_updated_at": when,
        "created_at": when,
        "lifecycle_deleted_at": None,
    }


def test_load_resume_maps_resume_id_and_preserves_child_record_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """四表快照应映射到公共模型，同时保留两类身份的明确边界。"""

    when = datetime(2026, 9, 2, 10, 0, 0)  # noqa: DTZ001
    cursor = ScriptedCursor(
        {
            "FROM resume ": [_profile_row()],
            "FROM resume_work_experience": [
                [
                    {
                        "id": 81001,
                        "internal_person_id": 70001,
                        "company_name": "示例科技有限公司",
                        "position_title": "项目经理",
                        "start_date": when,
                        "end_date": None,
                        "work_description": "工作职责：负责知识库建设。",
                        "is_employed": 1,
                        "business_updated_at": when,
                        "created_at": when,
                        "deleted_at": None,
                    }
                ]
            ],
            "FROM resume_project_experience": [
                [
                    {
                        "id": 82001,
                        "internal_person_id": 70001,
                        "project_title": "知识库项目",
                        "company_name": "示例科技有限公司",
                        "start_date": when,
                        "end_date": None,
                        "work_description": "项目目标：建设知识库。",
                        "business_updated_at": when,
                        "created_at": when,
                        "deleted_at": None,
                    }
                ]
            ],
            "FROM resume_education_experience": [
                [
                    {
                        "id": 83001,
                        "internal_person_id": 70001,
                        "school_name": "示例大学",
                        "major": "信息管理",
                        "business_updated_at": when,
                        "created_at": when,
                        "deleted_at": None,
                    }
                ]
            ],
        }
    )
    connection = FakeConnection(cursor)
    connect_kwargs: dict[str, object] = {}

    def fake_connect(**kwargs: object) -> FakeConnection:
        connect_kwargs.update(kwargs)
        return connection

    monkeypatch.setattr("resume_index.source.pymysql.connect", fake_connect)

    aggregate = MySQLResumeSourceAdapter(_source_config()).load_resume(70001)

    assert aggregate.profile.id == 70001
    assert aggregate.employments[0].expert_id == 70001
    assert aggregate.projects[0].expert_id == 70001
    assert aggregate.educations[0].expert_id == 70001
    assert aggregate.employments[0].id == 81001
    assert aggregate.projects[0].id == 82001
    assert connection.rollback_count == 1
    assert connection.closed is True
    assert connect_kwargs["autocommit"] is False

    statements = [sql for sql, _params in cursor.executed]
    assert all("SELECT *" not in sql for sql in statements)
    assert any("resume_deleted_at IS NULL" in sql for sql in statements)
    assert sum("deleted_at IS NULL" in sql for sql in statements) == 4
    assert all(params == (70001,) for sql, params in cursor.executed if sql.startswith("SELECT"))

    profile, employments, projects = normalize_aggregate(
        aggregate,
        data_version="resume-source-test",
        build_batch_id=21,
        dataset_spec=RESUME_SPEC,
    )
    assert profile.expert_id == 70001
    assert employments[0].source_table == "resume_work_experience"
    assert employments[0].source_field_id == "81001"
    assert projects[0].source_table == "resume_project_experience"
    assert projects[0].source_field_id == "82001"


def test_full_scan_candidates_keep_physical_cursor_separate_from_resume_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted_at = datetime(2026, 9, 2, 11, 0, 0)  # noqa: DTZ001
    cursor = ScriptedCursor(
        {
            "FROM resume ": [
                [
                    {"source_record_id": 41, "resume_id": 70001, "deleted_at": None},
                    {
                        "source_record_id": 42,
                        "resume_id": 90000007,
                        "deleted_at": deleted_at,
                    },
                ]
            ]
        }
    )
    connection = FakeConnection(cursor)
    monkeypatch.setattr(
        "resume_index.source.pymysql.connect",
        lambda **_kwargs: connection,
    )

    candidates = MySQLResumeSourceAdapter(_source_config()).load_full_scan_candidates(
        after_source_record_id=40,
        upper_bound_source_record_id=100,
        limit=20,
    )

    assert [(item.source_record_id, item.resume_id) for item in candidates] == [
        (41, 70001),
        (42, 90000007),
    ]
    assert candidates[0].is_deleted is False
    assert candidates[1].is_deleted is True
    sql, params = cursor.executed[0]
    assert "WHERE id > %s AND id <= %s" in sql
    assert "resume_id AS resume_id" in sql
    assert params == (40, 100, 20)
    assert connection.rollback_count == 0
    assert connection.closed is True


def test_full_scan_upper_bound_uses_profile_physical_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = ScriptedCursor({"FROM resume ": [{"upper_bound_id": 42}]})
    connection = FakeConnection(cursor)
    monkeypatch.setattr(
        "resume_index.source.pymysql.connect",
        lambda **_kwargs: connection,
    )

    upper_bound = MySQLResumeSourceAdapter(_source_config()).get_full_scan_upper_bound(
        max_resumes=20
    )

    assert upper_bound == 42
    sql, params = cursor.executed[0]
    assert "SELECT id FROM resume ORDER BY id ASC LIMIT %s" in sql
    assert "resume_id" not in sql
    assert params == (20,)


def test_load_resume_candidate_reads_lifecycle_without_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = ScriptedCursor(
        {"FROM resume ": [{"source_record_id": 17, "resume_id": 70001, "deleted_at": None}]}
    )
    connection = FakeConnection(cursor)
    monkeypatch.setattr(
        "resume_index.source.pymysql.connect",
        lambda **_kwargs: connection,
    )

    candidate = MySQLResumeSourceAdapter(_source_config()).load_resume_candidate(70001)

    assert candidate.source_record_id == 17
    assert candidate.resume_id == 70001
    sql, params = cursor.executed[0]
    assert "background" not in sql
    assert "WHERE resume_id = %s" in sql
    assert params == (70001,)


def test_missing_profile_rolls_back_and_closes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = ScriptedCursor({"FROM resume ": [None]})
    connection = FakeConnection(cursor)
    monkeypatch.setattr(
        "resume_index.source.pymysql.connect",
        lambda **_kwargs: connection,
    )

    with pytest.raises(ResumeNotFoundError, match="不存在或已删除"):
        MySQLResumeSourceAdapter(_source_config()).load_resume(70001)

    assert connection.rollback_count == 1
    assert connection.closed is True


def test_poll_candidate_scan_keeps_physical_id_and_resume_id_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    when = datetime(2026, 9, 2, 10, 0, 0)  # noqa: DTZ001
    cursor = ScriptedCursor(
        {
            "FROM resume ": [
                [
                    {
                        "id": 17,
                        "internal_person_id": 70001,
                        "updated_at": when,
                        "lifecycle_deleted_at": None,
                    }
                ]
            ]
        }
    )
    monkeypatch.setattr(
        "resume_index.source.pymysql.connect",
        lambda **_kwargs: FakeConnection(cursor),
    )
    adapter = MySQLResumeSourceAdapter(_source_config())

    candidates = adapter.load_source_change_candidates(
        "resume",
        after_cursor=SourceScanCursor(datetime(1970, 1, 1), 0),  # noqa: DTZ001
        upper_bound=SourceScanCursor(when, 17),
        limit=100,
    )

    assert len(candidates) == 1
    assert candidates[0].record_id == 17
    assert candidates[0].expert_id == 70001
    assert candidates[0].source_table == "resume"
    sql, params = cursor.executed[-1]
    assert "ORDER BY updated_at ASC, id ASC" in sql
    assert params is not None and params[-1] == 100


def test_poll_profile_delete_reads_by_physical_id_and_returns_empty_person_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    when = datetime(2026, 9, 2, 10, 0, 0)  # noqa: DTZ001
    profile = _profile_row()
    profile["physical_source_id"] = 17
    profile["lifecycle_deleted_at"] = when
    cursor = ScriptedCursor({"FROM resume ": [profile]})
    connection = FakeConnection(cursor)
    monkeypatch.setattr(
        "resume_index.source.pymysql.connect",
        lambda **_kwargs: connection,
    )
    scope = impact_scope_from_target(
        target_type="expert",
        expert_id=70001,
        dataset_spec=RESUME_SPEC,
    )

    facts = MySQLResumeSourceAdapter(_source_config()).read_current_facts(
        CurrentFactReadRequest(
            impact_scope=scope,
            trigger_source_table="resume",
            trigger_source_field_id="17",
            trigger_source_updated_at=when,
        )
    )

    assert facts.expert.id == 70001
    assert facts.expert_deleted is True
    assert not facts.employments and not facts.projects and not facts.educations
    assert cursor.executed[-1][1] == (17,)
    assert connection.rollback_count == 1


def test_poll_deleted_employment_returns_trigger_and_parent_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    when = datetime(2026, 9, 2, 10, 0, 0)  # noqa: DTZ001
    employment = {
        "id": 81001,
        "internal_person_id": 70001,
        "company_name": "示例科技有限公司",
        "position_title": "项目经理",
        "start_date": when,
        "end_date": None,
        "work_description": "工作职责：负责知识库建设。",
        "is_employed": 1,
        "business_updated_at": when,
        "scan_updated_at": when,
        "created_at": when,
        "deleted_at": when,
    }
    cursor = ScriptedCursor(
        {
            "FROM resume_work_experience": [employment],
            "FROM resume ": [_profile_row()],
        }
    )
    monkeypatch.setattr(
        "resume_index.source.pymysql.connect",
        lambda **_kwargs: FakeConnection(cursor),
    )
    scope = impact_scope_from_target(
        target_type="employment",
        expert_id=70001,
        source_field_id="81001",
        dataset_spec=RESUME_SPEC,
    )

    facts = MySQLResumeSourceAdapter(_source_config()).read_current_facts(
        CurrentFactReadRequest(
            impact_scope=scope,
            trigger_source_table="resume_work_experience",
            trigger_source_field_id="81001",
            trigger_source_updated_at=when,
        )
    )

    assert facts.employments[0].id == 81001
    assert facts.employments[0].deleted_at == when
    assert facts.expert.id == 70001


def test_adapter_rejects_non_resume_spec() -> None:
    from expert_index.dataset import EXPERT_SPEC

    with pytest.raises(ValueError, match="只能使用冻结的 RESUME_SPEC"):
        MySQLResumeSourceAdapter(_source_config(), dataset_spec=EXPERT_SPEC)


def test_adapter_rejects_index_database_role() -> None:
    index_config = DatabaseConfig(
        host="127.0.0.1",
        port=3306,
        user="resume_writer",
        password="test-only",
        database="resume_index_test",
        role=DatabaseRole.INDEX,
    )

    with pytest.raises(DatabaseBoundaryError, match="SOURCE 连接"):
        MySQLResumeSourceAdapter(index_config)
