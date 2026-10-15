"""Resume Index 初始化器的 Fake 数据库边界测试。"""

from __future__ import annotations

from typing import Any, Self

import pytest

from resume_index.config import ResumeSettings
from resume_index.schema_initializer import MySQLResumeSchemaInitializer


def _settings() -> ResumeSettings:
    return ResumeSettings.from_env(
        {
            "RESUME_SOURCE_DB_HOST": "source.mysql.local",
            "RESUME_SOURCE_DB_PORT": "3306",
            "RESUME_SOURCE_DB_NAME": "shared_source",
            "RESUME_SOURCE_DB_USER": "resume_reader",
            "RESUME_SOURCE_DB_PASSWORD": "source-secret",
            "RESUME_INDEX_DB_HOST": "127.0.0.1",
            "RESUME_INDEX_DB_PORT": "3307",
            "RESUME_INDEX_DB_NAME": "resume_index",
            "RESUME_INDEX_DB_USER": "resume_writer",
            "RESUME_INDEX_DB_PASSWORD": "index-secret",
            "RESUME_MILVUS_COLLECTION_NAME": "resume_candidates_v15_contract",
            "SOURCE_DB_HOST": "source.mysql.local",
            "SOURCE_DB_PORT": "3306",
            "SOURCE_DB_NAME": "shared_source",
            "INDEX_DB_HOST": "127.0.0.1",
            "INDEX_DB_PORT": "3306",
            "INDEX_DB_NAME": "expert_index",
            "MILVUS_COLLECTION_NAME": "expert_candidates_v15_contract",
        }
    )


class _Cursor:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.statements: list[str] = []
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str) -> None:
        self.statements.append(sql)
        if self.fail_at == len(self.statements):
            raise RuntimeError("fake ddl failure")

    def close(self) -> None:
        self.closed = True


class _Connection:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.cursor_instance = _Cursor(fail_at=fail_at)
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> _Cursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


class _ConnectionFactory:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _Connection:
        self.calls.append(kwargs)
        return self.connection


def test_initializer_executes_only_against_resume_index() -> None:
    connection = _Connection()
    factory = _ConnectionFactory(connection)

    result = MySQLResumeSchemaInitializer(
        _settings(),
        connection_factory=factory,
    ).initialize_schema()

    assert factory.calls[0]["database"] == "resume_index"
    assert factory.calls[0]["autocommit"] is False
    assert result.schema_name == "resume_index"
    assert result.statements_executed == 11
    assert len(connection.cursor_instance.statements) == 11
    assert all(
        "CREATE TABLE IF NOT EXISTS expert_index_" in statement
        or statement.startswith("INSERT INTO expert_index_batch_id_sequence")
        for statement in connection.cursor_instance.statements
    )
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.cursor_instance.closed
    assert connection.closed


def test_initializer_rolls_back_and_closes_on_failure() -> None:
    connection = _Connection(fail_at=2)

    with pytest.raises(RuntimeError, match="fake ddl failure"):
        MySQLResumeSchemaInitializer(
            _settings(),
            connection_factory=_ConnectionFactory(connection),
        ).initialize_schema()

    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.cursor_instance.closed
    assert connection.closed


def test_initializer_rejects_non_resume_settings_before_connect() -> None:
    connection = _Connection()
    factory = _ConnectionFactory(connection)

    with pytest.raises(TypeError, match="ResumeSettings"):
        MySQLResumeSchemaInitializer(object(), connection_factory=factory)  # type: ignore[arg-type]

    assert factory.calls == []
