"""Resume 独立只读预检和 CLI 范围测试。"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytest

from resume_index.cli import build_parser, main
from resume_index.config import ResumeSettings
from resume_index.preflight import run_resume_source_preflight
from resume_index.schema_initializer import ResumeSchemaInitializationResult
from resume_index.source_schema import ResumeSourceSchemaError, ResumeSourceSchemaReport


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


class _Connection:
    def __init__(self) -> None:
        self.closed = False

    def cursor(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _ConnectionFactory:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _Connection:
        self.calls.append(kwargs)
        return self.connection


def test_preflight_uses_resume_source_and_returns_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()
    factory = _ConnectionFactory(connection)
    captured: dict[str, object] = {}

    def fake_require_schema(
        received_connection: object,
        *,
        schema_name: str,
    ) -> ResumeSourceSchemaReport:
        captured["connection"] = received_connection
        captured["schema_name"] = schema_name
        return ResumeSourceSchemaReport(schema_name=schema_name, issues=())

    monkeypatch.setattr("resume_index.preflight.require_resume_source_schema", fake_require_schema)

    result = run_resume_source_preflight(_settings(), connection_factory=factory)

    assert captured == {"connection": connection, "schema_name": "shared_source"}
    assert factory.calls[0]["database"] == "shared_source"
    assert factory.calls[0]["autocommit"] is True
    assert result.source_schema == "shared_source"
    assert result.index_schema == "resume_index"
    assert result.milvus_collection_name == "resume_candidates_v15_contract"
    assert connection.closed


def test_preflight_closes_connection_when_schema_is_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()

    def reject_schema(*_args: object, **_kwargs: object) -> None:
        raise ResumeSourceSchemaError("缺少表 resume_profile")

    monkeypatch.setattr("resume_index.preflight.require_resume_source_schema", reject_schema)

    with pytest.raises(ResumeSourceSchemaError, match="resume_profile"):
        run_resume_source_preflight(
            _settings(),
            connection_factory=_ConnectionFactory(connection),
        )

    assert connection.closed


def test_resume_cli_exposes_only_current_safe_commands() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparsers.choices) == {"preflight-source", "init-db"}


def test_cli_prints_safe_json_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings()
    monkeypatch.setattr("resume_index.cli.ResumeSettings.from_env", lambda: settings)

    # 使用真实结果对象可以锁定 CLI 字段，同时避免连接任何服务。
    from resume_index.preflight import ResumeSourcePreflightResult

    expected = ResumeSourcePreflightResult(
        source_schema="shared_source",
        source_tables=("resume_profile",),
        index_schema="resume_index",
        milvus_collection_name="resume_candidates_v15_contract",
    )
    monkeypatch.setattr("resume_index.cli.run_resume_source_preflight", lambda _settings: expected)

    main(["preflight-source"])

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "index_schema": "resume_index",
        "milvus_collection_name": "resume_candidates_v15_contract",
        "source_schema": "shared_source",
        "source_tables": ["resume_profile"],
    }


def test_cli_init_db_uses_resume_initializer_and_prints_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings()
    captured: dict[str, object] = {}

    class FakeInitializer:
        def __init__(self, received_settings: ResumeSettings) -> None:
            captured["settings"] = received_settings

        def initialize_schema(self) -> ResumeSchemaInitializationResult:
            captured["initialized"] = True
            return ResumeSchemaInitializationResult(
                schema_name="resume_index",
                statements_executed=7,
            )

    monkeypatch.setattr("resume_index.cli.ResumeSettings.from_env", lambda: settings)
    monkeypatch.setattr("resume_index.cli.MySQLResumeSchemaInitializer", FakeInitializer)

    main(["init-db"])

    assert captured == {"settings": settings, "initialized": True}
    assert json.loads(capsys.readouterr().out) == {
        "schema_name": "resume_index",
        "statements_executed": 7,
    }
