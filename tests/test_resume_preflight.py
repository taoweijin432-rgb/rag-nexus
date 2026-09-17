"""Resume 独立只读预检和 CLI 范围测试。"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import pytest

from resume_index.cli import build_parser, main
from resume_index.config import ResumeSettings
from resume_index.pipeline import ResumeBuildResult, ResumeVectorSyncSummary
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
        raise ResumeSourceSchemaError("缺少表 resume")

    monkeypatch.setattr("resume_index.preflight.require_resume_source_schema", reject_schema)

    with pytest.raises(ResumeSourceSchemaError, match="resume"):
        run_resume_source_preflight(
            _settings(),
            connection_factory=_ConnectionFactory(connection),
        )

    assert connection.closed


def test_resume_cli_exposes_only_resume_commands() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparsers.choices) == {
        "preflight-source",
        "init-db",
        "init-checkpoints",
        "build",
        "build-batch",
        "incremental-once",
        "incremental-worker",
        "resume-batch",
        "retry-batch",
    }


def test_resume_cli_exposes_full_build_worker_count() -> None:
    """新建和续跑全量批次都应显式暴露页内并发度。"""

    parser = build_parser()
    build_args = parser.parse_args(
        [
            "build-batch",
            "--data-version",
            "resume-v1",
            "--worker-count",
            "4",
        ]
    )
    resume_args = parser.parse_args(
        [
            "resume-batch",
            "--batch-id",
            "2",
            "--worker-count",
            "3",
        ]
    )

    assert build_args.worker_count == 4
    assert resume_args.worker_count == 3


def test_resume_cli_exposes_incremental_source_worker_count() -> None:
    """Resume Poll 单轮和常驻入口都应显式暴露来源表级并发度。"""

    parser = build_parser()
    once_args = parser.parse_args(
        [
            "incremental-once",
            "--worker-id",
            "resume-worker",
            "--cycle-id",
            "cycle-1",
            "--max-attempts",
            "4",
            "--retry-delay-seconds",
            "30",
            "--task-lease-seconds",
            "300",
            "--worker-count",
            "2",
        ]
    )
    worker_args = parser.parse_args(
        [
            "incremental-worker",
            "--worker-id",
            "resume-worker",
            "--max-attempts",
            "4",
            "--retry-delay-seconds",
            "30",
            "--task-lease-seconds",
            "300",
            "--worker-count",
            "4",
        ]
    )

    assert once_args.worker_count == 2
    assert worker_args.worker_count == 4


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
        source_tables=("resume",),
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
        "source_tables": ["resume"],
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


def test_cli_build_uses_resume_id_and_prints_resume_semantics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings()
    captured: dict[str, object] = {}

    class FakePipeline:
        def build_single_resume(
            self,
            resume_id: int,
            *,
            data_version: str,
            batch_no: str | None,
        ) -> ResumeBuildResult:
            captured.update(
                resume_id=resume_id,
                data_version=data_version,
                batch_no=batch_no,
            )
            return ResumeBuildResult(
                batch_id=501,
                batch_no=batch_no or "generated",
                resume_id=resume_id,
                extracted_count=4,
                stage_count=0,
                chunk_count=3,
                partial_chunk_count=0,
                vector_sync=ResumeVectorSyncSummary(
                    resume_count=1,
                    current_count=3,
                    eligible_count=3,
                    embedded_count=3,
                    metadata_updated_count=0,
                    unchanged_count=0,
                    deleted_count=0,
                    delete_not_found_count=0,
                ),
            )

    monkeypatch.setattr("resume_index.cli.ResumeSettings.from_env", lambda: settings)
    monkeypatch.setattr(
        "resume_index.cli._build_runtime",
        lambda received: SimpleNamespace(
            pipeline=FakePipeline(),
            settings=received,
        ),
    )

    main(
        [
            "build",
            "--resume-id",
            "70001",
            "--data-version",
            "resume-v1",
            "--batch-no",
            "resume-single-test",
        ]
    )

    assert captured == {
        "resume_id": 70001,
        "data_version": "resume-v1",
        "batch_no": "resume-single-test",
    }
    output = json.loads(capsys.readouterr().out)
    assert output["resume_id"] == 70001
    assert output["vector_sync"]["resume_count"] == 1
    assert "expert_id" not in output


def test_cli_incremental_once_uses_resume_poll_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings()
    captured: dict[str, object] = {}

    class FakeResult:
        @staticmethod
        def to_summary() -> dict[str, object]:
            return {"cycle_id": "resume-cycle-1", "success_count": 2}

    class FakeRuntime:
        def run_once(self, worker_id: str, *, cycle_id: str) -> FakeResult:
            captured.update(worker_id=worker_id, cycle_id=cycle_id)
            return FakeResult()

    def build_poll_runtime(received: ResumeSettings, **kwargs: object) -> FakeRuntime:
        captured.update(settings=received, **kwargs)
        return FakeRuntime()

    monkeypatch.setattr("resume_index.cli.ResumeSettings.from_env", lambda: settings)
    monkeypatch.setattr("resume_index.cli._build_poll_runtime", build_poll_runtime)

    main(
        [
            "incremental-once",
            "--worker-id",
            "resume-worker-1",
            "--cycle-id",
            "resume-cycle-1",
            "--max-attempts",
            "3",
            "--retry-delay-seconds",
            "30",
            "--task-lease-seconds",
            "120",
            "--worker-count",
            "2",
        ]
    )

    assert captured == {
        "settings": settings,
        "max_attempts": 3,
        "retry_delay_seconds": 30,
        "task_lease_seconds": 120,
        "source_worker_count": 2,
        "worker_id": "resume-worker-1",
        "cycle_id": "resume-cycle-1",
    }
    assert json.loads(capsys.readouterr().out) == {
        "cycle_id": "resume-cycle-1",
        "success_count": 2,
    }
