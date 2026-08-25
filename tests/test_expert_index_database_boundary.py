"""正式入口、状态 Repository 和测试 Fixture 的数据库边界测试。"""

from __future__ import annotations

import argparse
from typing import Any, Self

import pytest

from expert_index.checkpoint import CheckpointManager
from expert_index.cli import build_parser
from expert_index.config import (
    DatabaseBoundaryError,
    DatabaseConfig,
    DatabaseRole,
    Settings,
)
from expert_index.pipeline import ExpertIndexPipeline
from expert_index.repositories import (
    MySQLBatchItemRepository,
    MySQLBatchRepository,
    MySQLBuildLockRepository,
    MySQLSourceStateRepository,
    MySQLSourceSyncTaskRepository,
)
from expert_index.schema_initializer import MySQLSchemaInitializer
from expert_index.test_fixture import ensure_complete_test_expert


def _config(
    database: str,
    role: DatabaseRole,
    *,
    host: str = "127.0.0.1",
) -> DatabaseConfig:
    """创建不连接真实数据库的角色化测试配置。"""

    return DatabaseConfig(host, 3306, "test", "test", database, role=role)


def _environment() -> dict[str, str]:
    """返回 Source 与 Index 物理分离的完整显式环境变量。"""

    return {
        "SOURCE_DB_HOST": "source.mysql.local",
        "SOURCE_DB_PORT": "3306",
        "SOURCE_DB_USER": "source_reader",
        "SOURCE_DB_PASSWORD": "source-secret",
        "SOURCE_DB_NAME": "expert_source",
        "INDEX_DB_HOST": "127.0.0.1",
        "INDEX_DB_PORT": "3307",
        "INDEX_DB_USER": "index_writer",
        "INDEX_DB_PASSWORD": "index-secret",
        "INDEX_DB_NAME": "expert_index_local",
    }


def test_formal_cli_does_not_expose_source_fixture_command() -> None:
    """正式命令集合不能提供任何写四张来源表的 Fixture 入口。"""

    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert "seed-test-expert" not in subparsers.choices


def test_settings_reads_index_only_from_explicit_index_variables() -> None:
    """正式配置为 Source/Index 标记角色，且不从通用 DB_* 推导状态库。"""

    settings = Settings.from_env(_environment())

    assert settings.source_db.role is DatabaseRole.SOURCE
    assert settings.index_db.role is DatabaseRole.INDEX
    assert settings.index_db.database == "expert_index_local"
    settings.validate_index_write_boundary()


@pytest.mark.parametrize(
    "missing_name",
    [
        "INDEX_DB_HOST",
        "INDEX_DB_PORT",
        "INDEX_DB_USER",
        "INDEX_DB_PASSWORD",
        "INDEX_DB_NAME",
    ],
)
def test_settings_rejects_missing_explicit_index_variable(missing_name: str) -> None:
    """任一 INDEX_DB_* 缺失都应在建立连接前停止正式入口。"""

    environment = _environment()
    environment.pop(missing_name)

    with pytest.raises(RuntimeError, match=missing_name):
        Settings.from_env(environment)


def test_index_write_boundary_rejects_same_schema_with_localhost_alias() -> None:
    """不同账号或本机地址写法不能绕过 Source/Index 同目标检查。"""

    settings = Settings(
        source_db=_config("expert", DatabaseRole.SOURCE, host="localhost"),
        index_db=_config("EXPERT", DatabaseRole.INDEX, host="127.0.0.1"),
    )

    with pytest.raises(DatabaseBoundaryError, match="同一 MySQL Schema"):
        settings.validate_index_write_boundary()


@pytest.mark.parametrize(
    "repository_type",
    [
        MySQLBatchRepository,
        MySQLBatchItemRepository,
        MySQLBuildLockRepository,
        MySQLSourceStateRepository,
        MySQLSourceSyncTaskRepository,
        CheckpointManager,
        MySQLSchemaInitializer,
    ],
)
def test_state_writer_rejects_source_role(repository_type: type[Any]) -> None:
    """所有状态写入组件都在构造阶段拒绝 Source 连接配置。"""

    with pytest.raises(DatabaseBoundaryError, match="INDEX_DB"):
        repository_type(_config("expert_source", DatabaseRole.SOURCE))


def test_pipeline_default_state_repositories_validate_target_before_connect() -> None:
    """使用真实 Repository 的 Pipeline 在连接数据库前拒绝同目标配置。"""

    settings = Settings(
        source_db=_config("expert", DatabaseRole.SOURCE),
        index_db=_config("expert", DatabaseRole.INDEX),
    )

    with pytest.raises(DatabaseBoundaryError, match="同一 MySQL Schema"):
        ExpertIndexPipeline(settings, vector_sync=object())  # type: ignore[arg-type]


class _FixtureCursor:
    """让每次 Fixture 查询都命中同一条虚构测试记录。"""

    lastrowid = 901

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, _sql: str, _params: tuple[object, ...] | None = None) -> None:
        return None

    def fetchone(self) -> dict[str, int]:
        return {"id": 901}


class _FixtureConnection:
    """记录隔离测试 Fixture 的事务收口，不执行 SQL。"""

    def __init__(self) -> None:
        self.cursor_instance = _FixtureCursor()
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> _FixtureCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "config",
    [
        _config("expert_source_test", DatabaseRole.SOURCE),
        _config("expert_source", DatabaseRole.ISOLATED_TEST),
    ],
)
def test_fixture_rejects_non_isolated_target_before_connect(
    monkeypatch: pytest.MonkeyPatch,
    config: DatabaseConfig,
) -> None:
    """测试角色和数据库名称任一不安全时都不能连接 MySQL。"""

    def unexpected_connect(**_kwargs: Any) -> None:
        raise AssertionError("边界校验失败时不应连接 MySQL")

    monkeypatch.setattr("expert_index.test_fixture.pymysql.connect", unexpected_connect)

    with pytest.raises(DatabaseBoundaryError):
        ensure_complete_test_expert(config)


def test_fixture_keeps_write_capability_for_explicit_isolated_test_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式隔离测试库仍可通过 Fake Connection 保留完整 Fixture 能力。"""

    connection = _FixtureConnection()
    monkeypatch.setattr(
        "expert_index.test_fixture.pymysql.connect",
        lambda **_kwargs: connection,
    )

    result = ensure_complete_test_expert(
        _config("expert_source_test", DatabaseRole.ISOLATED_TEST)
    )

    assert result.expert_id == 901
    assert result.work_experience_ids == (901, 901, 901)
    assert result.project_experience_ids == (901, 901, 901)
    assert result.education_experience_ids == (901, 901)
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.closed is True
