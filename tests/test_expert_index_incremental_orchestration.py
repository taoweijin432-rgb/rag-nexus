"""P3.8 四表候选扫描协调与正式配置边界测试。"""

from __future__ import annotations

import argparse

import pytest

from expert_index.cli import build_parser
from expert_index.config import (
    DatabaseBoundaryError,
    DatabaseConfig,
    DatabaseRole,
    IncrementalScanConfig,
    Settings,
)
from expert_index.scanner import (
    INCREMENTAL_SOURCE_TABLE_ORDER,
    IncrementalScanCoordinator,
    IncrementalScanDisabledError,
    IncrementalScanResult,
    build_incremental_scan_coordinator,
)


def _database(
    database: str,
    role: DatabaseRole,
    *,
    host: str = "127.0.0.1",
) -> DatabaseConfig:
    """构造不建立真实连接的角色化数据库配置。"""

    return DatabaseConfig(host, 3306, "test", "test", database, role=role)


def _environment() -> dict[str, str]:
    """返回 Source 与 Index 分离的最小正式环境变量。"""

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


class FakeSourceScanService:
    """记录逐表调用，并让指定来源表抛出不应外泄文本的异常。"""

    def __init__(self, failed_source_table: str | None = None) -> None:
        self.failed_source_table = failed_source_table
        self.calls: list[tuple[str, str, str]] = []

    def scan_source(
        self,
        source_table: str,
        lease_owner: str,
        *,
        batch_no: str,
    ) -> IncrementalScanResult:
        """返回零范围结果，或模拟一个来源表的隔离失败。"""

        self.calls.append((source_table, lease_owner, batch_no))
        if source_table == self.failed_source_table:
            raise RuntimeError("sensitive-source-row-must-not-leak")
        return IncrementalScanResult(
            source_table=source_table,
            batch_source=None,
            recovered=False,
            page_count=0,
            scheduled_candidate_count=0,
            lease_released=True,
        )


def test_incremental_scan_config_defaults_to_disabled_candidate_scanning() -> None:
    """缺省配置必须关闭增量候选写入并采用冻结的分页与租约初值。"""

    settings = Settings.from_env(_environment())

    assert settings.incremental_scan == IncrementalScanConfig(
        enabled=False,
        batch_size=500,
        lease_seconds=120,
    )


def test_incremental_scan_config_reads_explicit_safe_values() -> None:
    """显式环境变量应完整映射到候选扫描配置。"""

    environment = _environment()
    environment.update(
        {
            "INCREMENTAL_ENABLED": "true",
            "INCREMENTAL_BATCH_SIZE": "37",
            "INCREMENTAL_LEASE_SECONDS": "45",
        }
    )

    settings = Settings.from_env(environment)

    assert settings.incremental_scan.enabled is True
    assert settings.incremental_scan.batch_size == 37
    assert settings.incremental_scan.lease_seconds == 45


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("INCREMENTAL_ENABLED", "maybe"),
        ("INCREMENTAL_BATCH_SIZE", "0"),
        ("INCREMENTAL_BATCH_SIZE", "1001"),
        ("INCREMENTAL_LEASE_SECONDS", "0"),
        ("INCREMENTAL_LEASE_SECONDS", "seconds"),
    ],
)
def test_incremental_scan_config_rejects_ambiguous_or_unsafe_values(
    name: str,
    value: str,
) -> None:
    """非法开关、页大小和租约必须在建立数据库连接前失败。"""

    environment = _environment()
    environment[name] = value

    with pytest.raises(ValueError, match=name):
        Settings.from_env(environment)


def test_coordinator_scans_fixed_order_and_isolates_one_source_failure() -> None:
    """一个来源表失败不能阻断后续表，也不能在结果中泄露原始错误文本。"""

    service = FakeSourceScanService("experts_work_experience")
    coordinator = IncrementalScanCoordinator(scan_service=service)  # type: ignore[arg-type]

    result = coordinator.scan_all_sources(
        "worker-p3-8-attempt-001",
        cycle_id="20260825T120000Z-001",
    )

    assert [call[0] for call in service.calls] == list(INCREMENTAL_SOURCE_TABLE_ORDER)
    assert [call[1] for call in service.calls] == [
        "worker-p3-8-attempt-001.20260825T120000Z-001.experts",
        "worker-p3-8-attempt-001.20260825T120000Z-001.work",
        "worker-p3-8-attempt-001.20260825T120000Z-001.project",
        "worker-p3-8-attempt-001.20260825T120000Z-001.education",
    ]
    assert [call[2] for call in service.calls] == [
        "inc-20260825T120000Z-001-experts",
        "inc-20260825T120000Z-001-work",
        "inc-20260825T120000Z-001-project",
        "inc-20260825T120000Z-001-education",
    ]
    assert [outcome.source_table for outcome in result.outcomes] == list(
        INCREMENTAL_SOURCE_TABLE_ORDER
    )
    assert result.succeeded_source_count == 3
    assert result.failed_source_count == 1
    failed_outcome = result.outcomes[1]
    assert failed_outcome.succeeded is False
    assert failed_outcome.failure_type == "RuntimeError"
    assert "sensitive-source-row" not in repr(result)


@pytest.mark.parametrize(
    ("worker_id", "cycle_id"),
    [
        ("worker with spaces", "cycle-001"),
        ("worker-001", "cycle/001"),
        ("worker-001", "x" * 49),
    ],
)
def test_coordinator_rejects_invalid_identifiers_before_partial_scan(
    worker_id: str,
    cycle_id: str,
) -> None:
    """非法编排标识不能在四表周期中形成部分数据库状态。"""

    service = FakeSourceScanService()
    coordinator = IncrementalScanCoordinator(scan_service=service)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        coordinator.scan_all_sources(worker_id, cycle_id=cycle_id)

    assert service.calls == []


def test_formal_factory_stays_disabled_without_explicit_switch() -> None:
    """默认配置不能构造会创建 pending 任务的正式扫描入口。"""

    settings = Settings(
        source_db=_database("expert_source", DatabaseRole.SOURCE),
        index_db=_database("expert_index", DatabaseRole.INDEX),
    )

    with pytest.raises(IncrementalScanDisabledError, match="保持禁用"):
        build_incremental_scan_coordinator(settings)


def test_formal_factory_assembles_read_source_and_index_state_components() -> None:
    """启用后工厂应只组装 Source 读取和 Index 状态写入组件。"""

    source_db = _database("expert_source", DatabaseRole.SOURCE)
    index_db = _database("expert_index", DatabaseRole.INDEX)
    settings = Settings(
        source_db=source_db,
        index_db=index_db,
        incremental_scan=IncrementalScanConfig(
            enabled=True,
            batch_size=37,
            lease_seconds=45,
        ),
    )

    coordinator = build_incremental_scan_coordinator(settings)
    service = coordinator.scan_service

    assert service.source_adapter.config is source_db
    assert service.source_state_repository.config is index_db
    assert service.task_repository.config is index_db
    assert service.page_size == 37
    assert service.lease_seconds == 45
    assert service.policy_version == settings.chunk_policy_version


def test_formal_factory_rejects_source_index_same_target_before_assembly() -> None:
    """即使显式启用，扫描入口也不能把本地状态写入来源 Schema。"""

    settings = Settings(
        source_db=_database("expert", DatabaseRole.SOURCE, host="localhost"),
        index_db=_database("EXPERT", DatabaseRole.INDEX),
        incremental_scan=IncrementalScanConfig(enabled=True),
    )

    with pytest.raises(DatabaseBoundaryError, match="同一 MySQL Schema"):
        build_incremental_scan_coordinator(settings)


def test_coordinator_does_not_expose_task_processing_or_checkpoint_commit() -> None:
    """P3 候选协调器不能提前提供 P5 任务处理或自动水位提交方法。"""

    public_names = {
        name for name in dir(IncrementalScanCoordinator) if not name.startswith("_")
    }

    assert public_names == {"scan_all_sources"}


def test_formal_cli_does_not_expose_incremental_scan_before_p5() -> None:
    """可靠任务回执接入前，正式 CLI 不能开放候选扫描或自动提交命令。"""

    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert "scan-incremental" not in subparsers.choices
    assert "run-incremental" not in subparsers.choices
