"""在隔离的本地 Index Schema 验证来源扫描 lease 与批次恢复语义。"""

from __future__ import annotations

import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import pymysql

from expert_index.config import DatabaseConfig, DatabaseRole, Settings
from expert_index.models import BatchSourceScan, SourceCheckpoint, SourceScanCursor
from expert_index.repositories import (
    BatchSourceRecoveryConflictError,
    MySQLSourceStateRepository,
    SourceLeaseConflictError,
)

# 同时支持直接脚本和 python -m scripts.<入口>，不改变 SQL 解析规则。
if __package__:
    from .sql_files import load_sql_statements
else:
    from sql_files import load_sql_statements

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCHEMA = "expert_demo_p3_lease_verify"
CONTROL_MIGRATION = PROJECT_ROOT / "sql/012_complete_incremental_control_tables.sql"
LOOKUP_INDEX_MIGRATION = PROJECT_ROOT / "sql/013_add_incremental_scan_lookup_index.sql"


def _execute_file(cursor: Any, path: Path) -> None:
    """在当前临时 Schema 中按顺序执行一份迁移文件。"""

    for statement in load_sql_statements(path):
        cursor.execute(statement)


def _temporary_config(base: DatabaseConfig) -> DatabaseConfig:
    """派生仅指向固定临时 Schema 的 Index Repository 配置。"""

    return DatabaseConfig(
        host=base.host,
        port=base.port,
        user=base.user,
        password=base.password,
        database=TEST_SCHEMA,
        role=DatabaseRole.INDEX,
    )


def _assert_local_writable_server(cursor: Any) -> dict[str, Any]:
    """拒绝在远程或只读 MySQL 上创建验证 Schema。"""

    cursor.execute(
        "SELECT @@hostname AS hostname, @@read_only AS read_only, VERSION() AS version"
    )
    server = cursor.fetchone()
    local_hostnames = {socket.gethostname().lower(), socket.getfqdn().lower()}
    if str(server["hostname"]).lower() not in local_hostnames:
        raise RuntimeError("MySQL 服务端不是本机，拒绝创建 P3 临时测试 Schema")
    if int(server["read_only"]) != 0:
        raise RuntimeError("本机 MySQL 当前为只读，不能执行 P3 临时 Schema 验证")
    return server


def _insert_representative_history(cursor: Any, cutoff: datetime) -> None:
    """写入已完成历史范围，使恢复查询执行计划具有代表性。"""

    upper_bound = datetime.fromisoformat("2026-08-24 08:00:00")
    rows = [
        (
            batch_id,
            "experts",
            cutoff,
            0,
            upper_bound,
            batch_id,
            upper_bound,
            batch_id,
            0,
            "success",
        )
        for batch_id in range(10_000, 11_000)
    ]
    cursor.executemany(
        """
        INSERT INTO expert_index_batch_source(
            batch_id, source_table,
            start_updated_at, start_id,
            cursor_updated_at, cursor_id,
            upper_bound_updated_at, upper_bound_id,
            checkpoint_lock_version, status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )


def _assert_recovery_query_plan(cursor: Any) -> str:
    """确认恢复查询可使用 P3.2 新增的来源状态索引。"""

    cursor.execute(
        """
        EXPLAIN
        SELECT id, batch_id
        FROM expert_index_batch_source
        WHERE source_table=%s
          AND status IN ('pending', 'running', 'failed')
        ORDER BY id ASC
        LIMIT 2
        """,
        ("experts",),
    )
    plan = cursor.fetchone()
    possible_keys = str(plan.get("possible_keys") or "")
    expected_index = "idx_expert_batch_source_claim"
    if expected_index not in possible_keys:
        raise AssertionError(f"恢复查询未识别预期索引: {plan}")
    return str(plan.get("key") or "NONE")


def _verify_repository(config: DatabaseConfig, admin_connection: Any) -> str:
    """验证认领、续租、过期接管、原批次恢复和冲突保护。"""

    repository = MySQLSourceStateRepository(config)
    cutoff = datetime.fromisoformat("2026-08-13 22:00:00")
    upper_bound = datetime.fromisoformat("2026-08-24 08:00:00")
    repository.initialize_source_checkpoint(
        SourceCheckpoint(
            source_table="experts",
            last_updated_at=cutoff,
            last_id=0,
        )
    )

    acquired = repository.acquire_source_lease(
        "experts", "p3-worker-1", lease_seconds=60
    )
    assert acquired.lease_owner == "p3-worker-1"
    assert acquired.last_updated_at == cutoff
    assert acquired.last_id == 0
    assert acquired.lock_version == 0
    renewed = repository.renew_source_lease(
        "experts", "p3-worker-1", lease_seconds=120
    )
    assert renewed.lease_owner == "p3-worker-1"

    expected_scan = BatchSourceScan(
        batch_id=901,
        source_table="experts",
        start_cursor=SourceScanCursor(cutoff, 0),
        current_cursor=SourceScanCursor(cutoff, 0),
        upper_bound=SourceScanCursor(upper_bound, 300),
        checkpoint_lock_version=0,
    )
    repository.create_batch_source_scan(expected_scan)
    assert (
        repository.get_recoverable_batch_source_scan("experts", "p3-worker-1")
        == expected_scan
    )

    with admin_connection.cursor() as cursor:
        cursor.execute(
            "UPDATE expert_index_source_checkpoint "
            "SET lease_until=TIMESTAMPADD(SECOND, -1, CURRENT_TIMESTAMP(6)) "
            "WHERE source_table='experts'"
        )
    for stale_operation in (
        lambda: repository.renew_source_lease(
            "experts", "p3-worker-1", lease_seconds=60
        ),
        lambda: repository.release_source_lease("experts", "p3-worker-1"),
    ):
        try:
            stale_operation()
        except SourceLeaseConflictError:
            pass
        else:
            raise AssertionError("过期 Worker 不应续租或释放来源 lease")

    takeover = repository.acquire_source_lease(
        "experts", "p3-worker-2", lease_seconds=120
    )
    assert takeover.lease_owner == "p3-worker-2"
    assert (
        repository.get_recoverable_batch_source_scan("experts", "p3-worker-2")
        == expected_scan
    )

    with admin_connection.cursor() as cursor:
        _insert_representative_history(cursor, cutoff)
        selected_key = _assert_recovery_query_plan(cursor)
        cursor.execute(
            """
            INSERT INTO expert_index_batch_source(
                batch_id, source_table,
                start_updated_at, start_id,
                cursor_updated_at, cursor_id,
                upper_bound_updated_at, upper_bound_id,
                checkpoint_lock_version, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'failed')
            """,
            (902, "experts", cutoff, 0, cutoff, 0, upper_bound, 301, 0),
        )
    try:
        repository.get_recoverable_batch_source_scan("experts", "p3-worker-2")
    except BatchSourceRecoveryConflictError:
        pass
    else:
        raise AssertionError("多个未完成范围必须触发 fail-closed 冲突")

    checkpoint = repository.get_source_checkpoint("experts")
    assert checkpoint is not None
    assert (checkpoint.last_updated_at, checkpoint.last_id, checkpoint.lock_version) == (
        cutoff,
        0,
        0,
    )
    return selected_key


def main() -> None:
    """创建固定临时 Schema，完成真实事务验证并在最后清理。"""

    base = Settings.from_env().index_db
    if base.host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("只允许连接本机 Index MySQL")

    connection = pymysql.connect(
        host=base.host,
        port=base.port,
        user=base.user,
        password=base.password,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
    created = False
    try:
        with connection.cursor() as cursor:
            server = _assert_local_writable_server(cursor)
            cursor.execute(
                "SELECT COUNT(*) AS count FROM information_schema.SCHEMATA "
                "WHERE SCHEMA_NAME=%s",
                (TEST_SCHEMA,),
            )
            if int(cursor.fetchone()["count"]) != 0:
                raise RuntimeError(f"临时 Schema 已存在，拒绝覆盖: {TEST_SCHEMA}")
            cursor.execute(
                f"CREATE DATABASE `{TEST_SCHEMA}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
            )
            created = True
            cursor.execute(f"USE `{TEST_SCHEMA}`")
            _execute_file(cursor, CONTROL_MIGRATION)
            _execute_file(cursor, LOOKUP_INDEX_MIGRATION)
            selected_key = _verify_repository(_temporary_config(base), connection)
            print(
                "P3 lease 验证通过: "
                f"MySQL={server['version']}, recovery_key={selected_key}, "
                "checkpoint_unchanged=true"
            )
    finally:
        try:
            if created:
                with connection.cursor() as cursor:
                    cursor.execute(f"DROP DATABASE `{TEST_SCHEMA}`")
        finally:
            connection.close()


if __name__ == "__main__":
    main()
