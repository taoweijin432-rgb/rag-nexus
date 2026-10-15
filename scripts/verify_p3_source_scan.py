"""在隔离本地 Index Schema 验证 P3.5 固定范围任务调度和临时游标。"""

from __future__ import annotations

import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import pymysql

from expert_index.config import DatabaseConfig, DatabaseRole, Settings
from expert_index.models import SourceChangeCandidate, SourceCheckpoint, SourceScanCursor
from expert_index.repositories import (
    MySQLSourceStateRepository,
    MySQLSourceSyncTaskRepository,
)
from expert_index.scanner import IncrementalSourceScanService

# 同时支持直接脚本和 python -m scripts.<入口>，不改变 SQL 解析规则。
if __package__:
    from .sql_files import load_sql_statements
else:
    from sql_files import load_sql_statements

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCHEMA = "expert_demo_p3_5_scan_verify"
BASE_SCHEMA = PROJECT_ROOT / "sql/001_create_expert_index_tables.sql"


def _temporary_config(base: DatabaseConfig) -> DatabaseConfig:
    """派生仅指向固定临时 Schema 的 Index 配置。"""

    return DatabaseConfig(
        host=base.host,
        port=base.port,
        user=base.user,
        password=base.password,
        database=TEST_SCHEMA,
        role=DatabaseRole.INDEX,
    )


class FixedSourceAdapter:
    """提供固定上界和候选页，不连接或写入来源数据库。"""

    def __init__(
        self,
        upper_bound: SourceScanCursor,
        candidates: list[SourceChangeCandidate],
    ) -> None:
        self.upper_bound = upper_bound
        self.pages = [candidates, []]

    def get_source_scan_upper_bound(self, source_table: str) -> SourceScanCursor:
        """返回测试冻结的来源上界。"""

        assert source_table == "experts_work_experience"
        return self.upper_bound

    def load_source_change_candidates(
        self,
        source_table: str,
        *,
        after_cursor: SourceScanCursor,
        upper_bound: SourceScanCursor,
        limit: int,
    ) -> list[SourceChangeCandidate]:
        """按调用顺序返回候选页和空尾页。"""

        assert source_table == "experts_work_experience"
        assert after_cursor < upper_bound == self.upper_bound
        assert limit == 2
        return self.pages.pop(0)


def _assert_local_writable_server(cursor: Any) -> dict[str, Any]:
    """拒绝在远程或只读 MySQL 创建临时验证 Schema。"""

    cursor.execute(
        "SELECT @@hostname AS hostname, @@read_only AS read_only, VERSION() AS version"
    )
    server = cursor.fetchone()
    local_hostnames = {socket.gethostname().lower(), socket.getfqdn().lower()}
    if str(server["hostname"]).lower() not in local_hostnames:
        raise RuntimeError("MySQL 服务端不是本机，拒绝创建 P3.5 临时 Schema")
    if int(server["read_only"]) != 0:
        raise RuntimeError("本机 MySQL 当前为只读，不能执行 P3.5 验证")
    return server


def _verify_scan(config: DatabaseConfig, admin_connection: Any) -> None:
    """运行一页候选加空尾页，并核对任务、临时游标和安全水位。"""

    start = SourceScanCursor(datetime.fromisoformat("2026-08-13 22:00:00"), 0)
    first_time = datetime.fromisoformat("2026-08-24 08:00:00")
    second_time = datetime.fromisoformat("2026-08-24 08:00:01")
    upper_bound = SourceScanCursor(
        datetime.fromisoformat("2026-08-24 09:00:00"),
        999,
    )
    candidates = [
        SourceChangeCandidate(
            source_table="experts_work_experience",
            record_id=901,
            expert_id=101,
            updated_at=first_time,
            deleted_at=None,
        ),
        SourceChangeCandidate(
            source_table="experts_work_experience",
            record_id=902,
            expert_id=101,
            updated_at=second_time,
            deleted_at=None,
        ),
    ]
    state_repository = MySQLSourceStateRepository(config)
    task_repository = MySQLSourceSyncTaskRepository(config)
    state_repository.initialize_source_checkpoint(
        SourceCheckpoint(
            source_table="experts_work_experience",
            last_updated_at=start.updated_at,
            last_id=start.record_id,
        )
    )
    service = IncrementalSourceScanService(
        source_adapter=FixedSourceAdapter(  # type: ignore[arg-type]
            upper_bound,
            candidates,
        ),
        source_state_repository=state_repository,
        task_repository=task_repository,
        policy_version="1.3",
        page_size=2,
        lease_seconds=120,
    )

    result = service.scan_source(
        "experts_work_experience",
        "p3-5-mysql-worker-1",
        batch_no="p3-5-mysql-incremental-work-1",
    )
    assert result.batch_source is not None
    assert result.batch_source.start_cursor == start
    assert result.batch_source.current_cursor == upper_bound
    assert result.batch_source.candidate_count == 2
    assert result.scheduled_candidate_count == 2
    assert result.lease_released is False

    with admin_connection.cursor() as cursor:
        cursor.execute(
            "SELECT expert_index_batch.run_mode AS run_mode, "
            "expert_index_batch.status AS status, "
            "expert_index_batch_source.candidate_count AS candidate_count, "
            "expert_index_batch_source.cursor_updated_at AS cursor_updated_at, "
            "expert_index_batch_source.cursor_id AS cursor_id, "
            "expert_index_batch_source.upper_bound_updated_at AS upper_bound_updated_at, "
            "expert_index_batch_source.upper_bound_id AS upper_bound_id "
            "FROM expert_index_batch JOIN expert_index_batch_source "
            "ON expert_index_batch.id=expert_index_batch_source.batch_id"
        )
        batch = cursor.fetchone()
        assert batch["run_mode"] == "incremental"
        assert batch["status"] == "running"
        assert int(batch["candidate_count"]) == 2
        assert batch["cursor_updated_at"] == upper_bound.updated_at
        assert int(batch["cursor_id"]) == upper_bound.record_id
        assert batch["upper_bound_updated_at"] == upper_bound.updated_at
        assert int(batch["upper_bound_id"]) == upper_bound.record_id

        cursor.execute(
            "SELECT COUNT(*) AS count FROM expert_index_sync_task "
            "WHERE source_table='experts_work_experience' "
            "AND processing_stage='candidate' AND sync_status='pending'"
        )
        assert int(cursor.fetchone()["count"]) == 2

        cursor.execute(
            "SELECT last_updated_at, last_id, lock_version, lease_owner "
            "FROM expert_index_source_checkpoint "
            "WHERE source_table='experts_work_experience'"
        )
        checkpoint = cursor.fetchone()
        assert checkpoint["last_updated_at"] == start.updated_at
        assert int(checkpoint["last_id"]) == start.record_id
        assert int(checkpoint["lock_version"]) == 0
        assert checkpoint["lease_owner"] == "p3-5-mysql-worker-1"


def main() -> None:
    """创建固定临时 Schema，运行真实事务验证并在最后清理。"""

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
            for statement in load_sql_statements(BASE_SCHEMA):
                cursor.execute(statement)
            _verify_scan(_temporary_config(base), connection)
            print(
                "P3.5 固定范围验证通过: "
                f"MySQL={server['version']}, tasks=2, "
                "cursor_sealed=true, checkpoint_unchanged=true"
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
