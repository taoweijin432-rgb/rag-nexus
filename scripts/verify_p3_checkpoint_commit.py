"""在隔离本地 Index Schema 验证 P3.7 原子安全 Checkpoint 提交。"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Any

import pymysql

from expert_index.checkpoint import CheckpointManager, SourceBatchNotReadyError
from expert_index.config import DatabaseConfig, DatabaseRole, Settings

# 同时支持直接脚本和 python -m scripts.<入口>，不改变 SQL 解析规则。
if __package__:
    from .sql_files import load_sql_statements
else:
    from sql_files import load_sql_statements

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCHEMA = "expert_demo_p3_7_checkpoint_verify"
BASE_SCHEMA = PROJECT_ROOT / "sql/001_create_expert_index_tables.sql"
START_TIME = "2026-08-13 22:00:00.000000"
UPPER_TIME = "2026-08-25 10:00:00.000000"


def _temporary_config(base: DatabaseConfig) -> DatabaseConfig:
    """派生只指向固定临时 Schema 的 Index 配置。"""

    return DatabaseConfig(
        host=base.host,
        port=base.port,
        user=base.user,
        password=base.password,
        database=TEST_SCHEMA,
        role=DatabaseRole.INDEX,
    )


def _assert_local_writable_server(cursor: Any) -> dict[str, Any]:
    """拒绝在远程或只读 MySQL 创建临时验证 Schema。"""

    cursor.execute(
        "SELECT @@hostname AS hostname, @@read_only AS read_only, VERSION() AS version"
    )
    server = cursor.fetchone()
    local_hostnames = {socket.gethostname().lower(), socket.getfqdn().lower()}
    if str(server["hostname"]).lower() not in local_hostnames:
        raise RuntimeError("MySQL 服务端不是本机，拒绝创建 P3.7 临时 Schema")
    if int(server["read_only"]) != 0:
        raise RuntimeError("本机 MySQL 当前为只读，不能执行 P3.7 验证")
    return server


def _insert_batch(
    connection: Any,
    *,
    batch_no: str,
    source_table: str,
    lease_owner: str,
    candidate_count: int,
    task_statuses: tuple[str, ...],
) -> int:
    """插入一个封口来源范围及其预置任务状态。"""

    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO expert_index_batch(batch_no, run_mode, status) "
            "VALUES (%s, 'incremental', 'running')",
            (batch_no,),
        )
        batch_id = int(cursor.lastrowid)
        cursor.execute(
            "INSERT INTO expert_index_source_checkpoint("
            "source_table, last_updated_at, last_id, lock_version, "
            "lease_owner, lease_until"
            ") VALUES (%s, %s, 0, 0, %s, "
            "TIMESTAMPADD(SECOND, 300, CURRENT_TIMESTAMP(6)))",
            (source_table, START_TIME, lease_owner),
        )
        cursor.execute(
            "INSERT INTO expert_index_batch_source("
            "batch_id, source_table, start_updated_at, start_id, "
            "cursor_updated_at, cursor_id, upper_bound_updated_at, "
            "upper_bound_id, checkpoint_lock_version, status, "
            "candidate_count, success_count, failed_count, skipped_count, "
            "started_at"
            ") VALUES (%s, %s, %s, 0, %s, 902, %s, 902, 0, "
            "'running', %s, 0, 0, 0, CURRENT_TIMESTAMP(6))",
            (
                batch_id,
                source_table,
                START_TIME,
                UPPER_TIME,
                UPPER_TIME,
                candidate_count,
            ),
        )
        for offset, status in enumerate(task_statuses, start=1):
            record_id = 900 + offset
            is_success = status == "success"
            cursor.execute(
                "INSERT INTO expert_index_sync_task("
                "dedupe_key, batch_id, event_source, source_table, "
                "source_record_id, source_updated_at, expert_id, "
                "operation_type, target_type, processing_stage, sync_status, "
                "retry_count, policy_version, result_code, completed_at"
                ") VALUES (%s, %s, 'poll', %s, %s, %s, 101, 'update', %s, "
                "%s, %s, 0, '1.3', %s, %s)",
                (
                    hashlib.sha256(
                        f"{batch_id}:{source_table}:{record_id}".encode()
                    ).hexdigest(),
                    batch_id,
                    source_table,
                    record_id,
                    UPPER_TIME,
                    _target_type(source_table),
                    "completed" if is_success else "candidate",
                    status,
                    "downstream_applied" if is_success else None,
                    UPPER_TIME if is_success else None,
                ),
            )
    connection.commit()
    return batch_id


def _target_type(source_table: str) -> str:
    """返回验证使用的来源表级最小影响类型。"""

    return {
        "experts_work_experience": "employment",
        "experts_project_experience": "project",
    }[source_table]


def _assert_success_state(connection: Any, *, batch_id: int) -> None:
    """核对 Checkpoint、Batch Source 和 Batch 已在同一终态收口。"""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT last_updated_at, last_id, last_batch_id, lock_version, "
            "lease_owner, lease_until, last_success_time "
            "FROM expert_index_source_checkpoint "
            "WHERE source_table='experts_work_experience'"
        )
        checkpoint = cursor.fetchone()
        assert str(checkpoint["last_updated_at"]) == "2026-08-25 10:00:00"
        assert int(checkpoint["last_id"]) == 902
        assert int(checkpoint["last_batch_id"]) == batch_id
        assert int(checkpoint["lock_version"]) == 1
        assert checkpoint["lease_owner"] is None
        assert checkpoint["lease_until"] is None
        assert checkpoint["last_success_time"] is not None

        cursor.execute(
            "SELECT status, candidate_count, success_count, failed_count, "
            "skipped_count, finished_at FROM expert_index_batch_source "
            "WHERE batch_id=%s AND source_table='experts_work_experience'",
            (batch_id,),
        )
        batch_source = cursor.fetchone()
        assert batch_source["status"] == "success"
        assert int(batch_source["candidate_count"]) == 2
        assert int(batch_source["success_count"]) == 2
        assert int(batch_source["failed_count"]) == 0
        assert int(batch_source["skipped_count"]) == 0
        assert batch_source["finished_at"] is not None

        cursor.execute(
            "SELECT status, cursor_updated_at, cursor_id, candidate_count, "
            "success_count, failed_count, finished_at "
            "FROM expert_index_batch WHERE id=%s",
            (batch_id,),
        )
        batch = cursor.fetchone()
        assert batch["status"] == "success"
        assert str(batch["cursor_updated_at"]) == "2026-08-25 10:00:00"
        assert int(batch["cursor_id"]) == 902
        assert int(batch["candidate_count"]) == 2
        assert int(batch["success_count"]) == 2
        assert int(batch["failed_count"]) == 0
        assert batch["finished_at"] is not None


def _assert_not_ready_state(connection: Any, *, batch_id: int) -> None:
    """核对未完成任务被拒后所有来源批次状态均保持可恢复。"""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT last_updated_at, last_id, last_batch_id, lock_version, "
            "lease_owner FROM expert_index_source_checkpoint "
            "WHERE source_table='experts_project_experience'"
        )
        checkpoint = cursor.fetchone()
        assert str(checkpoint["last_updated_at"]) == "2026-08-13 22:00:00"
        assert int(checkpoint["last_id"]) == 0
        assert checkpoint["last_batch_id"] is None
        assert int(checkpoint["lock_version"]) == 0
        assert checkpoint["lease_owner"] == "p3-7-project-worker"

        cursor.execute(
            "SELECT status, success_count, failed_count, finished_at "
            "FROM expert_index_batch_source WHERE batch_id=%s",
            (batch_id,),
        )
        batch_source = cursor.fetchone()
        assert batch_source["status"] == "running"
        assert int(batch_source["success_count"]) == 0
        assert int(batch_source["failed_count"]) == 0
        assert batch_source["finished_at"] is None

        cursor.execute(
            "SELECT status, candidate_count, success_count, failed_count, "
            "finished_at FROM expert_index_batch WHERE id=%s",
            (batch_id,),
        )
        batch = cursor.fetchone()
        assert batch["status"] == "running"
        assert int(batch["candidate_count"]) == 0
        assert int(batch["success_count"]) == 0
        assert int(batch["failed_count"]) == 0
        assert batch["finished_at"] is None


def _verify_checkpoint_commit(config: DatabaseConfig, connection: Any) -> None:
    """验证成功原子收口、幂等重放和未完成任务整笔拒绝。"""

    manager = CheckpointManager(config)
    success_batch_id = _insert_batch(
        connection,
        batch_no="p3-7-work-success",
        source_table="experts_work_experience",
        lease_owner="p3-7-work-worker",
        candidate_count=2,
        task_statuses=("success", "success"),
    )
    result = manager.try_commit_source_batch(
        "experts_work_experience",
        success_batch_id,
        "p3-7-work-worker",
        checkpoint_lock_version=0,
    )
    assert result.already_committed is False
    assert result.candidate_count == 2
    assert result.checkpoint.lock_version == 1
    repeated = manager.try_commit_source_batch(
        "experts_work_experience",
        success_batch_id,
        "p3-7-response-retry",
        checkpoint_lock_version=0,
    )
    assert repeated.already_committed is True
    _assert_success_state(connection, batch_id=success_batch_id)

    pending_batch_id = _insert_batch(
        connection,
        batch_no="p3-7-project-pending",
        source_table="experts_project_experience",
        lease_owner="p3-7-project-worker",
        candidate_count=1,
        task_statuses=("pending",),
    )
    try:
        manager.try_commit_source_batch(
            "experts_project_experience",
            pending_batch_id,
            "p3-7-project-worker",
            checkpoint_lock_version=0,
        )
    except SourceBatchNotReadyError:
        pass
    else:
        raise AssertionError("pending 任务不应允许推进安全 Checkpoint")
    _assert_not_ready_state(connection, batch_id=pending_batch_id)


def main() -> None:
    """创建临时 Schema，执行真实事务验证，并在最后删除全部测试数据。"""

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
            _verify_checkpoint_commit(_temporary_config(base), connection)
            print(
                "P3.7 Checkpoint 提交验证通过: "
                f"MySQL={server['version']}, atomic_commit=true, "
                "idempotent_replay=true, pending_rejected=true"
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
