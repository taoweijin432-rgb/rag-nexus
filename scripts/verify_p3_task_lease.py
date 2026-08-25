"""在隔离本地 Index Schema 验证 P3.6 任务租约和 attempt token 守卫。"""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymysql

from expert_index.config import DatabaseConfig, DatabaseRole, Settings
from expert_index.repositories import (
    MySQLSourceSyncTaskRepository,
    SourceSyncTaskLeaseConflictError,
    SourceSyncTaskStageConflictError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCHEMA = "expert_demo_p3_6_task_lease_verify"
BASE_SCHEMA = PROJECT_ROOT / "sql/001_create_expert_index_tables.sql"


def _statements(path: Path) -> list[str]:
    """返回 SQL 文件内按顺序排列的可执行语句。"""

    statements: list[str] = []
    for raw_statement in path.read_text(encoding="utf-8").split(";"):
        statement = "\n".join(
            line
            for line in raw_statement.splitlines()
            if not line.lstrip().startswith("--")
        ).strip()
        if statement:
            statements.append(statement)
    return statements


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


def _token_factory(*tokens: str) -> Callable[[], str]:
    """返回按固定顺序生成测试 attempt token 的函数。"""

    remaining = list(tokens)
    return lambda: remaining.pop(0)


def _assert_local_writable_server(cursor: Any) -> dict[str, Any]:
    """拒绝在远程或只读 MySQL 创建临时验证 Schema。"""

    cursor.execute(
        "SELECT @@hostname AS hostname, @@read_only AS read_only, VERSION() AS version"
    )
    server = cursor.fetchone()
    local_hostnames = {socket.gethostname().lower(), socket.getfqdn().lower()}
    if str(server["hostname"]).lower() not in local_hostnames:
        raise RuntimeError("MySQL 服务端不是本机，拒绝创建 P3.6 临时 Schema")
    if int(server["read_only"]) != 0:
        raise RuntimeError("本机 MySQL 当前为只读，不能执行 P3.6 验证")
    return server


def _insert_verification_tasks(connection: Any) -> tuple[int, int, int, int]:
    """插入两个 pending 任务和一个预置 handed_off 任务。"""

    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO expert_index_batch(batch_no, run_mode, status) "
            "VALUES ('p3-6-task-lease-verify', 'incremental', 'running')"
        )
        batch_id = int(cursor.lastrowid)
        task_ids: list[int] = []
        for record_id, dedupe_key in ((901, "a" * 64), (902, "b" * 64)):
            cursor.execute(
                "INSERT INTO expert_index_sync_task("
                "dedupe_key, batch_id, event_source, source_table, "
                "source_record_id, source_updated_at, expert_id, "
                "operation_type, target_type, processing_stage, sync_status, "
                "retry_count, policy_version"
                ") VALUES (%s, %s, 'poll', 'experts_work_experience', %s, "
                "'2026-08-25 08:00:00.000000', 101, 'update', 'employment', "
                "'candidate', 'pending', 0, '1.3')",
                (dedupe_key, batch_id, record_id),
            )
            task_ids.append(int(cursor.lastrowid))
        cursor.execute(
            "INSERT INTO expert_index_sync_task("
            "dedupe_key, batch_id, event_source, source_table, "
            "source_record_id, source_updated_at, expert_id, "
            "operation_type, target_type, processing_stage, sync_status, "
            "retry_count, attempt_token, lease_until, policy_version, started_at"
            ") VALUES (%s, %s, 'poll', 'experts_work_experience', 903, "
            "'2026-08-25 08:00:01.000000', 101, 'update', 'employment', "
            "'handed_off', 'processing', 0, 'preset-p5-receipt-attempt', "
            "TIMESTAMPADD(SECOND, 300, CURRENT_TIMESTAMP(6)), '1.3', "
            "CURRENT_TIMESTAMP(6))",
            ("c" * 64, batch_id),
        )
        handed_off_task_id = int(cursor.lastrowid)
    connection.commit()
    return batch_id, task_ids[0], task_ids[1], handed_off_task_id


def _verify_task_state_machine(config: DatabaseConfig, admin_connection: Any) -> None:
    """验证跳锁领取、过期接管、旧 token 拒绝和终态阶段守卫。"""

    batch_id, first_task_id, second_task_id, handed_off_task_id = (
        _insert_verification_tasks(admin_connection)
    )
    with admin_connection.cursor() as cursor:
        cursor.execute("START TRANSACTION")
        cursor.execute(
            "SELECT id FROM expert_index_sync_task WHERE id=%s FOR UPDATE",
            (first_task_id,),
        )
        assert int(cursor.fetchone()["id"]) == first_task_id

    skip_locked_repository = MySQLSourceSyncTaskRepository(
        config,
        attempt_token_factory=_token_factory("skip-locked-attempt"),
    )
    claimed_while_locked = skip_locked_repository.claim_source_sync_tasks(
        batch_id,
        "experts_work_experience",
        limit=1,
        lease_seconds=120,
    )
    assert len(claimed_while_locked) == 1
    assert claimed_while_locked[0].task_id == second_task_id
    assert claimed_while_locked[0].attempt_token == "skip-locked-attempt"
    admin_connection.rollback()

    repository = MySQLSourceSyncTaskRepository(
        config,
        attempt_token_factory=_token_factory(
            "first-worker-attempt",
            "replacement-worker-attempt",
        ),
    )
    first_claim = repository.claim_source_sync_tasks(
        batch_id,
        "experts_work_experience",
        limit=1,
        lease_seconds=120,
    )
    assert len(first_claim) == 1
    assert first_claim[0].task_id == first_task_id
    assert first_claim[0].attempt_token == "first-worker-attempt"

    with admin_connection.cursor() as cursor:
        cursor.execute(
            "UPDATE expert_index_sync_task "
            "SET lease_until=TIMESTAMPADD(SECOND, -1, CURRENT_TIMESTAMP(6)) "
            "WHERE id=%s",
            (first_task_id,),
        )
    admin_connection.commit()

    replacement_claim = repository.claim_source_sync_tasks(
        batch_id,
        "experts_work_experience",
        limit=1,
        lease_seconds=120,
    )
    assert len(replacement_claim) == 1
    assert replacement_claim[0].task_id == first_task_id
    assert replacement_claim[0].attempt_token == "replacement-worker-attempt"
    assert replacement_claim[0].retry_count == 1

    try:
        repository.renew_source_sync_task_lease(
            first_task_id,
            "first-worker-attempt",
            lease_seconds=120,
        )
    except SourceSyncTaskLeaseConflictError:
        pass
    else:
        raise AssertionError("旧 Worker 不应能够续租已被接管的任务")

    try:
        repository.advance_source_sync_task_stage(
            first_task_id,
            "replacement-worker-attempt",
            next_stage="normalized",
        )
    except SourceSyncTaskStageConflictError:
        pass
    else:
        raise AssertionError("任务不应跳过 source_read 阶段")

    advanced = repository.advance_source_sync_task_stage(
        first_task_id,
        "replacement-worker-attempt",
        next_stage="source_read",
    )
    assert advanced.processing_stage == "source_read"
    repeated = repository.advance_source_sync_task_stage(
        first_task_id,
        "replacement-worker-attempt",
        next_stage="source_read",
    )
    assert repeated.processing_stage == "source_read"
    failed = repository.mark_source_sync_task_failed(
        first_task_id,
        "replacement-worker-attempt",
        error_stage="source",
        error_code="verification_failure",
        error_type="PresetVerificationError",
        safe_error_message="sanitized temporary verification failure",
    )
    assert failed.sync_status == "failed"
    assert failed.attempt_token is None

    succeeded = repository.mark_source_sync_task_success(
        handed_off_task_id,
        "preset-p5-receipt-attempt",
        result_code="preset_receipt_verified",
    )
    assert succeeded.processing_stage == "completed"
    assert succeeded.sync_status == "success"
    assert succeeded.completed_at is not None
    assert succeeded.attempt_token is None


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
            for statement in _statements(BASE_SCHEMA):
                cursor.execute(statement)
            _verify_task_state_machine(_temporary_config(base), connection)
            print(
                "P3.6 任务租约验证通过: "
                f"MySQL={server['version']}, skip_locked=true, "
                "stale_token_rejected=true, stage_guarded=true"
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
