"""在隔离本地 Schema 验证 014 双执行和 DROP 后恢复幂等性。"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pymysql

from expert_index.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCHEMA = "expert_demo_p5_014_verify"
CONTROL_MIGRATION = PROJECT_ROOT / "sql/012_complete_incremental_control_tables.sql"
ERROR_STAGE_MIGRATION = PROJECT_ROOT / "sql/014_expand_expert_sync_task_error_stages.sql"
CONSTRAINT_NAME = "chk_expert_sync_task_error_stage"
EXPECTED_STAGES = {
    "source", "parse", "pii", "chunk_builder", "generation", "validation",
    "embedding", "milvus", "checkpoint", "artifact", "handoff",
    "scope_consumer", "receipt",
}


def _statements(path: Path) -> list[str]:
    """读取迁移并返回去除注释后的独立 SQL 语句。"""

    statements = []
    for raw_statement in path.read_text(encoding="utf-8").split(";"):
        statement = "\n".join(
            line for line in raw_statement.splitlines()
            if not line.lstrip().startswith("--")
        ).strip()
        if statement:
            statements.append(statement)
    return statements


def _execute_file(cursor: Any, path: Path) -> None:
    """在当前临时 Schema 依次执行迁移语句。"""

    for statement in _statements(path):
        cursor.execute(statement)


def _constraint_clause(cursor: Any) -> str:
    """读取并验证目标 CHECK 的唯一最终谓词。"""

    cursor.execute(
        """
        SELECT CHECK_CLAUSE
        FROM information_schema.CHECK_CONSTRAINTS
        WHERE CONSTRAINT_SCHEMA=%s AND TABLE_NAME='expert_index_sync_task'
          AND CONSTRAINT_NAME=%s
        """,
        (TEST_SCHEMA, CONSTRAINT_NAME),
    )
    rows = cursor.fetchall()
    if len(rows) != 1:
        raise AssertionError(f"目标 CHECK 数量必须为 1，实际为 {len(rows)}")
    clause = str(rows[0]["CHECK_CLAUSE"])
    missing = {stage for stage in EXPECTED_STAGES if f"'{stage}'" not in clause}
    if missing:
        raise AssertionError(f"目标 CHECK 缺少 error_stage: {sorted(missing)}")
    return clause


def main() -> None:
    """创建临时 Schema，验证双执行和 DROP 后重跑，最后清理。"""

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
            cursor.execute("SELECT @@hostname AS hostname, @@read_only AS read_only, VERSION() AS version")
            server = cursor.fetchone()
            local_names = {socket.gethostname().lower(), socket.getfqdn().lower()}
            if str(server["hostname"]).lower() not in local_names or int(server["read_only"]) != 0:
                raise RuntimeError("MySQL 不是本机可写服务，拒绝创建临时 Schema")
            cursor.execute(
                "SELECT COUNT(*) AS count FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=%s",
                (TEST_SCHEMA,),
            )
            if int(cursor.fetchone()["count"]) != 0:
                raise RuntimeError(f"临时 Schema 已存在，拒绝覆盖: {TEST_SCHEMA}")
            cursor.execute(f"CREATE DATABASE `{TEST_SCHEMA}` CHARACTER SET utf8mb4")
            created = True
            cursor.execute(f"USE `{TEST_SCHEMA}`")
            _execute_file(cursor, CONTROL_MIGRATION)
            _execute_file(cursor, ERROR_STAGE_MIGRATION)
            first = _constraint_clause(cursor)
            _execute_file(cursor, ERROR_STAGE_MIGRATION)
            if _constraint_clause(cursor) != first:
                raise AssertionError("014 第二次执行后 CHECK 谓词发生漂移")
            cursor.execute(
                "ALTER TABLE expert_index_sync_task "
                f"DROP CHECK {CONSTRAINT_NAME}"
            )
            _execute_file(cursor, ERROR_STAGE_MIGRATION)
            if _constraint_clause(cursor) != first:
                raise AssertionError("014 未能从约束缺失状态恢复相同 CHECK")
            print(f"P5 014 验证通过: MySQL={server['version']}, idempotent=true, recovery=true")
    finally:
        try:
            if created:
                with connection.cursor() as cursor:
                    cursor.execute(f"DROP DATABASE `{TEST_SCHEMA}`")
        finally:
            connection.close()


if __name__ == "__main__":
    main()
