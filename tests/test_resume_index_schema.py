"""Resume Index 汇总 DDL 的静态契约测试，不连接或修改 MySQL。"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUME_INDEX_DDL = PROJECT_ROOT / "sql" / "020_create_resume_index_tables.sql"
RESUME_CDC_MIGRATION = PROJECT_ROOT / "sql" / "021_add_resume_cdc_control_tables.sql"
BATCH_ID_SEQUENCE_MIGRATION = PROJECT_ROOT / "sql" / "022_add_global_batch_id_sequence.sql"

EXPECTED_CONTROL_TABLES = {
    "expert_index_batch",
    "expert_index_source_checkpoint",
    "expert_index_batch_source",
    "expert_index_sync_task",
    "expert_index_sync_artifact",
    "expert_index_sync_dead_letter",
    "expert_index_batch_item",
    "expert_index_cdc_event",
    "expert_index_cdc_checkpoint",
    "expert_index_batch_id_sequence",
}
RESUME_SOURCE_TABLES = {
    "resume",
    "resume_work_experience",
    "resume_project_experience",
    "resume_education_experience",
}


def _read_sql() -> str:
    return RESUME_INDEX_DDL.read_text(encoding="utf-8")


def _create_table_block(sql: str, table_name: str) -> str:
    marker = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    start = sql.index(marker)
    end = sql.index(") ENGINE=InnoDB", start)
    return sql[start:end]


def test_resume_schema_creates_full_poll_and_cdc_control_tables() -> None:
    sql = _read_sql()
    created_tables = {
        line.split()[5]
        for line in sql.splitlines()
        if line.startswith("CREATE TABLE IF NOT EXISTS ")
    }

    assert created_tables == EXPECTED_CONTROL_TABLES
    assert "CREATE DATABASE" not in sql.upper()
    assert "CREATE SCHEMA" not in sql.upper()
    assert "USE " not in sql.upper()


def test_resume_schema_never_creates_or_modifies_source_tables() -> None:
    sql = _read_sql()

    for source_table in RESUME_SOURCE_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {source_table}" not in sql
        assert f"ALTER TABLE {source_table}" not in sql
        assert f"DROP TABLE {source_table}" not in sql


def test_resume_source_table_checks_use_only_resume_whitelist() -> None:
    sql = _read_sql()

    for table_name in (
        "expert_index_source_checkpoint",
        "expert_index_batch_source",
        "expert_index_sync_task",
        "expert_index_sync_dead_letter",
    ):
        block = _create_table_block(sql, table_name)
        for source_table in RESUME_SOURCE_TABLES:
            assert f"'{source_table}'" in block
        for expert_source_table in (
            "experts",
            "experts_work_experience",
            "experts_project_experience",
            "experts_education_experience",
        ):
            assert f"'{expert_source_table}'" not in block


def test_resume_schema_contains_current_poll_checkpoint_and_task_contract() -> None:
    sql = _read_sql()
    checkpoint = _create_table_block(sql, "expert_index_source_checkpoint")
    batch_source = _create_table_block(sql, "expert_index_batch_source")
    task = _create_table_block(sql, "expert_index_sync_task")

    assert "PRIMARY KEY (source_table)" in checkpoint
    assert "lock_version BIGINT UNSIGNED NOT NULL DEFAULT 0" in checkpoint
    assert "lease_owner VARCHAR(128) NULL" in checkpoint
    assert "KEY idx_expert_batch_source_claim (source_table, status, id)" in batch_source
    for field in (
        "dedupe_key",
        "source_record_id",
        "source_updated_at",
        "attempt_token",
        "lease_until",
        "next_retry_at",
        "policy_version",
        "result_code",
    ):
        assert field in task
    assert "event_source IN ('poll', 'cdc', 'reconcile', 'replay')" in task
    for stage in ("artifact", "handoff", "scope_consumer", "receipt"):
        assert f"'{stage}'" in task


def test_resume_schema_contains_latest_dead_letter_history_and_audit_fields() -> None:
    dead_letter = _create_table_block(
        _read_sql(),
        "expert_index_sync_dead_letter",
    )

    assert "UNIQUE KEY uk_expert_sync_dead_letter_attempt (task_id, retry_count)" in dead_letter
    assert "uk_expert_sync_dead_letter_task" not in dead_letter
    for field in (
        "resolution_operator",
        "resolution_request_id",
        "resolution_version",
    ):
        assert field in dead_letter


def test_resume_internal_expert_id_is_documented_as_resume_id_compatibility() -> None:
    sql = _read_sql()
    task = _create_table_block(sql, "expert_index_sync_task")
    item = _create_table_block(sql, "expert_index_batch_item")

    assert "固定承载 resume_id" in task
    assert "固定承载 resume_id" in item
    assert "本脚本只能由 resume_index.cli init-db" in sql


def test_resume_schema_contains_latest_cdc_event_and_checkpoint_contract() -> None:
    sql = _read_sql()
    event = _create_table_block(sql, "expert_index_cdc_event")
    checkpoint = _create_table_block(sql, "expert_index_cdc_checkpoint")

    for field in (
        "cdc_event_id",
        "local_batch_id",
        "task_id",
        "policy_version",
        "binlog_server_id",
    ):
        assert field in event
    assert "UNIQUE KEY uk_expert_cdc_event_id (cdc_event_id)" in event
    assert "KEY idx_expert_cdc_event_task (task_id, status)" in event
    assert "PRIMARY KEY (destination)" in checkpoint
    for source_table in RESUME_SOURCE_TABLES:
        assert f"'{source_table}'" in event


def test_resume_cdc_migration_is_resume_only_and_replay_safe() -> None:
    migration = RESUME_CDC_MIGRATION.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS expert_index_cdc_event" in migration
    assert "CREATE TABLE IF NOT EXISTS expert_index_cdc_checkpoint" in migration
    assert "information_schema.CHECK_CONSTRAINTS" in migration
    assert "event_source IN (''poll'', ''cdc'', ''reconcile'', ''replay'')" in migration
    for source_table in RESUME_SOURCE_TABLES:
        assert f"'{source_table}'" in migration
    for expert_source_table in (
        "experts",
        "experts_work_experience",
        "experts_project_experience",
        "experts_education_experience",
    ):
        assert f"'{expert_source_table}'" not in migration
    for source_table in RESUME_SOURCE_TABLES:
        assert f"ALTER TABLE {source_table}" not in migration


def test_global_batch_id_sequence_migration_uses_all_persisted_watermarks() -> None:
    """统一序列必须覆盖 Poll、全量、任务、CDC 和 Checkpoint 历史编号。"""

    migration = BATCH_ID_SEQUENCE_MIGRATION.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS expert_index_batch_id_sequence" in migration
    assert "PRIMARY KEY (sequence_name)" in migration
    for watermark in (
        "MAX(id) FROM expert_index_batch",
        "MAX(batch_id) FROM expert_index_batch_source",
        "MAX(batch_id) FROM expert_index_batch_item",
        "MAX(batch_id) FROM expert_index_sync_task",
        "MAX(batch_id) FROM expert_index_sync_dead_letter",
        "MAX(last_batch_id) FROM expert_index_source_checkpoint",
        "MAX(local_batch_id) FROM expert_index_cdc_event",
        "MAX(last_local_batch_id) FROM expert_index_cdc_checkpoint",
    ):
        assert watermark in migration
    assert "GREATEST(current_value, VALUES(current_value))" in migration
    assert "DROP " not in migration.upper()
