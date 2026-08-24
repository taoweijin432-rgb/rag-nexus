"""加工批次、增量任务、恢复证据与安全水位的静态 DDL 契约测试。

本文件只读取版本库中的 SQL 文本，不连接或修改 MySQL。它用于第一阶段审查：确保
新建库使用 001，现有库依次使用增量迁移补齐批次跟踪、同步统计、来源扫描范围和
Checkpoint、CDC 事件、同步任务、Artifact 和 Dead Letter 表，并确保新环境
不再创建已停用的 Stage/Chunk 表。
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_SCHEMA = PROJECT_ROOT / "sql" / "001_create_expert_index_tables.sql"
BATCH_MIGRATION = PROJECT_ROOT / "sql" / "005_add_full_batch_tracking.sql"
VECTOR_STATS_MIGRATION = PROJECT_ROOT / "sql" / "006_add_vector_sync_stats.sql"
SOURCE_CHECKPOINT_MIGRATION = PROJECT_ROOT / "sql" / "007_add_source_checkpoint.sql"
SOURCE_BATCH_MIGRATION = PROJECT_ROOT / "sql" / "008_add_source_batch_tracking.sql"
CDC_EVENT_MIGRATION = PROJECT_ROOT / "sql" / "009_add_cdc_event_tracking.sql"
SYNC_TASK_MIGRATION = PROJECT_ROOT / "sql" / "010_add_incremental_sync_task.sql"
CONTROL_TABLE_MIGRATION = (
    PROJECT_ROOT / "sql" / "012_complete_incremental_control_tables.sql"
)

VECTOR_STAT_FIELDS = {
    "vector_sync_expert_count",
    "vector_current_count",
    "vector_eligible_count",
    "vector_embedded_count",
    "vector_metadata_updated_count",
    "vector_unchanged_count",
    "vector_deleted_count",
    "vector_delete_not_found_count",
}


def _read(path: Path) -> str:
    """按 UTF-8 读取 SQL，供下面的结构断言复用。"""

    return path.read_text(encoding="utf-8")


def _create_table_block(sql: str, table_name: str) -> str:
    """截取一个 CREATE TABLE 语句，避免字段名在其他表出现时造成假通过。

    该帮助函数只服务静态契约测试，不是通用 SQL 解析器。项目建表文件中的表定义都以
    ``ENGINE=InnoDB`` 结束，因此可用固定边界提取目标表。
    """

    marker = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    start = sql.index(marker)
    end = sql.index(") ENGINE=InnoDB", start)
    return sql[start:end]


def test_initial_schema_contains_full_batch_summary_fields() -> None:
    """新建库的批次主表应能记录全量边界、进度和专家级结果总数。"""

    batch = _create_table_block(_read(INITIAL_SCHEMA), "expert_index_batch")
    required_fields = {
        "data_version",
        "cursor_id",
        "upper_bound_id",
        "candidate_count",
        "success_count",
        "failed_count",
        "deactivated_count",
        "rejected_count",
    }

    for field in required_fields:
        assert field in batch

    for field in VECTOR_STAT_FIELDS:
        assert field in batch


def test_initial_schema_contains_expert_level_batch_items() -> None:
    """专家明细表应覆盖全部已确认终态，并保证一个批次内专家唯一。"""

    item = _create_table_block(_read(INITIAL_SCHEMA), "expert_index_batch_item")

    for status in (
        "pending",
        "running",
        "success",
        "rejected",
        "deactivated",
        "failed",
    ):
        assert f"'{status}'" in item

    assert "UNIQUE KEY uk_expert_batch_item (batch_id, expert_id)" in item
    assert "KEY idx_expert_batch_item_status (batch_id, status)" in item
    assert "result_code VARCHAR(64) NULL" in item
    assert "error_message VARCHAR(512) NULL" in item


def test_initial_schema_contains_source_checkpoint() -> None:
    """新建库应为四张物理来源表保存独立的复合安全游标。"""

    checkpoint = _create_table_block(
        _read(INITIAL_SCHEMA),
        "expert_index_source_checkpoint",
    )

    for field in (
        "source_table",
        "last_updated_at",
        "last_id",
        "last_batch_id",
        "last_success_time",
        "lock_version",
        "lease_owner",
        "lease_until",
    ):
        assert field in checkpoint
    assert "PRIMARY KEY (source_table)" in checkpoint
    assert "KEY idx_expert_source_checkpoint_lease (lease_until, source_table)" in checkpoint
    assert "chk_expert_source_checkpoint_lease" in checkpoint
    for source_table in (
        "experts",
        "experts_work_experience",
        "experts_project_experience",
        "experts_education_experience",
    ):
        assert f"'{source_table}'" in checkpoint


def test_initial_schema_contains_source_batch_scan_range() -> None:
    """新建库应分开记录来源批次的起点、临时游标和固定上界。"""

    source_batch = _create_table_block(
        _read(INITIAL_SCHEMA),
        "expert_index_batch_source",
    )

    for field in (
        "batch_id",
        "source_table",
        "start_updated_at",
        "start_id",
        "cursor_updated_at",
        "cursor_id",
        "upper_bound_updated_at",
        "upper_bound_id",
        "checkpoint_lock_version",
        "status",
        "candidate_count",
        "success_count",
        "failed_count",
        "skipped_count",
    ):
        assert field in source_batch
    assert "UNIQUE KEY uk_expert_batch_source (batch_id, source_table)" in source_batch
    assert "KEY idx_expert_batch_source_status (batch_id, status)" in source_batch


def test_initial_schema_contains_minimal_cdc_event_tracking() -> None:
    """新建库应持久化 CDC 位点和任务状态，但不保存来源行或 Chunk 正文。"""

    event = _create_table_block(_read(INITIAL_SCHEMA), "expert_index_cdc_event")

    for field in (
        "destination",
        "binlog_file",
        "binlog_position",
        "row_index",
        "event_time",
        "source_schema",
        "source_table",
        "operation_type",
        "source_record_id",
        "expert_id",
        "status",
        "attempt_count",
        "processed_at",
    ):
        assert field in event
    assert "UNIQUE KEY uk_expert_cdc_event_position" in event
    assert "KEY idx_expert_cdc_event_queue (status, id)" in event
    for operation_type in ("insert", "update", "delete"):
        assert f"'{operation_type}'" in event
    for status in ("pending", "running", "retry", "success", "dead_letter"):
        assert f"'{status}'" in event
    for forbidden_field in ("before_columns", "after_columns", "raw_payload", "chunk_text"):
        assert forbidden_field not in event.lower()


def test_initial_schema_contains_source_record_sync_tasks() -> None:
    """新建库应按来源记录保存标准增量任务字段、状态和幂等键。"""

    task = _create_table_block(_read(INITIAL_SCHEMA), "expert_index_sync_task")

    for field in (
        "batch_id",
        "dedupe_key",
        "event_source",
        "source_table",
        "source_record_id",
        "source_updated_at",
        "expert_id",
        "operation_type",
        "target_type",
        "processing_stage",
        "sync_status",
        "retry_count",
        "attempt_token",
        "lease_until",
        "next_retry_at",
        "policy_version",
        "error_stage",
        "error_code",
        "error_type",
        "error_message",
        "result_code",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
    ):
        assert field in task
    assert "UNIQUE KEY uk_expert_sync_task_source" in task
    assert "UNIQUE KEY uk_expert_sync_task_dedupe (dedupe_key)" in task
    assert "batch_id, source_table, source_record_id" in task
    assert "KEY idx_expert_sync_task_queue (sync_status, id)" in task
    assert (
        "KEY idx_expert_sync_task_retry_queue (sync_status, next_retry_at, id)"
        in task
    )
    assert "KEY idx_expert_sync_task_history (expert_id, created_at)" in task
    assert "chk_expert_sync_task_attempt_lease" in task
    for operation_type in ("insert", "update", "delete"):
        assert f"'{operation_type}'" in task
    for status in ("pending", "processing", "success", "failed", "retrying", "dead"):
        assert f"'{status}'" in task
    for error_stage in (
        "source",
        "parse",
        "pii",
        "chunk_builder",
        "generation",
        "validation",
        "embedding",
        "milvus",
        "checkpoint",
    ):
        assert f"'{error_stage}'" in task
    for forbidden_field in (
        "raw_payload",
        "source_row",
        "chunk_text",
        "metadata json",
        "embedding float_vector",
    ):
        assert forbidden_field not in task.lower()


def test_initial_schema_contains_controlled_sync_artifacts() -> None:
    """Artifact 只保存二选一的 Contract Payload 或引用及其 Hash。"""

    artifact = _create_table_block(
        _read(INITIAL_SCHEMA),
        "expert_index_sync_artifact",
    )

    for field in (
        "task_id",
        "artifact_type",
        "artifact_version",
        "payload_json",
        "payload_reference",
        "payload_hash",
        "expires_at",
    ):
        assert field in artifact
    assert "UNIQUE KEY uk_expert_sync_artifact_version" in artifact
    assert "chk_expert_sync_artifact_payload" in artifact
    for forbidden_field in (
        "embedding_vector",
        "float_vector",
        "person_name",
        "phone_number",
        "email_address",
        "identity_card",
    ):
        assert forbidden_field not in artifact.lower()


def test_initial_schema_contains_sync_dead_letters() -> None:
    """Dead Letter 应以任务唯一，保存失败定位、重试和处置状态。"""

    dead_letter = _create_table_block(
        _read(INITIAL_SCHEMA),
        "expert_index_sync_dead_letter",
    )

    for field in (
        "task_id",
        "batch_id",
        "source_table",
        "source_record_id",
        "expert_id",
        "failure_stage",
        "reason_code",
        "reason_message",
        "retry_count",
        "artifact_id",
        "resolution_status",
        "resolved_at",
    ):
        assert field in dead_letter
    assert "UNIQUE KEY uk_expert_sync_dead_letter_task (task_id)" in dead_letter
    assert "KEY idx_expert_sync_dead_letter_resolution" in dead_letter
    for resolution_status in ("pending", "requeued", "resolved", "ignored"):
        assert f"'{resolution_status}'" in dead_letter


def test_initial_schema_only_creates_local_control_tables() -> None:
    """新环境只创建本地控制表，Stage/Chunk 继续走内存链路。"""

    schema = _read(INITIAL_SCHEMA)

    assert schema.count("CREATE TABLE IF NOT EXISTS") == 8
    for removed_table in (
        "expert_index_stage",
        "expert_index_employment_stage",
        "expert_index_project_stage",
        "expert_chunk",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {removed_table}" not in schema


def test_existing_database_migration_is_additive_and_complete() -> None:
    """005 只能增加批量结构，不能删除当前单专家链路使用的表或字段。"""

    migration = _read(BATCH_MIGRATION)

    assert "ALTER TABLE expert_index_batch" in migration
    assert "CREATE TABLE IF NOT EXISTS expert_index_batch_item" in migration
    for field in (
        "data_version",
        "upper_bound_id",
        "candidate_count",
        "success_count",
        "failed_count",
        "deactivated_count",
    ):
        assert f"COLUMN {field}" in migration

    # 第一阶段迁移必须保持纯新增，防止审查前意外破坏已有单专家数据。
    assert "DROP TABLE" not in migration.upper()
    assert "DROP COLUMN" not in migration.upper()


def test_vector_stats_migration_adds_only_batch_summary_fields() -> None:
    """现有库迁移应只为批次主表增加带默认值的同步统计字段。"""

    migration = _read(VECTOR_STATS_MIGRATION)

    assert "ALTER TABLE expert_index_batch" in migration
    for field in VECTOR_STAT_FIELDS:
        assert f"ADD COLUMN {field}" in migration
    assert migration.count("NOT NULL DEFAULT 0") == len(VECTOR_STAT_FIELDS)
    assert "expert_index_batch_item" not in migration
    assert "DROP " not in migration.upper()


def test_source_checkpoint_migration_only_creates_checkpoint_table() -> None:
    """现有库迁移只新增来源 Checkpoint，不初始化游标或修改来源表。"""

    migration = _read(SOURCE_CHECKPOINT_MIGRATION)
    checkpoint = _create_table_block(
        migration,
        "expert_index_source_checkpoint",
    )

    for field in (
        "source_table",
        "last_updated_at",
        "last_id",
        "last_batch_id",
        "last_success_time",
        "lock_version",
    ):
        assert field in checkpoint
    assert "INSERT INTO" not in migration.upper()
    assert "ALTER TABLE" not in migration.upper()
    assert "DROP " not in migration.upper()


def test_source_batch_migration_only_creates_scan_range_table() -> None:
    """现有库迁移只新增来源扫描范围，不创建任务或推进 Checkpoint。"""

    migration = _read(SOURCE_BATCH_MIGRATION)
    source_batch = _create_table_block(
        migration,
        "expert_index_batch_source",
    )

    for field in (
        "start_updated_at",
        "start_id",
        "cursor_updated_at",
        "cursor_id",
        "upper_bound_updated_at",
        "upper_bound_id",
        "checkpoint_lock_version",
    ):
        assert field in source_batch
    assert "expert_index_source_checkpoint" in migration
    assert "INSERT INTO" not in migration.upper()
    assert "ALTER TABLE" not in migration.upper()
    assert "DROP " not in migration.upper()


def test_cdc_event_migration_only_creates_minimal_event_table() -> None:
    """现有库迁移只新增 CDC 任务表，不消费事件或推进任何 Checkpoint。"""

    migration = _read(CDC_EVENT_MIGRATION)
    event = _create_table_block(migration, "expert_index_cdc_event")

    assert "UNIQUE KEY uk_expert_cdc_event_position" in event
    assert "destination, binlog_file, binlog_position, row_index" in event
    assert "KEY idx_expert_cdc_event_queue (status, id)" in event
    assert "INSERT INTO" not in migration.upper()
    assert "ALTER TABLE" not in migration.upper()
    assert "DROP " not in migration.upper()


def test_sync_task_migration_only_creates_task_state_table() -> None:
    """现有库迁移只新增任务结构，不执行任务或推进 Checkpoint。"""

    migration = _read(SYNC_TASK_MIGRATION)
    task = _create_table_block(migration, "expert_index_sync_task")

    assert "UNIQUE KEY uk_expert_sync_task_source" in task
    assert "KEY idx_expert_sync_task_queue (sync_status, id)" in task
    assert "retry_count SMALLINT UNSIGNED NOT NULL DEFAULT 0" in task
    assert "INSERT INTO" not in migration.upper()
    assert "ALTER TABLE" not in migration.upper()
    assert "DROP " not in migration.upper()


def test_control_table_migration_completes_p1_state_without_cdc_offset() -> None:
    """012 应补齐五张控制表，不猜测 CDC Checkpoint 或执行状态转换。"""

    migration = _read(CONTROL_TABLE_MIGRATION)

    for table_name in (
        "expert_index_source_checkpoint",
        "expert_index_batch_source",
        "expert_index_sync_task",
        "expert_index_sync_artifact",
        "expert_index_sync_dead_letter",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table_name}" in migration
    assert "expert_index_cdc_checkpoint" not in migration.lower()
    assert "ALTER TABLE expert_index_cdc_event" not in migration
    assert "CREATE TABLE IF NOT EXISTS expert_chunk" not in migration


def test_control_table_migration_guards_existing_columns_and_indexes() -> None:
    """012 应在 ALTER 前检查结构，重复执行时跳过已存在对象。"""

    migration = _read(CONTROL_TABLE_MIGRATION)

    for field in (
        "lease_owner",
        "lease_until",
        "candidate_count",
        "success_count",
        "failed_count",
        "skipped_count",
        "dedupe_key",
        "event_source",
        "source_updated_at",
        "target_type",
        "processing_stage",
        "attempt_token",
        "next_retry_at",
        "policy_version",
        "error_code",
    ):
        assert f"COLUMN_NAME = '{field}'" in migration
    assert "FROM information_schema.COLUMNS" in migration
    assert "FROM information_schema.STATISTICS" in migration
    assert "FROM information_schema.TABLE_CONSTRAINTS" in migration
    assert "PREPARE expert_index_p1_stmt FROM @expert_index_ddl" in migration
    assert "DEALLOCATE PREPARE expert_index_p1_stmt" in migration


def test_control_table_migration_backfills_legacy_task_identity_safely() -> None:
    """旧 010 任务先获得稳定兼容键和目标类型，再收紧非空与唯一约束。"""

    migration = _read(CONTROL_TABLE_MIGRATION)

    assert "UPDATE expert_index_sync_task\nSET dedupe_key = SHA2(" in migration
    assert "'legacy'" in migration
    assert "WHERE dedupe_key IS NULL" in migration
    assert "SET target_type = CASE" in migration
    assert "source_table = 'experts' AND operation_type = 'delete'" in migration
    assert "MODIFY COLUMN dedupe_key" in migration
    assert "MODIFY COLUMN target_type" in migration
    assert "ADD UNIQUE KEY uk_expert_sync_task_dedupe (dedupe_key)" in migration


def test_control_table_migration_never_modifies_source_tables_or_drops_state() -> None:
    """012 只能改本地索引状态表，不能改来源表或破坏已有证据。"""

    migration = _read(CONTROL_TABLE_MIGRATION)
    executable_sql = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )

    for source_table in (
        "experts",
        "experts_work_experience",
        "experts_project_experience",
        "experts_education_experience",
    ):
        pattern = rf"ALTER\s+TABLE\s+{source_table}(?:\s|$)"
        assert re.search(pattern, executable_sql, re.IGNORECASE) is None
    for destructive_statement in (
        "DROP TABLE",
        "DROP COLUMN",
        "TRUNCATE TABLE",
        "DELETE FROM",
    ):
        assert destructive_statement not in executable_sql.upper()
