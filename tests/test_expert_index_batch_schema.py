"""加工批次与来源 Checkpoint 状态表的静态 DDL 契约测试。

本文件只读取版本库中的 SQL 文本，不连接或修改 MySQL。它用于第一阶段审查：确保
新建库使用 001，现有库依次使用增量迁移补齐批次跟踪、同步统计和来源 Checkpoint，
并确保新环境不再创建已停用的 Stage/Chunk 持久化表。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_SCHEMA = PROJECT_ROOT / "sql" / "001_create_expert_index_tables.sql"
BATCH_MIGRATION = PROJECT_ROOT / "sql" / "005_add_full_batch_tracking.sql"
VECTOR_STATS_MIGRATION = PROJECT_ROOT / "sql" / "006_add_vector_sync_stats.sql"
SOURCE_CHECKPOINT_MIGRATION = PROJECT_ROOT / "sql" / "007_add_source_checkpoint.sql"

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
    ):
        assert field in checkpoint
    assert "PRIMARY KEY (source_table)" in checkpoint
    for source_table in (
        "experts",
        "experts_work_experience",
        "experts_project_experience",
        "experts_education_experience",
    ):
        assert f"'{source_table}'" in checkpoint


def test_initial_schema_only_creates_batch_state_tables() -> None:
    """新环境只创建批次和 Checkpoint 状态表，Stage/Chunk 继续走内存链路。"""

    schema = _read(INITIAL_SCHEMA)

    assert schema.count("CREATE TABLE IF NOT EXISTS") == 3
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
