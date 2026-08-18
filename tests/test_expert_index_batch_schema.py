"""第一版全量加工批次表的静态 DDL 契约测试。

本文件只读取版本库中的 SQL 文本，不连接或修改 MySQL。它用于第一阶段审查：确保
新建库使用的 001 和现有库使用的 005 都准备了相同的批次跟踪能力。后续 Repository
开始写入这些字段时，还需要另行补充 SQL 参数映射和真实 MySQL 集成测试。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_SCHEMA = PROJECT_ROOT / "sql" / "001_create_expert_index_tables.sql"
BATCH_MIGRATION = PROJECT_ROOT / "sql" / "005_add_full_batch_tracking.sql"


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


def test_initial_schema_does_not_create_legacy_expert_chunk_table() -> None:
    """新环境只创建 Batch/Stage；Chunk 通过内存调用交付下游。"""

    schema = _read(INITIAL_SCHEMA)

    assert "CREATE TABLE IF NOT EXISTS expert_chunk" not in schema


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
