"""015 DLQ 追加历史迁移的范围和幂等性静态检查。"""

from __future__ import annotations

from pathlib import Path

SQL = (
    Path(__file__).resolve().parents[1] / "sql" / "015_append_dead_letter_history.sql"
).read_text(encoding="utf-8")


def test_migration_replaces_task_unique_with_attempt_unique() -> None:
    assert "DROP INDEX uk_expert_sync_dead_letter_task" in SQL
    assert (
        "ADD UNIQUE KEY uk_expert_sync_dead_letter_attempt (task_id, retry_count)"
        in SQL
    )
    assert "information_schema.STATISTICS" in SQL
    assert "DROP TABLE" not in SQL.upper()
    assert "TRUNCATE" not in SQL.upper()
