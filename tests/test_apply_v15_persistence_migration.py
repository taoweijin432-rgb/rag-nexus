"""apply_v15_persistence_migration.py 纯逻辑单元测试。

只覆盖升级模式的静态校验与 information_schema 判断辅助函数，
不连接 MySQL、不执行任何 DDL。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "apply_v15_persistence_migration.py"
)
_spec = importlib.util.spec_from_file_location("apply_v15_migration", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


class _FakeCursor:
    """返回固定 COUNT 的最小 Cursor，记录最近一次执行的语句。"""

    def __init__(self, count: int) -> None:
        self._count = count
        self.last_sql = ""
        self.last_params: tuple[Any, ...] | None = None

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self.last_sql = " ".join(sql.split())
        self.last_params = params

    def fetchone(self) -> dict[str, int]:
        return {"n": self._count}


def test_classify_accepts_column_key_constraint() -> None:
    column = migration._classify_alter_add(
        "ALTER TABLE v15_vectorization_work_item "
        "ADD COLUMN source_event_id BIGINT UNSIGNED NULL"
    )
    assert column == ("v15_vectorization_work_item", "column", "source_event_id")

    index = migration._classify_alter_add(
        "ALTER TABLE v15_vectorization_work_item ADD KEY idx_x (chunk_type, status)"
    )
    assert index == ("v15_vectorization_work_item", "index", "idx_x")

    constraint = migration._classify_alter_add(
        "ALTER TABLE v15_vectorization_work_item\n"
        "    ADD CONSTRAINT chk_c CHECK (\n"
        "        chunk_type IS NULL OR chunk_type IN ('employment')\n"
        "    )"
    )
    assert constraint == ("v15_vectorization_work_item", "constraint", "chk_c")


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE v15_vectorization_work_item",
        "ALTER TABLE v15_vectorization_work_item DROP COLUMN chunk_type",
        "ALTER TABLE v15_vectorization_work_item MODIFY chunk_type VARCHAR(32)",
        "ALTER TABLE v15_vectorization_work_item CHANGE chunk_type c2 VARCHAR(64)",
        "TRUNCATE TABLE v15_vectorization_work_item",
        "UPDATE v15_vectorization_work_item SET status='running'",
    ],
)
def test_prepare_rejects_forbidden_or_non_whitelisted_shapes(statement: str) -> None:
    with pytest.raises(ValueError):
        migration._prepare_upgrade_statements(statement)


def test_prepare_rejects_table_outside_whitelist() -> None:
    with pytest.raises(ValueError, match="白名单"):
        migration._prepare_upgrade_statements(
            "ALTER TABLE some_other_table ADD COLUMN x INT;"
        )


def test_prepare_strips_comments_and_reports_targets() -> None:
    with pytest.raises(ValueError, match="未找到"):
        migration._prepare_upgrade_statements("-- 只有注释，没有任何语句")

    statements, targets = migration._prepare_upgrade_statements(
        "-- 头部注释\n"
        "ALTER TABLE v15_vectorization_work_item ADD COLUMN a INT;\n"
        "ALTER TABLE v15_vectorization_work_item ADD KEY k (a);"
    )
    assert len(statements) == 2
    assert targets == ["v15_vectorization_work_item"]


def test_object_exists_reads_information_schema() -> None:
    assert migration._object_exists(_FakeCursor(1), "t", "column", "c") is True
    assert migration._object_exists(_FakeCursor(0), "t", "index", "i") is False

    fake = _FakeCursor(1)
    assert migration._object_exists(fake, "tbl", "constraint", "chk") is True
    assert "information_schema.table_constraints" in fake.last_sql
    assert fake.last_params == ("tbl", "chk")
