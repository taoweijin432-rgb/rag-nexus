"""Resume 来源表结构预检的纯 Fake 测试，不连接或修改 MySQL。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import pytest

from resume_index.source_schema import (
    RESUME_SOURCE_COLUMN_REQUIREMENTS,
    RESUME_SOURCE_TABLES,
    ResumeSourceSchemaError,
    inspect_resume_source_schema,
    require_resume_source_schema,
)


class _Cursor:
    def __init__(
        self,
        column_rows: Sequence[Mapping[str, Any]],
        index_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self._column_rows = column_rows
        self._index_rows = index_rows
        self._result: Sequence[Mapping[str, Any]] = ()
        self.calls: list[tuple[str, Sequence[object] | None]] = []
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: Sequence[object] | None = None) -> None:
        self.calls.append((sql, params))
        self._result = (
            self._column_rows if "information_schema.COLUMNS" in sql else self._index_rows
        )

    def fetchall(self) -> Sequence[Mapping[str, Any]]:
        return self._result

    def close(self) -> None:
        self.closed = True


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self.cursor_instance = cursor

    def cursor(self) -> _Cursor:
        return self.cursor_instance


def _valid_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    columns: list[dict[str, Any]] = []
    for table, requirements in RESUME_SOURCE_COLUMN_REQUIREMENTS.items():
        for column, requirement in requirements.items():
            columns.append(
                {
                    "TABLE_NAME": table,
                    "COLUMN_NAME": column,
                    "DATA_TYPE": min(requirement.data_types),
                    "IS_NULLABLE": "YES" if requirement.nullable is True else "NO",
                }
            )

    indexes: list[dict[str, Any]] = []
    for table in RESUME_SOURCE_TABLES:
        definitions = [
            ("PRIMARY", 0, ("id",)),
            ("idx_scan", 1, ("updated_at", "id")),
        ]
        definitions.append(
            ("uk_resume_id", 0, ("resume_id",))
            if table == "resume_profile"
            else ("idx_resume_id", 1, ("resume_id",))
        )
        for index_name, non_unique, index_columns in definitions:
            for sequence, column in enumerate(index_columns, 1):
                indexes.append(
                    {
                        "TABLE_NAME": table,
                        "INDEX_NAME": index_name,
                        "NON_UNIQUE": non_unique,
                        "SEQ_IN_INDEX": sequence,
                        "COLUMN_NAME": column,
                    }
                )
    return columns, indexes


def test_valid_contract_uses_information_schema_selects_only() -> None:
    columns, indexes = _valid_rows()
    cursor = _Cursor(columns, indexes)

    report = require_resume_source_schema(_Connection(cursor), schema_name="resume_source")

    assert report.is_compatible
    assert len(cursor.calls) == 2
    assert all(call[0].lstrip().startswith("SELECT") for call in cursor.calls)
    assert all("information_schema." in call[0] for call in cursor.calls)
    assert all(call[1] == ("resume_source", *RESUME_SOURCE_TABLES) for call in cursor.calls)
    assert cursor.closed


def test_absent_tables_report_all_four_without_any_write() -> None:
    cursor = _Cursor((), ())

    report = inspect_resume_source_schema(_Connection(cursor), schema_name="resume_source")

    assert not report.is_compatible
    assert report.issues == tuple(f"缺少表 {table}" for table in RESUME_SOURCE_TABLES)
    assert all(call[0].lstrip().startswith("SELECT") for call in cursor.calls)


def test_missing_column_and_index_fail_closed() -> None:
    columns, indexes = _valid_rows()
    columns = [
        row
        for row in columns
        if not (row["TABLE_NAME"] == "resume_profile" and row["COLUMN_NAME"] == "resume_deleted_at")
    ]
    indexes = [
        row
        for row in indexes
        if not (row["TABLE_NAME"] == "resume_work_experience" and row["INDEX_NAME"] == "idx_scan")
    ]

    with pytest.raises(ResumeSourceSchemaError) as error:
        require_resume_source_schema(
            _Connection(_Cursor(columns, indexes)), schema_name="resume_source"
        )

    message = str(error.value)
    assert "resume_profile 缺少列 resume_deleted_at" in message
    assert "resume_work_experience 缺少索引 scan(updated_at,id)" in message


def test_incompatible_key_type_and_nullable_contract_are_reported() -> None:
    columns, indexes = _valid_rows()
    for row in columns:
        if row["TABLE_NAME"] == "resume_profile" and row["COLUMN_NAME"] == "resume_id":
            row["DATA_TYPE"] = "varchar"
            row["IS_NULLABLE"] = "YES"

    report = inspect_resume_source_schema(
        _Connection(_Cursor(columns, indexes)),
        schema_name="resume_source",
    )

    assert "resume_profile.resume_id 类型不兼容: varchar" in report.issues
    assert "resume_profile.resume_id 必须非空" in report.issues


@pytest.mark.parametrize("schema_name", ["", "   "])
def test_blank_schema_name_is_rejected_before_query(schema_name: str) -> None:
    cursor = _Cursor((), ())

    with pytest.raises(ValueError, match="schema_name"):
        inspect_resume_source_schema(_Connection(cursor), schema_name=schema_name)

    assert cursor.calls == []
