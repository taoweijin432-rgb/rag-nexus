from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import SecretStr

from rag_vector_store import (
    MYSQL_EXPERT_CHUNK_SELECT_COLUMNS,
    MySqlExpertChunkReader,
    MySQLSettings,
)


class FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.cursor_instance = FakeCursor(rows)
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


def _metadata() -> dict[str, object]:
    return {
        "company_name_raw": "Example Co",
        "company_name_norm": "example_co",
        "company_normalization_status": "verified",
        "role_name_raw": "Engineer",
        "role_code": "software_engineer",
        "role_level": "senior",
        "start_month": "2020-01",
        "end_month": None,
        "is_current": True,
        "industry_codes": ["software"],
        "topic_codes": [],
        "origin_type": "source_template",
        "verification_status": "validated",
        "weight_tier": "A",
    }


def _row(
    *,
    metadata: object | None = None,
    is_active: int = 1,
    content_hash: str | None = None,
    row_id: int = 65,
    chunk_id: str = "expert:7:employment:source:0",
    updated_at: datetime | None = None,
) -> tuple[Any, ...]:
    values: dict[str, Any] = {
        "id": row_id,
        "chunk_id": chunk_id,
        "expert_id": 7,
        "serial_number": None,
        "chunk_type": "employment",
        "chunk_subtype": "responsibility",
        "chunk_index": 0,
        "chunk_total": 1,
        "chunk_text": "unchanged text",
        "char_count": len("unchanged text"),
        "metadata_json": json.dumps(_metadata()) if metadata is None else metadata,
        "source_table": "resume_experience",
        "source_record_id": "record-7",
        "source_item_key": "experience-0",
        "source_field": "description",
        "source_field_id": "description-0",
        "source_version": "source-v1",
        "source_updated_at": datetime.fromisoformat("2026-08-11T16:28:36"),
        "source_hash": "a" * 64,
        "content_hash": content_hash or hashlib.sha256(b"unchanged text").hexdigest(),
        "chunk_policy_version": "chunk-v1",
        "metadata_schema_version": "metadata-v1",
        "dictionary_version": "dict-v1",
        "data_version": "data-v1",
        "is_active": is_active,
        "updated_at": updated_at
        or datetime.fromisoformat("2026-08-11T16:28:47.391385"),
    }
    return tuple(values[name] for name in MYSQL_EXPERT_CHUNK_SELECT_COLUMNS)


def _settings() -> MySQLSettings:
    return MySQLSettings(
        database="expert_dev",
        user="readonly",
        password=SecretStr("not-a-real-secret"),
    )


def test_reader_maps_mysql_row_without_using_id_as_chunk_id() -> None:
    fake_connection = FakeConnection([_row(content_hash="b" * 64)])
    reader = MySqlExpertChunkReader(
        _settings(),
        database_timezone=ZoneInfo("Asia/Shanghai"),
        connector_factory=lambda **_: fake_connection,
    )

    result = reader.read()

    assert result.input_count == 1
    assert result.errors == []
    record = result.records[0]
    assert record.chunk_id == "expert:7:employment:source:0"
    assert not hasattr(record, "id")
    assert record.content_hash == "b" * 64
    assert record.chunk_text == "unchanged text"
    assert record.metadata.role_code == "software_engineer"
    assert record.source.source_hash == "a" * 64
    assert record.source.source_updated_at == "2026-08-11T16:28:36+08:00"
    assert record.versions.data_version == "data-v1"
    assert record.updated_at == "2026-08-11T16:28:47.391385+08:00"
    assert record.is_active is True
    assert fake_connection.closed is True
    assert fake_connection.cursor_instance.closed is True


def test_reader_uses_explicit_read_only_columns_and_limit() -> None:
    fake_connection = FakeConnection([_row()])
    reader = MySqlExpertChunkReader(
        _settings(),
        database_timezone=ZoneInfo("Asia/Shanghai"),
        connector_factory=lambda **_: fake_connection,
    )

    reader.read(limit=1)

    query, params = fake_connection.cursor_instance.executed[0]
    upper_query = query.upper()
    assert "SELECT *" not in upper_query
    assert " FROM EXPERT_CHUNK ORDER BY ID ASC LIMIT %S" in upper_query
    assert params == (1,)
    assert not any(
        re.search(rf"\b{keyword}\b", upper_query)
        for keyword in ("INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "TRUNCATE")
    )


def test_reader_does_not_guess_timezone_for_naive_mysql_datetime() -> None:
    fake_connection = FakeConnection([_row()])
    reader = MySqlExpertChunkReader(
        _settings(),
        connector_factory=lambda **_: fake_connection,
    )

    result = reader.read()

    assert result.records == []
    assert len(result.errors) == 1
    assert result.errors[0].code == "datetime_timezone_missing"
    assert "unchanged text" not in result.errors[0].message


def test_reader_reports_invalid_json_without_repairing_row() -> None:
    fake_connection = FakeConnection([_row(metadata="not-json")])
    reader = MySqlExpertChunkReader(
        _settings(),
        database_timezone=ZoneInfo("Asia/Shanghai"),
        connector_factory=lambda **_: fake_connection,
    )

    result = reader.read()

    assert result.records == []
    assert result.errors[0].code == "json_field_invalid"
    assert result.errors[0].row_id == 65
    assert result.errors[0].chunk_id == "expert:7:employment:source:0"
