from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import SecretStr

from rag_vector_store.config import MySQLSettings
from rag_vector_store.mysql import (
    MYSQL_EXPERT_CHUNK_SELECT_COLUMNS,
    MySqlExpertChunkReader,
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _baseline(chunk_type: str, *, source_field_id: str) -> dict[str, Any]:
    text = {
        "profile_summary": "[专家能力概述]\n扁平表 profile 测试记录。",
        "employment": "[任职经历]\n公司:某公司\n岗位:产品经理\n职责简述。",
        "project": "[项目经历]\n项目名称:某项目\n项目描述简述。",
    }[chunk_type]
    index = 0
    common: dict[str, Any] = {
        "id": 65,
        "vector_doc_id": f"expert:7:{chunk_type}:{source_field_id}:{index}",
        "expert_id": 7,
        "serial_number": "SN-7",
        "chunk_type": chunk_type,
        "chunk_subtype": None,
        "chunk_index": index,
        "chunk_total": 1,
        "chunk_text": text,
        "char_count": len(text),
        "source_table": "experts",
        "source_field": "company+job_name+background",
        "source_field_id": source_field_id,
        "source_version": "2026-08-11T16:28:36+08:00",
        "source_updated_at": datetime.fromisoformat("2026-08-11T16:28:36"),
        "source_hash": "a" * 64,
        "content_hash": _sha(text),
        "metadata_hash": "b" * 64,
        "company_name_raw": "某公司",
        "company_name_norm": "某个标准公司",
        "role_name_raw": "产品经理",
        "role_code": "product_manager",
        "role_level": "manager",
        "job_status_norm": None,
        "abroad_flag_norm": None,
        "work_years": None,
        "project_name": None,
        "start_month": None,
        "end_month": None,
        "is_current": None,
        "industry_codes": ["software"],
        "function_codes": [],
        "topic_codes": [],
        "school_names_raw": [],
        "major_names_raw": [],
        "eligible_for_recommendation": None,
        "origin_type": "source_parsed",
        "verification_status": "verified",
        "weight_tier": "A",
        "chunk_policy_version": "1.3",
        "metadata_schema_version": "3.0",
        "is_active": 1,
        "updated_at": datetime.fromisoformat("2026-08-11T16:28:47.391385"),
    }
    if chunk_type == "profile_summary":
        common.update(
            {
                "source_field": "company+job_name+background",
                "job_status_norm": "active",
                "abroad_flag_norm": 0,
                "work_years": Decimal("12.50"),
                "project_name": None,
                "school_names_raw": ["测试大学"],
                "major_names_raw": ["软件工程"],
                "function_codes": ["product_management"],
                "eligible_for_recommendation": 1,
                "origin_type": "compressed_from_source",
                "verification_status": "source_summary",
                "weight_tier": "B",
            }
        )
    elif chunk_type == "employment":
        common.update(
            {
                "source_field": "company_name+position_title+start_date+end_date+work_description+is_employed",
                "start_month": "202301",
                "is_current": 1,
                "origin_type": "source_parsed",
                "verification_status": "verified",
                "weight_tier": "A",
            }
        )
    elif chunk_type == "project":
        common.update(
            {
                "source_table": "experts_project_experience",
                "source_field": "project_title+company_name+start_date+end_date+work_description",
                "project_name": "某项目",
                "start_month": "202401",
                "end_month": "202406",
                "origin_type": "source_parsed",
                "verification_status": "verified",
                "weight_tier": "A",
            }
        )
    return common


def _deep_baseline(chunk_type: str, source_field_id: str) -> dict[str, Any]:
    return _baseline(chunk_type, source_field_id=source_field_id)


def _row_for(chunk_type: str, source_field_id: str, **overrides: Any) -> tuple[Any, ...]:
    values = _deep_baseline(chunk_type, source_field_id)
    values.update(overrides)
    return tuple(values[name] for name in MYSQL_EXPERT_CHUNK_SELECT_COLUMNS)


def _settings() -> MySQLSettings:
    return MySQLSettings(
        database="expert_dev",
        user="readonly",
        password=SecretStr("not-a-real-secret"),
    )


def _reader(rows: list[tuple[Any, ...]], *, timezone: str | None = "Asia/Shanghai"):
    fake_connection = FakeConnection(rows)
    return (
        MySqlExpertChunkReader(
            _settings(),
            database_timezone=ZoneInfo(timezone) if timezone else None,
            connector_factory=lambda **_: fake_connection,
        ),
        fake_connection,
    )


# ---- profile / employment / project 扁平行 → Contract 4.0.0 ----

def test_profile_flat_row_maps_to_contract_v400() -> None:
    reader, fake = _reader(
        [_row_for("profile_summary", "900001", serial_number=None)]
    )

    result = reader.read()

    assert result.input_count == 1
    assert result.errors == []
    record = result.records[0]
    assert record.contract_version == "4.0.0"
    assert record.chunk_type == "profile_summary"
    assert record.vector_doc_id == "expert:7:profile_summary:900001:0"
    assert record.expert_id == 7
    assert record.serial_number is None
    assert record.metadata.job_status_norm == "active"
    assert record.metadata.abroad_flag_norm is False
    assert record.metadata.work_years == Decimal("12.50")
    assert record.metadata.school_names_raw == ["测试大学"]
    assert record.metadata.major_names_raw == ["软件工程"]
    assert record.metadata.function_codes == ["product_management"]
    assert record.metadata.eligible_for_recommendation is True
    assert record.metadata.origin_type == "compressed_from_source"
    assert record.metadata.verification_status == "source_summary"
    assert record.versions.chunk_policy_version == "1.3"
    assert record.versions.metadata_schema_version == "3.0"
    assert fake.cursor_instance.closed is True
    assert fake.closed is True


def test_employment_flat_row_maps_to_contract_v400() -> None:
    reader, _ = _reader(
        [_row_for("employment", "21307", is_current=0, end_month="202001")]
    )

    result = reader.read()

    assert result.errors == []
    record = result.records[0]
    assert record.chunk_type == "employment"
    assert record.metadata.start_month == "202301"
    assert record.metadata.end_month == "202001"
    assert record.metadata.is_current is False
    assert record.metadata.eligible_for_recommendation is None
    assert record.metadata.work_years is None
    assert record.metadata.job_status_norm is None
    assert record.metadata.school_names_raw == []
    assert record.metadata.origin_type == "source_parsed"
    assert record.metadata.verification_status == "verified"


def test_project_flat_row_maps_to_contract_v400() -> None:
    reader, _ = _reader([_row_for("project", "161")])

    result = reader.read()

    assert result.errors == []
    record = result.records[0]
    assert record.chunk_type == "project"
    assert record.metadata.project_name == "某项目"
    assert record.metadata.start_month == "202401"
    assert record.metadata.end_month == "202406"
    assert record.metadata.is_current is None
    assert record.metadata.eligible_for_recommendation is None
    assert record.metadata.verification_status == "verified"


# ---- 类型处理 ----

def test_work_years_is_decimal_not_float() -> None:
    reader, _ = _reader(
        [_row_for("profile_summary", "900001", work_years=Decimal("12.50"))]
    )

    record = reader.read().records[0]
    assert record.metadata.work_years == Decimal("12.50")
    assert isinstance(record.metadata.work_years, Decimal)
    assert not isinstance(record.metadata.work_years, float)


def test_work_years_accepts_decimal_string_for_driver() -> None:
    reader, _ = _reader(
        [_row_for("profile_summary", "900001", work_years="9.75")]
    )

    record = reader.read().records[0]
    assert record.metadata.work_years == Decimal("9.75")
    assert isinstance(record.metadata.work_years, Decimal)


def test_json_column_as_string_is_parsed_to_list() -> None:
    for column in ("industry_codes", "function_codes", "topic_codes",
                   "school_names_raw", "major_names_raw"):
        reader, _ = _reader(
            [
                _row_for(
                    "profile_summary",
                    "900001",
                    **{column: json.dumps(["a", "b"])},
                )
            ]
        )
        record = reader.read().records[0]
        assert getattr(record.metadata, column) == ["a", "b"]


def test_json_column_as_parsed_list_is_passed_through() -> None:
    reader, _ = _reader(
        [_row_for("profile_summary", "900001", school_names_raw=["大学"], industry_codes=["a"])]
    )

    record = reader.read().records[0]
    assert record.metadata.school_names_raw == ["大学"]
    assert record.metadata.industry_codes == ["a"]


def test_empty_array_is_preserved() -> None:
    reader, _ = _reader(
        [_row_for("project", "161", industry_codes=[], school_names_raw=[])]
    )

    record = reader.read().records[0]
    assert record.metadata.industry_codes == []
    assert record.metadata.school_names_raw == []


def test_tinyint_bool_conversion() -> None:
    reader, _ = _reader(
        [_row_for("employment", "21307", is_active=1, is_current=1)]
    )

    record = reader.read().records[0]
    assert record.is_active is True
    assert record.metadata.is_current is True


def test_tinyint_zero_is_false() -> None:
    reader, _ = _reader(
        [_row_for("employment", "21307", is_active=0, is_current=0)]
    )

    record = reader.read().records[0]
    assert record.is_active is False
    assert record.metadata.is_current is False


def test_invalid_bool_fails_closed() -> None:
    reader, _ = _reader([_row_for("employment", "21307", is_active=2)])

    result = reader.read()

    assert result.records == []
    assert result.errors[0].code == "boolean_field_invalid"
    assert result.errors[0].row_id == 65
    assert result.errors[0].vector_doc_id == "expert:7:employment:21307:0"


# ---- nullable 与 NULL 处理 ----

def test_nullable_metadata_fields_default_to_none_or_empty() -> None:
    reader, _ = _reader([_row_for("employment", "21307")])

    record = reader.read().records[0]
    assert record.metadata.job_status_norm is None
    assert record.metadata.abroad_flag_norm is None
    assert record.metadata.work_years is None
    assert record.metadata.eligible_for_recommendation is None
    assert record.metadata.function_codes == []


def test_source_updated_at_null_is_allowed() -> None:
    reader, _ = _reader([_row_for("employment", "21307", source_updated_at=None)])

    result = reader.read()

    assert result.errors == []
    assert result.records[0].source.source_updated_at is None


def test_updated_at_null_is_rejected() -> None:
    reader, _ = _reader([_row_for("employment", "21307", updated_at=None)])

    result = reader.read()

    assert result.records == []
    assert result.errors[0].code == "required_datetime_missing"


# ---- 时间 / 主键 / 哈希 / 独立列 ----

def test_update_and_source_times_are_asia_shanghai_iso() -> None:
    reader, _ = _reader(
        [
            _row_for(
                "employment",
                "21307",
                # 模拟驱动返回无时区 DATETIME，Reader 根据 Asia/Shanghai 附加时区。
                source_updated_at=datetime(2026, 8, 11, 16, 28, 36),  # noqa: DTZ001
            )
        ]
    )

    record = reader.read().records[0]
    assert record.source.source_updated_at == "2026-08-11T16:28:36+08:00"
    assert record.updated_at == "2026-08-11T16:28:47.391385+08:00"


def test_vector_doc_id_reads_independent_column() -> None:
    reader, _ = _reader([_row_for("project", "161")])

    record = reader.read().records[0]
    # 来自独立扁平列，而不是 metadata 副本。
    assert record.vector_doc_id == "expert:7:project:161:0"
    assert not hasattr(record.metadata, "vector_doc_id")


def test_metadata_hash_is_forwarded() -> None:
    reader, _ = _reader([_row_for("profile_summary", "900001", metadata_hash="c" * 64)])

    record = reader.read().records[0]
    assert record.metadata_hash == "c" * 64


def test_reader_does_not_depend_on_deprecated_columns() -> None:
    # SELECT 固定列不包含任何废弃字段。
    deprecated = {
        "metadata_json",
        "chunk_id",
        "source_item_key",
        "dictionary_version",
        "data_version",
        "token_count_estimate",
    }
    assert not (set(MYSQL_EXPERT_CHUNK_SELECT_COLUMNS) & deprecated)


def test_reader_uses_explicit_read_only_columns_and_limit() -> None:
    reader, fake = _reader([_row_for("employment", "21307")])

    reader.read(limit=1)

    query, params = fake.cursor_instance.executed[0]
    upper_query = query.upper()
    assert "SELECT *" not in upper_query
    assert " FROM EXPERT_CHUNK ORDER BY ID ASC LIMIT %S" in upper_query
    assert params == (1,)
    import re

    assert not any(
        re.search(rf"\b{keyword}\b", upper_query)
        for keyword in ("INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "TRUNCATE")
    )


def test_reader_does_not_guess_timezone_for_naive_mysql_datetime() -> None:
    reader, _ = _reader([_row_for("employment", "21307")], timezone=None)

    result = reader.read()

    assert result.records == []
    assert result.errors[0].code == "datetime_timezone_missing"
    assert "任职" not in result.errors[0].message