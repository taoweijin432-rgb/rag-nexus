"""Resume 四张来源表的只读结构契约与预检。

本模块只查询 ``information_schema``，不读取 Resume 正文，不创建或修改来源表。
来源表 DDL 由部署人员执行；程序只在结构不满足读取契约时 fail-closed。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class _Cursor(Protocol):
    def execute(self, sql: str, params: Sequence[object] | None = None) -> object: ...

    def fetchall(self) -> Sequence[Mapping[str, Any]]: ...

    def close(self) -> None: ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor: ...


_INTEGER_TYPES = frozenset({"tinyint", "smallint", "mediumint", "int", "integer", "bigint"})
_NUMBER_TYPES = _INTEGER_TYPES | frozenset({"decimal", "numeric", "float", "double", "real"})
_TEXT_TYPES = frozenset({"char", "varchar", "tinytext", "text", "mediumtext", "longtext"})
_TEMPORAL_TYPES = frozenset({"date", "datetime", "timestamp"})


@dataclass(frozen=True)
class ColumnRequirement:
    """单个必要列的兼容类型和关键可空约束。"""

    data_types: frozenset[str]
    nullable: bool | None = None


@dataclass(frozen=True)
class IndexRequirement:
    """按索引列前缀匹配的必要索引；不绑定部署侧索引名称。"""

    label: str
    columns: tuple[str, ...]
    unique: bool | None = None


@dataclass(frozen=True)
class ResumeSourceSchemaReport:
    """Resume 来源结构预检结果，不包含数据库凭据或业务数据。"""

    schema_name: str
    issues: tuple[str, ...]

    @property
    def is_compatible(self) -> bool:
        return not self.issues


class ResumeSourceSchemaError(RuntimeError):
    """四张来源表尚未部署或结构不满足读取契约。"""


RESUME_SOURCE_TABLES = (
    "resume_profile",
    "resume_work_experience",
    "resume_project_experience",
    "resume_education_experience",
)

_COMMON_TIMESTAMPS = {
    "created_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=False),
    "updated_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=False),
    "deleted_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
}

RESUME_SOURCE_COLUMN_REQUIREMENTS: Mapping[str, Mapping[str, ColumnRequirement]] = {
    "resume_profile": {
        "id": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "resume_id": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "serial_number": ColumnRequirement(_TEXT_TYPES, nullable=False),
        "company": ColumnRequirement(_TEXT_TYPES, nullable=False),
        "job_status": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "job_name": ColumnRequirement(_TEXT_TYPES, nullable=False),
        "contact_status": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "doubtful_flag": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "abroad_flag": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "background": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "resume_updated_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "resume_deleted_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "work_years": ColumnRequirement(_NUMBER_TYPES, nullable=True),
        **_COMMON_TIMESTAMPS,
    },
    "resume_work_experience": {
        "id": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "company_name": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "start_date": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "end_date": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "position_title": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "work_description": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "resume_work_experience_updated_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "resume_work_experience_deleted_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "resume_id": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "is_employed": ColumnRequirement(_INTEGER_TYPES, nullable=True),
        **_COMMON_TIMESTAMPS,
    },
    "resume_project_experience": {
        "id": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "company_name": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "start_date": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "end_date": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "project_title": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "work_description": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "resume_project_experience_updated_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "resume_project_experience_deleted_at": ColumnRequirement(_TEMPORAL_TYPES, nullable=True),
        "resume_id": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        **_COMMON_TIMESTAMPS,
    },
    "resume_education_experience": {
        "id": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        "school_name": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "major": ColumnRequirement(_TEXT_TYPES, nullable=True),
        "resume_id": ColumnRequirement(_INTEGER_TYPES, nullable=False),
        **_COMMON_TIMESTAMPS,
    },
}

_COMMON_INDEXES = (
    IndexRequirement("PRIMARY KEY(id)", ("id",), unique=True),
    IndexRequirement("scan(updated_at,id)", ("updated_at", "id")),
)
RESUME_SOURCE_INDEX_REQUIREMENTS: Mapping[str, tuple[IndexRequirement, ...]] = {
    "resume_profile": _COMMON_INDEXES
    + (IndexRequirement("UNIQUE(resume_id)", ("resume_id",), unique=True),),
    "resume_work_experience": _COMMON_INDEXES
    + (IndexRequirement("lookup(resume_id)", ("resume_id",)),),
    "resume_project_experience": _COMMON_INDEXES
    + (IndexRequirement("lookup(resume_id)", ("resume_id",)),),
    "resume_education_experience": _COMMON_INDEXES
    + (IndexRequirement("lookup(resume_id)", ("resume_id",)),),
}


def inspect_resume_source_schema(
    connection: _Connection,
    *,
    schema_name: str,
) -> ResumeSourceSchemaReport:
    """只读检查四表列和必要索引，表不存在时返回问题而不是尝试创建。"""

    normalized_schema = _validate_schema_name(schema_name)
    placeholders = ", ".join(["%s"] * len(RESUME_SOURCE_TABLES))
    params: tuple[object, ...] = (normalized_schema, *RESUME_SOURCE_TABLES)

    cursor = connection.cursor()
    try:
        cursor.execute(
            f"""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({placeholders})
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """,
            params,
        )
        column_rows = cursor.fetchall()
        cursor.execute(
            f"""
            SELECT TABLE_NAME, INDEX_NAME, NON_UNIQUE, SEQ_IN_INDEX, COLUMN_NAME
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({placeholders})
            ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
            """,
            params,
        )
        index_rows = cursor.fetchall()
    finally:
        cursor.close()

    issues = _validate_columns(column_rows)
    issues.extend(_validate_indexes(index_rows, column_rows=column_rows))
    return ResumeSourceSchemaReport(schema_name=normalized_schema, issues=tuple(issues))


def require_resume_source_schema(
    connection: _Connection,
    *,
    schema_name: str,
) -> ResumeSourceSchemaReport:
    """执行只读预检；不兼容时阻止 Resume 继续启动。"""

    report = inspect_resume_source_schema(connection, schema_name=schema_name)
    if not report.is_compatible:
        raise ResumeSourceSchemaError("Resume 来源表结构不兼容: " + "; ".join(report.issues))
    return report


def _validate_schema_name(schema_name: str) -> str:
    if not isinstance(schema_name, str) or not schema_name.strip():
        raise ValueError("schema_name 必须是非空字符串")
    return schema_name.strip()


def _validate_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    actual: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        table = str(row["TABLE_NAME"])
        column = str(row["COLUMN_NAME"])
        actual[table][column] = row

    issues: list[str] = []
    for table in RESUME_SOURCE_TABLES:
        if table not in actual:
            issues.append(f"缺少表 {table}")
            continue
        for column, requirement in RESUME_SOURCE_COLUMN_REQUIREMENTS[table].items():
            row = actual[table].get(column)
            if row is None:
                issues.append(f"{table} 缺少列 {column}")
                continue
            data_type = str(row["DATA_TYPE"]).casefold()
            if data_type not in requirement.data_types:
                issues.append(f"{table}.{column} 类型不兼容: {data_type}")
            if requirement.nullable is not None:
                is_nullable = str(row["IS_NULLABLE"]).upper() == "YES"
                if is_nullable != requirement.nullable:
                    expected = "可空" if requirement.nullable else "非空"
                    issues.append(f"{table}.{column} 必须{expected}")
    return issues


def _validate_indexes(
    rows: Sequence[Mapping[str, Any]],
    *,
    column_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    present_tables = {str(row["TABLE_NAME"]) for row in column_rows}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["TABLE_NAME"]), str(row["INDEX_NAME"]))].append(row)

    actual: dict[str, list[tuple[tuple[str, ...], bool]]] = defaultdict(list)
    for (table, _index_name), members in grouped.items():
        ordered = sorted(members, key=lambda item: int(item["SEQ_IN_INDEX"]))
        columns = tuple(str(item["COLUMN_NAME"]) for item in ordered)
        unique = int(ordered[0]["NON_UNIQUE"]) == 0
        actual[table].append((columns, unique))

    issues: list[str] = []
    for table in RESUME_SOURCE_TABLES:
        if table not in present_tables:
            continue
        for requirement in RESUME_SOURCE_INDEX_REQUIREMENTS[table]:
            if not any(
                columns[: len(requirement.columns)] == requirement.columns
                and (requirement.unique is None or unique == requirement.unique)
                for columns, unique in actual[table]
            ):
                issues.append(f"{table} 缺少索引 {requirement.label}")
    return issues
