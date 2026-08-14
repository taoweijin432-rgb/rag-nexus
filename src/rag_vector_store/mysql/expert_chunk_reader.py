"""只读读取 MySQL ``expert_chunk`` 扁平表并映射为正式输入 Contract 4.0.0。

Reader 位于 MySQL 与 Contract Validator 之间：它只读取明确列出的正式列、解析
JSON 数组与 DECIMAL/bool/datetime，并从扁平列组装 Contract 的顶层、
``metadata``/``source``/``versions``，然后交给严格 Contract 模型。它不清洗、
摘要、截断、重算 ``source_hash``，也不负责 Embedding、Task State 或 Milvus
写入；数据库 ``id`` 只保留为 Row identity，``vector_doc_id`` 用于错误定位。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, tzinfo
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from expert_contracts.models import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    ExpertChunkEmbeddingInput,
)

from ..config import MySQLSettings
from .connection import MySQLConnection

# 本 Reader 实际消费的正式列集合（expert_chunk 表的子集，非全部 48 列）。
# 未列出的真实表字段不属于输入 Contract，也不会被 SELECT * 隐式读取；这里刻意
# 排除已废弃列（chunk_id/metadata_json/source_item_key/dictionary_version/data_version
# /token_count_estimate）以及下游回填字段（embedding_model_*/indexed_at）和
# 批次/建表字段（build_batch_id/created_at）。固定顺序同时约束 tuple Row 映射。
MYSQL_EXPERT_CHUNK_SELECT_COLUMNS = (
    "id",
    "vector_doc_id",
    "expert_id",
    "serial_number",
    "chunk_type",
    "chunk_subtype",
    "chunk_index",
    "chunk_total",
    "chunk_text",
    "char_count",
    "source_table",
    "source_field",
    "source_field_id",
    "source_version",
    "source_updated_at",
    "source_hash",
    "content_hash",
    "metadata_hash",
    "company_name_raw",
    "company_name_norm",
    "role_name_raw",
    "role_code",
    "role_level",
    "job_status_norm",
    "abroad_flag_norm",
    "work_years",
    "project_name",
    "start_month",
    "end_month",
    "is_current",
    "industry_codes",
    "function_codes",
    "topic_codes",
    "school_names_raw",
    "major_names_raw",
    "eligible_for_recommendation",
    "origin_type",
    "verification_status",
    "weight_tier",
    "chunk_policy_version",
    "metadata_schema_version",
    "is_active",
    "updated_at",
)
_JSON_LIST_COLUMNS = frozenset(
    {
        "industry_codes",
        "function_codes",
        "topic_codes",
        "school_names_raw",
        "major_names_raw",
    }
)
_SELECT_FIELDS_PREFIX = (
    "SELECT "
    + ", ".join(MYSQL_EXPERT_CHUNK_SELECT_COLUMNS)
    + " FROM expert_chunk"
)
_SELECT_PREFIX = _SELECT_FIELDS_PREFIX + " ORDER BY id ASC"


class MySQLRowMappingError(ValueError):
    """单条 MySQL Row 无法无损映射到输入 Contract（安全定位信息）。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        row_id: int | None = None,
        vector_doc_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.row_id = row_id
        self.vector_doc_id = vector_doc_id


@dataclass(frozen=True)
class MySQLReadError:
    """安全的单行 Reader 错误，不保存正文或大 JSON。"""

    row_id: int | None
    vector_doc_id: str | None
    code: str
    message: str


@dataclass(frozen=True)
class MySQLReadResult:
    """一次全量或限量读取的映射结果。

    ``input_count`` 统计数据库返回的 Row 数，``records`` 只包含成功映射的
    Contract，``errors`` 则保留安全定位信息。因此调用方可以区分“数据库读到
    多少行”和“多少行真正满足输入契约”。
    """

    input_count: int
    records: list[ExpertChunkEmbeddingInput]
    errors: list[MySQLReadError]

    @property
    def valid_count(self) -> int:
        """返回成功映射为输入 Contract 的记录数。"""

        return len(self.records)


class MySqlExpertChunkReader:
    """使用同步 MySQL 连接只读读取 ``expert_chunk`` 扁平表。

    调用时机：
        - 正式批处理/E2E 入口（如 ``scripts/task2_acceptance.py``）在调用
          ``ContractValidator`` / Ingestion 之前读取真实数据；
        - 任何需要从 MySQL 重建 ``ExpertChunkEmbeddingInput 4.0.0`` 的读取端。

    输入：
        ``MySQLSettings``（只读连接配置，见 ``config.py``）。

    数据流：
        MySQL ``expert_chunk`` 行
        -> ``read()``（固定 SELECT，按 id 排序）
        -> ``map_row()``（Python 类型转换：JSON 数组 / DECIMAL / bool / datetime）
        -> ``ExpertChunkEmbeddingInput 4.0.0``（Pydantic 严格校验）
        -> ``MySQLReadResult``（records + errors 分离）

    职责边界：
        - Reader 只做读取和映射，**不做 Embedding**，也不做 Eligibility、Task
          State 或 Milvus 写入；
        - 不清洗、摘要、截断、重算 ``source_hash``；
        - ``id`` 只作为数据库 Row identity 用于排序和错误定位，不会写入 Contract；
        - ``vector_doc_id`` 直接读取独立列，用于错误定位，不生成、不重算。
    """

    def __init__(
        self,
        settings: MySQLSettings,
        *,
        database_timezone: tzinfo | None = None,
        connector_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.database_timezone = database_timezone
        self.connector_factory = connector_factory

    def read(self, *, limit: int | None = None) -> MySQLReadResult:
        """按 ``id`` 顺序读取全部或前 ``limit`` 条记录。

        使用固定 SELECT 而不是 ``SELECT *``，避免上游新增列时改变 tuple 位置或
        把不属于输入契约的数据意外带入映射层。单条映射失败只进入 errors，
        由调用方（如验收脚本）决定是否 fatal。

        输出：
            ``MySQLReadResult`` —— input_count（数据库行数）/ records（成功映射
            Contract 列表）/ errors（安全定位错误）。
        """

        if limit is not None and (isinstance(limit, bool) or limit < 1):
            raise ValueError("limit 必须是正整数或 None")

        with MySQLConnection(
            self.settings,
            connector_factory=self.connector_factory,
        ) as connection:
            cursor = connection.cursor()
            try:
                if limit is None:
                    cursor.execute(_SELECT_PREFIX)
                else:
                    cursor.execute(f"{_SELECT_PREFIX} LIMIT %s", (limit,))
                rows = cursor.fetchall()
            finally:
                cursor.close()

        records: list[ExpertChunkEmbeddingInput] = []
        errors: list[MySQLReadError] = []
        for row in rows:
            try:
                records.append(self.map_row(row))
            except MySQLRowMappingError as exc:
                errors.append(
                    MySQLReadError(
                        row_id=exc.row_id,
                        vector_doc_id=exc.vector_doc_id,
                        code=exc.code,
                        message=str(exc),
                    )
                )

        return MySQLReadResult(
            input_count=len(rows),
            records=records,
            errors=errors,
        )

    def map_row(self, row: Any) -> ExpertChunkEmbeddingInput:
        """将一行固定列顺序的 MySQL 扁平结果映射为 Contract 4.0.0。

        从独立扁平列组装 ``vector_doc_id``（顶层）与业务 ``metadata``，
        来源/版本字段扁平透传；数据库 ``id`` 不进入 payload。最终仍由
        Pydantic Contract 做严格类型/枚举/公式校验。
        """

        values = self._row_values(row)
        row_id = _optional_row_id(values.get("id"))
        vector_doc_id = _optional_text(values.get("vector_doc_id"))

        # 扁平列 → Contract 组装点：顶层身份、业务 metadata、来源/版本、生命周期。
        # 数据库 id 只用于错误定位，不进入 payload；vector_doc_id 直接透传。
        try:
            payload: dict[str, Any] = {
                "contract_version": EXPERT_CHUNK_CONTRACT_VERSION,
                "vector_doc_id": values["vector_doc_id"],
                "expert_id": values["expert_id"],
                "serial_number": _optional_text(values["serial_number"]),
                "chunk_type": values["chunk_type"],
                "chunk_subtype": _optional_text(values["chunk_subtype"]),
                "chunk_index": values["chunk_index"],
                "chunk_total": values["chunk_total"],
                "chunk_text": values["chunk_text"],
                "char_count": values["char_count"],
                "content_hash": values["content_hash"],
                "metadata_hash": values["metadata_hash"],
                "metadata": {
                    "company_name_raw": _optional_text(values["company_name_raw"]),
                    "company_name_norm": _optional_text(values["company_name_norm"]),
                    "role_name_raw": _optional_text(values["role_name_raw"]),
                    "role_code": _optional_text(values["role_code"]),
                    "role_level": _optional_text(values["role_level"]),
                    "job_status_norm": _optional_text(values["job_status_norm"]),
                    "abroad_flag_norm": _optional_bool(
                        values["abroad_flag_norm"], "abroad_flag_norm"
                    ),
                    "work_years": _decimal_or_none(values["work_years"], "work_years"),
                    "project_name": _optional_text(values["project_name"]),
                    "start_month": _optional_text(values["start_month"]),
                    "end_month": _optional_text(values["end_month"]),
                    "is_current": _optional_bool(values["is_current"], "is_current"),
                    "industry_codes": _json_string_list(
                        values["industry_codes"], "industry_codes"
                    ),
                    "function_codes": _json_string_list(
                        values["function_codes"], "function_codes"
                    ),
                    "topic_codes": _json_string_list(
                        values["topic_codes"], "topic_codes"
                    ),
                    "school_names_raw": _json_string_list(
                        values["school_names_raw"], "school_names_raw"
                    ),
                    "major_names_raw": _json_string_list(
                        values["major_names_raw"], "major_names_raw"
                    ),
                    "eligible_for_recommendation": _optional_bool(
                        values["eligible_for_recommendation"],
                        "eligible_for_recommendation",
                    ),
                    "origin_type": values["origin_type"],
                    "verification_status": values["verification_status"],
                    "weight_tier": values["weight_tier"],
                },
                "source": {
                    "source_table": values["source_table"],
                    "source_field": values["source_field"],
                    "source_field_id": values["source_field_id"],
                    "source_version": values["source_version"],
                    "source_updated_at": _format_datetime(
                        values["source_updated_at"],
                        "source_updated_at",
                        database_timezone=self.database_timezone,
                    ),
                    "source_hash": values["source_hash"],
                },
                "versions": {
                    "chunk_policy_version": values["chunk_policy_version"],
                    "metadata_schema_version": values["metadata_schema_version"],
                },
                "is_active": _mysql_bool(values["is_active"], "is_active"),
                "updated_at": _format_datetime(
                    values["updated_at"],
                    "updated_at",
                    database_timezone=self.database_timezone,
                ),
            }
            return ExpertChunkEmbeddingInput.model_validate(payload)
        except MySQLRowMappingError as exc:
            if exc.row_id is not None or exc.vector_doc_id is not None:
                raise
            raise MySQLRowMappingError(
                exc.code,
                str(exc),
                row_id=row_id,
                vector_doc_id=vector_doc_id,
            ) from exc
        except ValidationError as exc:
            raise MySQLRowMappingError(
                "contract_model_invalid",
                "MySQL Row 不符合 ExpertChunkEmbeddingInput 4.0.0 结构约束",
                row_id=row_id,
                vector_doc_id=vector_doc_id,
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise MySQLRowMappingError(
                "field_mapping_invalid",
                "MySQL Row 字段无法映射为 Contract 输入",
                row_id=row_id,
                vector_doc_id=vector_doc_id,
            ) from exc

    @staticmethod
    def _row_values(row: Any) -> dict[str, Any]:
        """把驱动返回的 dict/tuple 归一化为固定列名映射。"""

        if isinstance(row, Mapping):
            missing = [
                name
                for name in MYSQL_EXPERT_CHUNK_SELECT_COLUMNS
                if name not in row
            ]
            if missing:
                raise MySQLRowMappingError(
                    "row_columns_missing",
                    "MySQL Row 缺少 Reader 所需列",
                )
            return {name: row[name] for name in MYSQL_EXPERT_CHUNK_SELECT_COLUMNS}

        if not isinstance(row, (tuple, list)) or len(row) != len(
            MYSQL_EXPERT_CHUNK_SELECT_COLUMNS
        ):
            raise MySQLRowMappingError(
                "row_shape_invalid",
                "MySQL Row 列数与固定 SELECT 不一致",
            )
        return dict(zip(MYSQL_EXPERT_CHUNK_SELECT_COLUMNS, row, strict=True))


def _json_string_list(value: Any, field_name: str) -> list[str]:
    """解析 MySQL JSON 数组列并只接受字符串数组。

    数据库驱动可能返回已解析的 ``list``、JSON 文本字符串或 ``bytes``；
    统一反序列化为 ``list[str]``。数组列在正式表中为 ``NOT NULL``，不适用某
    ``chunk_type`` 时上游写入 ``[]``，因此 ``None`` 属于失败（fail-closed）。
    """

    if value is None:
        raise MySQLRowMappingError(
            "json_field_null",
            f"{field_name} 是 NOT NULL JSON 数组，不能为 NULL",
        )
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_constant=_reject_non_finite_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise MySQLRowMappingError(
                "json_field_invalid",
                f"{field_name} 不是合法 JSON 数组",
            ) from exc
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise MySQLRowMappingError(
            "json_field_invalid",
            f"{field_name} 必须是 list[str] 或 JSON 数组",
        )
    return list(value)


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限数值: {value}")


def _mysql_bool(value: Any, field_name: str) -> bool:
    """把 ``NOT NULL TINYINT(1)`` 的 0/1 映射为 Python bool（fail-closed）。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise MySQLRowMappingError(
        "boolean_field_invalid",
        f"{field_name} 必须是 MySQL TINYINT(1) 的 0/1",
    )


def _optional_bool(value: Any, field_name: str) -> bool | None:
    """把可空 ``TINYINT(1)`` 的 0/1/NULL 映射为 bool 或 None（fail-closed）。"""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise MySQLRowMappingError(
        "boolean_field_invalid",
        f"{field_name} 必须是 MySQL TINYINT(1) 的 NULL/0/1",
    )


def _decimal_or_none(value: Any, field_name: str) -> Decimal | None:
    """把 ``DECIMAL(5,2)`` 映射为 ``Decimal | None``，禁止转 float。

    数据库驱动对 DECIMAL 通常返回 ``Decimal``；这里只接受 ``Decimal`` / 整型 /
    十进制字符串，直接构造 ``Decimal``，不做任何浮点中间转换，保证精度。布尔值
    不是合法工作年限。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise MySQLRowMappingError(
            "decimal_field_invalid",
            f"{field_name} 必须是 DECIMAL(5,2) 数值或 NULL",
        )
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            return Decimal(value)
        except (ValueError, ArithmeticError) as exc:
            raise MySQLRowMappingError(
                "decimal_field_invalid",
                f"{field_name} 不是合法 DECIMAL(5,2) 数值",
            ) from exc
    raise MySQLRowMappingError(
        "decimal_field_invalid",
        f"{field_name} 必须是 DECIMAL(5,2) 数值或 NULL",
    )


def _format_datetime(
    value: Any,
    field_name: str,
    *,
    database_timezone: tzinfo | None,
) -> str | None:
    """将数据库时间转换为带时区的 Contract 字符串（按 ``Asia/Shanghai`` 解释）。

    ``updated_at`` 不允许为 ``NULL``；``source_updated_at`` 允许为 ``NULL``。
    Reader 层允许驱动返回无时区 ``DATETIME``，但只有显式提供
    ``database_timezone`` 时才附加时区；这样不会悄悄猜测数据库时区。该规则
    只用于 Contract 输出，保留数据库返回的时间精度，且不修改 ``source_version``。
    """

    if value is None:
        if field_name == "updated_at":
            raise MySQLRowMappingError(
                "required_datetime_missing",
                "updated_at 不允许为 NULL",
            )
        return None

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MySQLRowMappingError(
                "datetime_invalid",
                f"{field_name} 不是合法 ISO 时间",
            ) from exc
    else:
        raise MySQLRowMappingError(
            "datetime_invalid",
            f"{field_name} 不是 datetime 或 ISO 时间字符串",
        )

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if database_timezone is None:
            raise MySQLRowMappingError(
                "datetime_timezone_missing",
                f"{field_name} 是无时区 DATETIME，Reader 未提供 database_timezone",
            )
        parsed = parsed.replace(tzinfo=database_timezone)
    return parsed.isoformat()


def _optional_row_id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "MYSQL_EXPERT_CHUNK_SELECT_COLUMNS",
    "MySQLReadError",
    "MySQLReadResult",
    "MySQLRowMappingError",
    "MySqlExpertChunkReader",
]