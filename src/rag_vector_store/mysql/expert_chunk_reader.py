"""只读读取 MySQL ``expert_chunk`` 并映射到正式输入 Contract。

Reader 位于 MySQL 与 Contract Validator 之间：它只读取明确列出的字段、解析
``metadata_json``、组装 ``source``/``versions``，然后交给严格 Contract 模型。
它不清洗、摘要、截断、重算 ``source_hash``，也不负责 Embedding、Task State 或
Milvus 写入；数据库 ``id`` 只保留为 Row identity 和错误定位信息。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Any

from pydantic import ValidationError

from ..chunking.contracts import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    ExpertChunkEmbeddingInput,
)
from ..config import MySQLSettings
from .connection import MySQLConnection

# 这些列是本 Reader 实际消费的字段。未列出的真实表字段不是输入 Contract 的来源，
# 也不会被 Reader 通过 SELECT * 隐式读取；固定顺序同时约束 tuple Row 的映射。
MYSQL_EXPERT_CHUNK_SELECT_COLUMNS = (
    "id",
    "chunk_id",
    "expert_id",
    "serial_number",
    "chunk_type",
    "chunk_subtype",
    "chunk_index",
    "chunk_total",
    "chunk_text",
    "char_count",
    "metadata_json",
    "source_table",
    "source_record_id",
    "source_item_key",
    "source_field",
    "source_field_id",
    "source_version",
    "source_updated_at",
    "source_hash",
    "content_hash",
    "chunk_policy_version",
    "metadata_schema_version",
    "dictionary_version",
    "data_version",
    "is_active",
    "updated_at",
)
_SELECT_FIELDS_PREFIX = (
    "SELECT "
    + ", ".join(MYSQL_EXPERT_CHUNK_SELECT_COLUMNS)
    + " FROM expert_chunk"
)
_SELECT_PREFIX = _SELECT_FIELDS_PREFIX + " ORDER BY id ASC"


class MySQLRowMappingError(ValueError):
    """单条 MySQL Row 无法无损映射到输入 Contract。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        row_id: int | None = None,
        chunk_id: str | None = None,
        chunk_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.row_id = row_id
        self.chunk_id = chunk_id
        self.chunk_type = chunk_type


@dataclass(frozen=True)
class MySQLReadError:
    """安全的单行 Reader 错误，不保存正文或大 JSON。"""

    row_id: int | None
    chunk_id: str | None
    chunk_type: str | None
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
    """使用同步 MySQL 连接只读读取 ``expert_chunk``。

    ``id`` 只作为数据库 Row identity 用于排序和错误定位，不会写入
    :class:`ExpertChunkEmbeddingInput`，也不会替代 ``chunk_id``。Reader 不实现
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
        把不属于输入契约的数据意外带入映射层。单条映射失败只进入 errors。
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
                        chunk_id=exc.chunk_id,
                        chunk_type=exc.chunk_type,
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
        """将一行固定列顺序的 MySQL 结果映射为 Contract 输入。

        ``metadata_json`` 必须是 JSON object；来源和版本使用真实表中的扁平列
        组装成 Contract 嵌套对象。数据库 ``id`` 不进入 payload，避免把 Row identity
        与领域 ``chunk_id`` 混淆；最终仍由 Pydantic Contract 做严格类型/枚举检查。
        """

        values = self._row_values(row)
        row_id = _optional_row_id(values.get("id"))
        chunk_id = _optional_text(values.get("chunk_id"))
        chunk_type = _optional_text(values.get("chunk_type"))

        try:
            # 这里只解析和组装，不修复 JSON、不填默认值；无法无损表示时必须拒绝
            # 该 Row，避免把上游数据错误伪装成合法 Contract。
            metadata = _json_object(values["metadata_json"], "metadata_json")
            payload: dict[str, Any] = {
                "contract_version": EXPERT_CHUNK_CONTRACT_VERSION,
                "chunk_id": values["chunk_id"],
                "vector_doc_id": metadata.get("vector_doc_id"),
                "expert_id": values["expert_id"],
                "serial_number": values["serial_number"],
                "chunk_type": values["chunk_type"],
                "chunk_subtype": values["chunk_subtype"],
                "chunk_index": values["chunk_index"],
                "chunk_total": values["chunk_total"],
                "chunk_text": values["chunk_text"],
                "char_count": values["char_count"],
                "content_hash": values["content_hash"],
                "metadata": metadata,
                "source": {
                    "source_table": values["source_table"],
                    "source_record_id": values["source_record_id"],
                    "source_item_key": values["source_item_key"],
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
                    field: values[field]
                    for field in (
                        "data_version",
                        "chunk_policy_version",
                        "metadata_schema_version",
                        "dictionary_version",
                    )
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
            if exc.row_id is not None or exc.chunk_id is not None:
                raise
            raise MySQLRowMappingError(
                exc.code,
                str(exc),
                row_id=row_id,
                chunk_id=chunk_id,
                chunk_type=chunk_type,
            ) from exc
        except ValidationError as exc:
            raise MySQLRowMappingError(
                "contract_model_invalid",
                "MySQL Row 不符合 ExpertChunkEmbeddingInput 结构约束",
                row_id=row_id,
                chunk_id=chunk_id,
                chunk_type=chunk_type,
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise MySQLRowMappingError(
                "field_mapping_invalid",
                "MySQL Row 字段无法映射为 Contract 输入",
                row_id=row_id,
                chunk_id=chunk_id,
                chunk_type=chunk_type,
            ) from exc

    @staticmethod
    def _row_values(row: Any) -> dict[str, Any]:
        """把驱动返回的 dict/tuple 归一化为固定列名映射。"""

        if isinstance(row, Mapping):
            missing = [name for name in MYSQL_EXPERT_CHUNK_SELECT_COLUMNS if name not in row]
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


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    """解析 MySQL JSON 列并只接受 object 根节点。"""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise MySQLRowMappingError(
            "json_field_invalid",
            f"{field_name} 必须是 JSON object",
        )
    try:
        parsed = json.loads(value, parse_constant=_reject_non_finite_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MySQLRowMappingError(
            "json_field_invalid",
            f"{field_name} 不是合法 JSON object",
        ) from exc
    if not isinstance(parsed, dict):
        raise MySQLRowMappingError(
            "json_field_invalid",
            f"{field_name} 必须是 JSON object",
        )
    return parsed


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限数值: {value}")


def _mysql_bool(value: Any, field_name: str) -> bool:
    """把 MySQL ``TINYINT(1)`` 的 0/1 映射为 Python bool。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise MySQLRowMappingError(
        "boolean_field_invalid",
        f"{field_name} 必须是 MySQL TINYINT(1) 的 0/1",
    )


def _format_datetime(
    value: Any,
    field_name: str,
    *,
    database_timezone: tzinfo | None,
) -> str | None:
    """将数据库时间转换为带时区的 Contract 字符串。

    Reader 层允许驱动返回无时区 ``DATETIME``，但只有显式提供
    ``database_timezone`` 时才会附加时区；这样不会悄悄猜测数据库时区。该规则
    只用于 Contract 输出，保留数据库返回的时间精度。
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
