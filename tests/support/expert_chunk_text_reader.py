"""读取临时 ``expert_chunk_text`` 测试表的 Flat Row 适配器。

这个 Reader 只服务于临时测试链，不构造 ``ExpertChunkEmbeddingInput``，也不依赖
正式 Contract 的字段全集。它通过查询结果的列名动态组装 metadata，因此测试表
后续增删业务列不会被固化进生产模型。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from expert_index.config import DatabaseConfig

from .mysql_connection import TestMySQLConnection


@dataclass(frozen=True)
class FlatTestChunk:
    """临时测试链使用的最小 Flat Row DTO。"""

    vector_doc_id: str
    chunk_text: str
    metadata: dict[str, object]


class ExpertChunkTextTestReader:
    """按列名读取临时 ``expert_chunk_text`` 表。

    ``id`` 只用于排序，不进入结果；``vector_doc_id`` 和 ``chunk_text`` 是两个
    顶层测试字段；查询结果中的其他列全部保留到 metadata。Reader 不补字段、不
    重新计算 hash，也不执行正式 Contract 的类型转换。
    """

    _TABLE_NAME = "expert_chunk_text"
    _REQUIRED_COLUMNS = frozenset({"id", "vector_doc_id", "chunk_text"})
    _EXCLUDED_COLUMNS = frozenset({"id", "vector_doc_id", "chunk_text"})
    _JSON_LIST_COLUMNS = frozenset({"school_names_raw", "major_names_raw"})

    def __init__(
        self,
        settings: DatabaseConfig,
        *,
        connector_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.connector_factory = connector_factory
        self.last_columns: tuple[str, ...] = ()

    def read(self, *, limit: int | None = None) -> list[FlatTestChunk]:
        """读取当前表的全部行，或按 ``id`` 顺序读取前 ``limit`` 行。"""

        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit 必须是正整数或 None")

        with TestMySQLConnection(
            self.settings,
            connector_factory=self.connector_factory,
        ) as connection:
            cursor = connection.cursor()
            try:
                query = f"SELECT * FROM {self._TABLE_NAME} ORDER BY id ASC"
                if limit is None:
                    cursor.execute(query)
                else:
                    cursor.execute(f"{query} LIMIT %s", (limit,))
                columns = _column_names(cursor.description)
                rows = cursor.fetchall()
            finally:
                cursor.close()

        self.last_columns = columns
        missing = self._REQUIRED_COLUMNS - set(columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"expert_chunk_text 缺少必需测试列: {missing_text}")

        chunks = [self._map_row(row, columns) for row in rows]
        vector_doc_ids = [chunk.vector_doc_id for chunk in chunks]
        if len(vector_doc_ids) != len(set(vector_doc_ids)):
            raise ValueError("expert_chunk_text.vector_doc_id 在本批次内必须唯一")
        return chunks

    def _map_row(self, row: Any, columns: tuple[str, ...]) -> FlatTestChunk:
        values = _row_mapping(row, columns)
        vector_doc_id = _required_text(values["vector_doc_id"], "vector_doc_id")
        chunk_text = _required_text(values["chunk_text"], "chunk_text")

        metadata: dict[str, object] = {}
        for column in columns:
            if column in self._EXCLUDED_COLUMNS:
                continue
            value = values[column]
            if column in self._JSON_LIST_COLUMNS:
                value = _json_string_list_or_none(value, column)
            metadata[column] = value

        return FlatTestChunk(
            vector_doc_id=vector_doc_id,
            chunk_text=chunk_text,
            metadata=metadata,
        )


def _column_names(description: Any) -> tuple[str, ...]:
    if not description:
        raise ValueError("expert_chunk_text 查询未返回列描述")

    columns = tuple(column[0] for column in description)
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("expert_chunk_text 查询返回了无效列名")
    if len(columns) != len(set(columns)):
        raise ValueError("expert_chunk_text 查询返回了重复列名")
    return columns


def _row_mapping(row: Any, columns: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(row, Mapping):
        missing = [column for column in columns if column not in row]
        if missing:
            raise ValueError(f"查询行缺少列: {', '.join(missing)}")
        return {column: row[column] for column in columns}

    if not isinstance(row, (tuple, list)) or len(row) != len(columns):
        raise ValueError("查询行列数与列描述不一致")
    return dict(zip(columns, row, strict=True))


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value


def _json_string_list_or_none(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{field_name} 不是合法 JSON 数组") from exc
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} 必须是 list[str] 或 NULL")
    return list(value)


__all__ = ["ExpertChunkTextTestReader", "FlatTestChunk"]
