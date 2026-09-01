"""向量状态 Hash 的统一投影与严格读取工具。"""

from __future__ import annotations

from collections.abc import Mapping

HASH_FIELDS = ("content_hash", "metadata_hash", "source_hash")


def project_hash_fields(row: Mapping[str, object]) -> dict[str, object]:
    """把 V15 metadata 中的 Hash 投影到逻辑返回顶层，不修改原行。"""

    record = dict(row)
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return record
    for field in HASH_FIELDS:
        top_value = record.get(field)
        if isinstance(top_value, str) and top_value:
            continue
        flat_value = metadata.get(field)
        if isinstance(flat_value, str) and flat_value:
            record[field] = flat_value
    return record


def read_hash_field(
    record: Mapping[str, object],
    field: str,
    *,
    required: bool,
) -> str | None:
    """从顶层或 metadata 严格读取 Hash。

    顶层优先；字段存在但为空或类型错误时 fail-closed。``source_hash`` 可通过
    ``required=False`` 兼容历史缺失记录。
    """

    metadata = record.get("metadata")
    sources: tuple[Mapping[str, object], ...] = (
        record,
        metadata if isinstance(metadata, Mapping) else {},
    )
    invalid_type: str | None = None
    for source in sources:
        if field not in source:
            continue
        value = source[field]
        if value is None:
            continue
        if isinstance(value, str) and value:
            return value
        # required 字段沿用历史回放兼容语义：顶层坏值仍允许从 metadata
        # 读取；optional 字段则保持“存在但类型错误立即失败”。
        invalid_type = type(value).__name__
        if not required:
            raise ValueError(f"已存记录 {field} 类型非法：{invalid_type}")
    if required:
        raise ValueError(f"已存记录缺少 {field}")
    return None


__all__ = ["HASH_FIELDS", "project_hash_fields", "read_hash_field"]
