"""将 expert_chunk 映射为向量化输入契约 JSONL。

Exporter 不清洗、不摘要、不拼接 Metadata，只重新校验哈希/字符数并封装固定结构。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import CONTRACT_VERSION
from .normalize import sha256_value


def export_jsonl(rows: list[dict[str, object]], output_path: Path, timezone_name: str) -> int:
    """原子写出 UTF-8 JSONL；失败时不会留下半份正式文件。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                contract = row_to_contract(row, timezone_name)
                handle.write(json.dumps(contract, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return len(rows)


def row_to_contract(row: dict[str, object], timezone_name: str) -> dict[str, object]:
    """执行固定字段映射，并纵深校验文本哈希与字符数。"""

    text = str(row["chunk_text"])
    if len(text) != int(row["char_count"]):
        raise ValueError(f"{row['chunk_id']} char_count 不一致")
    if sha256_value(text) != row["content_hash"]:
        raise ValueError(f"{row['chunk_id']} content_hash 不一致")
    metadata = row["metadata_json"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return {
        "contract_version": CONTRACT_VERSION,
        "chunk_id": row["chunk_id"],
        "expert_id": int(row["expert_id"]),
        "serial_number": row["serial_number"],
        "chunk_type": row["chunk_type"],
        "chunk_subtype": row["chunk_subtype"],
        "chunk_index": int(row["chunk_index"]),
        "chunk_total": int(row["chunk_total"]),
        "chunk_text": text,
        "char_count": int(row["char_count"]),
        "content_hash": row["content_hash"],
        "metadata": metadata,
        "source": {
            "source_table": row["source_table"],
            "source_field": row["source_field"],
            "source_field_id": row["source_field_id"],
            "source_item_key": row["source_item_key"],
            "source_version": row["source_version"],
            "source_updated_at": _iso(row["source_updated_at"], timezone_name),
            "source_hash": row["source_hash"],
        },
        "versions": {
            "data_version": row["data_version"],
            "chunk_policy_version": row["chunk_policy_version"],
            "metadata_schema_version": row["metadata_schema_version"],
            "dictionary_version": row["dictionary_version"],
        },
        "is_active": bool(row["is_active"]),
        "updated_at": _iso(row["updated_at"], timezone_name),
    }


def _iso(value: object, timezone_name: str) -> str | None:
    """将 MySQL naive DATETIME 按业务时区转换为 ISO 8601。"""

    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError(f"期望 datetime，实际为 {type(value).__name__}")
    aware = value if value.tzinfo else value.replace(tzinfo=ZoneInfo(timezone_name))
    return aware.isoformat(timespec="microseconds")
