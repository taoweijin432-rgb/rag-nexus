"""将扁平化 ``expert_chunk`` 映射为向量化输入契约 JSONL。

MySQL 不再保存 ``metadata_json``；Exporter 只在输出边界按 ``chunk_type`` 从显式列
重建 Metadata 对象，供仍以 JSONL 交换的调试/下游流程使用。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .chunk_builder import build_vector_doc_id
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
    expected_vector_doc_id = build_vector_doc_id(
        expert_id=int(row["expert_id"]),
        chunk_type=str(row["chunk_type"]),
        source_field_id=str(row["source_field_id"]),
        chunk_index=int(row["chunk_index"]),
    )
    if row["vector_doc_id"] != expected_vector_doc_id:
        raise ValueError(f"{row['chunk_id']} vector_doc_id 不一致")
    metadata = _metadata_from_columns(row)
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
        "metadata_hash": row["metadata_hash"],
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


def _metadata_from_columns(row: dict[str, object]) -> dict[str, object]:
    """按三类规范从列并集重建类型专属 Metadata。

    数据库表采用“字段并集”，但 JSONL 不应把其他类型的无关字段大量输出为 null。
    因此公共字段先统一加入，再按 ``chunk_type`` 选择 profile、employment 或 project
    的专属字段。列表列兼容 PyMySQL 返回 JSON 字符串或测试中的 Python list。
    """

    common = {
        "vector_doc_id": row["vector_doc_id"],
        "expert_id": int(row["expert_id"]),
        "serial_number": row["serial_number"],
        "chunk_type": row["chunk_type"],
        "chunk_subtype": row["chunk_subtype"],
        "chunk_index": int(row["chunk_index"]),
        "chunk_total": int(row["chunk_total"]),
        "origin_type": row["origin_type"],
        "verification_status": row["verification_status"],
        "weight_tier": row["weight_tier"],
        "metadata_hash": row["metadata_hash"],
    }
    chunk_type = str(row["chunk_type"])
    if chunk_type == "profile_summary":
        common.update({
            "company_name_raw": row["company_name_raw"],
            "company_name_norm": row["company_name_norm"],
            "role_name_raw": row["role_name_raw"],
            "role_code": row["role_code"],
            "role_level": row["role_level"],
            "job_status_norm": row["job_status_norm"],
            "abroad_flag_norm": _optional_bool(row["abroad_flag_norm"]),
            "work_years": row["work_years"],
            "industry_codes": _json_list(row["industry_codes"]),
            "function_codes": _json_list(row["function_codes"]),
            "topic_codes": _json_list(row["topic_codes"]),
            "school_names_raw": _json_list(row["school_names_raw"]),
            "major_names_raw": _json_list(row["major_names_raw"]),
            "eligible_for_recommendation": _optional_bool(
                row["eligible_for_recommendation"]
            ),
        })
    elif chunk_type == "employment":
        common.update({
            "company_name_raw": row["company_name_raw"],
            "company_name_norm": row["company_name_norm"],
            "role_name_raw": row["role_name_raw"],
            "role_code": row["role_code"],
            "role_level": row["role_level"],
            "start_month": row["start_month"],
            "end_month": row["end_month"],
            "is_current": _optional_bool(row["is_current"]),
            "industry_codes": _json_list(row["industry_codes"]),
            "topic_codes": _json_list(row["topic_codes"]),
        })
    elif chunk_type == "project":
        common.update({
            "project_name": row["project_name"],
            "company_name_raw": row["company_name_raw"],
            "company_name_norm": row["company_name_norm"],
            "start_month": row["start_month"],
            "end_month": row["end_month"],
            "industry_codes": _json_list(row["industry_codes"]),
            "topic_codes": _json_list(row["topic_codes"]),
        })
    else:
        raise ValueError(f"{row['chunk_id']} chunk_type 不支持：{chunk_type}")
    return common


def _json_list(value: object) -> list[object]:
    """读取 MySQL JSON 列；空值安全降级为空数组。"""

    if value is None:
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)):
        raise TypeError("列表型 Metadata 列必须是 JSON 数组")
    return list(parsed)


def _optional_bool(value: object) -> bool | None:
    """保留 SQL NULL，并把 MySQL TINYINT(1) 统一转为 Python bool。"""

    return None if value is None else bool(value)


def _iso(value: object, timezone_name: str) -> str | None:
    """将 MySQL naive DATETIME 按业务时区转换为 ISO 8601。"""

    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError(f"期望 datetime，实际为 {type(value).__name__}")
    aware = value if value.tzinfo else value.replace(tzinfo=ZoneInfo(timezone_name))
    return aware.isoformat(timespec="microseconds")
