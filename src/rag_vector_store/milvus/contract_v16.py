"""Expert V16 Entity、字段双落映射和反向还原。"""

from __future__ import annotations

import copy
import json
import math
import struct
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.adapters import VectorStorageAdapter
from ..domains import ExpertContractAdapter
from ..model_space import FORMAL_EMBEDDING_DIMENSION
from .contract_v15 import ContractV15RowMapper

V16_METADATA_MAX_BYTES = 65536
V16_WRITABLE_FIELDS = (
    "vector_doc_id",
    "expert_id",
    "chunk_text",
    "chunk_type",
    "is_active",
    "eligible_for_recommendation",
    "dense_embedding",
    "job_status_norm",
    "abroad_flag_norm",
    "work_years",
    "metadata",
)


class MilvusV16Entity(BaseModel):
    """V16 可写字段；Function 输出 sparse_embedding 刻意不在模型中。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    vector_doc_id: str
    expert_id: int
    chunk_text: str
    chunk_type: str
    is_active: bool
    eligible_for_recommendation: bool
    dense_embedding: list[float] = Field(min_length=1)
    job_status_norm: str | None
    abroad_flag_norm: bool | None
    work_years: float | None
    metadata: dict[str, object]

    @field_validator("dense_embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("dense_embedding 必须只包含有限浮点数")
        return value


def map_flat_v15_row_to_v16(
    row: Mapping[str, object],
    *,
    expected_dimension: int = FORMAL_EMBEDDING_DIMENSION,
) -> dict[str, object]:
    """由一份 V15 flat 行同时生成 V16 metadata 与顶层规范化副本。"""

    vector_doc_id = _required_text(row, "vector_doc_id", max_bytes=2048)
    if not vector_doc_id.startswith("expert:"):
        raise ValueError("V16 Expert vector_doc_id 必须以 'expert:' 开头")
    chunk_text = _required_text(row, "chunk_text", max_bytes=8192)
    metadata_raw = row.get("metadata")
    if not isinstance(metadata_raw, Mapping):
        raise TypeError("V16 行缺少合法 metadata JSON 对象")
    metadata = copy.deepcopy(dict(metadata_raw))
    _validate_metadata_size(metadata)

    embedding_raw = row.get("embedding", row.get("dense_embedding"))
    if not isinstance(embedding_raw, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in embedding_raw
    ):
        raise TypeError("V16 行缺少合法 Dense 向量")
    dense_embedding = [float(item) for item in embedding_raw]
    if len(dense_embedding) != expected_dimension or not all(
        math.isfinite(item) for item in dense_embedding
    ):
        raise ValueError(f"dense_embedding 必须是 {expected_dimension} 维有限向量")

    expert_id = _positive_int(metadata.get("expert_id"), "expert_id")
    chunk_type = _required_metadata_text(metadata, "chunk_type", max_bytes=64)
    is_active = _bool_with_default(metadata, "is_active", null_is_default=False)
    eligible = _bool_with_default(
        metadata,
        "eligible_for_recommendation",
        null_is_default=True,
    )
    job_status = _nullable_text(metadata, "job_status_norm", max_bytes=64)
    abroad_flag = _nullable_bool(metadata, "abroad_flag_norm")
    work_years = _nullable_finite_float(metadata, "work_years")

    entity = MilvusV16Entity(
        vector_doc_id=vector_doc_id,
        expert_id=expert_id,
        chunk_text=chunk_text,
        chunk_type=chunk_type,
        is_active=is_active,
        eligible_for_recommendation=eligible,
        dense_embedding=dense_embedding,
        job_status_norm=job_status,
        abroad_flag_norm=abroad_flag,
        work_years=work_years,
        metadata=metadata,
    )
    return entity.model_dump(mode="python")


def validate_v16_mirrors(row: Mapping[str, object]) -> None:
    """校验已存 V16 顶层副本等于同一 metadata 的规范化结果。"""

    metadata_raw = row.get("metadata")
    if not isinstance(metadata_raw, Mapping):
        raise TypeError("V16 行缺少合法 metadata JSON 对象")
    expected = {
        "expert_id": _positive_int(metadata_raw.get("expert_id"), "expert_id"),
        "chunk_type": _required_metadata_text(metadata_raw, "chunk_type", max_bytes=64),
        "is_active": _bool_with_default(
            metadata_raw, "is_active", null_is_default=False
        ),
        "eligible_for_recommendation": _bool_with_default(
            metadata_raw,
            "eligible_for_recommendation",
            null_is_default=True,
        ),
        "job_status_norm": _nullable_text(
            metadata_raw, "job_status_norm", max_bytes=64
        ),
        "abroad_flag_norm": _nullable_bool(metadata_raw, "abroad_flag_norm"),
        "work_years": _nullable_finite_float(metadata_raw, "work_years"),
    }
    for field, value in expected.items():
        actual = row.get(field)
        if not _valid_mirror_type(field, actual) or actual != value:
            raise ValueError(f"V16 顶层字段 {field!r} 与 metadata 规范化结果不一致")


class ContractV16RowMapper:
    """把 Expert Embedded Contract 映射为一次写入的完整 V16 行。"""

    def __init__(
        self,
        adapter: VectorStorageAdapter[Any, Any, Any] | None = None,
    ) -> None:
        self.adapter = adapter or ExpertContractAdapter()
        self._flat_mapper = ContractV15RowMapper(self.adapter)

    def to_entity(self, chunk: Any) -> MilvusV16Entity:
        flat_row = self._flat_mapper.to_row(chunk)
        return MilvusV16Entity.model_validate(map_flat_v15_row_to_v16(flat_row))

    def to_row(self, chunk: Any) -> dict[str, object]:
        return self.to_entity(chunk).model_dump(mode="python")

    def to_rows(self, chunks: Sequence[Any]) -> list[dict[str, Any]]:
        rows = [self.to_row(chunk) for chunk in chunks]
        ids = [str(row["vector_doc_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("一个 V16 mutation batch 不能包含重复 vector_doc_id")
        return rows

    def from_row(self, row: Mapping[str, object]) -> Any:
        validate_v16_mirrors(row)
        dense = row.get("dense_embedding")
        if dense is None:
            raise ValueError("V16 行缺少 dense_embedding，无法执行 metadata-only")
        return self._flat_mapper.from_row(
            {
                "vector_doc_id": row.get("vector_doc_id"),
                "chunk_text": row.get("chunk_text"),
                "embedding": dense,
                "metadata": copy.deepcopy(row.get("metadata")),
            }
        )


class MilvusV16RowMapper(ContractV16RowMapper):
    """Expert V16 正式 RowMapper。"""


def _valid_mirror_type(field: str, value: object) -> bool:
    if field == "expert_id":
        return isinstance(value, int) and not isinstance(value, bool)
    if field in {"is_active", "eligible_for_recommendation"}:
        return isinstance(value, bool)
    if field == "abroad_flag_norm":
        return value is None or isinstance(value, bool)
    if field == "work_years":
        return value is None or isinstance(value, float)
    if field == "job_status_norm":
        return value is None or isinstance(value, str)
    return isinstance(value, str)


def _required_text(row: Mapping[str, object], key: str, *, max_bytes: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} 必须是非空字符串")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{key} 超过 {max_bytes} bytes")
    return value


def _required_metadata_text(
    metadata: Mapping[str, object],
    key: str,
    *,
    max_bytes: int,
) -> str:
    return _required_text(metadata, key, max_bytes=max_bytes)


def _positive_int(value: object, key: str) -> int:
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{key} 必须是正整数")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{key} 必须是正整数") from exc
    if result < 1:
        raise ValueError(f"{key} 必须是正整数")
    return result


def _bool_with_default(
    metadata: Mapping[str, object],
    key: str,
    *,
    null_is_default: bool,
) -> bool:
    if key not in metadata or (null_is_default and metadata.get(key) is None):
        return False
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"metadata.{key} 必须是布尔值")
    return value


def _nullable_text(
    metadata: Mapping[str, object],
    key: str,
    *,
    max_bytes: int,
) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"metadata.{key} 必须是非空字符串或 null")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"metadata.{key} 超过 {max_bytes} bytes")
    return value


def _nullable_bool(metadata: Mapping[str, object], key: str) -> bool | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"metadata.{key} 必须是布尔值或 null")
    return value


def _nullable_finite_float(metadata: Mapping[str, object], key: str) -> float | None:
    return normalize_v16_work_years(metadata.get(key))


def normalize_v16_work_years(value: object) -> float | None:
    """正式存储和实验映射共用的有限 float32 工作年限规范化规则。"""

    key = "work_years"
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"metadata.{key} 必须是有限数值或 null")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise TypeError(f"metadata.{key} 必须是有限数值或 null") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"metadata.{key} 必须是有限数值或 null")
    result = float(decimal_value)
    if not math.isfinite(result):
        raise ValueError(f"metadata.{key} 必须是有限数值或 null")
    # Milvus FLOAT 是 IEEE-754 单精度。写前即归一到 float32，保证顶层副本在
    # 回读后仍能与 metadata 的规范化结果做确定性一致性比较。
    try:
        normalized = struct.unpack("!f", struct.pack("!f", result))[0]
    except OverflowError as exc:
        raise ValueError(f"metadata.{key} 必须是 Milvus FLOAT 可表示的有限数值") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"metadata.{key} 必须是 Milvus FLOAT 可表示的有限数值")
    return normalized


def _validate_metadata_size(metadata: Mapping[str, object]) -> None:
    try:
        payload = json.dumps(
            metadata,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("metadata 必须可序列化为 JSON") from exc
    if len(payload) > V16_METADATA_MAX_BYTES:
        raise ValueError("metadata 超过 Milvus JSON 65536 bytes 限制")


__all__ = [
    "V16_WRITABLE_FIELDS",
    "ContractV16RowMapper",
    "MilvusV16Entity",
    "MilvusV16RowMapper",
    "map_flat_v15_row_to_v16",
    "normalize_v16_work_years",
    "validate_v16_mirrors",
]
