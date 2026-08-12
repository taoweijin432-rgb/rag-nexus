"""将 JSON-compatible Mapping 适配为正式专家 Chunk Contract。"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from .models import ExpertChunkEmbeddingInput

__all__ = [
    "ExpertChunkMappingAdapter",
    "ExpertChunkMappingError",
    "adapt_expert_chunk_mapping",
]

_MISSING = object()


class ExpertChunkMappingError(ValueError):
    """Mapping 无法无损适配为 ``ExpertChunkEmbeddingInput``。"""


class ExpertChunkMappingAdapter:
    """只做结构适配和 Contract 校验，不生成或重算业务主键。"""

    def from_mapping(
        self,
        value: Mapping[str, object],
    ) -> ExpertChunkEmbeddingInput:
        """把上游 Mapping 转换成正式 Contract。

        上游必须在 ``metadata.vector_doc_id`` 提供主键。若 Mapping 已经包含顶层
        ``vector_doc_id``，这里只验证它与 Metadata 快照相等；缺失时只做同值提取，
        不根据任何 Chunk 字段重新计算。
        """

        if not isinstance(value, Mapping):
            raise TypeError("ExpertChunkMappingAdapter 只接受 Mapping")

        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ExpertChunkMappingError("metadata 必须是 Mapping")

        metadata_vector_doc_id = metadata.get("vector_doc_id", _MISSING)
        if metadata_vector_doc_id is _MISSING:
            raise ExpertChunkMappingError("metadata.vector_doc_id 必须由上游提供")
        if not isinstance(metadata_vector_doc_id, str) or not metadata_vector_doc_id.strip():
            raise ExpertChunkMappingError("metadata.vector_doc_id 必须是非空字符串")

        top_level_vector_doc_id = value.get("vector_doc_id", _MISSING)
        if (
            top_level_vector_doc_id is not _MISSING
            and top_level_vector_doc_id != metadata_vector_doc_id
        ):
            raise ExpertChunkMappingError(
                "metadata.vector_doc_id 必须与顶层 vector_doc_id 一致"
            )

        payload = dict(value)
        payload["metadata"] = dict(metadata)
        payload["vector_doc_id"] = metadata_vector_doc_id
        for field_name in ("source", "versions"):
            nested = payload.get(field_name)
            if isinstance(nested, Mapping):
                payload[field_name] = dict(nested)

        try:
            return ExpertChunkEmbeddingInput.model_validate(payload)
        except ValidationError as exc:
            raise ExpertChunkMappingError(_safe_validation_message(exc)) from exc


def adapt_expert_chunk_mapping(
    value: Mapping[str, object],
) -> ExpertChunkEmbeddingInput:
    """适配一条 JSON-compatible Mapping，返回正式 Contract。"""

    return ExpertChunkMappingAdapter().from_mapping(value)


def _safe_validation_message(error: ValidationError) -> str:
    """只返回字段路径，避免异常文本携带完整输入值。"""

    locations: list[str] = []
    for item in error.errors(include_context=False):
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        if location not in locations:
            locations.append(location)
    return "Mapping 不符合 ExpertChunkEmbeddingInput v2: " + ", ".join(locations)
