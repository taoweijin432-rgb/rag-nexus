"""将 JSON-compatible Mapping 适配为正式专家 Chunk Contract（4.0.0）。"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from .models import ExpertChunkEmbeddingInput

__all__ = [
    "ExpertChunkMappingAdapter",
    "ExpertChunkMappingError",
    "adapt_expert_chunk_mapping",
]


class ExpertChunkMappingError(ValueError):
    """Mapping 无法无损适配为 ``ExpertChunkEmbeddingInput``。"""


class ExpertChunkMappingAdapter:
    """只做结构适配和 Contract 校验，不生成或重算业务主键。"""

    def from_mapping(
        self,
        value: Mapping[str, object],
    ) -> ExpertChunkEmbeddingInput:
        """把上游 Mapping 转换成正式 Contract（4.0.0）。

        上游必须在顶层提供 ``vector_doc_id``；它只存在 Contract 顶层，
        ``metadata`` 不再保留 ``vector_doc_id`` 副本。一致性由正式公式统一校验，
        这里只做同值结构适配，不根据任何 Chunk 字段重新计算主键。
        """

        if not isinstance(value, Mapping):
            raise TypeError("ExpertChunkMappingAdapter 只接受 Mapping")

        vector_doc_id = value.get("vector_doc_id")
        if not isinstance(vector_doc_id, str) or not vector_doc_id.strip():
            raise ExpertChunkMappingError("顶层 vector_doc_id 必须由上游提供")

        payload = dict(value)
        for field_name in ("metadata", "source", "versions"):
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
    """返回安全字段路径与错误摘要，绝不携带完整输入值。

    跨字段/主键一致性校验（例如公式不符）由 ``model_validator`` 在根节点抛出，
    这里把其 ``msg`` 一并带上，方便调用方定位；``include_input=False`` 防止
    原始输入或正文泄漏。
    """

    details: list[str] = []
    for item in error.errors(include_input=False, include_context=False):
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        message = item.get("msg") or ""
        message = message.removeprefix("Value error, ")
        details.append(f"{location}: {message}" if message else location)
    return "Mapping 不符合 ExpertChunkEmbeddingInput 4.0.0: " + "; ".join(details)
