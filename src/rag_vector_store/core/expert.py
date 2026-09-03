"""专家 Contract 到领域无关内核的兼容适配器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from expert_contracts.models import (
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
    ExpertChunkSource,
    ExpertChunkVersions,
    metadata_model_for_chunk_type,
)


class ExpertContractAdapter:
    """把现有专家 Contract 接入通用向量化内核。

    ``ExpertContractAdapter`` 同时作为兼容 Contract 的基础实现。Resume 领域
    使用同一个内部 ``ExpertChunkEmbeddingInput``，只把物理 owner 字段和主键
    命名空间替换为 ``resume_id`` / ``resume:``。
    """

    def __init__(
        self,
        validator: Any | None = None,
        eligibility_validator: Any | None = None,
        *,
        storage_owner_field: str = "expert_id",
        id_prefix: str = "expert:",
    ) -> None:
        # Repository profile 只需要无参构造 Adapter；专家 ingestion Facade
        # 则注入已创建的校验器，确保两条入口使用完全相同的规则实例。
        if validator is None or eligibility_validator is None:
            from ..chunking.contract_pipeline import (
                ContractValidator,
                EmbeddingEligibilityValidator,
            )

            validator = validator or ContractValidator()
            eligibility_validator = eligibility_validator or EmbeddingEligibilityValidator()
        self.validator = validator
        self.eligibility_validator = eligibility_validator
        if not isinstance(storage_owner_field, str) or not storage_owner_field.strip():
            raise ValueError("storage_owner_field 必须是非空字符串")
        if not isinstance(id_prefix, str) or not id_prefix.strip():
            raise ValueError("id_prefix 必须是非空字符串")
        self._storage_owner_field = storage_owner_field
        self._id_prefix = id_prefix

    @property
    def contract_owner_field(self) -> str:
        """返回兼容 Contract 中承载 owner 的字段名。"""

        return "expert_id"

    def is_input(self, value: object) -> bool:
        if not isinstance(value, ExpertChunkEmbeddingInput):
            return False
        vector_doc_id = value.vector_doc_id
        return isinstance(vector_doc_id, str) and vector_doc_id.startswith(self._id_prefix)

    def validate(self, value: ExpertChunkEmbeddingInput) -> ContractPreparedChunk:
        return self.validator.validate(value)

    def vector_doc_id(self, value: object) -> str:
        vector_doc_id = getattr(value, "vector_doc_id", None)
        if not isinstance(vector_doc_id, str) or not vector_doc_id.strip():
            raise TypeError("Contract 缺少合法 vector_doc_id")
        if not vector_doc_id.startswith(self._id_prefix):
            raise ValueError(
                f"vector_doc_id {vector_doc_id!r} 不符合当前领域前缀 {self._id_prefix!r}"
            )
        return vector_doc_id

    def chunk_text(self, value: ContractPreparedChunk) -> str:
        return value.chunk_text

    def skip_reason(self, value: ContractPreparedChunk) -> str | None:
        return self.eligibility_validator.get_skip_reason(value)

    def attach_embedding(
        self,
        value: ContractPreparedChunk,
        embedding: Sequence[float],
        *,
        embedding_model_alias: str,
        embedding_model_version: str,
        embedded_at: str,
    ) -> EmbeddedContractPreparedChunk:
        return EmbeddedContractPreparedChunk(
            **value.model_dump(),
            embedding=list(embedding),
            embedding_dim=len(embedding),
            embedding_model_alias=embedding_model_alias,
            embedding_model_version=embedding_model_version,
            embedded_at=embedded_at,
        )

    def is_embedded(self, value: object) -> bool:
        return isinstance(value, EmbeddedContractPreparedChunk)

    def is_prepared(self, value: object) -> bool:
        return isinstance(value, ContractPreparedChunk)

    def model_space(self, value: EmbeddedContractPreparedChunk) -> dict[str, object]:
        return {
            "embedding_model_alias": value.embedding_model_alias,
            "embedding_model_version": value.embedding_model_version,
            "embedding_dim": value.embedding_dim,
        }

    def embedding(self, value: EmbeddedContractPreparedChunk) -> Sequence[float]:
        return value.embedding

    def embedded_at(self, value: EmbeddedContractPreparedChunk) -> str:
        return value.embedded_at

    def metadata(self, value: EmbeddedContractPreparedChunk) -> dict[str, object]:
        metadata = value.metadata.model_dump(mode="json")
        # V15 Profile 将 owner 提升为 flat metadata key；业务 Contract 本身
        # 保持原有 per-type Metadata，不增加内部 owner 字段副本。
        metadata[self._storage_owner_field] = self.owner_id(value)
        return metadata

    def storage_projection(
        self,
        value: EmbeddedContractPreparedChunk,
    ) -> dict[str, Mapping[str, object]]:
        """返回领域无关 RowMapper 所需的四段 flat metadata 投影。"""

        top_level = value.model_dump(mode="json")
        top_level_flat = {
            key: item
            for key, item in top_level.items()
            if key not in {"vector_doc_id", "chunk_text", "embedding"}
            and key not in {"metadata", "source", "versions"}
            and key != self.contract_owner_field
        }
        return {
            "contract": top_level_flat,
            "metadata": self.metadata(value),
            "source": value.source.model_dump(mode="json"),
            "versions": value.versions.model_dump(mode="json"),
        }

    def restore_embedded(
        self,
        row: Mapping[str, object],
    ) -> EmbeddedContractPreparedChunk:
        """从 V15 flat 行还原兼容 Embedded Contract。"""

        flat = row.get("metadata")
        if not isinstance(flat, Mapping):
            raise TypeError("Milvus V15 行缺少合法 metadata JSON 对象")
        if self._storage_owner_field != self.contract_owner_field and (
            self.contract_owner_field in flat
        ):
            raise ValueError(
                f"Resume 行不得包含内部 owner 字段 {self.contract_owner_field!r}"
            )
        metadata_model = metadata_model_for_chunk_type(flat.get("chunk_type"))
        payload = {
            "contract_version": flat.get("contract_version"),
            "vector_doc_id": row.get("vector_doc_id"),
            "expert_id": flat.get(self._storage_owner_field),
            "serial_number": flat.get("serial_number"),
            "chunk_type": flat.get("chunk_type"),
            "chunk_subtype": flat.get("chunk_subtype"),
            "chunk_index": flat.get("chunk_index"),
            "chunk_total": flat.get("chunk_total"),
            "chunk_text": row.get("chunk_text"),
            "char_count": flat.get("char_count"),
            "content_hash": flat.get("content_hash"),
            "metadata_hash": flat.get("metadata_hash"),
            "metadata": _pick_flat_fields(flat, metadata_model),
            "source": _pick_flat_fields(flat, ExpertChunkSource),
            "versions": _pick_flat_fields(flat, ExpertChunkVersions),
            "is_active": flat.get("is_active"),
            "updated_at": flat.get("updated_at"),
            "embedding": row.get("embedding"),
            "embedding_dim": flat.get("embedding_dim"),
            "embedding_model_alias": flat.get("embedding_model_alias"),
            "embedding_model_version": flat.get("embedding_model_version"),
            "embedded_at": flat.get("embedded_at"),
        }
        return EmbeddedContractPreparedChunk.model_validate(payload)

    def with_active(
        self,
        value: EmbeddedContractPreparedChunk,
        is_active: bool,
    ) -> EmbeddedContractPreparedChunk:
        return value.model_copy(update={"is_active": is_active})

    def content_hash(self, value: object) -> str:
        return value.content_hash

    def metadata_hash(self, value: object) -> str:
        return value.metadata_hash

    def source_hash(self, value: object) -> str | None:
        source = getattr(value, "source", None)
        result = getattr(source, "source_hash", None)
        return result if isinstance(result, str) else None

    def text_and_identity_equal(self, left: object, right: object) -> bool:
        return (
            self.vector_doc_id(left) == self.vector_doc_id(right)
            and getattr(left, "content_hash", None) == getattr(right, "content_hash", None)
            and getattr(left, "chunk_text", None) == getattr(right, "chunk_text", None)
        )

    def owner_id(self, value: object) -> int:
        owner_id = getattr(value, "expert_id", None)
        if isinstance(owner_id, bool) or not isinstance(owner_id, int) or owner_id < 1:
            raise ValueError("expert_id 必须是正整数")
        return owner_id

    def owner_field(self) -> str:
        return self._storage_owner_field

    def id_prefix(self) -> str:
        return self._id_prefix


class ResumeContractAdapter(ExpertContractAdapter):
    """Resume 领域的兼容 Adapter。

    Resume 暂时复用 Expert Contract 结构，只有物理持久化身份使用
    ``resume_id``/``resume:``，避免修改公共 Contract。
    """

    def __init__(
        self,
        validator: Any | None = None,
        eligibility_validator: Any | None = None,
    ) -> None:
        super().__init__(
            validator,
            eligibility_validator,
            storage_owner_field="resume_id",
            id_prefix="resume:",
        )


def _pick_flat_fields(
    flat: Mapping[str, object],
    model_cls: type[Any],
) -> dict[str, object]:
    """从 flat metadata 中挑选指定 Contract 模型声明的字段。"""

    return {
        field_name: flat[field_name]
        for field_name in model_cls.model_fields
        if field_name in flat
    }


__all__ = ["ExpertContractAdapter", "ResumeContractAdapter"]
