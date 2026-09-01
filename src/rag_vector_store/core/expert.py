"""专家 Contract 到领域无关内核的兼容适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from expert_contracts.models import (
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
)


class ExpertContractAdapter:
    """把现有专家 Contract 接入通用向量化内核。"""

    def __init__(
        self,
        validator: Any | None = None,
        eligibility_validator: Any | None = None,
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

    def is_input(self, value: object) -> bool:
        return isinstance(value, ExpertChunkEmbeddingInput)

    def validate(self, value: ExpertChunkEmbeddingInput) -> ContractPreparedChunk:
        return self.validator.validate(value)

    def vector_doc_id(self, value: object) -> str:
        vector_doc_id = getattr(value, "vector_doc_id", None)
        if not isinstance(vector_doc_id, str) or not vector_doc_id.strip():
            raise TypeError("Contract 缺少合法 vector_doc_id")
        return vector_doc_id

    def identity(self, value: object) -> str:
        return self.vector_doc_id(value)

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
        # 保持原有 per-type Metadata，不增加 expert_id 副本。
        metadata["expert_id"] = value.expert_id
        return metadata

    def source(self, value: EmbeddedContractPreparedChunk) -> dict[str, object]:
        return value.source.model_dump(mode="json")

    def versions(self, value: EmbeddedContractPreparedChunk) -> dict[str, object]:
        return value.versions.model_dump(mode="json")

    def from_embedded_row(self, row: dict[str, object]) -> EmbeddedContractPreparedChunk:
        raise NotImplementedError(
            "专家 V15 行还原仍由 MilvusV15RowMapper 负责；不要绕过 Mapper"
        )

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
        return "expert_id"

    def id_prefix(self) -> str:
        return "expert:"


__all__ = ["ExpertContractAdapter"]
