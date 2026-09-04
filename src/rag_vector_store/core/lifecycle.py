"""领域无关的向量生命周期核心。

该模块只依赖 Contract Adapter、行 Mapper 和 Repository 协议，不导入专家或
Resume 类型。专家 Facade 负责把这里的领域无关结果映射回既有公共模型。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .._utils import batched, dedupe_nonempty_strings
from ..model_space import build_model_space_id
from ..models import MutationResult
from .adapters import VectorContractAdapter

InputT = TypeVar("InputT")
PreparedT = TypeVar("PreparedT")
EmbeddedT = TypeVar("EmbeddedT")


class DomainLifecycleRepository(Protocol[EmbeddedT]):
    def get_many_by_vector_doc_ids(
        self,
        vector_doc_ids: Sequence[str],
        *,
        include_embedding: bool = True,
    ) -> Mapping[str, Mapping[str, object]]:
        ...

    def upsert_batch(self, records: Sequence[EmbeddedT]) -> MutationResult:
        ...


class DomainRowMapper(Protocol[EmbeddedT]):
    def from_row(self, row: dict[str, object]) -> EmbeddedT:
        ...


@dataclass(frozen=True)
class DomainLifecycleItem:
    vector_doc_id: str
    status: str
    error_message: str | None = None
    recommendation: str | None = None


@dataclass(frozen=True)
class DomainLifecycleResult:
    results: list[DomainLifecycleItem]


@dataclass(frozen=True)
class DomainDeletionResult:
    requested_count: int
    deleted_count: int
    not_found_count: int


class DomainVectorLifecycleService(Generic[InputT, PreparedT, EmbeddedT]):
    """承载 metadata-only 与逻辑下线的通用核心。"""

    _MUTATION_BATCH_SIZE = 200

    def __init__(
        self,
        repository: DomainLifecycleRepository[EmbeddedT],
        row_mapper: DomainRowMapper[EmbeddedT],
        adapter: VectorContractAdapter[InputT, PreparedT, EmbeddedT],
        *,
        expected_dimension: int,
        formal_model_space_id: str,
    ) -> None:
        self.repository = repository
        self.row_mapper = row_mapper
        self.adapter = adapter
        self.expected_dimension = expected_dimension
        self.formal_model_space_id = formal_model_space_id

    def disable_vectors(self, vector_doc_ids: Sequence[str]) -> DomainDeletionResult:
        cleaned = dedupe_nonempty_strings(
            vector_doc_ids,
            prefix=self.adapter.id_prefix(),
        )
        old_rows = self.repository.get_many_by_vector_doc_ids(
            cleaned,
            include_embedding=True,
        )
        disabled_count = 0
        not_found_count = 0
        pending: list[EmbeddedT] = []
        for vector_doc_id in cleaned:
            old_row = old_rows.get(vector_doc_id)
            if old_row is None:
                not_found_count += 1
                continue
            old_chunk = self.row_mapper.from_row(dict(old_row))
            if not self.adapter.is_embedded(old_chunk):
                raise TypeError("Repository 行未还原为当前领域 Embedded Contract")
            if self._is_inactive(old_chunk):
                disabled_count += 1
                continue
            new_chunk = self.adapter.with_active(old_chunk, False)
            pending.append(new_chunk)
            disabled_count += 1
        for batch in batched(pending, size=self._MUTATION_BATCH_SIZE):
            mutation = self.repository.upsert_batch(batch)
            self._validate_mutation(batch, mutation)
        return DomainDeletionResult(
            requested_count=len(cleaned),
            deleted_count=disabled_count,
            not_found_count=not_found_count,
        )

    def update_metadata(
        self,
        chunks: Iterable[InputT],
    ) -> DomainLifecycleResult:
        source = list(chunks)
        prepared_by_id: dict[str, PreparedT] = {}
        ordered_ids: list[str] = []
        for chunk in source:
            if not self.adapter.is_input(chunk):
                raise TypeError("update_metadata 只接受当前领域 Contract")
            vector_doc_id = self.adapter.vector_doc_id(chunk)
            if vector_doc_id in prepared_by_id:
                raise ValueError("update_metadata 一个批次不能包含重复 vector_doc_id")
            validated = self.adapter.validate(chunk)
            if not self.adapter.is_prepared(validated):
                raise TypeError("Contract Adapter 返回了非法 Prepared Contract")
            prepared_by_id[vector_doc_id] = validated
            ordered_ids.append(vector_doc_id)

        old_rows = self.repository.get_many_by_vector_doc_ids(
            ordered_ids,
            include_embedding=True,
        )
        items: list[DomainLifecycleItem] = []
        pending: list[EmbeddedT] = []
        for vector_doc_id in ordered_ids:
            validated = prepared_by_id[vector_doc_id]
            old_row = old_rows.get(vector_doc_id)
            if old_row is None:
                items.append(
                    DomainLifecycleItem(
                        vector_doc_id=vector_doc_id,
                        status="failed",
                        error_message="Milvus 实体不存在，无法复用旧向量",
                        recommendation="call_upsert",
                    )
                )
                continue

            old_chunk = self.row_mapper.from_row(dict(old_row))
            if not self.adapter.is_embedded(old_chunk):
                raise TypeError("Repository 行未还原为当前领域 Embedded Contract")
            inconsistency = self._identity_error(validated, old_chunk)
            if inconsistency is not None:
                raise ValueError(inconsistency)

            gate_error = self._metadata_gate_error(old_chunk)
            if gate_error is not None:
                items.append(
                    DomainLifecycleItem(
                        vector_doc_id=vector_doc_id,
                        status="failed",
                        error_message=gate_error,
                        recommendation="call_upsert",
                    )
                )
                continue

            old_space = self.adapter.model_space(old_chunk)
            new_chunk = self.adapter.attach_embedding(
                validated,
                self.adapter.embedding(old_chunk),
                embedding_model_alias=_required_space_text(
                    old_space, "embedding_model_alias"
                ),
                embedding_model_version=_required_space_text(
                    old_space, "embedding_model_version"
                ),
                embedded_at=self.adapter.embedded_at(old_chunk),
            )
            if not self.adapter.is_embedded(new_chunk):
                raise TypeError("Contract Adapter 返回了非法 Embedded Contract")
            pending.append(new_chunk)
            items.append(
                DomainLifecycleItem(
                    vector_doc_id=vector_doc_id,
                    status="metadata_updated",
                )
            )
        for batch in batched(pending, size=self._MUTATION_BATCH_SIZE):
            mutation = self.repository.upsert_batch(batch)
            self._validate_mutation(batch, mutation)
        return DomainLifecycleResult(results=items)

    def _metadata_gate_error(self, old_chunk: EmbeddedT) -> str | None:
        space = self.adapter.model_space(old_chunk)
        alias = _required_space_text(space, "embedding_model_alias")
        version = _required_space_text(space, "embedding_model_version")
        dimension = space.get("embedding_dim")
        old_space = build_model_space_id(
            model_alias=alias,
            dimension=dimension,
            model_version=version,
        )
        if old_space != self.formal_model_space_id:
            return f"旧向量 model_space 不一致: {old_space} != {self.formal_model_space_id}"
        if dimension != self.expected_dimension:
            return f"旧向量 embedding_dim 不一致: {dimension} != {self.expected_dimension}"
        return None

    def _identity_error(self, validated: PreparedT, old_chunk: EmbeddedT) -> str | None:
        if not self.adapter.text_and_identity_equal(validated, old_chunk):
            if self.adapter.vector_doc_id(validated) != self.adapter.vector_doc_id(old_chunk):
                return (
                    "update_metadata 只允许 metadata-only 更新；vector_doc_id 不一致；"
                    "正文/主键变化必须走 upsert_chunks 并重新 Embedding"
                )
            if self.adapter.content_hash(validated) != self.adapter.content_hash(old_chunk):
                return (
                    "update_metadata 只允许 metadata-only 更新；content_hash 不一致；"
                    "正文变化必须走 upsert_chunks 并重新 Embedding"
                )
            return (
                "update_metadata 只允许 metadata-only 更新；chunk_text 不一致；"
                "正文变化必须走 upsert_chunks 并重新 Embedding"
            )
        return None

    def _is_inactive(self, value: EmbeddedT) -> bool:
        return getattr(value, "is_active", None) is False

    def _validate_mutation(
        self,
        records: Sequence[EmbeddedT],
        mutation: MutationResult,
    ) -> None:
        if not isinstance(mutation, MutationResult):
            raise TypeError("Repository 返回了非法 MutationResult")
        expected_ids = [self.adapter.vector_doc_id(item) for item in records]
        if mutation.affected_count != len(records):
            raise ValueError("Repository affected_count 与请求数量不一致")
        if mutation.primary_keys != expected_ids:
            raise ValueError("Repository 返回的 primary_keys 与请求主键不一致")


def _required_space_text(space: dict[str, object], key: str) -> str:
    value = space.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model_space 缺少合法的 {key}")
    return value


__all__ = [
    "DomainDeletionResult",
    "DomainLifecycleItem",
    "DomainLifecycleRepository",
    "DomainLifecycleResult",
    "DomainRowMapper",
    "DomainVectorLifecycleService",
]
