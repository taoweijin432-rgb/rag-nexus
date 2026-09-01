"""向量持久化 Profile 的领域无关协议。

Milvus Repository 基类通过 Profile 获得物理 Schema、行映射和身份隔离规则；本
模块不包含任何领域字段或 Collection 名称。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeVar

EmbeddedT = TypeVar("EmbeddedT")


class VectorRowMapper(Protocol[EmbeddedT]):
    """领域 Embedded Contract 与物理行之间的双向映射。"""

    def to_rows(self, records: Sequence[EmbeddedT]) -> list[dict[str, Any]]:
        ...

    def from_row(self, row: Mapping[str, object]) -> EmbeddedT:
        ...


class VectorRepositoryProfile(Protocol[EmbeddedT]):
    """Repository 所需的完整持久化 profile。"""

    _schema_field_specs: Any
    _schema_builder: Any
    _index_specs_fn: Any
    _row_mapper_cls: type[VectorRowMapper[EmbeddedT]]
    _full_scan_filter: str
    _model_space_fields: tuple[str, ...]
    _contract_adapter_cls: type[Any]
    _owner_field: str
    _vector_doc_id_prefix: str
    _owner_filter_fn: Callable[[Any], str]
    _owner_output_fields: tuple[str, ...]


__all__ = ["VectorRepositoryProfile", "VectorRowMapper"]
