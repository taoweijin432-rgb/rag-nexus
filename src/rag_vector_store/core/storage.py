"""向量持久化 Profile 的领域无关模型与协议。

Milvus Repository 基类通过 Profile 获得物理 Schema、行映射和身份隔离规则；本
模块不包含任何领域字段或 Collection 名称。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

EmbeddedT = TypeVar("EmbeddedT")


class VectorRowMapper(Protocol[EmbeddedT]):
    """领域 Embedded Contract 与物理行之间的双向映射。"""

    def to_rows(self, records: Sequence[EmbeddedT]) -> list[dict[str, Any]]:
        ...

    def from_row(self, row: Mapping[str, object]) -> EmbeddedT:
        ...


@dataclass(frozen=True)
class VectorRepositoryProfile(Generic[EmbeddedT]):
    """Repository 所需的完整、不可变持久化 profile。

    Repository 在构造时把该对象复制到实例状态，后续不会从请求数据或全局变量
    动态切换领域、Collection 或 Contract 类型。
    """

    schema_field_specs: Any
    schema_builder: Any
    index_specs_fn: Any
    row_mapper_cls: type[VectorRowMapper[EmbeddedT]]
    full_scan_filter: str
    model_space_fields: tuple[str, ...]
    contract_adapter_cls: type[Any]
    owner_field: str
    vector_doc_id_prefix: str
    owner_filter_fn: Callable[[Any], str]
    owner_output_fields: tuple[str, ...]


__all__ = ["VectorRepositoryProfile", "VectorRowMapper"]
