"""Milvus Contract Collection 的向量与标量索引定义。

本模块只描述“应该有哪些索引及参数”，不直接调用 Client。Repository 负责执行
这些声明，Compatibility 模块负责比较服务端实际结果；这样索引变更不会和 CRUD
流程混在一起。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import MilvusSettings
from .schemas.contract import CONTRACT_SCALAR_INDEX_FIELDS


@dataclass(frozen=True)
class IndexSpec:
    """一个 Milvus Index 的声明，不包含任何 Client 操作。

    向量索引携带 metric 和构建参数；scalar 索引只声明字段和类型。当前 Contract
    集合由一个向量索引和冻结的 11 个 INVERTED 索引组成。
    """

    field_name: str
    index_type: str
    index_name: str | None = None
    metric_type: str = ""
    params: Mapping[str, object] = field(default_factory=dict)

    def summary(self, *, include_name: bool) -> dict[str, object]:
        """返回与现有 Repository 公共摘要兼容的结构。"""

        summary: dict[str, object] = {
            "field_name": self.field_name,
            "index_type": self.index_type.upper(),
            "metric_type": self.metric_type.upper(),
            "params": dict(self.params),
        }
        if include_name:
            summary = {"index_name": self.index_name, **summary}
        return summary


def vector_index_spec(settings: MilvusSettings) -> IndexSpec:
    """返回正式 Contract Collection 的向量索引定义。

    索引类型、metric 和参数来自环境配置，但写入前会由 Compatibility Gate 与
    已有 Collection 做只读比较；本函数本身不创建索引。
    """

    return IndexSpec(
        field_name="embedding",
        index_name="embedding_vector",
        index_type=settings.index_type,
        metric_type=settings.metric_type,
        params=dict(settings.index_params),
    )


def index_specs(settings: MilvusSettings) -> tuple[IndexSpec, ...]:
    """返回正式 Contract Collection 应具备的完整索引集合。

    谁使用：
        - ``Repository.initialize``：创建 Collection 时的索引声明；
        - ``Repository.compare_contract_indexes``：只读索引对比基准。

    构成：
        1 个向量索引（``embedding_vector``，HNSW/COSINE，参数来自环境配置）+ 3 个
        INVERTED 标量索引（``expert_id`` / ``chunk_type`` /
        ``eligible_for_recommendation``，见 ``contract.CONTRACT_SCALAR_INDEX_FIELDS``）。
    """

    vector = vector_index_spec(settings)
    return (
        vector,
        *(
            IndexSpec(
                field_name=field_name,
                index_name=f"{field_name}_inverted",
                index_type="INVERTED",
            )
            for field_name in CONTRACT_SCALAR_INDEX_FIELDS
        ),
    )


def build_index_params(
    client: Any,
    specs: Sequence[IndexSpec],
) -> Any:
    """将索引定义交给 PyMilvus 参数对象，仍不执行 Collection 创建。

    返回的是 SDK 的参数容器；真正创建 Collection 的动作只发生在 Repository
    明确允许创建且目标 Collection 不存在时。
    """

    index_params = client.prepare_index_params()
    for spec in specs:
        kwargs: dict[str, object] = {
            "field_name": spec.field_name,
            "index_type": spec.index_type,
        }
        if spec.index_name is not None:
            kwargs["index_name"] = spec.index_name
        if spec.metric_type:
            kwargs["metric_type"] = spec.metric_type
            # 向量索引即使参数为空，也保持旧实现向 SDK 传递 params={} 的行为。
            kwargs["params"] = dict(spec.params)
        index_params.add_index(**kwargs)
    return index_params
