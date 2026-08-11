"""根据字段定义构造 PyMilvus Collection Schema。

Builder 只把已经选定的 ``MilvusFieldSpec`` 转换为 SDK 对象；字段集合和约束在
``schemas.contract`` 中定义，Repository 决定是否在真实环境执行创建。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .specs import MilvusFieldSpec


def build_schema_from_specs(specs: Sequence[MilvusFieldSpec]) -> Any:
    """将已选定的字段定义转换为 PyMilvus Schema。

    Schema 使用 ``auto_id=False`` 和关闭动态字段，保证每个写入字段都来自正式
    Contract；未知字段不会被服务端动态接收而绕过校验。
    """

    try:
        from pymilvus import DataType, MilvusClient
    except ImportError as exc:  # pragma: no cover - 依赖已在 pyproject 中声明
        raise RuntimeError(
            "pymilvus is required to create a Milvus collection; run `uv sync` first"
        ) from exc

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    for spec in specs:
        field_kwargs: dict[str, Any] = {
            "field_name": spec.name,
            "datatype": getattr(DataType, spec.datatype),
        }
        if spec.primary_key:
            field_kwargs["is_primary"] = True
        if spec.nullable:
            field_kwargs["nullable"] = True
        if spec.max_length is not None:
            field_kwargs["max_length"] = spec.max_length
        if spec.dim is not None:
            field_kwargs["dim"] = spec.dim
        schema.add_field(**field_kwargs)
    return schema
