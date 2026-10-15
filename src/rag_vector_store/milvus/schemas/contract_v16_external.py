"""外部 Jieba 正式 Schema；正文与 BM25 token 分开。"""

from __future__ import annotations

from typing import Any

from ...config import MilvusSettings
from ...model_space import FORMAL_EMBEDDING_DIMENSION
from ...tokenization.jieba_index import BM25_TEXT_MAX_BYTES, JiebaIndexConfig
from .contract_v16 import V16_BM25_FUNCTION_NAME, v16_milvus_field_specs
from .specs import MilvusFieldSpec


def external_field_specs(
    vector_dimension: int = FORMAL_EMBEDDING_DIMENSION,
) -> tuple[MilvusFieldSpec, ...]:
    fields = list(v16_milvus_field_specs(vector_dimension))
    fields.insert(3, MilvusFieldSpec("chunk_text_bm25", "VARCHAR", max_length=BM25_TEXT_MAX_BYTES))
    return tuple(fields)


def schema_description(config: JiebaIndexConfig) -> str:
    return f"v16_external_jieba_accurate_v1:{config.fingerprint}"


def build_external_schema(
    settings: MilvusSettings,
    vector_dimension: int | None,
    *,
    config: JiebaIndexConfig,
) -> Any:
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    if vector_dimension is None or settings.expert_storage_profile != "v16_external_formal":
        raise ValueError("外部分词 Schema 需要绑定维度和外部 Jieba Profile")
    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=False, description=schema_description(config)
    )
    for spec in external_field_specs(vector_dimension):
        options: dict[str, Any] = {}
        if spec.primary_key:
            options["is_primary"] = True
        if spec.nullable:
            options["nullable"] = True
        if spec.max_length is not None:
            options["max_length"] = spec.max_length
        if spec.dim is not None:
            options["dim"] = spec.dim
        if spec.name == "chunk_text_bm25":
            options.update(enable_analyzer=True, analyzer_params={"tokenizer": "whitespace"})
        schema.add_field(spec.name, getattr(DataType, spec.datatype), **options)
    schema.add_function(
        Function(
            name=V16_BM25_FUNCTION_NAME,
            function_type=FunctionType.BM25,
            input_field_names=["chunk_text_bm25"],
            output_field_names=["sparse_embedding"],
        )
    )
    return schema
