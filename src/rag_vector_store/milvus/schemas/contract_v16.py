"""Expert V16 的 12 字段 Schema 与 BM25 Function 定义。"""

from __future__ import annotations

from typing import Any

from ...config import MilvusSettings
from ...model_space import FORMAL_EMBEDDING_DIMENSION
from ..v16_analyzer import V16AnalyzerConfig, resolve_v16_analyzer
from .specs import MilvusFieldSpec

V16_VECTOR_DOC_ID_MAX_BYTES = 2048
V16_CHUNK_TEXT_MAX_BYTES = 8192
V16_BM25_FUNCTION_NAME = "chunk_text_bm25"


def v16_milvus_field_specs(
    vector_dimension: int = FORMAL_EMBEDDING_DIMENSION,
) -> tuple[MilvusFieldSpec, ...]:
    """返回冻结的 Expert V16 物理字段契约。"""

    return (
        MilvusFieldSpec(
            "vector_doc_id",
            "VARCHAR",
            primary_key=True,
            max_length=V16_VECTOR_DOC_ID_MAX_BYTES,
        ),
        MilvusFieldSpec("expert_id", "INT64"),
        MilvusFieldSpec("chunk_text", "VARCHAR", max_length=V16_CHUNK_TEXT_MAX_BYTES),
        MilvusFieldSpec("chunk_type", "VARCHAR", max_length=64),
        MilvusFieldSpec("is_active", "BOOL"),
        MilvusFieldSpec("eligible_for_recommendation", "BOOL"),
        MilvusFieldSpec("dense_embedding", "FLOAT_VECTOR", dim=vector_dimension),
        MilvusFieldSpec("sparse_embedding", "SPARSE_FLOAT_VECTOR"),
        MilvusFieldSpec("job_status_norm", "VARCHAR", nullable=True, max_length=64),
        MilvusFieldSpec("abroad_flag_norm", "BOOL", nullable=True),
        MilvusFieldSpec("work_years", "FLOAT", nullable=True),
        MilvusFieldSpec("metadata", "JSON"),
    )


def build_v16_schema(
    settings: MilvusSettings,
    vector_dimension: int | None,
    *,
    analyzer_config: V16AnalyzerConfig | None = None,
) -> Any:
    """构造 12 字段 Schema；原始 chunk_text 经指定 Analyzer 进入 BM25。"""

    analyzer_config = analyzer_config or resolve_v16_analyzer(settings)
    if vector_dimension is None:
        raise ValueError("vector dimension is not configured")
    try:
        from pymilvus import DataType, Function, FunctionType, MilvusClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pymilvus is required to create V16 schema") from exc

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        "vector_doc_id",
        DataType.VARCHAR,
        is_primary=True,
        max_length=V16_VECTOR_DOC_ID_MAX_BYTES,
    )
    schema.add_field("expert_id", DataType.INT64)
    schema.add_field(
        "chunk_text",
        DataType.VARCHAR,
        max_length=V16_CHUNK_TEXT_MAX_BYTES,
        enable_analyzer=True,
        analyzer_params=analyzer_config.params,
    )
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=64)
    schema.add_field("is_active", DataType.BOOL)
    schema.add_field("eligible_for_recommendation", DataType.BOOL)
    schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=vector_dimension)
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("job_status_norm", DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field("abroad_flag_norm", DataType.BOOL, nullable=True)
    schema.add_field("work_years", DataType.FLOAT, nullable=True)
    schema.add_field("metadata", DataType.JSON)
    schema.add_function(
        Function(
            name=V16_BM25_FUNCTION_NAME,
            function_type=FunctionType.BM25,
            input_field_names=["chunk_text"],
            output_field_names=["sparse_embedding"],
        )
    )
    return schema


__all__ = [
    "V16_BM25_FUNCTION_NAME",
    "V16_CHUNK_TEXT_MAX_BYTES",
    "V16_VECTOR_DOC_ID_MAX_BYTES",
    "build_v16_schema",
    "v16_milvus_field_specs",
]
