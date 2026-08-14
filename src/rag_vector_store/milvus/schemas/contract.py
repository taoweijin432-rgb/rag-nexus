"""正式 Contract Schema 定义（导出契约 4.0.0）。

这里是当前唯一正式业务 Collection 的字段基线：输入字段、模型输出、高频标量
投影和三组 JSON extension。函数只返回声明或构造 SDK Schema，不负责连接、创建、
迁移或修改已存在 Collection。

Schema 只保留正式 Contract 4.0.0 字段；已废弃或不属于正式契约的字段
（chunk_id / source_item_key / data_version / dictionary_version /
company_normalization_status / project_role / metadata.vector_doc_id）在新
Schema 中一律不存在。

标量投影只服务于有明确检索/过滤/增量语义的字段（Vector PK、专家过滤、生命周期、
业务状态、推荐过滤、检索权重数据、增量哈希、搜索正文），其余 metadata/source/
versions 内容统一由三份 JSON extension 无损承载，不因为旧 v13 Collection 有
标量字段就自动保留。
"""

from __future__ import annotations

from typing import Any

from ...config import MilvusSettings
from .builder import build_schema_from_specs
from .specs import MilvusFieldSpec

CONTRACT_VECTOR_DIMENSION = 1024
CONTRACT_VECTOR_DOC_ID_MAX_BYTES = 2048


def contract_milvus_field_specs(
    vector_dimension: int = CONTRACT_VECTOR_DIMENSION,
) -> tuple[MilvusFieldSpec, ...]:
    """返回无损承载正式 Chunk 契约输出所需的字段定义。

    谁使用：
        - ``schemas/builder.py``：构造 SDK Schema（Repository 创建 Collection）；
        - ``MilvusRowMapper``：存储容量校验的字段基线；
        - ``Repository`` / 验证脚本：只读 Schema 对比。

    当前 v14 设计：
        - ``vector_doc_id`` 是唯一 PK（auto_id=False，VARCHAR 2048 bytes）；
        - ``embedding`` 为 FLOAT_VECTOR（默认 1024 维）；
        - ``metadata_ext`` / ``source_ext`` / ``versions_ext`` 为三组 JSON 完整快照；
        - Scalar Index 字段：``expert_id`` / ``chunk_type`` /
          ``eligible_for_recommendation``（见 ``CONTRACT_SCALAR_INDEX_FIELDS``）；
        - 已废弃字段（chunk_id 等）在新 Schema 中一律不存在。

    维度由调用方传入并写入向量字段；其余字段必须保持稳定，Repository 的
    Compatibility Gate 会在任何写入前逐项核对。

    ``vector_doc_id`` 是唯一 Milvus PK（``auto_id=False``），不再存在 ``chunk_id``
    或其他业务主键。``embedding`` 保持 FLOAT_VECTOR。标量字段只保留具有明确检索/
    过滤/增量语义的投影；其余内容全部 JSON 化。
    """

    return (
        # 身份与向量主键：vector_doc_id 是唯一 PK
        MilvusFieldSpec(
            "vector_doc_id",
            "VARCHAR",
            primary_key=True,
            max_length=CONTRACT_VECTOR_DOC_ID_MAX_BYTES,
        ),
        MilvusFieldSpec("expert_id", "INT64"),
        MilvusFieldSpec("serial_number", "VARCHAR", nullable=True, max_length=64),
        MilvusFieldSpec("chunk_type", "VARCHAR", max_length=32),
        MilvusFieldSpec("chunk_subtype", "VARCHAR", nullable=True, max_length=32),
        MilvusFieldSpec("chunk_index", "INT64"),
        MilvusFieldSpec("chunk_total", "INT64"),
        # 检索正文：搜索返回需要，正文只取 Embedding 输入同一段 chunk_text
        MilvusFieldSpec("chunk_text", "VARCHAR", max_length=8192),
        MilvusFieldSpec("char_count", "INT64"),
        # 增量判断字段
        MilvusFieldSpec("content_hash", "VARCHAR", max_length=64),
        MilvusFieldSpec("metadata_hash", "VARCHAR", max_length=64),
        # 审计与契约版本
        MilvusFieldSpec("contract_version", "VARCHAR", max_length=32),
        # 业务状态（正式三值）与推荐过滤
        MilvusFieldSpec("verification_status", "VARCHAR", max_length=32),
        MilvusFieldSpec(
            "eligible_for_recommendation",
            "BOOL",
            nullable=True,
        ),
        # 检索权重数据（不做 filter/index，检索策略最终由正式检索定）
        MilvusFieldSpec("weight_tier", "VARCHAR", max_length=1),
        MilvusFieldSpec("is_active", "BOOL"),
        MilvusFieldSpec("updated_at", "VARCHAR", max_length=64),
        MilvusFieldSpec("embedded_at", "VARCHAR", max_length=64),
        # Embedding 输出
        MilvusFieldSpec("embedding", "FLOAT_VECTOR", dim=vector_dimension),
        MilvusFieldSpec("embedding_dim", "INT64"),
        MilvusFieldSpec("embedding_model_alias", "VARCHAR", max_length=512),
        MilvusFieldSpec("embedding_model_version", "VARCHAR", max_length=512),
        # 完整 JSON 快照（对应 Contract 4.0.0，无废弃字段）
        MilvusFieldSpec("metadata_ext", "JSON"),
        MilvusFieldSpec("source_ext", "JSON"),
        MilvusFieldSpec("versions_ext", "JSON"),
    )


# 真实检索/过滤需要建立 INVERTED scalar index 的字段：
# expert_id（专家过滤）、chunk_type（基础检索过滤）、
# eligible_for_recommendation（推荐过滤）。
#
# is_active / verification_status 作为生命周期/业务状态**保留独立 Scalar 列**（随行
# 保存、round-trip 需要），但因 Eligibility 在写入前已排除 is_active=false 与
# rejected，Collection 内这两字段多样性被压平，暂不预建 INVERTED Index；待增量
# “软停用 vs 删除”策略确定、确实需要按状态过滤时再补。
# weight_tier / content_hash / metadata_hash 也只作为标量数据承载，不建索引。
CONTRACT_SCALAR_INDEX_FIELDS = (
    "expert_id",
    "chunk_type",
    "eligible_for_recommendation",
)


def build_contract_schema(
    settings: MilvusSettings,
    vector_dimension: int | None,
) -> Any:
    """构造正式 Contract Collection Schema，不执行 Collection 操作。"""

    if vector_dimension is None:
        raise ValueError("vector dimension is not configured")
    return build_schema_from_specs(contract_milvus_field_specs(vector_dimension))