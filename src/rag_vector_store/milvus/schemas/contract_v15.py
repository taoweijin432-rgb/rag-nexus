"""V15 精简物理 Schema 定义（导出契约 4.0.0 的 V15 存储形态）。

V15 与 v14 的关系：
    - v14 是当前正式 Collection 的 25 字段 Schema（含高频标量投影和三份 JSON
      extension），保持不变，继续由 ``schemas/contract.py`` 与
      ``milvus/contract.py`` 服务，作为迁移期历史实现。
    - V15 是本项目冻结的纯 4 字段物理 Schema：
      ``vector_doc_id`` / ``chunk_text`` / ``embedding`` / ``metadata``。
      其余 Contract 信息（含 ``expert_id`` / ``content_hash`` /
      ``metadata_hash`` 等）全部合并进单个 flat ``metadata`` JSON。

本模块只声明 V15 字段与 SDK Schema 构造，不负责连接、创建、迁移或修改已存在
Collection。V15 的 Entity / RowMapper 在 ``milvus/contract_v15.py``。

设计约束（与 ``milvus/contract_v15.py`` 配合）：
    - ``metadata`` 必须是单层 flat JSON，不使用 ``contract`` / ``metadata`` /
      ``source`` / ``versions`` 等 namespace 键；
    - 一级字段（``vector_doc_id`` / ``chunk_text`` / ``embedding``）不得重复
      出现在 ``metadata`` 中；
    - 从一级字段迁移进 metadata 的 ``expert_id`` / ``content_hash`` /
      ``metadata_hash`` 等不得导致 hash 重算或 hash payload 变化；
    - dynamic field = False；
    - 不创建真实 Collection，不修改 .env，不迁移真实数据。
"""

from __future__ import annotations

from typing import Any

from ...config import MilvusSettings
from .builder import build_schema_from_specs
from .specs import MilvusFieldSpec

V15_CONTRACT_VECTOR_DIMENSION = 1024
V15_VECTOR_DOC_ID_MAX_BYTES = 2048
# chunk_text 的 Milvus VARCHAR 字节容量沿用 v14 与 ContractValidator 的 UTF-8
# 上限（8192 bytes），保证任意合法 Contract chunk_text 都能落库。
V15_CHUNK_TEXT_MAX_BYTES = 8192

# V15 不保留任何 scalar index：expert_id / content_hash / metadata_hash 均已
# 进入 flat metadata JSON，本阶段冻结纯 4 字段 persistence correctness。
# JSON path index / per-expert query 性能在真实 V15 E2E 后单独评估。
V15_SCALAR_INDEX_FIELDS: tuple[str, ...] = ()


def v15_milvus_field_specs(
    vector_dimension: int = V15_CONTRACT_VECTOR_DIMENSION,
) -> tuple[MilvusFieldSpec, ...]:
    """返回 V15 精简物理 Schema 的 4 个字段定义。

    谁使用：
        - ``schemas/builder.py``：构造 SDK Schema（后续创建 V15 Collection）；
        - ``MilvusV15RowMapper``：存储容量校验的字段基线；
        - Repository / 验证脚本：只读 Schema 对比。

    字段集合（4 个，顺序固定）：
        1. ``vector_doc_id`` — VARCHAR PK，auto_id=False；
        2. ``chunk_text`` — VARCHAR(8192)，检索正文（与 Embedding 输入同一段文本）；
        3. ``embedding`` — FLOAT_VECTOR(1024)；
        4. ``metadata`` — JSON，单层 flat，承载其余全部 Contract 信息。

    ``embedding`` 维度由调用方传入并写入向量字段；其余字段必须保持稳定。
    """

    return (
        # 身份与向量主键：vector_doc_id 是唯一 PK（所有 CRUD 统一按它定位）
        MilvusFieldSpec(
            "vector_doc_id",
            "VARCHAR",
            primary_key=True,
            max_length=V15_VECTOR_DOC_ID_MAX_BYTES,
        ),
        # 检索正文：搜索返回需要，正文只取 Embedding 输入同一段 chunk_text
        MilvusFieldSpec("chunk_text", "VARCHAR", max_length=V15_CHUNK_TEXT_MAX_BYTES),
        # Embedding 输出
        MilvusFieldSpec("embedding", "FLOAT_VECTOR", dim=vector_dimension),
        # 其余全部 Contract 信息（expert_id / content_hash / metadata_hash /
        # metadata / source / versions / 嵌入字段等）合并进 flat JSON
        MilvusFieldSpec("metadata", "JSON"),
    )


def build_v15_schema(
    settings: MilvusSettings,
    vector_dimension: int | None,
) -> Any:
    """构造 V15 精简 Contract Collection Schema，不执行 Collection 操作。

    签名与 v14 的 ``build_contract_schema`` 保持一致；``settings`` 当前仅用于
    保持调用形态统一（V15 的字段集合与 collection 配置无关）。
    """

    del settings  # 与 v14 build_contract_schema 保持签名一致的占位参数
    if vector_dimension is None:
        raise ValueError("vector dimension is not configured")
    return build_schema_from_specs(v15_milvus_field_specs(vector_dimension))
