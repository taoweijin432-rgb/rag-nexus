"""V15 精简 Contract Collection 的 Milvus Repository 实现。

V15 与 v14 的关系：
    - ``ContractMilvusVectorRepository``（``milvus/repository.py``）是 V14 正式
      Repository，保持不变：25 字段 Schema、``MilvusRowMapper``、``index_specs``；
    - 本类是同一套 CRUD / Query / Safety 编排的 V15 变体，通过整体替换 base
      Repository 的持久化 profile 实现，不复制任何 CRUD 逻辑，也不在方法内
      按版本分支：
        * Schema：``v15_milvus_field_specs``（7 字段）+ ``build_v15_schema``；
        * 行映射：``MilvusV15RowMapper``（7 字段 + flat metadata JSON）；
        * 索引：``index_specs_v15``（embedding HNSW/COSINE + expert_id INVERTED）；
        * 全量扫描 filter：基于一级 ``expert_id INT64`` 的恒真合法范围
          （``expert_id >= 0``），Full Reconciliation 状态扫描不依赖 JSON；
          模型空间一致性检查仍允许读取 metadata JSON（V15 的
          ``embedding_model_alias`` / ``embedding_model_version`` /
          ``embedding_dim`` 位于其中）。

V15 物理 Schema（7 字段，dynamic=False）：

    vector_doc_id  VARCHAR PK (auto_id=False)
    expert_id      INT64                  <- 唯一 INVERTED scalar index
    content_hash   VARCHAR(64)
    metadata_hash  VARCHAR(64)
    chunk_text     VARCHAR(8192)
    embedding      FLOAT_VECTOR(1024)
    metadata       JSON（单层 flat，其余全部 Contract 信息）

职责边界与 base Repository 完全一致：不负责 Contract Validation、Eligibility、
Embedding、Task State；不创建真实 Collection（是否创建由调用方结合
``allow_collection_creation`` 决定）；不修改 .env；不迁移真实数据。

检索（``search``）暂未接通：V15 没有 ``chunk_type`` / ``expert_id`` 之外的
一级过滤字段（``chunk_type`` 已进入 metadata JSON），检索语义需要单独设计，
本任务不进入检索功能，调用直接抛 ``NotImplementedError``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contract_v15 import MilvusV15RowMapper
from .indexes import index_specs_v15
from .repository import ContractMilvusVectorRepository
from .schemas.contract_v15 import build_v15_schema, v15_milvus_field_specs

__all__ = ["ContractMilvusV15VectorRepository"]


class ContractMilvusV15VectorRepository(ContractMilvusVectorRepository):
    """V15 精简 Contract Collection 的 Repository 类型。

    构造参数、initialize / upsert_batch / get_by_vector_doc_id /
    delete_by_vector_doc_ids / query_by_expert_id / list_all_vector_states /
    assert_destructive_maintenance_allowed / close 的语义与 base Repository
    一致；差异只体现在持久化 profile 上（见模块 docstring）。

    ``search`` 暂不支持（V15 检索通路未接通），调用抛 ``NotImplementedError``。
    """

    # 持久化 profile：整体替换为 V15 定义（禁止方法内按版本分支）。
    _schema_field_specs = staticmethod(v15_milvus_field_specs)
    _schema_builder = staticmethod(build_v15_schema)
    _index_specs_fn = staticmethod(index_specs_v15)
    _row_mapper_cls = MilvusV15RowMapper
    # 全量扫描 filter：不依赖 JSON——使用一级 expert_id INT64 的恒真合法范围
    # （所有落库 Entity 的 expert_id 都 >= 1，见 Contract 校验）。
    # 模型空间一致性检查仍允许读取 metadata JSON（model_space 三个字段在
    # flat metadata 内），这是 Mapping 已冻结的 V15 事实，不重新提升为一级。
    _full_scan_filter = "expert_id >= 0"
    _model_space_fields = ("metadata",)

    def _model_space_from_row(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, object]:
        """从 V15 行的 flat metadata JSON 提取模型空间字段。

        与 base Repository 的区别：V15 的 ``embedding_model_alias`` /
        ``embedding_model_version`` / ``embedding_dim`` 位于 metadata JSON 内，
        而不是一级字段；提取失败（缺少合法 metadata 对象）一律 fail-closed。
        """

        flat = row.get("metadata")
        if not isinstance(flat, Mapping):
            raise TypeError("Milvus V15 行缺少合法 metadata JSON 对象")
        return {
            "embedding_model_alias": flat.get("embedding_model_alias"),
            "embedding_model_version": flat.get("embedding_model_version"),
            "embedding_dim": flat.get("embedding_dim"),
        }

    def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int = 3,
        chunk_type: str | None = None,
    ) -> list[Any]:
        """V15 检索通路尚未接通：chunk_type 已进入 metadata JSON，语义需单独设计。"""

        del embedding, limit, chunk_type
        raise NotImplementedError(
            "V15 Collection 的检索通路尚未接通（chunk_type 已进入 metadata JSON）；"
            "当前任务只接通 initialize/CRUD/Query/Safety"
        )