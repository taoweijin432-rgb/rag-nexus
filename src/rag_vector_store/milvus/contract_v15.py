"""V15 精简物理 Schema 对应的 Milvus Entity 与行映射。

本模块是 Contract 输出到 V15 物理存储之间唯一的字段映射边界：4 个一级字段
（``vector_doc_id`` / ``chunk_text`` / ``embedding`` / ``metadata``），其余
Contract 信息（含 ``expert_id`` / ``content_hash`` / ``metadata_hash`` /
业务 metadata / source / versions / 嵌入字段等）全部合并进单个 flat
``metadata`` JSON。它不创建 Collection、不调用 Milvus Client，也不决定
Task State。

与 v14 的关系：
    - v14 的 25 字段物理形态（``MilvusExpertChunkEntity`` / ``MilvusRowMapper``）
      已退役删除；真实 v14 Collection 仍受保护性检查保护，不被迁移。
    - V15 是当前唯一正式映射，Schema 定义见 ``schemas/contract_v15.py``。

flat ``metadata`` 设计约束：
    - 单层 flat JSON，不使用 ``contract`` / ``metadata`` / ``source`` /
      ``versions`` 等 namespace 键；
    - 一级字段（``vector_doc_id`` / ``chunk_text`` / ``embedding``）不得重复
      出现在 ``metadata`` 中；
    - ``expert_id`` / ``content_hash`` / ``metadata_hash`` 等从一级字段迁移进
      metadata 后，其计算语义完全不变（只透传，不重算、不改变 hash payload）；
    - Metadata 为 Contract 4.1.0 的 per-type 模型：Mapper 只机械投影当前
      ``chunk_type`` 声明字段，不包含任何 ``if chunk_type`` 分支或字段删除名单；
    - 合并时发现真实 key 冲突一律 fail-closed（抛 ValueError），不静默覆盖；
    - ``null`` / ``[]`` 原样保留；``work_years``（Decimal）沿用当前
      ``model_dump(mode="json")`` 序列化为字符串（保留两位小数精度）；
      datetime 保持当前 ISO 8601 字符串语义。

``from_row`` 必须满足 ``update_metadata`` 的现有依赖：能还原完整的
``EmbeddedContractPreparedChunk``（含 embedding 原样保留、模型空间字段、
metadata/source/versions），且不重新计算 vector_doc_id、不改变 hash。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.adapters import VectorStorageAdapter
from ..core.flat_metadata import merge_flat_metadata
from ..domains import ExpertContractAdapter, ResumeContractAdapter
from .schemas.contract_v15 import (
    V15_CONTRACT_VECTOR_DIMENSION,
    V15_VECTOR_DOC_ID_MAX_BYTES,
    v15_milvus_field_specs,
)

__all__ = [
    "V15_CONTRACT_VECTOR_DIMENSION",
    "V15_VECTOR_DOC_ID_MAX_BYTES",
    "ContractV15RowMapper",
    "MilvusV15Entity",
    "MilvusV15RowMapper",
    "ResumeMilvusV15RowMapper",
    "v15_milvus_field_specs",
]

# 一级物理字段：这些 key 禁止出现在 flat metadata JSON 中。
# V15 纯 4 字段：expert_id / content_hash / metadata_hash 不再是物理字段，
# 因此允许（并要求）出现在 metadata 中。
_V15_PHYSICAL_KEYS = frozenset({"vector_doc_id", "chunk_text", "embedding"})
# 禁止的 namespace 键：flat metadata 不允许 contract/metadata/source/versions 分组。
_V15_NAMESPACE_KEYS = frozenset({"contract", "metadata", "source", "versions"})


class MilvusV15Entity(BaseModel):
    """V15 精简 Schema 在 Milvus 中的完整 Entity 数据模型（4 个一级字段）。

    谁使用：
        ``MilvusV15RowMapper`` 的输出模型；后续 V15 Repository 通过 Mapper
        把 ``EmbeddedContractPreparedChunk`` 映射为行后再交给 PyMilvus。

    内容：
        4 个物理字段；``metadata`` 为单层 flat JSON，承载其余全部 Contract
        信息（含 expert_id / content_hash / metadata_hash）。构造时会拒绝
        非有限向量。

    身份：
        只有 ``vector_doc_id``（唯一 PK），不存在 ``chunk_id`` 或其他业务主键。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    vector_doc_id: str
    chunk_text: str
    embedding: list[float] = Field(min_length=1)
    metadata: dict[str, object]

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        """确保 Milvus Entity 向量只包含有限浮点数。"""

        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding 必须只包含有限浮点数")
        return value


class ContractV15RowMapper:
    """使用领域 Adapter 将 Embedded Contract 无损映射为 V15 行。

    这是 V15 主链路中唯一的“业务 Contract → Milvus 物理存储模型”映射边界：

        EmbeddedContractPreparedChunk
        -> to_entity / to_rows
        -> MilvusV15Entity（再经 ``model_dump(mode="python")`` 成行）

    调用方：
        后续 V15 Repository（写入前、落 SDK 前调用）；当前阶段仅提供映射能力。

    映射规则：
        - 4 个一级字段：``vector_doc_id`` / ``chunk_text`` / ``embedding`` /
          ``metadata``；
        - ``metadata`` 为单层 flat JSON：顶层 Contract 的其余字段
          （``expert_id`` / ``content_hash`` / ``metadata_hash`` /
          ``contract_version`` / ``serial_number`` / ``chunk_type`` /
          ``chunk_subtype`` / ``chunk_index`` / ``chunk_total`` / ``char_count`` /
          ``is_active`` / ``updated_at`` / ``embedded_at``）与 Embedding 输出字段
          （``embedding_dim`` / ``embedding_model_alias`` /
          ``embedding_model_version``）+ 业务 metadata 字段 + ``source_*`` 字段 +
          ``versions_*`` 字段全部平铺为单层；
        - flat 合并使用 fail-closed 冲突检测：重复 key、一级字段 key、namespace
          key 一律抛 ``ValueError``，不静默覆盖；
        - Decimal ``work_years`` 经 ``model_dump(mode="json")`` 稳定序列化为字符串
          （保留两位小数精度），不转 float；
        - datetime 保持 ISO 8601 字符串语义；
        - 字节/JSON 大小 fail-closed：VARCHAR 按 UTF-8 bytes、metadata JSON 按
          Milvus 服务端限制校验，超限拒绝写入。

    职责边界：
        Mapper 只负责字段投影、flat JSON 组装和存储容量检查；不负责调用
        Embedding、生成或重算主键、修改输入对象或执行 Milvus CRUD。
    """

    def __init__(
        self,
        adapter: VectorStorageAdapter[Any, Any, Any],
    ) -> None:
        self.adapter = adapter

    def to_entity(self, chunk: Any) -> MilvusV15Entity:
        """创建包含 4 个物理字段和 flat metadata JSON 的 V15 Entity。"""

        if not self.adapter.is_embedded(chunk):
            raise TypeError(
                "MilvusV15RowMapper 只接受 EmbeddedContractPreparedChunk"
            )

        projection = self.adapter.storage_projection(chunk)
        required_parts = ("contract", "metadata", "source", "versions")
        if any(key not in projection for key in required_parts):
            raise ValueError(
                "Contract Adapter 的 storage_projection 缺少 V15 必需分区"
            )
        flat = _merge_flat_metadata(
            tuple(
                (origin, projection[origin])
                for origin in required_parts
            )
        )

        entity = MilvusV15Entity(
            vector_doc_id=self.adapter.vector_doc_id(chunk),
            chunk_text=self.adapter.chunk_text(chunk),
            embedding=list(self.adapter.embedding(chunk)),
            metadata=flat,
        )
        self._validate_storage_limits(entity)
        return entity

    def to_row(self, chunk: Any) -> dict[str, object]:
        """返回可直接交给 PyMilvus 的 V15 字典行。"""

        return self.to_entity(chunk).model_dump(mode="python")

    def to_rows(
        self,
        chunks: Sequence[Any],
    ) -> list[dict[str, object]]:
        """批量映射并拒绝同批重复主键。

        调用方：
            后续 V15 Repository 在落 SDK 前调用；这是正式写入路径的主入口。
        """

        entities = [self.to_entity(chunk) for chunk in chunks]
        primary_keys = [entity.vector_doc_id for entity in entities]
        if len(primary_keys) != len(set(primary_keys)):
            raise ValueError("一个 Milvus batch 不能包含重复 vector_doc_id")
        return [entity.model_dump(mode="python") for entity in entities]

    def from_row(self, row: Mapping[str, object]) -> Any:
        """将已有 V15 行还原为 Contract 输出对象（反向映射，非主写入路径）。

        用途：
            Metadata-only 更新必须保留原向量，因此需要先从 Milvus 读取当前
            Entity，再用新输入的 Metadata/source/versions 生成完整 Upsert 行。

        边界：
            该方法只做字段还原，不会调用 Embedding，也不会修改 Milvus；
            ``metadata`` 中 Decimal 化的 ``work_years`` 字符串会由 Contract
            模型重新构造为 ``Decimal``。不重新计算 vector_doc_id、不改变 hash。
        """

        return self.adapter.restore_embedded(row)

    @staticmethod
    def _validate_storage_limits(entity: MilvusV15Entity) -> None:
        """按 Milvus VARCHAR 字节和 JSON 大小限制校验映射结果。

        Milvus ``VARCHAR`` 的容量按 UTF-8 bytes 而不是 Python 字符数计算；JSON
        字段也必须在服务端限制内。超限直接拒绝写入，禁止静默截断。
        """

        specs = v15_milvus_field_specs(len(entity.embedding))
        values = entity.model_dump(mode="python")
        for spec in specs:
            if spec.max_length is None:
                continue
            value = values[spec.name]
            if value is not None and len(str(value).encode("utf-8")) > spec.max_length:
                raise ValueError(
                    f"{spec.name} 超过 Milvus VARCHAR 实现容量 {spec.max_length} bytes"
                )
        payload = json.dumps(
            values["metadata"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > 65536:
            raise ValueError("metadata 超过 Milvus JSON 65536 bytes 限制")


def _merge_flat_metadata(
    parts: Sequence[tuple[str, Mapping[str, object]]],
) -> dict[str, object]:
    """把多个来源的字段映射合并为单层 flat dict，冲突时 fail-closed。

    规则：
        - 任意两个来源出现相同 key → ``ValueError``（不允许重复 key 后覆盖）；
        - key 属于一级物理字段（``vector_doc_id`` / ``chunk_text`` /
          ``embedding``）→ ``ValueError``；
        - key 属于 namespace 键（``contract`` / ``metadata`` / ``source`` /
          ``versions``）→ ``ValueError``；
        - ``null`` / ``[]`` / Decimal 字符串 / ISO 8601 字符串原样保留。
    """

    return merge_flat_metadata(
        parts,
        physical_keys=_V15_PHYSICAL_KEYS,
        namespace_keys=_V15_NAMESPACE_KEYS,
    )


class MilvusV15RowMapper(ContractV15RowMapper):
    """Expert 领域的 V15 RowMapper 兼容包装。"""

    def __init__(
        self,
        adapter: VectorStorageAdapter[Any, Any, Any] | None = None,
    ) -> None:
        super().__init__(adapter or ExpertContractAdapter())


class ResumeMilvusV15RowMapper(ContractV15RowMapper):
    """Resume 领域的 V15 RowMapper 兼容包装。"""

    def __init__(
        self,
        adapter: VectorStorageAdapter[Any, Any, Any] | None = None,
    ) -> None:
        super().__init__(adapter or ResumeContractAdapter())
