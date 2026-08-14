"""正式 Chunk Contract 对应的 Milvus Entity 与行映射。

本模块是 Contract 输出到物理存储之间唯一的字段映射边界：高频标量被投影为
独立列，``metadata/source/versions`` 作为完整 JSON 快照保留，且不包含任何
已废弃字段（chunk_id / source_item_key / data_version / dictionary_version /
company_normalization_status / project_role / metadata.vector_doc_id）。它不创建
Collection、不调用 Milvus Client，也不决定 Task State；Repository 只消费这里的
Entity/row 结果。

``work_years`` 在 Contract 中保持 ``Decimal``；写入 JSON extension 时依赖
Pydantic ``mode="json"`` 稳定序列化为字符串（保留两位小数精度），不转 float。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from expert_contracts.models import EmbeddedContractPreparedChunk

from .schemas.contract import (
    CONTRACT_SCALAR_INDEX_FIELDS,
    CONTRACT_VECTOR_DIMENSION,
    CONTRACT_VECTOR_DOC_ID_MAX_BYTES,
    contract_milvus_field_specs,
)
from .schemas.specs import MilvusFieldSpec

__all__ = [
    "CONTRACT_SCALAR_INDEX_FIELDS",
    "CONTRACT_VECTOR_DIMENSION",
    "CONTRACT_VECTOR_DOC_ID_MAX_BYTES",
    "MilvusExpertChunkEntity",
    "MilvusFieldSpec",
    "MilvusRowMapper",
    "contract_milvus_field_specs",
]


class MilvusExpertChunkEntity(BaseModel):
    """正式 Contract 在 Milvus 中的完整 Entity 数据模型。

    该模型同时表达向量、标量投影和三组 JSON extension，确保过滤字段可直接使用
    且原始嵌套结构可完整回查。构造时会拒绝非有限向量或错误维度。

    身份字段只有 ``vector_doc_id``（唯一 PK），不存在 ``chunk_id`` 或其他业务
    主键；已废弃字段不得出现在本模型中。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    vector_doc_id: str
    expert_id: int
    serial_number: str | None
    chunk_type: str
    chunk_subtype: str | None
    chunk_index: int
    chunk_total: int

    chunk_text: str
    char_count: int
    content_hash: str
    metadata_hash: str
    contract_version: str

    verification_status: str
    eligible_for_recommendation: bool | None
    weight_tier: str
    is_active: bool

    updated_at: str
    embedded_at: str
    embedding: list[float] = Field(min_length=1)
    embedding_dim: int = Field(ge=1)
    embedding_model_alias: str
    embedding_model_version: str

    metadata_ext: dict[str, object]
    source_ext: dict[str, object]
    versions_ext: dict[str, object]

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        """确保 Milvus Entity 向量只包含有限浮点数。"""

        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding 必须只包含有限浮点数")
        return value

    @model_validator(mode="after")
    def validate_embedding_dimension(self) -> MilvusExpertChunkEntity:
        """确保 Entity 的维度声明与向量长度一致。"""

        if self.embedding_dim != len(self.embedding):
            raise ValueError("embedding_dim 必须等于 embedding 实际长度")
        return self


class MilvusRowMapper:
    """将 ``EmbeddedContractPreparedChunk`` 无损映射为 Milvus 行。

    Mapper 负责上游主键透传、字段投影、JSON 快照和存储容量检查；不负责调用
    Embedding、生成或重算主键、修改输入对象或执行 Milvus CRUD。
    """

    def to_entity(
        self,
        chunk: EmbeddedContractPreparedChunk,
    ) -> MilvusExpertChunkEntity:
        """创建包含高频标量和三份完整 JSON 快照的 Milvus Entity。"""

        if not isinstance(chunk, EmbeddedContractPreparedChunk):
            raise TypeError("MilvusRowMapper 只接受 EmbeddedContractPreparedChunk")

        # scalar projection 服务于高频过滤；完整对象另存为 JSON，不能因为只投影
        # 过滤字段就丢失 source/versions 中的追溯信息。model_dump(mode="json")
        # 会把 Decimal work_years 稳定序列化为字符串，不转 float。
        metadata = chunk.metadata.model_dump(mode="json")
        source = chunk.source.model_dump(mode="json")
        versions = chunk.versions.model_dump(mode="json")
        entity = MilvusExpertChunkEntity(
            vector_doc_id=chunk.vector_doc_id,
            expert_id=chunk.expert_id,
            serial_number=chunk.serial_number,
            chunk_type=chunk.chunk_type,
            chunk_subtype=chunk.chunk_subtype,
            chunk_index=chunk.chunk_index,
            chunk_total=chunk.chunk_total,
            chunk_text=chunk.chunk_text,
            char_count=chunk.char_count,
            content_hash=chunk.content_hash,
            metadata_hash=chunk.metadata_hash,
            contract_version=chunk.contract_version,
            verification_status=chunk.metadata.verification_status,
            eligible_for_recommendation=(
                chunk.metadata.eligible_for_recommendation
            ),
            weight_tier=chunk.metadata.weight_tier,
            is_active=chunk.is_active,
            updated_at=chunk.updated_at,
            embedded_at=chunk.embedded_at,
            embedding=list(chunk.embedding),
            embedding_dim=chunk.embedding_dim,
            embedding_model_alias=chunk.embedding_model_alias,
            embedding_model_version=chunk.embedding_model_version,
            metadata_ext=metadata,
            source_ext=source,
            versions_ext=versions,
        )
        self._validate_storage_limits(entity)
        return entity

    def to_row(self, chunk: EmbeddedContractPreparedChunk) -> dict[str, object]:
        """返回可直接交给 PyMilvus 的字典行。"""

        return self.to_entity(chunk).model_dump(mode="python")

    def to_rows(
        self,
        chunks: Sequence[EmbeddedContractPreparedChunk],
    ) -> list[dict[str, object]]:
        """批量映射并拒绝同批重复主键。"""

        entities = [self.to_entity(chunk) for chunk in chunks]
        primary_keys = [entity.vector_doc_id for entity in entities]
        if len(primary_keys) != len(set(primary_keys)):
            raise ValueError("一个 Milvus batch 不能包含重复 vector_doc_id")
        return [entity.model_dump(mode="python") for entity in entities]

    def from_row(self, row: Mapping[str, object]) -> EmbeddedContractPreparedChunk:
        """将已有 Milvus 行还原为 Contract 输出对象。

        Metadata-only 更新必须保留原向量，因此需要先从 Milvus 读取当前 Entity，
        再用新输入的 Metadata/source/versions 生成完整 Upsert 行。该方法只做字段
        还原，不会调用 Embedding，也不会修改 Milvus。``metadata_ext`` 中 Decimal
        化的 ``work_years`` 字符串会由 Contract 模型重新构造为 ``Decimal``。
        """

        metadata = row.get("metadata_ext")
        source = row.get("source_ext")
        versions = row.get("versions_ext")
        if not isinstance(metadata, Mapping):
            raise TypeError("Milvus 行缺少合法 metadata_ext JSON 对象")
        if not isinstance(source, Mapping):
            raise TypeError("Milvus 行缺少合法 source_ext JSON 对象")
        if not isinstance(versions, Mapping):
            raise TypeError("Milvus 行缺少合法 versions_ext JSON 对象")

        # 反向映射只还原已有 Entity，不生成新向量。Metadata-only/audit-only 更新
        # 依赖旧向量和新 Contract 字段重新构造完整 Upsert 行。
        payload = {
            "contract_version": row.get("contract_version"),
            "vector_doc_id": row.get("vector_doc_id"),
            "expert_id": row.get("expert_id"),
            "serial_number": row.get("serial_number"),
            "chunk_type": row.get("chunk_type"),
            "chunk_subtype": row.get("chunk_subtype"),
            "chunk_index": row.get("chunk_index"),
            "chunk_total": row.get("chunk_total"),
            "chunk_text": row.get("chunk_text"),
            "char_count": row.get("char_count"),
            "content_hash": row.get("content_hash"),
            "metadata_hash": row.get("metadata_hash"),
            "metadata": dict(metadata),
            "source": dict(source),
            "versions": dict(versions),
            "is_active": row.get("is_active"),
            "updated_at": row.get("updated_at"),
            "embedding": row.get("embedding"),
            "embedding_dim": row.get("embedding_dim"),
            "embedding_model_alias": row.get("embedding_model_alias"),
            "embedding_model_version": row.get("embedding_model_version"),
            "embedded_at": row.get("embedded_at"),
        }
        return EmbeddedContractPreparedChunk.model_validate(payload)

    @staticmethod
    def _validate_storage_limits(entity: MilvusExpertChunkEntity) -> None:
        """按 Milvus VARCHAR 字节和 JSON 大小限制校验映射结果。

        Milvus ``VARCHAR`` 的容量按 UTF-8 bytes 而不是 Python 字符数计算；JSON
        extension 也必须在服务端限制内。超限直接拒绝写入，禁止静默截断。
        """

        specs = contract_milvus_field_specs(entity.embedding_dim)
        values = entity.model_dump(mode="python")
        for spec in specs:
            if spec.max_length is None:
                continue
            value = values[spec.name]
            if value is not None and len(str(value).encode("utf-8")) > spec.max_length:
                raise ValueError(
                    f"{spec.name} 超过 Milvus VARCHAR 实现容量 {spec.max_length} bytes"
                )
        for field_name in ("metadata_ext", "source_ext", "versions_ext"):
            payload = json.dumps(
                values[field_name],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(payload) > 65536:
                raise ValueError(f"{field_name} 超过 Milvus JSON 65536 bytes 限制")