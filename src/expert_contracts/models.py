"""专家 Chunk 向量化输入契约 v4.1.0 的共享强类型定义。

本模块只表达 Contract 的字段、枚举和校验规则，不读取数据库、不读取文件，
也不依赖任何业务 Builder 或向量存储实现。上游生成的 ``vector_doc_id`` 只存在
Contract 顶层并按正式公式生成；``metadata`` 不再保留 ``vector_doc_id`` 副本。

4.1.0 变更：``metadata`` 从三类字段统一 superset 改为按 ``chunk_type`` 的
per-type 模型（``ProfileSummaryMetadata`` / ``EmploymentMetadata`` /
``ProjectMetadata``），不适用于当前类型的字段不再输出占位。
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EXPERT_CHUNK_CONTRACT_VERSION = "4.1.0"

_VECTOR_DOC_ID_MAX_BYTES = 2048

ChunkType = Literal["profile_summary", "employment", "project"]
ChunkSubtype = Literal[
    "responsibility",
    "achievement",
    "context",
    "action",
    "result",
    "segment",
]
# 正式导出契约枚举：profile 固定 compressed_from_source；employment/project 固定 source_parsed。
OriginType = Literal[
    "compressed_from_source",
    "source_parsed",
]
# 正式导出契约枚举：profile 为 source_summary / rejected；employment/project 固定 verified。
VerificationStatus = Literal[
    "source_summary",
    "rejected",
    "verified",
]
# 检索权重等级。本契约只负责承载正式字段，不把它当作 Embedding Eligibility；
# 在未取得真实 DDL/校验约束前，不擅自删除当前数据之外可能存在的其他取值。
WeightTier = Literal["A", "B", "C"]

_MONTH_PATTERN = r"^[0-9]{4}(0[1-9]|1[0-2])$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CODE_PATTERN = r"^[a-z][a-z0-9_]*$"


def vector_doc_id_prefix_for_source_table(source_table: str) -> str:
    """根据冻结来源表标签返回 ``vector_doc_id`` 命名空间。

    Contract 不携带数据集字段，为了不改变 4.1.0 的公开字段和 Milvus 物理 Schema，
    复用已经必填的 ``source.source_table`` 判断命名空间。所有 Resume 来源标签均以
    ``resume_`` 开头（Profile 聚合标签的第一个组成表也满足该规则）；未知或历史
    Expert 标签继续回退 ``expert``，确保既有 Expert Contract 行为完全兼容。
    """

    if not isinstance(source_table, str) or not source_table.strip():
        raise ValueError("source_table 必须是非空字符串")
    source_tables = tuple(part.strip() for part in source_table.split("+"))
    if source_tables and all(part.startswith("resume_") for part in source_tables):
        return "resume"
    return "expert"


__all__ = [
    "EXPERT_CHUNK_CONTRACT_VERSION",
    "EXPERT_CHUNK_METADATA_BY_TYPE",
    "ChunkSubtype",
    "ChunkType",
    "ContractPreparedChunk",
    "EmbeddedContractPreparedChunk",
    "EmploymentMetadata",
    "ExpertChunkEmbeddingInput",
    "ExpertChunkSource",
    "ExpertChunkVersions",
    "OriginType",
    "ProfileSummaryMetadata",
    "ProjectMetadata",
    "VerificationStatus",
    "WeightTier",
    "metadata_model_for_chunk_type",
    "vector_doc_id_prefix_for_source_table",
]


class _ContractModel(BaseModel):
    """所有共享 Contract 模型共用的严格对象配置。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class _ChunkMetadataBase(_ContractModel):
    """三类 Chunk 公共的 Metadata 字段。

    只包含对三种 ``chunk_type`` 都有意义、且都参与的字段；类型专属字段由
    ``ProfileSummaryMetadata`` / ``EmploymentMetadata`` / ``ProjectMetadata``
    各自声明。不适用于当前 ``chunk_type`` 的字段不得出现在该类型的 Metadata 中
    （不输出、不填 ``None``/``[]`` 占位）。
    """

    company_name_raw: str | None = Field(..., max_length=512)
    company_name_norm: str | None = Field(..., max_length=512)
    industry_codes: list[str] = Field(...)
    topic_codes: list[str] = Field(...)
    origin_type: OriginType = Field(...)
    verification_status: VerificationStatus = Field(...)
    weight_tier: WeightTier = Field(...)

    @field_validator("industry_codes", "topic_codes")
    @classmethod
    def validate_code_arrays(cls, value: list[str]) -> list[str]:
        """校验标签代码是合法的小写代码，并且已经去重、排序。

        ``school_names_raw``/``major_names_raw`` 是自由文本，不受本规则约束。
        """

        if any(not re.fullmatch(_CODE_PATTERN, item) for item in value):
            raise ValueError("industry/topic_codes 必须是小写 snake_case 代码")
        if value != sorted(set(value)):
            raise ValueError("industry/topic_codes 必须去重并按字典序升序排列")
        return value


class ProfileSummaryMetadata(_ChunkMetadataBase):
    """profile_summary 的专属业务 Metadata。

    只声明 profile 有意义的字段：岗位、工作年限、海外标记、职能/学校/专业数组
    与推荐资格；不包含 employment/project 的 ``project_name``/``start_month``/
    ``end_month``/``is_current``。
    """

    role_name_raw: str | None = Field(..., max_length=256)
    role_code: str | None = Field(..., max_length=128)
    role_level: str | None = Field(..., max_length=64)
    job_status_norm: str | None = Field(..., max_length=32)
    abroad_flag_norm: bool | None = Field(...)
    work_years: Decimal | None = Field(...)
    function_codes: list[str] = Field(...)
    school_names_raw: list[str] = Field(...)
    major_names_raw: list[str] = Field(...)
    eligible_for_recommendation: bool | None = Field(...)

    @field_validator("function_codes")
    @classmethod
    def validate_function_codes(cls, value: list[str]) -> list[str]:
        """职能代码同样要求小写 snake_case、去重、升序。"""

        if any(not re.fullmatch(_CODE_PATTERN, item) for item in value):
            raise ValueError("function_codes 必须是小写 snake_case 代码")
        if value != sorted(set(value)):
            raise ValueError("function_codes 必须去重并按字典序升序排列")
        return value

    @field_validator("work_years", mode="before")
    @classmethod
    def coerce_work_years(cls, value: object) -> Decimal | None:
        """把 JSON 兼容数值表示转换为 ``Decimal``，禁止转 float。

        ``work_years`` 的正式类型是 ``Decimal | None``；JSON/Mapping 输入无法直接
        携带 ``Decimal``，这里只接受数值字面量或十进制字符串，并直接构造 ``Decimal``，
        不经过 ``float``。布尔值不是合法工作年限。
        """

        if value is None:
            return None
        if isinstance(value, bool) or (
            not isinstance(value, (Decimal, int, str))
        ):
            raise TypeError("work_years 必须是 DECIMAL(5,2) 数值或 NULL")
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        try:
            return Decimal(value)
        except (ValueError, ArithmeticError) as exc:
            raise ValueError("work_years 必须是 DECIMAL(5,2) 数值或 NULL") from exc


class EmploymentMetadata(_ChunkMetadataBase):
    """employment 的专属业务 Metadata。

    岗位字段（profile 也有）、任职起止月与是否当前任职；不包含 profile 的
    ``job_status_norm``/``abroad_flag_norm``/``work_years``/``school_names_raw``/
    ``major_names_raw``/``eligible_for_recommendation``，也不包含 project 的
    ``project_name``。
    """

    role_name_raw: str | None = Field(..., max_length=256)
    role_code: str | None = Field(..., max_length=128)
    role_level: str | None = Field(..., max_length=64)
    start_month: str | None = Field(..., pattern=_MONTH_PATTERN)
    end_month: str | None = Field(..., pattern=_MONTH_PATTERN)
    is_current: bool | None = Field(...)


class ProjectMetadata(_ChunkMetadataBase):
    """project 的专属业务 Metadata。

    项目名称与起止月；不包含岗位字段（``role_name_raw``/``role_code``/
    ``role_level``）、``is_current``、教育数组与推荐资格。
    """

    project_name: str | None = Field(default=None, max_length=512)
    start_month: str | None = Field(..., pattern=_MONTH_PATTERN)
    end_month: str | None = Field(..., pattern=_MONTH_PATTERN)


# 三类 Metadata 的联合类型与按 chunk_type 的注册表。
ChunkMetadata = ProfileSummaryMetadata | EmploymentMetadata | ProjectMetadata

EXPERT_CHUNK_METADATA_BY_TYPE: dict[str, type[ChunkMetadata]] = {
    "profile_summary": ProfileSummaryMetadata,
    "employment": EmploymentMetadata,
    "project": ProjectMetadata,
}


def metadata_model_for_chunk_type(chunk_type: str) -> type[ChunkMetadata]:
    """返回指定 ``chunk_type`` 对应的 Metadata 模型类（fail-closed）。"""

    try:
        return EXPERT_CHUNK_METADATA_BY_TYPE[chunk_type]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"未知 chunk_type: {chunk_type!r}") from exc


class ExpertChunkSource(_ContractModel):
    """契约中的来源追溯字段。"""

    source_table: str = Field(..., min_length=1, max_length=256)
    source_field: str = Field(..., min_length=1, max_length=512)
    source_field_id: str = Field(..., min_length=1, max_length=128)
    source_version: str = Field(..., min_length=1, max_length=64)
    source_updated_at: str | None = Field(...)
    source_hash: str = Field(..., pattern=_SHA256_PATTERN)

    @field_validator("source_updated_at")
    @classmethod
    def validate_source_updated_at(cls, value: str | None) -> str | None:
        """校验来源更新时间是带时区的 ISO 8601 时间。"""

        if value is not None:
            _validate_timezone_datetime(value, "source_updated_at")
        return value


class ExpertChunkVersions(_ContractModel):
    """正式导出契约的版本字段（不含已废弃的 data/dictionary 版本）。"""

    chunk_policy_version: str = Field(..., min_length=1, max_length=32)
    metadata_schema_version: str = Field(..., min_length=1, max_length=32)


class ExpertChunkEmbeddingInput(_ContractModel):
    """一条待向量化的正式内存输入记录（Contract 4.1.0）。

    在主链路中的位置：
        - 产生：由上游 Builder（expert_index 侧）生成，或由
          上游 ``contract_bridge`` 从 ``ChunkRecord`` 直接组装。
        - 消费：``ContractValidator.validate``（校验）→ ``ContractEmbeddingWorker``
          （Eligibility + Embedding）→ ``ContractIngestionService`` /
          ``ContractBatchIngestionService``（入库编排）。
        - 下一站：通过 Validator 后转换为 ``ContractPreparedChunk``。

    设计要点：
        - 顶层 ``vector_doc_id`` 是唯一身份键（主线公式见 ``validate_contract_consistency``），
          ``metadata`` 不保留副本；不存在 ``chunk_id`` 旧语义。
        - 叠加 strict + extra=forbid：未声明字段或类型不符都会直接失败（fail-closed）。
        - 本类型只表达结构，不负责校验“是否应 Embedding”，也不负责向量化。
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        title="Expert Chunk Embedding Input",
        json_schema_extra={
            "$id": "https://internal.example/schemas/"
            "expert_chunk_embedding_input-4.1.0.json"
        },
    )

    contract_version: Literal["4.1.0"]
    vector_doc_id: str = Field(..., min_length=1, max_length=_VECTOR_DOC_ID_MAX_BYTES)
    expert_id: int = Field(..., ge=1)
    serial_number: str | None = Field(..., max_length=64)
    chunk_type: ChunkType
    chunk_subtype: ChunkSubtype | None = Field(...)
    chunk_index: int = Field(..., ge=0)
    chunk_total: int = Field(..., ge=1)
    chunk_text: str = Field(..., min_length=1, max_length=1000)
    char_count: int = Field(..., ge=1, le=1000)
    content_hash: str = Field(..., pattern=_SHA256_PATTERN)
    metadata_hash: str = Field(..., pattern=_SHA256_PATTERN)
    metadata: ChunkMetadata
    source: ExpertChunkSource
    versions: ExpertChunkVersions
    is_active: bool
    updated_at: str = Field(..., min_length=1)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: str) -> str:
        """校验 Chunk 更新时间是带时区的 ISO 8601 时间。"""

        _validate_timezone_datetime(value, "updated_at")
        return value

    @field_validator("vector_doc_id")
    @classmethod
    def validate_vector_doc_id_capacity(cls, value: str) -> str:
        """校验上游主键不超过正式 Milvus 的 UTF-8 字节容量。"""

        return _validate_vector_doc_id_capacity(value)

    @model_validator(mode="after")
    def validate_contract_consistency(self) -> ExpertChunkEmbeddingInput:
        """校验正式契约版本，并依据正式公式校验顶层向量主键。

        ``metadata`` 不保留 ``vector_doc_id`` 副本，因此这里不再比对快照，而是用
        ``{namespace}:{expert_id}:{chunk_type}:{source_field_id}:{chunk_index}`` 公式
        直接校验顶层 ``vector_doc_id``。namespace 由冻结来源表标签决定：Expert 为
        ``expert``，Resume 为 ``resume``。
        """

        if self.contract_version != EXPERT_CHUNK_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version 必须是 {EXPERT_CHUNK_CONTRACT_VERSION}"
            )
        # Metadata 类型必须与 chunk_type 匹配：不输出其他类型的字段占位。
        expected_metadata_type = metadata_model_for_chunk_type(self.chunk_type)
        if not isinstance(self.metadata, expected_metadata_type):
            # pydantic model_validator 抛 ValueError 会被封装为 ValidationError，
            # 符合契约 fail-closed 语义；此处 isinstance 检查是业务语义校验，
            # 不是 Python 类型错误，保留 ValueError（TRY004 误报，显式抑制）。
            raise ValueError(  # noqa: TRY004
                f"{self.chunk_type} 的 metadata 必须是 "
                f"{expected_metadata_type.__name__}"
            )
        namespace = vector_doc_id_prefix_for_source_table(self.source.source_table)
        expected_vector_doc_id = (
            f"{namespace}:{self.expert_id}:{self.chunk_type}:"
            f"{self.source.source_field_id}:{self.chunk_index}"
        )
        if self.vector_doc_id != expected_vector_doc_id:
            raise ValueError(
                "vector_doc_id 必须等于正式公式 "
                f"{namespace}:{{expert_id}}:{{chunk_type}}:"
                "{source_field_id}:{chunk_index}"
                f"（namespace={namespace}）"
            )
        return self


class ContractPreparedChunk(ExpertChunkEmbeddingInput):
    """通过契约校验、尚未生成向量的 Chunk。

    在主链路中的位置：
        - 产生：``ContractValidator.validate`` / ``validate_mapping`` 成功时，
          由 ``ExpertChunkEmbeddingInput`` 转换而来。
        - 消费：``EmbeddingEligibilityValidator.get_skip_reason`` 判断是否允许
          Embedding。
        - 下一站：eligible 记录经 ``ContractEmbeddingWorker.embed`` 附加向量后
          变为 ``EmbeddedContractPreparedChunk``。

    含义：它的存在代表“该记录已通过消费端 Validator”，但尚未具备向量。
    """


class EmbeddedContractPreparedChunk(ContractPreparedChunk):
    """保留完整输入契约并附加向量化输出字段的记录。

    在主链路中的位置：
        - 产生：``ContractEmbeddingWorker.embed`` 对每条 eligible 记录调用
          ``EmbeddingModel.embed`` 后构造。
        - 消费：``ContractIngestionService`` / ``ContractBatchIngestionService``
          把 embedded 记录交给 ``ContractMilvusV15VectorRepository.upsert_batch``。
        - 下一站：``MilvusV15RowMapper.to_entity`` 映射为 Milvus 物理行。

    ``embedded_at`` 是内部 Embedding 完成时间，与下游回填的 ``indexed_at``
    （Milvus upsert 成功时间）保持独立。向量字段带模型空间信息，保证写入前
    可跨批次核对模型空间一致。
    """

    embedding: list[float] = Field(..., min_length=1)
    embedding_dim: int = Field(..., ge=1)
    embedding_model_alias: str = Field(..., min_length=1, max_length=128)
    embedding_model_version: str = Field(..., min_length=1, max_length=128)
    embedded_at: str = Field(..., min_length=1)

    @field_validator("embedding")
    @classmethod
    def validate_embedding_values(cls, value: list[float]) -> list[float]:
        """确保输出向量非空，并且每个元素都是有限数值。"""

        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding 必须只包含有限数值")
        return value

    @field_validator("embedded_at")
    @classmethod
    def validate_embedded_at(cls, value: str) -> str:
        """校验向量生成时间是带时区的 ISO 8601 时间。"""

        _validate_timezone_datetime(value, "embedded_at")
        return value

    @model_validator(mode="after")
    def validate_embedding_dimension(self) -> EmbeddedContractPreparedChunk:
        """确保声明的维度与实际向量长度一致。"""

        if self.embedding_dim != len(self.embedding):
            raise ValueError("embedding_dim 必须等于 embedding 的实际长度")
        return self


def _validate_timezone_datetime(value: str, field_name: str) -> None:
    """校验时间字符串可解析且包含时区信息。"""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} 必须是合法的 ISO 8601 时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} 必须包含时区")


def _validate_vector_doc_id_capacity(value: str) -> str:
    """统一校验上游主键的 Milvus UTF-8 字节容量。"""

    if len(value.encode("utf-8")) > _VECTOR_DOC_ID_MAX_BYTES:
        raise ValueError(
            f"vector_doc_id 超过 Milvus VARCHAR 实现容量 {_VECTOR_DOC_ID_MAX_BYTES} bytes"
        )
    return value
