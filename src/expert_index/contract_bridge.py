"""把上游 ``ChunkRecord`` 无损转换为下游正式 Contract 4.1.0。

Bridge 只负责扁平字段到嵌套 Contract 的结构和类型转换，不重新生成正文、主键或
哈希。这样上游仍是业务事实与哈希的唯一权威，下游严格模型负责最终 fail-closed
校验。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from expert_contracts import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    EmploymentMetadata,
    ExpertChunkEmbeddingInput,
    ExpertChunkSource,
    ExpertChunkVersions,
    ProfileSummaryMetadata,
    ProjectMetadata,
)

from .dataset import EXPERT_SPEC, DatasetSpec
from .models import ChunkRecord
from .validators import ChunkValidationError, validate_chunks


def chunk_record_to_contract(
    chunk: ChunkRecord,
    *,
    updated_at: datetime,
    timezone_name: str,
) -> ExpertChunkEmbeddingInput:
    """把一条已校验的上游 Chunk 转换为严格向量化输入 Contract。

    ``work_years`` 在上游当前领域模型中是 ``float``，而正式 Contract 禁止 float；
    这里通过十进制字符串构造 ``Decimal``，避免二进制浮点误差。MySQL naive 时间按
    配置的业务时区解释，输出带时区的 ISO 8601 字符串。
    """

    contract_updated_at = _iso_with_timezone(
        updated_at,
        timezone_name=timezone_name,
        field_name="updated_at",
    )
    if contract_updated_at is None:  # pragma: no cover - updated_at 参数不允许为 None
        raise ValueError("updated_at 不能为空")

    work_years = (
        Decimal(str(chunk.work_years))
        if chunk.work_years is not None
        else None
    )
    # 按 chunk_type 构造对应 Metadata：不输出其他类型的字段占位。
    if chunk.chunk_type == "profile_summary":
        metadata = ProfileSummaryMetadata(
            company_name_raw=chunk.company_name_raw,
            company_name_norm=chunk.company_name_norm,
            role_name_raw=chunk.role_name_raw,
            role_code=chunk.role_code,
            role_level=chunk.role_level,
            job_status_norm=chunk.job_status_norm,
            abroad_flag_norm=chunk.abroad_flag_norm,
            work_years=work_years,
            industry_codes=list(chunk.industry_codes),
            function_codes=list(chunk.function_codes),
            topic_codes=list(chunk.topic_codes),
            school_names_raw=list(chunk.school_names_raw),
            major_names_raw=list(chunk.major_names_raw),
            eligible_for_recommendation=chunk.eligible_for_recommendation,
            origin_type=chunk.origin_type,
            verification_status=chunk.verification_status,
            weight_tier=chunk.weight_tier,
        )
    elif chunk.chunk_type == "employment":
        metadata = EmploymentMetadata(
            company_name_raw=chunk.company_name_raw,
            company_name_norm=chunk.company_name_norm,
            role_name_raw=chunk.role_name_raw,
            role_code=chunk.role_code,
            role_level=chunk.role_level,
            start_month=chunk.start_month,
            end_month=chunk.end_month,
            is_current=chunk.is_current,
            industry_codes=list(chunk.industry_codes),
            topic_codes=list(chunk.topic_codes),
            origin_type=chunk.origin_type,
            verification_status=chunk.verification_status,
            weight_tier=chunk.weight_tier,
        )
    else:
        metadata = ProjectMetadata(
            company_name_raw=chunk.company_name_raw,
            company_name_norm=chunk.company_name_norm,
            project_name=chunk.project_name,
            start_month=chunk.start_month,
            end_month=chunk.end_month,
            industry_codes=list(chunk.industry_codes),
            topic_codes=list(chunk.topic_codes),
            origin_type=chunk.origin_type,
            verification_status=chunk.verification_status,
            weight_tier=chunk.weight_tier,
        )
    return ExpertChunkEmbeddingInput(
        contract_version=EXPERT_CHUNK_CONTRACT_VERSION,
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
        metadata=metadata,
        source=ExpertChunkSource(
            source_table=chunk.source_table,
            source_field=chunk.source_field,
            source_field_id=chunk.source_field_id,
            source_version=chunk.source_version,
            source_updated_at=_iso_with_timezone(
                chunk.source_updated_at,
                timezone_name=timezone_name,
                field_name="source_updated_at",
            ),
            source_hash=chunk.source_hash,
        ),
        versions=ExpertChunkVersions(
            chunk_policy_version=chunk.chunk_policy_version,
            metadata_schema_version=chunk.metadata_schema_version,
        ),
        is_active=chunk.is_active,
        updated_at=contract_updated_at,
    )


def chunk_records_to_contracts(
    chunks: Sequence[ChunkRecord],
    *,
    updated_at: datetime,
    timezone_name: str,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> list[ExpertChunkEmbeddingInput]:
    """校验并转换一位专家的完整 Chunk 快照。

    ``dataset_spec`` 用于把上游 Validator 与 Dataset 的 ID 命名空间保持一致；省略时
    使用 Expert 默认规格，Resume 入口必须显式传入 ``RESUME_SPEC``。
    空快照直接返回空列表；非空快照先复用上游 Validator 检查 ID、Hash 和集合唯一性，
    再拒绝混入多个专家的输入。校验完成后按原顺序转换，并为全部记录使用同一交付
    时间。任一记录失败时不返回部分 Contract。
    """

    prepared_chunks = list(chunks)
    # Expert 仍走历史单参数入口，保持外部 monkeypatch/扩展兼容；Resume 等显式规格
    # 才把命名空间传入 Validator。这样共享桥接不会改变 Expert 的调用形态。
    if dataset_spec is EXPERT_SPEC:
        validate_chunks(prepared_chunks)
    else:
        validate_chunks(prepared_chunks, dataset_spec=dataset_spec)
    expert_ids = {chunk.expert_id for chunk in prepared_chunks}
    if len(expert_ids) > 1:
        raise ChunkValidationError("内存 Contract 批次只能包含同一专家的完整 Chunk 快照")

    return [
        chunk_record_to_contract(
            chunk,
            updated_at=updated_at,
            timezone_name=timezone_name,
        )
        for chunk in prepared_chunks
    ]


def _iso_with_timezone(
    value: datetime | None,
    *,
    timezone_name: str,
    field_name: str,
) -> str | None:
    """把 datetime 转成带时区 ISO 8601；naive 值按业务时区解释。"""

    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} 必须是 datetime 或 None")
    aware = value
    if aware.tzinfo is None or aware.utcoffset() is None:
        aware = aware.replace(tzinfo=ZoneInfo(timezone_name))
    return aware.isoformat(timespec="microseconds")


__all__ = ["chunk_record_to_contract", "chunk_records_to_contracts"]
