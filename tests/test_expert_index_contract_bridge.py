"""上游 ``ChunkRecord`` 到下游正式 Contract 的无外部依赖测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal

import pytest

from expert_contracts import (
    EmploymentMetadata,
    ExpertChunkEmbeddingInput,
    ProfileSummaryMetadata,
    ProjectMetadata,
)
from expert_index.chunk_builder import ChunkFactory, calculate_metadata_hash
from expert_index.config import DatabaseConfig, Settings
from expert_index.contract_bridge import (
    chunk_record_to_contract,
    chunk_records_to_contracts,
)
from expert_index.models import (
    ChunkRecord,
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
)
from expert_index.normalize import normalize_aggregate
from expert_index.validators import ChunkValidationError, validate_chunks


def _settings() -> Settings:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def test_chunk_record_maps_to_strict_contract_v410() -> None:
    """Bridge 应补齐嵌套结构、Decimal 和带时区时间，不改 ID/正文/hash。"""

    source_time = datetime(2026, 8, 18, 9, 30)  # noqa: DTZ001 - 模拟 MySQL naive DATETIME
    aggregate = ExpertAggregate(
        profile=ExpertSource(
            id=700001,
            serial_number="TEST700001",
            company="虚构桥接测试公司",
            job_status=1,
            job_name="项目经理",
            contact_status=3,
            doubtful_flag=1,
            abroad_flag=2,
            background="仅用于 Contract Bridge 单元测试的虚构背景。",
            work_years=8.5,
            updated_at=source_time,
            created_at=source_time,
            deleted_at=None,
        ),
        employments=(),
        projects=(),
        educations=(),
    )
    settings = _settings()
    profile, employments, projects = normalize_aggregate(
        aggregate,
        data_version="20260818_01",
        build_batch_id=701,
    )
    chunk = ChunkFactory(settings).build_all(profile, employments, projects)[0]

    contract = chunk_record_to_contract(
        chunk,
        updated_at=datetime(2026, 8, 18, 10, 0),  # noqa: DTZ001 - 验证 Bridge 补时区
        timezone_name=settings.timezone,
    )

    assert isinstance(contract, ExpertChunkEmbeddingInput)
    assert contract.contract_version == "4.1.0"
    assert contract.vector_doc_id == chunk.vector_doc_id
    assert contract.chunk_text == chunk.chunk_text
    assert contract.content_hash == chunk.content_hash
    assert contract.metadata_hash == chunk.metadata_hash
    assert contract.metadata.work_years == Decimal("8.5")
    assert not isinstance(contract.metadata.work_years, float)
    assert contract.source.source_updated_at == "2026-08-18T09:30:00.000000+08:00"
    assert contract.updated_at == "2026-08-18T10:00:00.000000+08:00"
    assert contract.metadata.industry_codes == list(chunk.industry_codes)
    assert "build_batch_id" not in contract.model_dump()
    assert "embedding_model_alias" not in contract.model_dump()


def _three_type_chunks() -> list[ChunkRecord]:
    """通过正式主链路生成三类字段完整、时间边界不同的 Chunk。"""

    profile_time = datetime(2026, 5, 2, 3, 4, 5, 123456)  # noqa: DTZ001
    education_time = datetime(2026, 5, 3, 4, 5, 6, 234567)  # noqa: DTZ001
    employment_time = datetime(2026, 5, 4, 5, 6, 7, 345678)  # noqa: DTZ001
    aggregate = ExpertAggregate(
        profile=ExpertSource(
            id=710001,
            serial_number="CONTRACT-710001",
            company="北京京东世纪贸易有限公司",
            job_status=2,
            job_name="项目经理",
            contact_status=4,
            doubtful_flag=1,
            abroad_flag=1,
            background="负责严格 Contract 逐字段交付验证。",
            work_years=12.5,
            updated_at=profile_time,
            created_at=profile_time,
            deleted_at=None,
        ),
        employments=(
            EmploymentSource(
                id=711001,
                expert_id=710001,
                company_name="虚构任职公司",
                position_title="研发工程师",
                start_date=datetime(2021, 2, 1),  # noqa: DTZ001
                end_date=datetime(2025, 3, 1),  # noqa: DTZ001
                work_description=(
                    "工作职责：负责 Contract 映射。\n"
                    "代表性成果：全部字段验证通过。"
                ),
                is_employed=2,
                updated_at=employment_time,
                created_at=datetime(2021, 2, 1),  # noqa: DTZ001
                deleted_at=None,
            ),
        ),
        projects=(
            ProjectSource(
                id=712001,
                expert_id=710001,
                project_title="Contract 严格交付项目",
                company_name=None,
                start_date=None,
                end_date=datetime(2025, 12, 1),  # noqa: DTZ001
                work_description=(
                    "项目目标：验证分类 Metadata。\n"
                    "关键动作：逐字段比较输入输出。\n"
                    "项目结果：映射保持一致。"
                ),
                updated_at=None,
                created_at=None,
                deleted_at=None,
            ),
        ),
        educations=(
            EducationSource(
                id=713001,
                expert_id=710001,
                school_name="虚构契约大学",
                major="信息系统",
                updated_at=education_time,
                created_at=education_time,
                deleted_at=None,
            ),
        ),
    )
    profile, employments, projects = normalize_aggregate(
        aggregate,
        data_version="contract-delivery-excluded",
        build_batch_id=710,
    )
    chunks = ChunkFactory(_settings()).build_all(profile, employments, projects)
    validate_chunks(chunks)
    assert [item.chunk_type for item in chunks] == [
        "profile_summary",
        "employment",
        "project",
    ]
    return chunks


def _expected_metadata(chunk: ChunkRecord) -> dict[str, object]:
    """按冻结字段契约返回当前类型应交付的 Metadata。"""

    expected: dict[str, object] = {
        "company_name_raw": chunk.company_name_raw,
        "company_name_norm": chunk.company_name_norm,
        "industry_codes": list(chunk.industry_codes),
        "topic_codes": list(chunk.topic_codes),
        "origin_type": chunk.origin_type,
        "verification_status": chunk.verification_status,
        "weight_tier": chunk.weight_tier,
    }
    if chunk.chunk_type == "profile_summary":
        expected.update(
            {
                "role_name_raw": chunk.role_name_raw,
                "role_code": chunk.role_code,
                "role_level": chunk.role_level,
                "job_status_norm": chunk.job_status_norm,
                "abroad_flag_norm": chunk.abroad_flag_norm,
                "work_years": Decimal(str(chunk.work_years)),
                "function_codes": list(chunk.function_codes),
                "school_names_raw": list(chunk.school_names_raw),
                "major_names_raw": list(chunk.major_names_raw),
                "eligible_for_recommendation": chunk.eligible_for_recommendation,
            }
        )
    elif chunk.chunk_type == "employment":
        expected.update(
            {
                "role_name_raw": chunk.role_name_raw,
                "role_code": chunk.role_code,
                "role_level": chunk.role_level,
                "start_month": chunk.start_month,
                "end_month": chunk.end_month,
                "is_current": chunk.is_current,
            }
        )
    else:
        expected.update(
            {
                "project_name": chunk.project_name,
                "start_month": chunk.start_month,
                "end_month": chunk.end_month,
            }
        )
    return expected


@pytest.mark.parametrize(
    ("chunk_type", "metadata_type", "expected_source_updated_at"),
    [
        (
            "profile_summary",
            ProfileSummaryMetadata,
            "2026-05-03T04:05:06.234567+08:00",
        ),
        (
            "employment",
            EmploymentMetadata,
            "2026-05-04T05:06:07.345678+08:00",
        ),
        ("project", ProjectMetadata, None),
    ],
)
def test_each_chunk_type_maps_every_contract_field(
    chunk_type: str,
    metadata_type: type[object],
    expected_source_updated_at: str | None,
) -> None:
    """三类 Chunk 的顶层、Metadata、Source 和 Versions 必须逐字段无损交付。"""

    chunk = next(item for item in _three_type_chunks() if item.chunk_type == chunk_type)
    contract = chunk_record_to_contract(
        chunk,
        updated_at=datetime(2026, 5, 5, 6, 7, 8, 456789),  # noqa: DTZ001
        timezone_name=_settings().timezone,
    )

    assert isinstance(contract, ExpertChunkEmbeddingInput)
    assert isinstance(contract.metadata, metadata_type)
    assert contract.contract_version == "4.1.0"
    assert contract.vector_doc_id == chunk.vector_doc_id
    assert contract.expert_id == chunk.expert_id
    assert contract.serial_number == chunk.serial_number
    assert contract.chunk_type == chunk.chunk_type
    assert contract.chunk_subtype == chunk.chunk_subtype
    assert contract.chunk_index == chunk.chunk_index
    assert contract.chunk_total == chunk.chunk_total
    assert contract.chunk_text == chunk.chunk_text
    assert contract.char_count == chunk.char_count
    assert contract.content_hash == chunk.content_hash
    assert contract.metadata_hash == chunk.metadata_hash
    assert contract.is_active is chunk.is_active
    assert contract.updated_at == "2026-05-05T06:07:08.456789+08:00"

    assert contract.metadata.model_dump() == _expected_metadata(chunk)
    assert contract.source.model_dump() == {
        "source_table": chunk.source_table,
        "source_field": chunk.source_field,
        "source_field_id": chunk.source_field_id,
        "source_version": chunk.source_version,
        "source_updated_at": expected_source_updated_at,
        "source_hash": chunk.source_hash,
    }
    assert contract.versions.model_dump() == {
        "chunk_policy_version": chunk.chunk_policy_version,
        "metadata_schema_version": chunk.metadata_schema_version,
    }
    assert contract.vector_doc_id == (
        f"expert:{chunk.expert_id}:{chunk.chunk_type}:"
        f"{chunk.source_field_id}:{chunk.chunk_index}"
    )

    dumped = contract.model_dump()
    assert "build_batch_id" not in dumped
    assert "embedding_model_alias" not in dumped
    assert "embedding_model_version" not in dumped
    assert "indexed_at" not in dumped


def test_memory_batch_preserves_order_hashes_and_delivery_time() -> None:
    """内存批次转换必须保持输入顺序，并为每条记录使用同一交付时间。"""

    chunks = _three_type_chunks()
    contracts = chunk_records_to_contracts(
        chunks,
        updated_at=datetime(2026, 5, 6, 7, 8, 9, 567890),  # noqa: DTZ001
        timezone_name=_settings().timezone,
    )

    assert [item.vector_doc_id for item in contracts] == [item.vector_doc_id for item in chunks]
    assert [item.source.source_hash for item in contracts] == [item.source_hash for item in chunks]
    assert [item.content_hash for item in contracts] == [item.content_hash for item in chunks]
    assert [item.metadata_hash for item in contracts] == [item.metadata_hash for item in chunks]
    assert {item.updated_at for item in contracts} == {
        "2026-05-06T07:08:09.567890+08:00"
    }


def test_empty_memory_batch_returns_empty_contract_list() -> None:
    """空输入是合法内存批次，不应补造 Contract。"""

    assert chunk_records_to_contracts(
        [],
        updated_at=datetime(2026, 5, 6, 7, 8, 9),  # noqa: DTZ001
        timezone_name=_settings().timezone,
    ) == []


def test_memory_batch_rejects_cross_expert_input() -> None:
    """声明为单专家快照的批次不得混入另一个 expert_id。"""

    chunks = _three_type_chunks()
    original = chunks[1]
    other_without_hash = replace(
        original,
        expert_id=710002,
        vector_doc_id=(
            f"expert:710002:{original.chunk_type}:"
            f"{original.source_field_id}:{original.chunk_index}"
        ),
        metadata_hash="",
    )
    other = replace(
        other_without_hash,
        metadata_hash=calculate_metadata_hash(other_without_hash),
    )
    validate_chunks([other])

    with pytest.raises(ChunkValidationError, match="同一专家"):
        chunk_records_to_contracts(
            [chunks[0], other],
            updated_at=datetime(2026, 5, 6, 7, 8, 9),  # noqa: DTZ001
            timezone_name=_settings().timezone,
        )


def test_memory_batch_rejects_duplicate_vector_doc_id() -> None:
    """内存批次必须在转换前拒绝重复 vector_doc_id。"""

    chunk = _three_type_chunks()[0]

    with pytest.raises(ChunkValidationError, match="重复 vector_doc_id"):
        chunk_records_to_contracts(
            [chunk, chunk],
            updated_at=datetime(2026, 5, 6, 7, 8, 9),  # noqa: DTZ001
            timezone_name=_settings().timezone,
        )


@pytest.mark.parametrize(
    "hash_path",
    ["source_hash", "content_hash", "metadata_hash"],
)
def test_formal_contract_rejects_invalid_hash(hash_path: str) -> None:
    """正式 Contract 必须拒绝任一不符合小写 SHA-256 的 Hash。"""

    chunk = _three_type_chunks()[0]
    contract = chunk_record_to_contract(
        chunk,
        updated_at=datetime(2026, 5, 6, 7, 8, 9),  # noqa: DTZ001
        timezone_name=_settings().timezone,
    )
    payload = contract.model_dump()
    if hash_path == "source_hash":
        payload["source"]["source_hash"] = "INVALID"
    else:
        payload[hash_path] = "INVALID"

    with pytest.raises(ValueError, match=hash_path):
        ExpertChunkEmbeddingInput.model_validate(payload)
