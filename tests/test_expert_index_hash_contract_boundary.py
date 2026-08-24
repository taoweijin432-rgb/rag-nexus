"""验证三 Hash 的字段参与、排除和相互分离边界。"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime

import pytest

from expert_index.chunk_builder import (
    ChunkFactory,
    calculate_metadata_hash,
    metadata_hash_payload,
)
from expert_index.config import DatabaseConfig, Settings
from expert_index.contract_bridge import chunk_record_to_contract
from expert_index.models import (
    ChunkRecord,
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
)
from expert_index.normalize import normalize_aggregate, sha256_value
from expert_index.validators import ChunkValidationError, validate_chunks

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _settings() -> Settings:
    """构造不连接数据库或下游服务的固定配置。"""

    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _aggregate() -> ExpertAggregate:
    """返回可对单字段执行成对变异的固定四表快照。"""

    source_time = datetime(2026, 4, 5, 6, 7, 8, 123456)  # noqa: DTZ001
    return ExpertAggregate(
        profile=ExpertSource(
            id=990001,
            serial_number="HASH-BOUNDARY-001",
            company="虚构边界科技有限公司",
            job_status=1,
            job_name="项目经理",
            contact_status=3,
            doubtful_flag=1,
            abroad_flag=2,
            background="负责知识库能力建设。",
            work_years=9.0,
            updated_at=source_time,
            created_at=datetime(2020, 1, 1),  # noqa: DTZ001
            deleted_at=None,
        ),
        employments=(
            EmploymentSource(
                id=991001,
                expert_id=990001,
                company_name="虚构边界科技有限公司",
                position_title="项目经理",
                start_date=datetime(2021, 1, 1),  # noqa: DTZ001
                end_date=datetime(2025, 1, 1),  # noqa: DTZ001
                work_description=(
                    "工作职责：负责检索系统交付。\n"
                    "代表性成果：召回率提升10%。"
                ),
                is_employed=2,
                updated_at=source_time,
                created_at=datetime(2021, 1, 1),  # noqa: DTZ001
                deleted_at=None,
            ),
        ),
        projects=(
            ProjectSource(
                id=992001,
                expert_id=990001,
                project_title="边界验证项目",
                company_name="虚构客户公司",
                start_date=datetime(2024, 1, 1),  # noqa: DTZ001
                end_date=datetime(2024, 12, 1),  # noqa: DTZ001
                work_description=(
                    "项目角色：技术负责人\n"
                    "项目目标：验证哈希边界。\n"
                    "关键动作：执行固定输入测试。\n"
                    "项目结果：全部断言通过。"
                ),
                updated_at=source_time,
                created_at=datetime(2024, 1, 1),  # noqa: DTZ001
                deleted_at=None,
            ),
        ),
        educations=(
            EducationSource(
                id=993001,
                expert_id=990001,
                school_name="虚构测试大学",
                major="软件工程",
                updated_at=source_time,
                created_at=datetime(2016, 9, 1),  # noqa: DTZ001
                deleted_at=None,
            ),
        ),
    )


def _normalize(
    aggregate: ExpertAggregate,
    *,
    data_version: str = "hash-boundary-v1",
    build_batch_id: int = 1,
) -> tuple[object, list[object], list[object]]:
    """通过正式 Normalize 入口返回三类 Stage。"""

    return normalize_aggregate(
        aggregate,
        data_version=data_version,
        build_batch_id=build_batch_id,
    )


def _build(
    aggregate: ExpertAggregate,
    *,
    data_version: str = "hash-boundary-v1",
    build_batch_id: int = 1,
) -> list[ChunkRecord]:
    """通过正式上游主链路生成并校验 Chunk。"""

    profile, employments, projects = _normalize(
        aggregate,
        data_version=data_version,
        build_batch_id=build_batch_id,
    )
    chunks = ChunkFactory(_settings()).build_all(profile, employments, projects)
    validate_chunks(chunks)
    return chunks


def _chunk(chunks: list[ChunkRecord], chunk_type: str) -> ChunkRecord:
    """返回指定类型的唯一固定测试 Chunk。"""

    matches = [item for item in chunks if item.chunk_type == chunk_type]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("id", 990002),
        ("serial_number", "HASH-BOUNDARY-002"),
        ("company", "另一家虚构公司"),
        ("job_status", 2),
        ("job_name", "研发工程师"),
        ("contact_status", 1),
        ("doubtful_flag", 2),
        ("abroad_flag", 1),
        ("background", "负责另一类知识库能力建设。"),
        ("work_years", 10.0),
    ],
)
def test_profile_source_hash_includes_each_frozen_profile_field(
    field_name: str,
    changed_value: object,
) -> None:
    """Profile allowlist 任一标准化值变化都必须改变 source_hash。"""

    aggregate = _aggregate()
    baseline, _, _ = _normalize(replace(aggregate, employments=(), projects=(), educations=()))
    changed_profile = replace(aggregate.profile, **{field_name: changed_value})
    changed, _, _ = _normalize(
        ExpertAggregate(changed_profile, (), (), ()),
    )

    assert baseline.source_hash != changed.source_hash


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("company_name", "另一家虚构任职公司"),
        ("position_title", "研发工程师"),
        ("start_date", datetime(2022, 2, 1)),  # noqa: DTZ001
        ("end_date", datetime(2025, 2, 1)),  # noqa: DTZ001
        ("work_description", "工作职责：负责另一项工作。\n代表性成果：效率提升20%。"),
        ("is_employed", 1),
    ],
)
def test_employment_source_hash_includes_each_frozen_business_field(
    field_name: str,
    changed_value: object,
) -> None:
    """Employment allowlist 任一标准化值变化都必须改变 source_hash。"""

    aggregate = _aggregate()
    _, baseline, _ = _normalize(aggregate)
    changed_item = replace(aggregate.employments[0], **{field_name: changed_value})
    _, changed, _ = _normalize(replace(aggregate, employments=(changed_item,)))

    assert baseline[0].source_hash != changed[0].source_hash


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("project_title", "另一项边界验证项目"),
        ("company_name", "另一家虚构客户"),
        ("start_date", datetime(2024, 2, 1)),  # noqa: DTZ001
        ("end_date", datetime(2025, 1, 1)),  # noqa: DTZ001
        (
            "work_description",
            "项目目标：验证另一边界。\n关键动作：执行另一测试。\n项目结果：结果通过。",
        ),
    ],
)
def test_project_source_hash_includes_each_frozen_business_field(
    field_name: str,
    changed_value: object,
) -> None:
    """Project allowlist 任一标准化值变化都必须改变 source_hash。"""

    aggregate = _aggregate()
    _, _, baseline = _normalize(aggregate)
    changed_item = replace(aggregate.projects[0], **{field_name: changed_value})
    _, _, changed = _normalize(replace(aggregate, projects=(changed_item,)))

    assert baseline[0].source_hash != changed[0].source_hash


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("id", 993002),
        ("school_name", "另一所虚构测试大学"),
        ("major", "人工智能"),
    ],
)
def test_profile_source_hash_includes_each_frozen_education_field(
    field_name: str,
    changed_value: object,
) -> None:
    """有效教育记录的 ID、学校和专业均参与 Profile source_hash。"""

    aggregate = _aggregate()
    baseline, _, _ = _normalize(aggregate)
    changed_education = replace(aggregate.educations[0], **{field_name: changed_value})
    changed, _, _ = _normalize(replace(aggregate, educations=(changed_education,)))

    assert baseline.source_hash != changed.source_hash


def test_source_time_changes_metadata_hash_without_changing_text_or_source_hash() -> None:
    """来源版本时间是 Metadata 输入，但不是 source_hash 或正文输入。"""

    aggregate = _aggregate()
    baseline = _chunk(_build(aggregate), "employment")
    changed_item = replace(
        aggregate.employments[0],
        updated_at=datetime(2026, 4, 6, 7, 8, 9, 654321),  # noqa: DTZ001
    )
    changed = _chunk(_build(replace(aggregate, employments=(changed_item,))), "employment")

    assert changed.vector_doc_id == baseline.vector_doc_id
    assert changed.source_hash == baseline.source_hash
    assert changed.chunk_text == baseline.chunk_text
    assert changed.content_hash == baseline.content_hash
    assert changed.metadata_hash != baseline.metadata_hash


def test_content_only_change_does_not_change_source_or_metadata_hash() -> None:
    """仅修改最终正文时只有 content_hash 发生变化。"""

    baseline = _chunk(_build(_aggregate()), "employment")
    changed_text = f"{baseline.chunk_text}\n补充固定正文。"
    changed = replace(
        baseline,
        chunk_text=changed_text,
        char_count=len(changed_text),
        content_hash=sha256_value(changed_text),
    )
    validate_chunks([changed])

    assert changed.source_hash == baseline.source_hash
    assert changed.content_hash != baseline.content_hash
    assert changed.metadata_hash == baseline.metadata_hash


def test_metadata_only_change_does_not_change_source_or_content_hash() -> None:
    """仅修改正式 Metadata 时只重新计算 metadata_hash。"""

    baseline = _chunk(_build(_aggregate()), "employment")
    changed_without_hash = replace(baseline, is_active=False, metadata_hash="")
    changed = replace(
        changed_without_hash,
        metadata_hash=calculate_metadata_hash(changed_without_hash),
    )
    validate_chunks([changed])

    assert changed.source_hash == baseline.source_hash
    assert changed.chunk_text == baseline.chunk_text
    assert changed.content_hash == baseline.content_hash
    assert changed.metadata_hash != baseline.metadata_hash


def test_metadata_hash_payload_keys_match_each_contract_type() -> None:
    """三类 payload 必须严格等于冻结的公共字段与类型专属字段并集。"""

    common_fields = {
        "vector_doc_id",
        "expert_id",
        "serial_number",
        "chunk_type",
        "chunk_subtype",
        "chunk_index",
        "chunk_total",
        "source_table",
        "source_field",
        "source_field_id",
        "source_version",
        "source_updated_at",
        "source_hash",
        "company_name_raw",
        "company_name_norm",
        "industry_codes",
        "topic_codes",
        "origin_type",
        "verification_status",
        "weight_tier",
        "chunk_policy_version",
        "metadata_schema_version",
        "is_active",
    }
    type_fields = {
        "profile_summary": {
            "role_name_raw",
            "role_code",
            "role_level",
            "job_status_norm",
            "abroad_flag_norm",
            "work_years",
            "function_codes",
            "school_names_raw",
            "major_names_raw",
            "eligible_for_recommendation",
        },
        "employment": {
            "role_name_raw",
            "role_code",
            "role_level",
            "start_month",
            "end_month",
            "is_current",
        },
        "project": {
            "project_name",
            "start_month",
            "end_month",
        },
    }

    for chunk in _build(_aggregate()):
        assert set(metadata_hash_payload(chunk)) == common_fields | type_fields[chunk.chunk_type]


def test_downstream_runtime_fields_do_not_change_metadata_hash() -> None:
    """Embedding、索引时间和构建批次属于 metadata_hash 排除项。"""

    baseline = _chunk(_build(_aggregate()), "employment")
    changed = replace(
        baseline,
        embedding_model_alias="excluded-model",
        embedding_model_version="excluded-version",
        indexed_at=datetime(2026, 4, 9, 10, 11, 12),  # noqa: DTZ001
        build_batch_id=999,
    )
    validate_chunks([changed])

    assert calculate_metadata_hash(changed) == baseline.metadata_hash


def test_excluded_runtime_fields_do_not_change_any_hash() -> None:
    """批次、数据版本和非回退 created_at 不得污染上游输出 Hash。"""

    aggregate = _aggregate()
    baseline = _build(aggregate, data_version="run-a", build_batch_id=11)
    changed_profile = replace(
        aggregate.profile,
        created_at=datetime(2022, 2, 2),  # noqa: DTZ001
    )
    changed = _build(
        replace(aggregate, profile=changed_profile),
        data_version="run-b",
        build_batch_id=22,
    )

    assert [item.vector_doc_id for item in changed] == [item.vector_doc_id for item in baseline]
    assert [item.chunk_text for item in changed] == [item.chunk_text for item in baseline]
    assert [item.source_hash for item in changed] == [item.source_hash for item in baseline]
    assert [item.content_hash for item in changed] == [item.content_hash for item in baseline]
    assert [item.metadata_hash for item in changed] == [item.metadata_hash for item in baseline]


def test_removed_pii_does_not_change_profile_hashes_or_contract() -> None:
    """被清洗掉的 PII 变化不得进入 Source、Chunk、Contract 或任一 Hash。"""

    aggregate = _aggregate()
    baseline = _chunk(_build(aggregate), "profile_summary")
    changed_profile = replace(
        aggregate.profile,
        background=f"{aggregate.profile.background} golden@example.com",
    )
    changed = _chunk(_build(replace(aggregate, profile=changed_profile)), "profile_summary")

    assert changed == baseline
    contract = chunk_record_to_contract(
        changed,
        updated_at=datetime(2026, 4, 7, 8, 9, 10),  # noqa: DTZ001
        timezone_name=_settings().timezone,
    )
    assert "golden@example.com" not in contract.model_dump_json()


def test_discarded_project_role_does_not_change_any_hash() -> None:
    """不进入当前正文和 Metadata 的项目角色变化必须保持三个 Hash 不变。"""

    aggregate = _aggregate()
    baseline = _chunk(_build(aggregate), "project")
    changed_description = aggregate.projects[0].work_description.replace(
        "项目角色：技术负责人",
        "项目角色：观察员",
    )
    changed_project = replace(
        aggregate.projects[0],
        work_description=changed_description,
    )
    changed = _chunk(_build(replace(aggregate, projects=(changed_project,))), "project")

    assert changed == baseline


def test_contract_delivery_time_does_not_rewrite_hashes() -> None:
    """不同交付时间只能改变 Contract updated_at，不能改写上游三个 Hash。"""

    chunk = _chunk(_build(_aggregate()), "profile_summary")
    first = chunk_record_to_contract(
        chunk,
        updated_at=datetime(2026, 4, 7, 8, 9, 10),  # noqa: DTZ001
        timezone_name=_settings().timezone,
    )
    second = chunk_record_to_contract(
        chunk,
        updated_at=datetime(2026, 4, 8, 9, 10, 11),  # noqa: DTZ001
        timezone_name=_settings().timezone,
    )

    assert first.updated_at != second.updated_at
    assert first.source.source_hash == second.source.source_hash == chunk.source_hash
    assert first.content_hash == second.content_hash == chunk.content_hash
    assert first.metadata_hash == second.metadata_hash == chunk.metadata_hash


def test_all_hashes_are_lowercase_sha256() -> None:
    """所有生成结果必须提供 64 位小写 SHA-256。"""

    for chunk in _build(_aggregate()):
        assert _SHA256.fullmatch(chunk.source_hash)
        assert _SHA256.fullmatch(chunk.content_hash)
        assert _SHA256.fullmatch(chunk.metadata_hash)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_message"),
    [
        ("source_hash", "A" * 64, "source_hash 格式错误"),
        ("content_hash", "0" * 64, "content_hash 不一致"),
        ("metadata_hash", "not-a-sha256", "metadata_hash 格式错误"),
    ],
)
def test_upstream_validator_rejects_invalid_hashes(
    field_name: str,
    invalid_value: str,
    expected_message: str,
) -> None:
    """上游 Validator 必须拒绝格式错误或与实际内容不一致的 Hash。"""

    chunk = _chunk(_build(_aggregate()), "profile_summary")
    invalid = replace(chunk, **{field_name: invalid_value})

    with pytest.raises(ChunkValidationError, match=expected_message):
        validate_chunks([invalid])


def test_upstream_validator_rejects_vector_doc_id_formula_mismatch() -> None:
    """上游 Validator 必须拒绝不符合统一公式的 vector_doc_id。"""

    chunk = _chunk(_build(_aggregate()), "profile_summary")
    invalid = replace(chunk, vector_doc_id="expert:990001:profile_summary:wrong:0")

    with pytest.raises(ChunkValidationError, match="vector_doc_id 不符合统一生成规则"):
        validate_chunks([invalid])
