"""Source、Stage 与最终 Chunk 的显式领域模型。

使用 dataclass 而不是在各模块间传递任意 dict，可以让字段来源一眼可见，并减少
公司、角色或 source_id 串段的风险。模型刻意不包含姓名、联系方式及报价字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ExpertSource:
    """从 experts 白名单读取的主档；不读取旧的经历长文本字段。"""

    expert_id: int
    serial_number: str | None
    company: str | None
    role: str | None
    background: str | None
    education: str | None
    work_years: float | None
    updated_at: datetime | None
    created_at: datetime | None


@dataclass(frozen=True)
class EmploymentSource:
    """同一段任职的来源对象；所有业务字段必须来自这一个对象。"""

    expert_id: int
    source_table: str
    source_record_id: str
    source_item_key: str
    company: str | None
    role: str | None
    start_date: datetime | None
    end_date: datetime | None
    description: str | None
    responsibilities: str | None
    achievements: str | None
    is_current: bool | None
    updated_at: datetime | None
    created_at: datetime | None
    parse_status: str = "complete"


@dataclass(frozen=True)
class ProjectSource:
    """同一个项目的来源对象，不把其他项目或主档事实混入。"""

    expert_id: int
    source_table: str
    source_record_id: str
    source_item_key: str
    project_name: str | None
    company: str | None
    project_role: str | None
    start_date: datetime | None
    end_date: datetime | None
    description: str | None
    responsibilities: str | None
    results: str | None
    updated_at: datetime | None
    created_at: datetime | None
    parse_status: str = "complete"


@dataclass(frozen=True)
class ExpertAggregate:
    """重建一位专家所需的完整当前快照。"""

    profile: ExpertSource
    employments: tuple[EmploymentSource, ...]
    projects: tuple[ProjectSource, ...]


@dataclass(frozen=True)
class ProfileStage:
    """已清洗、可落中间表的专家级允许字段。"""

    expert_id: int
    serial_number: str | None
    company_name_raw: str | None
    company_name_norm: str | None
    company_normalization_status: str
    role_name_raw: str | None
    role_code: str | None
    role_level: str | None
    background_clean: str | None
    school_names: tuple[str, ...]
    work_years: float | None
    source_hash: str
    source_updated_at: datetime | None
    data_version: str
    build_batch_id: int


@dataclass(frozen=True)
class EmploymentStage:
    """一行严格对应 experts_work_experience 中的一段任职。"""

    expert_id: int
    source_table: str
    source_record_id: str
    source_item_key: str
    company_name_raw: str | None
    company_name_norm: str | None
    company_normalization_status: str
    role_name_raw: str | None
    role_code: str | None
    role_level: str | None
    start_month: str | None
    end_month: str | None
    is_current: bool | None
    responsibilities_clean: str | None
    achievements_clean: str | None
    industry_codes: tuple[str, ...]
    topic_codes: tuple[str, ...]
    parse_status: str
    source_hash: str
    source_updated_at: datetime | None
    data_version: str
    build_batch_id: int


@dataclass(frozen=True)
class ProjectStage:
    """一个项目的标准化中间记录；未知结果保持 None，不做推断。"""

    expert_id: int
    source_table: str
    source_record_id: str
    source_item_key: str
    project_name: str | None
    company_or_client_raw: str | None
    company_or_client_norm: str | None
    company_normalization_status: str
    project_role: str | None
    start_month: str | None
    end_month: str | None
    objective_clean: str | None
    actions_clean: str | None
    results_clean: str | None
    industry_codes: tuple[str, ...]
    topic_codes: tuple[str, ...]
    parse_status: str
    source_hash: str
    source_updated_at: datetime | None
    data_version: str
    build_batch_id: int


@dataclass(frozen=True)
class ChunkRecord:
    """与 expert_chunk 表一一对应的最终记录，不包含 embedding。"""

    chunk_id: str
    expert_id: int
    serial_number: str | None
    chunk_type: str
    chunk_subtype: str | None
    chunk_index: int
    chunk_total: int
    chunk_text: str
    char_count: int
    metadata: dict[str, Any]
    source_table: str
    source_record_id: str
    source_item_key: str
    source_field: str
    source_field_id: str
    source_version: str
    source_updated_at: datetime | None
    source_hash: str
    content_hash: str
    company_name_norm: str | None
    role_code: str | None
    start_month: str | None
    end_month: str | None
    is_current: bool | None
    industry_codes: tuple[str, ...]
    topic_codes: tuple[str, ...]
    origin_type: str
    verification_status: str
    weight_tier: str
    chunk_policy_version: str
    metadata_schema_version: str
    dictionary_version: str
    data_version: str
    build_batch_id: int
