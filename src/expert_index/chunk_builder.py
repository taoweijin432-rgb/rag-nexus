"""三类标准 Chunk 的确定性构建器。

构建器只消费中间模型，不直接读取数据库。超长任职和项目在同一个来源记录内
切分，并重复相同的公司/岗位/项目头；不同来源记录永不合并。profile_summary
按契约最多一条，使用“抽取式压缩”选择能完整放入上限的前若干句，不让模型改写。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Settings
from .models import ChunkRecord, EmploymentStage, ProfileStage, ProjectStage
from .normalize import sha256_value

HARD_LIMITS = {"profile_summary": 600, "employment": 900, "project": 1000}


class ChunkFactory:
    """把一位专家的三类 Stage 对象转换为完整 Chunk 集合。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_all(
        self,
        profile: ProfileStage,
        employments: Iterable[EmploymentStage],
        projects: Iterable[ProjectStage],
    ) -> list[ChunkRecord]:
        """按 profile、employment、project 的稳定顺序生成 Chunk。"""

        chunks: list[ChunkRecord] = []
        chunks.extend(self.build_profile(profile))
        for employment in employments:
            chunks.extend(self.build_employment(profile.serial_number, employment))
        for project in projects:
            chunks.extend(self.build_project(profile.serial_number, project))
        return chunks

    def build_profile(self, item: ProfileStage) -> list[ChunkRecord]:
        """每位专家最多构建一条导航画像；完全空白时不生成。"""

        if not any((item.company_name_norm, item.role_name_raw, item.background_clean)):
            return []
        lines = ["[专家能力概述]"]
        if item.company_name_norm or item.role_name_raw:
            current = " · ".join(part for part in (item.company_name_norm, item.role_name_raw) if part)
            lines.append(f"当前/最近任职：{current}")
        prefix = "\n".join(lines)
        if item.background_clean:
            budget = HARD_LIMITS["profile_summary"] - len(prefix) - len("\n核心背景：")
            # 抽取完整句而非在字符中间硬截断；若首句自身过长，再按标点短语切分。
            background = _extractive_fit(item.background_clean, max(budget, 1))
            if background:
                lines.append(f"核心背景：{background}")
        text = "\n".join(lines)
        metadata = _metadata(
            company_raw=item.company_name_raw,
            company_norm=item.company_name_norm,
            company_status=item.company_normalization_status,
            role_raw=item.role_name_raw,
            role_code=item.role_code,
            role_level=item.role_level,
            industry_codes=(),
            topic_codes=(),
            origin_type="source_template",
            verification_status="source_evidence",
            weight_tier="B",
        )
        return [
            self._record(
                expert_id=item.expert_id,
                serial_number=item.serial_number,
                chunk_type="profile_summary",
                source_item_key=str(item.expert_id),
                text=text,
                metadata=metadata,
                source_table="experts",
                source_record_id=str(item.expert_id),
                source_field="background+company+job_name",
                source_field_id="experts.background",
                source_updated_at=item.source_updated_at,
                source_hash=item.source_hash,
                company_name_norm=item.company_name_norm,
                role_code=item.role_code,
                start_month=None,
                end_month=None,
                is_current=None,
                industry_codes=(),
                topic_codes=(),
                origin_type="source_template",
                verification_status="source_evidence",
                weight_tier="B",
                data_version=item.data_version,
                build_batch_id=item.build_batch_id,
            )
        ]

    def build_employment(self, serial_number: str | None, item: EmploymentStage) -> list[ChunkRecord]:
        """生成同一任职段的一个或多个 Chunk，不跨 source_item_key。"""

        # 公司和岗位至少要有一个，且职责/成果可以为空；绝不补造不存在的职责。
        if not any((item.company_name_raw, item.company_name_norm, item.role_name_raw)):
            return []
        header = ["[任职经历]"]
        _append(header, "公司", item.company_name_norm or item.company_name_raw)
        _append(header, "岗位", item.role_name_raw)
        if item.start_month:
            _append(header, "时间", f"{item.start_month}—{item.end_month or ('至今' if item.is_current else '未知')}")
        if item.is_current is not None:
            _append(header, "任职状态", "当前任职" if item.is_current else "历史任职")
        _append(header, "业务领域", "、".join(item.industry_codes) or None)
        sections = [
            ("工作职责", item.responsibilities_clean, "responsibility"),
            ("代表性成果", item.achievements_clean, "achievement"),
        ]
        pieces = _render_pieces(header, sections, HARD_LIMITS["employment"])
        # 即使源行能解析，也只有公司、岗位、开始月份和在职状态完整且日期不倒置时，
        # 才能成为 A 级硬条件证据；is_employed=0 等未知枚举安全降级为 partial。
        dates_valid = not (item.start_month and item.end_month and item.start_month > item.end_month)
        complete = (
            item.parse_status == "complete"
            and bool(item.company_name_raw or item.company_name_norm)
            and bool(item.role_name_raw)
            and bool(item.start_month)
            and item.is_current is not None
            and dates_valid
        )
        verification = "source_evidence" if complete else "partial"
        metadata = _metadata(
            company_raw=item.company_name_raw,
            company_norm=item.company_name_norm,
            company_status=item.company_normalization_status,
            role_raw=item.role_name_raw,
            role_code=item.role_code,
            role_level=item.role_level,
            start_month=item.start_month,
            end_month=item.end_month,
            is_current=item.is_current,
            industry_codes=item.industry_codes,
            topic_codes=item.topic_codes,
            origin_type="source_parsed",
            verification_status=verification,
            weight_tier="A",
        )
        return [
            self._record(
                expert_id=item.expert_id,
                serial_number=serial_number,
                chunk_type="employment",
                source_item_key=item.source_item_key,
                text=text,
                metadata=metadata,
                source_table=item.source_table,
                source_record_id=item.source_record_id,
                source_field="work_experience",
                source_field_id=f"{item.source_table}:{item.source_record_id}",
                source_updated_at=item.source_updated_at,
                source_hash=item.source_hash,
                company_name_norm=item.company_name_norm,
                role_code=item.role_code,
                start_month=item.start_month,
                end_month=item.end_month,
                is_current=item.is_current,
                industry_codes=item.industry_codes,
                topic_codes=item.topic_codes,
                origin_type="source_parsed",
                verification_status=verification,
                weight_tier="A",
                data_version=item.data_version,
                build_batch_id=item.build_batch_id,
                chunk_subtype=subtype,
                chunk_index=index,
                chunk_total=len(pieces),
            )
            for index, (subtype, text) in enumerate(pieces)
        ]

    def build_project(self, serial_number: str | None, item: ProjectStage) -> list[ChunkRecord]:
        """生成单项目 Chunk；项目名或角色缺失时保留 partial 但不捏造占位值。"""

        if not any((item.project_name, item.objective_clean, item.actions_clean)):
            return []
        header = ["[项目经历]"]
        _append(header, "项目名称", item.project_name)
        _append(header, "所属公司/客户", item.company_or_client_norm or item.company_or_client_raw)
        if item.start_month or item.end_month:
            _append(header, "时间", f"{item.start_month or '未知'}—{item.end_month or '未知'}")
        _append(header, "专家角色", item.project_role)
        _append(header, "业务领域", "、".join(item.industry_codes) or None)
        sections = [
            # 正确源表没有目标/结果列，只有字段注释明确的“工作内容”。
            ("项目描述", item.objective_clean, "context"),
            ("项目工作内容", item.actions_clean, "action"),
            ("项目结果", item.results_clean, "result"),
        ]
        pieces = _render_pieces(header, sections, HARD_LIMITS["project"])
        complete = item.parse_status == "complete" and bool(item.project_name and item.project_role)
        verification = "source_evidence" if complete else "partial"
        metadata = _metadata(
            company_raw=item.company_or_client_raw,
            company_norm=item.company_or_client_norm,
            company_status=item.company_normalization_status,
            role_raw=None,
            role_code=None,
            role_level=None,
            start_month=item.start_month,
            end_month=item.end_month,
            is_current=None,
            industry_codes=item.industry_codes,
            topic_codes=item.topic_codes,
            origin_type="source_parsed",
            verification_status=verification,
            weight_tier="A",
            project_name=item.project_name,
            project_role=item.project_role,
        )
        return [
            self._record(
                expert_id=item.expert_id,
                serial_number=serial_number,
                chunk_type="project",
                source_item_key=item.source_item_key,
                text=text,
                metadata=metadata,
                source_table=item.source_table,
                source_record_id=item.source_record_id,
                source_field="project_experience",
                source_field_id=f"{item.source_table}:{item.source_record_id}",
                source_updated_at=item.source_updated_at,
                source_hash=item.source_hash,
                company_name_norm=item.company_or_client_norm,
                role_code=None,
                start_month=item.start_month,
                end_month=item.end_month,
                is_current=None,
                industry_codes=item.industry_codes,
                topic_codes=item.topic_codes,
                origin_type="source_parsed",
                verification_status=verification,
                weight_tier="A",
                data_version=item.data_version,
                build_batch_id=item.build_batch_id,
                chunk_subtype=subtype,
                chunk_index=index,
                chunk_total=len(pieces),
            )
            for index, (subtype, text) in enumerate(pieces)
        ]

    def _record(
        self,
        *,
        expert_id: int,
        serial_number: str | None,
        chunk_type: str,
        source_item_key: str,
        text: str,
        metadata: dict[str, object],
        source_table: str,
        source_record_id: str,
        source_field: str,
        source_field_id: str,
        source_updated_at: datetime | None,
        source_hash: str,
        company_name_norm: str | None,
        role_code: str | None,
        start_month: str | None,
        end_month: str | None,
        is_current: bool | None,
        industry_codes: tuple[str, ...],
        topic_codes: tuple[str, ...],
        origin_type: str,
        verification_status: str,
        weight_tier: str,
        data_version: str,
        build_batch_id: int,
        chunk_subtype: str | None = None,
        chunk_index: int = 0,
        chunk_total: int = 1,
    ) -> ChunkRecord:
        """集中生成 ID、哈希和版本，避免三类 Builder 产生细微漂移。"""

        normalized = text.strip()
        return ChunkRecord(
            chunk_id=f"expert:{expert_id}:{chunk_type}:{source_item_key}:{chunk_index}",
            expert_id=expert_id,
            serial_number=serial_number,
            chunk_type=chunk_type,
            chunk_subtype=chunk_subtype,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            chunk_text=normalized,
            char_count=len(normalized),
            metadata=metadata,
            source_table=source_table,
            source_record_id=source_record_id,
            source_item_key=source_item_key,
            source_field=source_field,
            source_field_id=source_field_id,
            source_version=_iso_with_zone(source_updated_at, self.settings.timezone),
            source_updated_at=source_updated_at,
            source_hash=source_hash,
            content_hash=sha256_value(normalized),
            company_name_norm=company_name_norm,
            role_code=role_code,
            start_month=start_month,
            end_month=end_month,
            is_current=is_current,
            industry_codes=industry_codes,
            topic_codes=topic_codes,
            origin_type=origin_type,
            verification_status=verification_status,
            weight_tier=weight_tier,
            chunk_policy_version=self.settings.chunk_policy_version,
            metadata_schema_version=self.settings.metadata_schema_version,
            dictionary_version=self.settings.dictionary_version,
            data_version=data_version,
            build_batch_id=build_batch_id,
        )


def _metadata(
    *,
    company_raw: str | None,
    company_norm: str | None,
    company_status: str | None,
    role_raw: str | None,
    role_code: str | None,
    role_level: str | None,
    industry_codes: tuple[str, ...],
    topic_codes: tuple[str, ...],
    origin_type: str,
    verification_status: str,
    weight_tier: str,
    start_month: str | None = None,
    end_month: str | None = None,
    is_current: bool | None = None,
    project_name: str | None = None,
    project_role: str | None = None,
) -> dict[str, object]:
    """按契约一次性组装 metadata，显式列随后复用相同值。"""

    result: dict[str, object] = {
        "company_name_raw": company_raw,
        "company_name_norm": company_norm,
        "company_normalization_status": company_status,
        "role_name_raw": role_raw,
        "role_code": role_code,
        "role_level": role_level,
        "start_month": start_month,
        "end_month": end_month,
        "is_current": is_current,
        "industry_codes": list(industry_codes),
        "topic_codes": list(topic_codes),
        "origin_type": origin_type,
        "verification_status": verification_status,
        "weight_tier": weight_tier,
    }
    if project_name is not None or project_role is not None:
        result["project_name"] = project_name
        result["project_role"] = project_role
    return result


def _append(lines: list[str], label: str, value: str | None) -> None:
    """只输出有来源值的模板行，避免“未知”占位污染语义。"""

    if value:
        lines.append(f"{label}：{value}")


def _extractive_fit(text: str, limit: int) -> str:
    """按句子/短语选择完整来源片段，输出长度不超过 limit。"""

    if len(text) <= limit:
        return text
    units = [unit.strip() for unit in re.split(r"(?<=[。！？；;\n])", text) if unit.strip()]
    selected: list[str] = []
    for unit in units:
        candidate = "".join(selected) + unit
        if len(candidate) <= limit:
            selected.append(unit)
        else:
            break
    if selected:
        return "".join(selected).strip()
    # 极端情况下首句超长，按更细标点提取；最终仍只复制来源前缀，不增写省略号。
    phrases = [part for part in re.split(r"(?<=[，,、：:])", text) if part]
    result = ""
    for phrase in phrases:
        if len(result) + len(phrase) > limit:
            break
        result += phrase
    return result.strip() or text[:limit].strip()


def _render_pieces(
    header_lines: list[str],
    sections: list[tuple[str, str | None, str]],
    hard_limit: int,
) -> list[tuple[str | None, str]]:
    """先尝试单块；超长时按语义区段和句子切为多个同源子块。"""

    body_lines = [f"{label}：{value}" for label, value, _ in sections if value]
    full = "\n".join(header_lines + body_lines)
    if len(full) <= hard_limit:
        return [(None, full)]
    prefix = "\n".join(header_lines)
    capacity = hard_limit - len(prefix) - 1
    if capacity <= 20:
        raise ValueError("Chunk 公共头已超过长度上限，无法安全切分")
    pieces: list[tuple[str | None, str]] = []
    for label, value, subtype in sections:
        if not value:
            continue
        label_prefix = f"{label}："
        for part in _split_text(value, capacity - len(label_prefix)):
            pieces.append((subtype, f"{prefix}\n{label_prefix}{part}"))
    return pieces or [(None, prefix)]


def _split_text(text: str, limit: int) -> list[str]:
    """按段落/句子装箱；单句超长时才按字符边界切分。"""

    if len(text) <= limit:
        return [text]
    units = [unit.strip() for unit in re.split(r"(?<=[。！？；;\n])", text) if unit.strip()]
    result: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > limit:
            if current:
                result.append(current)
                current = ""
            result.extend(unit[start : start + limit] for start in range(0, len(unit), limit))
        elif not current or len(current) + len(unit) <= limit:
            current += unit
        else:
            result.append(current)
            current = unit
    if current:
        result.append(current)
    return [part.strip() for part in result if part.strip()]


def _iso_with_zone(value: datetime | None, timezone_name: str) -> str:
    """数据库 DATETIME 无时区；按配置时区解释后输出契约要求的 ISO 8601。"""

    if value is None:
        # 来源表允许 updated_at 为空，source_version 仍需稳定且非空。
        return "unknown"
    aware = value if value.tzinfo else value.replace(tzinfo=ZoneInfo(timezone_name))
    return aware.isoformat(timespec="microseconds")
