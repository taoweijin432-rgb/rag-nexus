"""把标准化后的 Stage 数据组装为三类最终 Chunk。

可以把本文件理解成“文本排版器”：normalize.py 已经负责清洗和标准化字段，本文件
只决定这些字段以什么标题、顺序和长度组成 chunk_text，再补齐 Metadata、唯一 ID
和哈希，最终得到可交给 Validator 和 Contract Bridge 的 ChunkRecord。

构建器不读取数据库，也不会使用模型改写事实。超长任职和项目只在同一来源记录内部
切分，并在每个子块重复公司、岗位或项目公共头；不同来源记录永不合并。
profile_summary 每位专家最多一条，超长背景采用抽取式压缩，尽量保留完整原句。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Settings
from .dataset import EXPERT_SPEC, DatasetSpec
from .models import ChunkRecord, EmploymentStage, ProfileStage, ProjectStage
from .normalize import canonical_json, sha256_value

# 三类 Chunk 的字符硬上限。Profile 超长时做抽取式压缩；任职和项目超长时拆成多个
# 同源子块。Validator 会使用同一份配置再次检查，防止超长正文进入下游 Contract。
HARD_LIMITS = {"profile_summary": 600, "employment": 900, "project": 1000}


def build_vector_doc_id(
    *,
    expert_id: int,
    chunk_type: str,
    source_field_id: str,
    chunk_index: int,
) -> str:
    """按产品规则生成 Milvus 记录的稳定唯一标识。

    产品公式中的旧 ``source_record_id`` 已由新溯源契约的 ``source_field_id``
    等价替代。该 ID 只依赖业务来源和分片位置，因此重复构建同一 Chunk 时保持稳定。
    """

    return f"expert:{expert_id}:{chunk_type}:{source_field_id}:{chunk_index}"


class ChunkFactory:
    """把一位专家的三类 Stage 对象转换为最终 Chunk 集合。

    Pipeline 每处理一位专家都会调用本对象；本类只做确定性构建，不负责校验、
    数据库事务或向量化，因此相同 Stage 输入会得到相同正文、ID 和哈希。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        dataset_spec: DatasetSpec = EXPERT_SPEC,
    ) -> None:
        """绑定构建设置和来源规格；现有调用默认保持 Expert 规格。"""

        self.settings = settings
        self.dataset_spec = dataset_spec

    def build_all(
        self,
        profile: ProfileStage,
        employments: Iterable[EmploymentStage],
        projects: Iterable[ProjectStage],
    ) -> list[ChunkRecord]:
        """按画像、工作、项目的固定顺序生成这位专家的全部 Chunk。

        一条工作或项目经历可能因为正文过长返回多个 Chunk，所以这里使用 extend
        展开结果。固定顺序便于测试、导出和人工检查，但每条记录的真正唯一性仍由
        vector_doc_id 保证。
        """

        chunks: list[ChunkRecord] = []
        # Profile 最多一条；即使为空不生成，也继续处理后面的工作和项目经历。
        chunks.extend(self.build_profile(profile))
        # 每条经历独立构建，不把不同公司的工作内容拼进同一个 Chunk。
        for employment in employments:
            chunks.extend(self.build_employment(profile.serial_number, employment))
        for project in projects:
            chunks.extend(self.build_project(profile.serial_number, project))
        return chunks

    def build_profile(self, item: ProfileStage) -> list[ChunkRecord]:
        """用主档关键事实生成最多一条 profile_summary。

        正文只展示当前/最近公司、岗位和背景；工作年限、教育、联系资格等字段放在
        Metadata 中，不重复塞入正文。公司、岗位和背景全部为空时返回空列表，表示
        这类 Chunk 没有足够内容，而不是构造只有标题的无意义文本。
        """

        # 教育信息目前只进入 Metadata，不能单独支撑一条画像正文。
        if not any((item.company_name_norm, item.role_name_raw, item.background_clean)):
            return []
        # 使用列表逐行组装，最后统一以换行连接，便于控制顺序并跳过空字段。
        lines = ["[专家能力概述]"]
        if item.company_name_norm or item.role_name_raw:
            current = " · ".join(part for part in (item.company_name_norm, item.role_name_raw) if part)
            lines.append(f"当前/最近任职：{current}")
        prefix = "\n".join(lines)
        if item.background_clean:
            # 先扣除标题、任职行和“核心背景”标签占用的字符，把剩余空间留给背景原文。
            budget = HARD_LIMITS["profile_summary"] - len(prefix) - len("\n核心背景：")
            # 抽取完整句而非在字符中间硬截断；若首句自身过长，再按标点短语切分。
            background = _extractive_fit(item.background_clean, max(budget, 1))
            if background:
                lines.append(f"核心背景：{background}")
        text = "\n".join(lines)
        return [
            self._record(
                expert_id=item.expert_id,
                serial_number=item.serial_number,
                chunk_type="profile_summary",
                text=text,
                # profile 同时压缩主档与教育表信息。source_field_id 仍使用专家 ID，
                # 因为教育表是通过 expert_id 聚合，vector_doc_id 因而保持稳定。
                source_table=self.dataset_spec.profile_source_label,
                source_field=self.dataset_spec.profile_source_field,
                source_field_id=str(item.expert_id),
                source_updated_at=item.source_updated_at,
                source_hash=item.source_hash,
                company_name_raw=item.company_name_raw,
                company_name_norm=item.company_name_norm,
                role_name_raw=item.role_name_raw,
                role_code=item.role_code,
                role_level=item.role_level,
                job_status_norm=item.job_status_norm,
                abroad_flag_norm=item.abroad_flag_norm,
                work_years=item.work_years,
                project_name=None,
                start_month=None,
                end_month=None,
                is_current=None,
                industry_codes=(),
                function_codes=(),
                topic_codes=(),
                school_names_raw=item.school_names_raw,
                major_names_raw=item.major_names_raw,
                eligible_for_recommendation=item.eligible_for_recommendation,
                origin_type="compressed_from_source",
                # profile 的 verification_status 由 experts.doubtful_flag 决定：
                # 1=普通→source_summary，2/未知=存疑→rejected。
                verification_status=item.verification_status,
                weight_tier="B",
                build_batch_id=item.build_batch_id,
            )
        ]

    def build_employment(self, serial_number: str | None, item: EmploymentStage) -> list[ChunkRecord]:
        """把一条工作经历排版成一个或多个 employment Chunk。

        header 保存每个子块都必须具备的公司、岗位、时间等上下文；sections 保存可能
        很长的职责和成果。正文没有超限时生成一条，超限时只拆 sections，并在每个
        子块重复 header，保证单独检索任一子块时仍然知道它属于哪段任职。
        """

        # 公司和岗位至少要有一个，且职责/成果可以为空；绝不补造不存在的职责。
        if not any((item.company_name_raw, item.company_name_norm, item.role_name_raw)):
            return []
        # header 是任职的公共身份信息；_append 会自动跳过没有来源值的行。
        header = ["[任职经历]"]
        _append(header, "公司", item.company_name_norm or item.company_name_raw)
        _append(header, "岗位", item.role_name_raw)
        if item.start_month:
            _append(header, "时间", f"{item.start_month}—{item.end_month or ('至今' if item.is_current else '未知')}")
        if item.is_current is not None:
            _append(header, "任职状态", "当前任职" if item.is_current else "历史任职")
        _append(header, "业务领域", "、".join(item.industry_codes) or None)
        # subtype 只在超长切分后标记子块语义；未超长时整条 Chunk 的 subtype 为 None。
        sections = [
            ("工作职责", item.responsibilities_clean, "responsibility"),
            ("代表性成果", item.achievements_clean, "achievement"),
        ]
        pieces = _render_pieces(header, sections, HARD_LIMITS["employment"])
        # enumerate 从 0 生成稳定 chunk_index；相同源记录重复构建时 vector_doc_id 不变。
        return [
            self._record(
                expert_id=item.expert_id,
                serial_number=serial_number,
                chunk_type="employment",
                text=text,
                source_table=item.source_table,
                source_field=item.source_field,
                source_field_id=item.source_field_id,
                source_updated_at=item.source_updated_at,
                source_hash=item.source_hash,
                company_name_raw=item.company_name_raw,
                company_name_norm=item.company_name_norm,
                role_name_raw=item.role_name_raw,
                role_code=item.role_code,
                role_level=item.role_level,
                job_status_norm=None,
                abroad_flag_norm=None,
                work_years=None,
                project_name=None,
                start_month=item.start_month,
                end_month=item.end_month,
                is_current=item.is_current,
                industry_codes=item.industry_codes,
                function_codes=(),
                topic_codes=item.topic_codes,
                school_names_raw=(),
                major_names_raw=(),
                eligible_for_recommendation=None,
                origin_type="source_parsed",
                verification_status="verified",
                weight_tier="A",
                build_batch_id=item.build_batch_id,
                chunk_subtype=subtype,
                chunk_index=index,
                chunk_total=len(pieces),
            )
            for index, (subtype, text) in enumerate(pieces)
        ]

    def build_project(self, serial_number: str | None, item: ProjectStage) -> list[ChunkRecord]:
        """把一条项目经历排版成一个或多个 project Chunk。

        项目名称、客户和日期组成公共头，目标、工作内容、结果组成正文区段。缺少的
        结构化事实保持为空，不从上下文猜测；超长时沿用任职经历相同的分块方式，
        且不同 source_field_id 的项目永远不会合并。
        """

        # 名称、目标、工作内容至少存在一项才尝试生成；最终契约仍由 Validator 复核。
        if not any((item.project_name, item.objective_clean, item.actions_clean)):
            return []
        # 每个项目子块都重复这一公共头，避免向量召回到子块后失去项目上下文。
        header = ["[项目经历]"]
        _append(header, "项目名称", item.project_name)
        _append(header, "所属公司/客户", item.company_or_client_norm or item.company_or_client_raw)
        if item.start_month or item.end_month:
            _append(header, "时间", f"{item.start_month or '未知'}—{item.end_month or '未知'}")
        _append(header, "业务领域", "、".join(item.industry_codes) or None)
        sections = [
            # 正确源表没有目标/结果列，只有字段注释明确的“工作内容”。
            ("项目描述", item.objective_clean, "context"),
            ("项目工作内容", item.actions_clean, "action"),
            ("项目结果", item.results_clean, "result"),
        ]
        pieces = _render_pieces(header, sections, HARD_LIMITS["project"])
        # pieces 的数量同时写入每条记录的 chunk_total，便于下游了解同源分片总数。
        return [
            self._record(
                expert_id=item.expert_id,
                serial_number=serial_number,
                chunk_type="project",
                text=text,
                source_table=item.source_table,
                source_field=item.source_field,
                source_field_id=item.source_field_id,
                source_updated_at=item.source_updated_at,
                source_hash=item.source_hash,
                company_name_raw=item.company_or_client_raw,
                company_name_norm=item.company_or_client_norm,
                role_name_raw=None,
                role_code=None,
                role_level=None,
                job_status_norm=None,
                abroad_flag_norm=None,
                work_years=None,
                project_name=item.project_name,
                start_month=item.start_month,
                end_month=item.end_month,
                is_current=None,
                industry_codes=item.industry_codes,
                function_codes=(),
                topic_codes=item.topic_codes,
                school_names_raw=(),
                major_names_raw=(),
                eligible_for_recommendation=None,
                origin_type="source_parsed",
                verification_status="verified",
                weight_tier="A",
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
        text: str,
        source_table: str,
        source_field: str,
        source_field_id: str,
        source_updated_at: datetime | None,
        source_hash: str,
        company_name_raw: str | None,
        company_name_norm: str | None,
        role_name_raw: str | None,
        role_code: str | None,
        role_level: str | None,
        job_status_norm: str | None,
        abroad_flag_norm: bool | None,
        work_years: float | None,
        project_name: str | None,
        start_month: str | None,
        end_month: str | None,
        is_current: bool | None,
        industry_codes: tuple[str, ...],
        function_codes: tuple[str, ...],
        topic_codes: tuple[str, ...],
        school_names_raw: tuple[str, ...],
        major_names_raw: tuple[str, ...],
        eligible_for_recommendation: bool | None,
        origin_type: str,
        verification_status: str,
        weight_tier: str,
        build_batch_id: int,
        chunk_subtype: str | None = None,
        chunk_index: int = 0,
        chunk_total: int = 1,
    ) -> ChunkRecord:
        """把已经排版好的正文和 Metadata 统一封装成 ChunkRecord。

        三类 Builder 都走这个入口，因此 ID 公式、字符数、正文哈希、Metadata 哈希、
        版本号和初始激活状态只有一份实现。以后修改公共字段时只需检查这里，避免
        profile、employment、project 各自实现后出现细微差异。
        """

        # 去掉模板拼接后可能留下的首尾空白；后面的字符数和 content_hash 都以该值为准。
        normalized = text.strip()
        # 唯一 ID 只使用稳定业务键和分片序号，不使用数据库自增 ID 或构建批次号。
        vector_doc_id = build_vector_doc_id(
            expert_id=expert_id,
            chunk_type=chunk_type,
            source_field_id=source_field_id,
            chunk_index=chunk_index,
        )
        # 把数据库无时区时间转换成接口契约使用的带时区版本字符串。
        source_version = _iso_with_zone(source_updated_at, self.settings.timezone)
        # metadata_hash 只覆盖最终会作为 Metadata 暴露的稳定扁平字段，不包含
        # chunk_text/content_hash，因此可以独立识别“正文不变、Metadata 变化”。
        # indexed_at 与 embedding 模型信息由下游回填，也不参与上游计算。
        record = ChunkRecord(
            expert_id=expert_id,
            serial_number=serial_number,
            chunk_type=chunk_type,
            chunk_subtype=chunk_subtype,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            chunk_text=normalized,
            char_count=len(normalized),
            vector_doc_id=vector_doc_id,
            source_table=source_table,
            source_field=source_field,
            source_field_id=source_field_id,
            source_version=source_version,
            source_updated_at=source_updated_at,
            source_hash=source_hash,
            content_hash=sha256_value(normalized),
            # 先使用临时空值构造不可变对象，再由统一函数基于全部扁平字段计算。
            # 这样 Validator 可复用完全相同的算法，避免两处字段清单发生漂移。
            metadata_hash="",
            company_name_raw=company_name_raw,
            company_name_norm=company_name_norm,
            role_name_raw=role_name_raw,
            role_code=role_code,
            role_level=role_level,
            job_status_norm=job_status_norm,
            abroad_flag_norm=abroad_flag_norm,
            # 下游 Contract 使用 Decimal 表示工作年限，先保留稳定的十进制语义。
            # 生成时也统一为 float，避免 8 与 8.0 的 JSON 表示差异导致停用重算哈希漂移。
            work_years=float(work_years) if work_years is not None else None,
            project_name=project_name,
            start_month=start_month,
            end_month=end_month,
            is_current=is_current,
            industry_codes=industry_codes,
            function_codes=function_codes,
            topic_codes=topic_codes,
            school_names_raw=school_names_raw,
            major_names_raw=major_names_raw,
            eligible_for_recommendation=eligible_for_recommendation,
            origin_type=origin_type,
            verification_status=verification_status,
            weight_tier=weight_tier,
            chunk_policy_version=self.settings.chunk_policy_version,
            metadata_schema_version=self.settings.metadata_schema_version,
            embedding_model_alias=None,
            embedding_model_version=None,
            indexed_at=None,
            is_active=True,
            build_batch_id=build_batch_id,
        )
        # ChunkRecord 是不可变 dataclass，先以空哈希创建，再通过 replace 返回补齐哈希的
        # 新对象；这样计算函数可以直接读取完整 Metadata，而不会修改原对象。
        return replace(record, metadata_hash=calculate_metadata_hash(record))


def metadata_hash_payload(chunk: ChunkRecord) -> dict[str, object]:
    """返回参与 ``metadata_hash`` 的稳定字段。

    ``content_hash`` 被刻意排除，确保正文变化与 Metadata 变化可以分别判断；
    ``indexed_at``、Embedding 模型字段和数据库更新时间由下游或数据库产生，也不应
    导致上游相同业务 Metadata 的哈希漂移。

    4.1.0 变更：Metadata 改为按 ``chunk_type`` 分类组装后，hash payload 只包含
    “公共字段 + 当前 ``chunk_type`` 实际存在的 Metadata 字段”；不适用于当前类型
    的字段不再以 ``None``/``[]`` 占位参与哈希。
    """

    payload: dict[str, object] = {
        # 身份 / 切分 / 来源 / 版本 / 有效性公共字段。
        "vector_doc_id": chunk.vector_doc_id,
        "expert_id": chunk.expert_id,
        "serial_number": chunk.serial_number,
        "chunk_type": chunk.chunk_type,
        "chunk_subtype": chunk.chunk_subtype,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk.chunk_total,
        "source_table": chunk.source_table,
        "source_field": chunk.source_field,
        "source_field_id": chunk.source_field_id,
        "source_version": chunk.source_version,
        "source_updated_at": chunk.source_updated_at,
        "source_hash": chunk.source_hash,
        # 公共业务 Metadata。
        "company_name_raw": chunk.company_name_raw,
        "company_name_norm": chunk.company_name_norm,
        "industry_codes": list(chunk.industry_codes),
        "topic_codes": list(chunk.topic_codes),
        "origin_type": chunk.origin_type,
        "verification_status": chunk.verification_status,
        "weight_tier": chunk.weight_tier,
        "chunk_policy_version": chunk.chunk_policy_version,
        "metadata_schema_version": chunk.metadata_schema_version,
        "is_active": chunk.is_active,
    }
    # 当前 chunk_type 专属业务 Metadata：只参与本类型实际存在的字段。
    if chunk.chunk_type == "profile_summary":
        payload.update(
            {
                "role_name_raw": chunk.role_name_raw,
                "role_code": chunk.role_code,
                "role_level": chunk.role_level,
                "job_status_norm": chunk.job_status_norm,
                "abroad_flag_norm": chunk.abroad_flag_norm,
                "work_years": chunk.work_years,
                "function_codes": list(chunk.function_codes),
                "school_names_raw": list(chunk.school_names_raw),
                "major_names_raw": list(chunk.major_names_raw),
                "eligible_for_recommendation": chunk.eligible_for_recommendation,
            }
        )
    elif chunk.chunk_type == "employment":
        payload.update(
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
        payload.update(
            {
                "project_name": chunk.project_name,
                "start_month": chunk.start_month,
                "end_month": chunk.end_month,
            }
        )
    return payload


def calculate_metadata_hash(chunk: ChunkRecord) -> str:
    """把 Metadata 按固定字段顺序编码后计算 SHA-256。

    相同 Metadata 会得到相同哈希，便于下游识别“正文不变、只有 Metadata 变化”的
    情况，从而跳过重复 Embedding，只更新 Milvus Metadata。
    """

    return sha256_value(canonical_json(metadata_hash_payload(chunk)))


def _append(lines: list[str], label: str, value: str | None) -> None:
    """有来源值时才追加“标签：值”，为空时不生成这一行。

    该小函数让三类正文采用相同的空值策略，避免出现“公司：None”之类无意义文本。
    """

    if value:
        lines.append(f"{label}：{value}")


def _extractive_fit(text: str, limit: int) -> str:
    """在长度上限内尽量保留完整原句，用于压缩 Profile 背景。

    这里只做“从前往后选取”，不会总结或改写：先按句号等句末标点选择完整句；
    第一整句仍放不下时再按逗号等短语边界选择，最后才按字符上限截取。
    """

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
    """把公共头和正文区段渲染为一个或多个同源文本块。

    未超限时保留完整文本并返回一块；超限时分别拆职责、成果或项目区段，每一块都
    带上相同公共头。返回值同时携带 subtype，供 Builder 写入 Chunk Metadata。
    """

    # 先拼出最理想的完整正文；长度允许时无需拆分，subtype 用 None 表示整块内容。
    body_lines = [f"{label}：{value}" for label, value, _ in sections if value]
    full = "\n".join(header_lines + body_lines)
    if len(full) <= hard_limit:
        return [(None, full)]
    prefix = "\n".join(header_lines)
    # capacity 是扣除公共头和换行后，单个正文区段真正可使用的字符数。
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
    """把长正文尽量按完整句子装入多个不超过 limit 的片段。

    多个短句会在不超限的前提下合并；只有单句本身超过上限时才按字符边界硬切，
    因此通常不会把一句正常长度的话拆到两个 Chunk 中。
    """

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
    """把数据库 DATETIME 按项目时区解释为稳定的 ISO 8601 版本字符串。

    该值用于 source_version 和 Metadata 追溯，不参与正文排版。来源时间为空时返回
    固定的 unknown，而不是使用当前运行时间，保证重复构建仍得到相同结果。
    """

    if value is None:
        # 来源表允许 updated_at 为空，source_version 仍需稳定且非空。
        return "unknown"
    aware = value if value.tzinfo else value.replace(tzinfo=ZoneInfo(timezone_name))
    return aware.isoformat(timespec="microseconds")
