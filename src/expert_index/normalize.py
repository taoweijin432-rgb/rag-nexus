"""确定性的文本清洗、PII 防护、标准化与来源哈希。

清洗器只删除噪声和禁止信息，不改写数字、单位、日期、公司主体或责任等级。
词典命中必须有原文证据；无法确认的公司/岗位保留 raw 并标记 unknown。
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from .models import (
    EmploymentSource,
    EmploymentStage,
    ExpertAggregate,
    ProfileStage,
    ProjectSource,
    ProjectStage,
)

# 正则只用于“识别并删除”，命中日志只能记录类别和数量，绝不能打印原值。
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    "bank_card": re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)"),
    "url": re.compile(r"https?://[^\s<>]+", re.IGNORECASE),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "wechat": re.compile(r"(?:微信|wechat|wx)\s*[号：:]?\s*[A-Za-z][-_A-Za-z0-9]{5,19}", re.IGNORECASE),
}

_EMPTY_VALUES = {"", "null", "none", "--", "/"}
_INVISIBLE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_HTML_TAG = re.compile(r"<[^>]+>")
_MONTH = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-/.年]?\s*(0?[1-9]|1[0-2])")

# 第一版词典保持小而可审计；未命中时绝不做模糊猜测。
COMPANY_ALIASES = {
    "北京京东世纪贸易有限公司": "京东零售",
    "京东零售集团有限公司": "京东零售",
    "ibm中国": "IBM中国",
}

ROLE_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("首席执行官", "ceo", "总经理", "总裁"), "executive_management", "executive"),
    (("项目经理",), "project_management", "manager"),
    (("运营负责人",), "operation_leader", "head"),
    (("研发工程师",), "research_engineering", "individual_contributor"),
    (("架构师",), "solution_architecture", "expert"),
)

INDUSTRY_RULES: dict[str, tuple[str, ...]] = {
    "artificial_intelligence": ("人工智能", "大模型", "aigc", "ai应用"),
    "automotive": ("汽车", "整车", "新能源车"),
    "ecommerce": ("电商", "电子商务", "gmv", "直播带货"),
    "environmental_protection": ("环保", "环境治理", "废水", "废气"),
    "finance": ("金融", "银行", "保险", "证券"),
    "hydrogen_energy": ("氢能", "燃料电池", "制氢"),
    "software": ("软件", "java", "云计算", "信息系统"),
}

TOPIC_RULES: dict[str, tuple[str, ...]] = {
    "cost_reduction": ("降本", "成本下降", "降低成本"),
    "gmv": ("gmv", "成交总额"),
    "project_delivery": ("项目交付", "项目管理", "交付管理"),
    "quality_management": ("质量管理", "质量体系", "良率"),
    "research_and_development": ("研发", "技术开发", "实验"),
    "user_growth": ("用户增长", "获客", "转化率"),
}


def clean_text(value: Any) -> str:
    """按固定顺序清洗文本；返回空串表示没有可用内容。"""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    if text.strip().lower() in _EMPTY_VALUES:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = html.unescape(_HTML_TAG.sub(" ", text))
    for pattern in PII_PATTERNS.values():
        text = pattern.sub("", text)
    # 删除 PII 后留下的孤立联系方式标签，避免无意义词进入向量文本。
    text = re.sub(r"(?:联系方式|联系电话|手机|邮箱|微信)\s*[，,；;：:]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([，,；;：:])\s*\1+", r"\1", text)
    return text.strip(" \n，,；;：:")


def pii_hits(value: str) -> dict[str, int]:
    """返回仍然命中的 PII 类别计数；调用方不得记录具体匹配内容。"""

    return {name: len(pattern.findall(value)) for name, pattern in PII_PATTERNS.items() if pattern.search(value)}


def canonical_json(value: Any) -> str:
    """生成 source_hash 使用的确定性 JSON，日期统一为 ISO 字符串。"""

    def default(item: Any) -> str:
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        raise TypeError(f"无法序列化 {type(item).__name__}")

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=default)


def sha256_value(value: str) -> str:
    """统一 SHA-256 小写十六进制实现。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_hash(allowed_fields: dict[str, Any]) -> str:
    """只对允许且已清洗的来源字段计算哈希。"""

    return sha256_value(canonical_json(allowed_fields))


def normalize_company(value: Any) -> tuple[str | None, str | None, str]:
    """返回 raw、norm、status；未知公司保留原文，不做后缀猜测。"""

    raw = clean_text(value) or None
    if raw is None:
        return None, None, "unknown"
    if raw in COMPANY_ALIASES:
        return raw, COMPANY_ALIASES[raw], "verified"
    return raw, raw, "unknown"


def normalize_role(value: Any) -> tuple[str | None, str | None, str | None]:
    """岗位规则只在包含明确关键词时映射，原岗位始终保留。"""

    raw = clean_text(value) or None
    if raw is None:
        return None, None, None
    lowered = raw.lower()
    for aliases, code, level in ROLE_RULES:
        if any(alias.lower() in lowered for alias in aliases):
            return raw, code, level
    return raw, None, None


def month_value(value: datetime | date | str | None) -> str | None:
    """把结构化日期或文本中的首个年月按 V1.3 规范转换为 ``YYYYMM``。"""

    if isinstance(value, (datetime, date)):
        return f"{value.year:04d}{value.month:02d}"
    cleaned = clean_text(value)
    match = _MONTH.search(cleaned)
    return f"{int(match.group(1)):04d}{int(match.group(2)):02d}" if match else None


def normalize_job_status(value: object) -> str | None:
    """标准化 ``experts.job_status``：1=现任，2=曾任。"""

    if value == 1:
        return "active"
    if value == 2:
        return "former"
    return None


def recommendation_eligibility(value: object) -> bool | None:
    """由 ``contact_status`` 计算推荐资格。

    1=未接触、2=接触中，尚未形成可推荐合作状态；3=接受访谈、4=访谈合作，
    已达到推荐准入条件。未知脏值保持 ``None``，不默认为可推荐。
    """

    if value in (1, 2):
        return False
    if value in (3, 4):
        return True
    return None


def profile_verification_status(value: object) -> str:
    """由 ``doubtful_flag`` 生成画像核验状态：1=普通，2=存疑。"""

    if value == 1:
        return "source_summary"
    if value == 2:
        return "rejected"
    # 未知值不能按正常专家处理，使用 rejected 阻止进入推荐/向量消费。
    return "rejected"


def normalize_abroad_flag(value: object) -> bool | None:
    """标准化 ``abroad_flag``：1=海外专家，2=非海外专家。"""

    if value == 1:
        return True
    if value == 2:
        return False
    return None


def employment_status(value: object) -> bool | None:
    """按源表注释标准化 is_employed：1=在职，2=不在职，其他=未知。"""

    if value == 1:
        return True
    if value == 2:
        return False
    return None


def parse_employment_description(value: object) -> tuple[str | None, str | None]:
    """解析显式的工作职责/代表性成果标签；普通文本整体作为职责。"""

    text = str(value).strip() if value is not None else ""
    if not text:
        return None, None
    sections = _labeled_sections(text, ("工作职责", "代表性成果"))
    if not sections:
        return text, None
    return sections.get("工作职责"), sections.get("代表性成果")


def parse_project_description(
    value: object,
) -> tuple[str | None, str | None, str | None, str | None]:
    """从同一项目原文提取角色、目标、动作、结果四类显式事实。

    无标签正文整体作为项目动作；项目角色不再进入 V1.3 输出契约。
    """

    text = str(value).strip() if value is not None else ""
    if not text:
        return None, None, None, None
    labels = ("项目角色", "项目目标", "关键动作", "项目结果")
    sections = _labeled_sections(text, labels)
    if not sections:
        return None, None, text, None
    return (
        sections.get("项目角色"),
        sections.get("项目目标"),
        sections.get("关键动作"),
        sections.get("项目结果"),
    )


def _labeled_sections(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """确定性解析中文标签区段，支持跨行且不改写原始事实。"""

    alternation = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"(?:^|\n)\s*({alternation})\s*[：:]\s*(.*?)"
        rf"(?=(?:\n\s*(?:{alternation})\s*[：:])|\Z)",
        re.DOTALL,
    )
    result: dict[str, str] = {}
    for match in pattern.finditer(text.replace("\r\n", "\n").replace("\r", "\n")):
        cleaned = match.group(2).strip()
        if cleaned:
            result[match.group(1)] = cleaned
    return result


def _codes(texts: Iterable[str], rules: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """从当前来源文本提取有直接关键词证据的代码，并按字典序稳定输出。"""

    evidence = "\n".join(texts).lower()
    return tuple(sorted(code for code, words in rules.items() if any(word.lower() in evidence for word in words)))


def normalize_aggregate(
    aggregate: ExpertAggregate,
    *,
    data_version: str,
    build_batch_id: int,
) -> tuple[ProfileStage, list[EmploymentStage], list[ProjectStage]]:
    """将一位专家完整快照转换为三类中间对象。"""

    profile = aggregate.profile
    company_raw, company_norm, company_status = normalize_company(profile.company)
    role_raw, role_code, role_level = normalize_role(profile.job_name)
    background = clean_text(profile.background) or None
    # 教育经历是独立结构化表，不再解析 experts.edu_experience 自由文本。稳定去重时
    # 保留源表排序中的首次出现，既避免重复 Metadata，也保证相同输入得到相同哈希。
    school_names = _clean_unique(item.school_name for item in aggregate.educations)
    major_names = _clean_unique(item.major for item in aggregate.educations)
    education_facts = [
        {
            "id": item.id,
            "school_name": clean_text(item.school_name) or None,
            "major": clean_text(item.major) or None,
        }
        for item in aggregate.educations
    ]
    profile_allowed = {
        "expert_id": profile.id,
        "serial_number": clean_text(profile.serial_number) or None,
        "company": company_raw,
        "job_status": profile.job_status,
        "role": role_raw,
        "contact_status": profile.contact_status,
        "doubtful_flag": profile.doubtful_flag,
        "abroad_flag": profile.abroad_flag,
        "background": background,
        "educations": education_facts,
        "work_years": profile.work_years,
    }
    profile_stage = ProfileStage(
        expert_id=profile.id,
        serial_number=profile_allowed["serial_number"],
        company_name_raw=company_raw,
        company_name_norm=company_norm,
        company_normalization_status=company_status,
        role_name_raw=role_raw,
        role_code=role_code,
        role_level=role_level,
        job_status_norm=normalize_job_status(profile.job_status),
        abroad_flag_norm=normalize_abroad_flag(profile.abroad_flag),
        background_clean=background,
        school_names_raw=school_names,
        major_names_raw=major_names,
        work_years=profile.work_years,
        eligible_for_recommendation=recommendation_eligibility(profile.contact_status),
        verification_status=profile_verification_status(profile.doubtful_flag),
        source_hash=source_hash(profile_allowed),
        # 任一教育记录变化都应推动 profile_summary 的来源版本前进。
        source_updated_at=_latest_source_time(profile, aggregate.educations),
        data_version=data_version,
        build_batch_id=build_batch_id,
    )

    employment_stages = [
        _normalize_employment(item, data_version=data_version, build_batch_id=build_batch_id)
        for item in aggregate.employments
    ]
    project_stages = [
        _normalize_project(item, data_version=data_version, build_batch_id=build_batch_id)
        for item in aggregate.projects
    ]
    return profile_stage, employment_stages, project_stages


def _normalize_employment(item: EmploymentSource, *, data_version: str, build_batch_id: int) -> EmploymentStage:
    """标准化单段任职；成果仅接受显式来源标签解析值。"""

    raw_company, norm_company, company_status = normalize_company(item.company_name)
    raw_role, role_code, role_level = normalize_role(item.position_title)
    parsed_responsibilities, parsed_achievements = parse_employment_description(
        item.work_description
    )
    responsibilities = clean_text(parsed_responsibilities) or None
    start_month, end_month = month_value(item.start_date), month_value(item.end_date)
    # 新表 is_employed 是权威标量：1=在职、2=不在职、其他未知。当前任职的
    # end_month 强制为空；历史任职即使缺结束时间也保留 false，不自行补日期。
    is_current = employment_status(item.is_employed)
    if is_current is True:
        end_month = None
    achievements = clean_text(parsed_achievements) or None
    allowed = {
        "company": raw_company,
        "role": raw_role,
        "start_month": start_month,
        "end_month": end_month,
        "is_current": is_current,
        "responsibilities": responsibilities,
        "achievements": achievements,
    }
    return EmploymentStage(
        expert_id=item.expert_id,
        source_table="experts_work_experience",
        source_field=(
            "company_name+position_title+start_date+end_date+"
            "work_description+is_employed"
        ),
        source_field_id=str(item.id),
        source_item_key=str(item.id),
        company_name_raw=raw_company,
        company_name_norm=norm_company,
        company_normalization_status=company_status,
        role_name_raw=raw_role,
        role_code=role_code,
        role_level=role_level,
        start_month=start_month,
        end_month=end_month,
        is_current=is_current,
        responsibilities_clean=responsibilities,
        achievements_clean=achievements,
        # V1.3 只保留行业/主题字段，获取规则尚未确定，不能继续使用旧关键词规则
        # 产生看似正式的分类结果。待产品确认数据源后再接入受版本控制的映射器。
        industry_codes=(),
        topic_codes=(),
        source_hash=source_hash(allowed),
        source_updated_at=item.updated_at or item.created_at,
        data_version=data_version,
        build_batch_id=build_batch_id,
    )


def _normalize_project(item: ProjectSource, *, data_version: str, build_batch_id: int) -> ProjectStage:
    """标准化单个项目；角色和结果只能来自显式来源标签。"""

    raw_company, norm_company, company_status = normalize_company(item.company_name)
    project_name = clean_text(item.project_title) or None
    _parsed_role, parsed_objective, parsed_actions, parsed_results = parse_project_description(
        item.work_description
    )
    objective = clean_text(parsed_objective) or None
    actions = clean_text(parsed_actions) or None
    results = clean_text(parsed_results) or None
    allowed = {
        "project_name": project_name,
        "company": raw_company,
        "start_month": month_value(item.start_date),
        "end_month": month_value(item.end_date),
        "description": objective,
        "responsibilities": actions,
        "results": results,
    }
    return ProjectStage(
        expert_id=item.expert_id,
        source_table="experts_project_experience",
        source_field="project_title+company_name+start_date+end_date+work_description",
        source_field_id=str(item.id),
        source_item_key=str(item.id),
        project_name=project_name,
        company_or_client_raw=raw_company,
        company_or_client_norm=norm_company,
        company_normalization_status=company_status,
        start_month=month_value(item.start_date),
        end_month=month_value(item.end_date),
        objective_clean=objective,
        actions_clean=actions,
        results_clean=results,
        # 与任职一致：字段保留，但本期不自行推断行业和主题代码。
        industry_codes=(),
        topic_codes=(),
        source_hash=source_hash(allowed),
        source_updated_at=item.updated_at or item.created_at,
        data_version=data_version,
        build_batch_id=build_batch_id,
    )


def _clean_unique(values: Iterable[object]) -> tuple[str, ...]:
    """清洗并按首次出现顺序去重，供学校、专业等列表型 Metadata 使用。"""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return tuple(result)


def _latest_source_time(profile: object, educations: Iterable[object]) -> datetime | None:
    """返回主档与教育经历中最新的业务时间，用作 profile 来源版本时间。"""

    candidates = [
        getattr(profile, "updated_at", None) or getattr(profile, "created_at", None),
        *(
            getattr(item, "updated_at", None) or getattr(item, "created_at", None)
            for item in educations
        ),
    ]
    known = [item for item in candidates if item is not None]
    return max(known) if known else None
