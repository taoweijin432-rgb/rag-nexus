"""把四张源表的原始字段转换为可生成 Chunk 的确定性 Stage 数据。

本模块位于 ``source.py`` 与 ``chunk_builder.py`` 之间，核心目的不是“润色”专家资料，
而是在不补造事实的前提下建立稳定、可审计的数据边界：

* 清理 HTML、不可见字符、空值占位符以及禁止进入索引的 PII；
* 将公司、岗位、日期和业务枚举转换成下游约定的统一表示；
* 从带有明确中文标签的经历描述中提取职责、成果、目标、动作和结果；
* 生成 Profile/Employment/Project Stage，供 ``ChunkFactory`` 构造 ``chunk_text``；
* 对实际允许进入链路的标准化字段计算 ``source_hash``，支持增量变化判断。

所有规则都必须是确定性的：相同输入应得到相同 Stage 和哈希。清洗器只删除噪声和
禁止信息，不改写数字、单位、日期、公司主体或责任等级；词典没有可靠命中时保留
原始值，并把标准化状态标记为 ``unknown``，避免产生看似规范但无法溯源的内容。
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

# PII 规则同时服务于两道防线：clean_text 在标准化阶段删除，validators.py 在落库前
# 通过 pii_hits 再检查一次。命中信息只能记录类别和数量，绝不能打印或写日志原值。
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    "bank_card": re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)"),
    "url": re.compile(r"https?://[^\s<>]+", re.IGNORECASE),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "wechat": re.compile(r"(?:微信|wechat|wx)\s*[号：:]?\s*[A-Za-z][-_A-Za-z0-9]{5,19}", re.IGNORECASE),
}

# 这些字符串在测试库和业务导入数据中通常表示“没有值”。统一转为空串后，下游可以
# 使用同一套“有值才生成文本行”的规则，不会把 null、-- 等噪声写入向量正文。
_EMPTY_VALUES = {"", "null", "none", "--", "/"}
# 隐藏控制符和双向文本控制符既会造成肉眼不可见差异，也可能让哈希或展示结果漂移；
# 在所有其他文本处理前删除，可保证数据库、JSON 和 Milvus 看到一致字符序列。
_INVISIBLE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_HTML_TAG = re.compile(r"<[^>]+>")
# 日期统一到月份粒度。只接受 1900—2099 年和合法月份，无法识别时返回 None，
# 不根据上下文猜测日期。
_MONTH = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-/.年]?\s*(0?[1-9]|1[0-2])")

# 第一版词典保持小而可审计。显式别名命中才能改变标准公司名；未命中时 raw/norm
# 都保留清洗后的原值，使检索仍能使用公司文本，同时用 unknown 表示尚未经过词典确认。
COMPANY_ALIASES = {
    "北京京东世纪贸易有限公司": "京东零售",
    "京东零售集团有限公司": "京东零售",
    "ibm中国": "IBM中国",
}

# 每条岗位规则由“可识别关键词、标准岗位代码、责任等级”组成。只做包含关系匹配，
# 没有证据时不生成 role_code/role_level，原岗位名称仍然保留。
ROLE_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("首席执行官", "ceo", "总经理", "总裁"), "executive_management", "executive"),
    (("项目经理",), "project_management", "manager"),
    (("运营负责人",), "operation_leader", "head"),
    (("研发工程师",), "research_engineering", "individual_contributor"),
    (("架构师",), "solution_architecture", "expert"),
)

# 行业和主题规则是早期可审计词典，目前 V1.3 的 Stage 构建明确不调用它们；对应
# Metadata 字段暂时输出空集合。保留常量和 _codes 辅助函数，供产品确定来源后复用。
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
    """按固定顺序把任意源值清洗成可安全进入 Stage 的文本。

    目的与效果
    ----------
    统一全角/半角、换行和空白表示，删除 HTML、不可见字符与 PII，使相同业务内容
    不会因为展示格式不同而生成不同哈希。返回空串表示该值没有可用内容，调用方会
    转为 ``None`` 或直接跳过对应 Chunk 文本行。

    使用位置
    --------
    Profile、Employment、Project 和教育字段标准化均复用本函数；Validator 会在最终
    Chunk 上再次执行 PII 检查，但不会再次改写文本。
    """

    # None 是数据库空值，不应转换成字符串 "None" 进入向量正文。
    if value is None:
        return ""
    # NFKC 把全角字母/数字等兼容字符转为稳定表示，但不会改写业务语义。
    text = unicodedata.normalize("NFKC", str(value))
    # 在进一步处理前识别常见空值占位符，避免生成只有“null”或“--”的 Chunk。
    if text.strip().lower() in _EMPTY_VALUES:
        return ""
    # 内部只保留 LF，确保跨操作系统运行时正文、字符数和 content_hash 一致。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    # 先移除标签再解码 HTML 实体，保留标签之间的可见业务文本。
    text = html.unescape(_HTML_TAG.sub(" ", text))
    # 多类 PII 逐类删除；这里只修改内存文本，不打印任何被删除的原值。
    for pattern in PII_PATTERNS.values():
        text = pattern.sub("", text)
    # 删除 PII 后留下的孤立联系方式标签，避免无意义词进入向量文本。
    text = re.sub(r"(?:联系方式|联系电话|手机|邮箱|微信)\s*[，,；;：:]?", "", text, flags=re.IGNORECASE)
    # 最后统一横向空白、行首尾空白、连续空行和重复标点。顺序固定可以保证幂等，
    # 即对已经清洗的文本再次调用 clean_text，结果仍保持不变。
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([，,；;：:])\s*\1+", r"\1", text)
    return text.strip(" \n，,；;：:")


def pii_hits(value: str) -> dict[str, int]:
    """检查清洗后文本是否仍含 PII，只返回类别及命中数量。

    本函数由 ``validators.py`` 在 Chunk 落库前调用，是 ``clean_text`` 之后的第二道
    安全防线。返回值不包含匹配原文，因此可以用于报错定位而不会再次泄露敏感信息。
    """

    return {name: len(pattern.findall(value)) for name, pattern in PII_PATTERNS.items() if pattern.search(value)}


def canonical_json(value: Any) -> str:
    """把字典等结构编码为字段有序、无多余空白的确定性 JSON。

    排序键名并统一日期表示，可以消除 Python 字典构造顺序和默认 JSON 空格造成的
    差异。相同业务值因此始终得到相同字符串，供 ``source_hash`` 和 Metadata Hash
    共用；本函数只用于哈希输入，不作为最终 Chunk 正文。
    """

    def default(item: Any) -> str:
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        raise TypeError(f"无法序列化 {type(item).__name__}")

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=default)


def sha256_value(value: str) -> str:
    """返回文本的 SHA-256 小写十六进制值，统一项目内哈希算法与编码。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_hash(allowed_fields: dict[str, Any]) -> str:
    """只对允许进入索引的标准化来源字段计算稳定哈希。

    该边界避免姓名、联系方式等未授权字段影响哈希，也让后续增量任务能够判断当前
    来源事实是否发生变化。它表示来源内容变化，不等同于最终 ``chunk_text`` 的
    ``content_hash``。
    """

    return sha256_value(canonical_json(allowed_fields))


def normalize_company(value: Any) -> tuple[str | None, str | None, str]:
    """生成公司原值、标准名和映射状态，供正文展示与 Metadata 过滤。

    ``raw`` 是清洗后的来源名称，``norm`` 只有在显式别名表命中时才替换；未命中时
    两者相同且状态为 ``unknown``。这样既保留可检索内容，也避免通过删除“有限公司”
    等后缀进行不可靠的主体合并。
    """

    raw = clean_text(value) or None
    if raw is None:
        return None, None, "unknown"
    if raw in COMPANY_ALIASES:
        return raw, COMPANY_ALIASES[raw], "verified"
    return raw, raw, "unknown"


def normalize_role(value: Any) -> tuple[str | None, str | None, str | None]:
    """生成岗位原名、标准代码和责任等级，供 Chunk Metadata 使用。

    只有清洗后的岗位包含明确规则关键词时才输出 ``role_code``/``role_level``；无法
    确认时返回空代码但保留原岗位，使下游能够展示来源事实而不会误分类。
    """

    raw = clean_text(value) or None
    if raw is None:
        return None, None, None
    lowered = raw.lower()
    for aliases, code, level in ROLE_RULES:
        if any(alias.lower() in lowered for alias in aliases):
            return raw, code, level
    return raw, None, None


def month_value(value: datetime | date | str | None) -> str | None:
    """把结构化日期或文本中的首个年月转换为 V1.3 的 ``YYYYMM``。

    Employment/Project 的正文、Metadata 和哈希都复用这一结果，从而避免同一日期同时
    出现 ``2024-01``、``2024/1`` 等格式。无法识别时返回 ``None``，不补造月份。
    """

    if isinstance(value, (datetime, date)):
        return f"{value.year:04d}{value.month:02d}"
    cleaned = clean_text(value)
    match = _MONTH.search(cleaned)
    return f"{int(match.group(1)):04d}{int(match.group(2)):02d}" if match else None


def normalize_job_status(value: object) -> str | None:
    """把 ``experts.job_status`` 映射为可读规范码：1=active，2=former。

    结果只进入 profile_summary Metadata；未知枚举返回 ``None``，避免误判在职状态。
    """

    if value == 1:
        return "active"
    if value == 2:
        return "former"
    return None


def recommendation_eligibility(value: object) -> bool | None:
    """由 ``contact_status`` 计算推荐资格。

    目的是把源库联系进度转换为 profile_summary 的推荐过滤字段。1=未接触、2=接触中，
    尚未形成可推荐合作状态；3=接受访谈、4=访谈合作，已达到推荐准入条件。未知脏值
    保持 ``None``，不默认为可推荐。
    """

    if value in (1, 2):
        return False
    if value in (3, 4):
        return True
    return None


def profile_verification_status(value: object) -> str:
    """由 ``doubtful_flag`` 生成 profile_summary 的核验状态。

    普通专家输出 ``source_summary``；存疑或未知值输出 ``rejected``。这样下游执行
    eligible-only 查询时可以排除存疑画像，同时数据库仍保留 Chunk 供审计和复核。
    这里的 rejected 是 Metadata 状态，不等同于批次中“完全没有 Chunk”的 rejected。
    """

    if value == 1:
        return "source_summary"
    if value == 2:
        return "rejected"
    # 未知值不能按正常专家处理，使用 rejected 阻止进入推荐/向量消费。
    return "rejected"


def normalize_abroad_flag(value: object) -> bool | None:
    """把 ``abroad_flag`` 映射为可空布尔值：1=True，2=False。

    结果只进入 profile_summary Metadata；其他值返回 ``None``，不把未知状态当成非海外。
    """

    if value == 1:
        return True
    if value == 2:
        return False
    return None


def employment_status(value: object) -> bool | None:
    """把任职表 ``is_employed`` 转为 ``is_current``：1=True，2=False。

    该权威标量决定任职正文显示“当前任职”还是“历史任职”，并用于处理结束月份；
    其他值返回 ``None``，不会根据日期自行猜测当前状态。
    """

    if value == 1:
        return True
    if value == 2:
        return False
    return None


def parse_employment_description(value: object) -> tuple[str | None, str | None]:
    """把一段工作描述确定性拆为职责和代表性成果。

    只有原文显式出现“工作职责：”或“代表性成果：”时才按标签拆分；没有标签时整段
    作为职责，成果保持空值。这样 ``chunk_builder.py`` 可以按语义切分超长任职内容，
    同时不会凭普通描述推断并不存在的成果。
    """

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

    只有带明确标签的区段才进入对应语义字段；无标签正文整体作为项目动作。该结果
    会生成“项目描述/项目工作内容/项目结果”等正文段。项目角色虽然会被解析以保证
    标签边界正确，但 V1.3 已取消该输出字段，因此调用方明确忽略返回的角色值。
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
    """按允许的中文标签提取跨行区段，并原样保留区段内业务事实。

    正则只识别调用方提供的标签，区段在下一个允许标签或文本结尾处结束。统一换行后
    再解析，可让 Windows/Unix 来源文本得到相同结果；未命中的文本不会被猜测拆分。
    """

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
    """从来源文本提取有关键词证据的代码，并去重排序后稳定输出。

    当前 V1.3 的 Employment/Project 构建没有调用本函数，行业和主题字段暂时为空；
    它只作为后续产品确认词典来源后的确定性实现基础，不能视为当前生效规则。
    """

    evidence = "\n".join(texts).lower()
    return tuple(sorted(code for code, words in rules.items() if any(word.lower() in evidence for word in words)))


def normalize_aggregate(
    aggregate: ExpertAggregate,
    *,
    data_version: str,
    build_batch_id: int,
) -> tuple[ProfileStage, list[EmploymentStage], list[ProjectStage]]:
    """把一位专家的四表一致性快照转换为三类 Stage。

    构成与效果
    ----------
    * 主档与全部有效教育经历聚合为一个 ``ProfileStage``；
    * 每条有效工作经历独立转换为一个 ``EmploymentStage``；
    * 每条有效项目经历独立转换为一个 ``ProjectStage``；
    * 所有允许字段在这里完成清洗、枚举映射和来源哈希，经历之间不会合并。

    使用位置
    --------
    ``pipeline.py::_process_expert`` 在 Source 读取之后调用本函数，随后把返回结果交给
    ``ChunkFactory.build_all``。因此这里决定 Chunk 使用什么标准化事实，但不负责文本
    模板、长度切分或 MySQL 写入。
    """

    profile = aggregate.profile
    # 先清洗和映射当前公司、岗位及背景，供 Profile Stage、正文和 Metadata 共用，
    # 避免不同下游字段各自清洗后产生不一致结果。
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
    # profile_allowed 是画像来源哈希的唯一输入白名单。它同时纳入教育事实，使学校或
    # 专业变化也能改变 source_hash；未列出的 PII 和备注字段不会影响索引增量判断。
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
    # ProfileStage 是主档与教育的标准化聚合结果：ChunkFactory 用公司、岗位和背景生成
    # 正文，其余状态、教育、年限等字段进入 profile_summary Metadata。
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

    # 一条源经历对应一个 Stage，保持 source_field_id 可回查；经历为空时自然得到空列表，
    # 属于合法业务情况，不会阻止 Profile 或其他类型 Chunk 生成。
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
    """把一条有效工作经历转换为可溯源的 ``EmploymentStage``。

    公司、岗位、日期、在职状态和描述只处理当前源记录，不与其他任职合并。普通描述
    作为职责；代表性成果只接受显式标签值。生成结果由 ``ChunkFactory`` 构造一个或
    多个 employment Chunk，``source_field_id`` 始终保持源经历主键。
    """

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
    # allowed 只包含实际参与该任职 Chunk/Metadata 的标准化事实。源记录的创建人、
    # 更新人、删除时间等审计列不进入哈希，避免无关管理字段触发正文重建。
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
    """把一条有效项目经历转换为可溯源的 ``ProjectStage``。

    项目名称、公司、日期和描述仅来自当前源记录。目标、动作、结果按照显式标签拆分；
    无标签描述作为项目动作。结果由 ``ChunkFactory`` 生成 project Chunk，源主键继续
    保存在 ``source_field_id`` 中，便于从最终 Chunk 回查原项目。
    """

    raw_company, norm_company, company_status = normalize_company(item.company_name)
    project_name = clean_text(item.project_title) or None
    _parsed_role, parsed_objective, parsed_actions, parsed_results = parse_project_description(
        item.work_description
    )
    objective = clean_text(parsed_objective) or None
    actions = clean_text(parsed_actions) or None
    results = clean_text(parsed_results) or None
    # 项目来源哈希只覆盖当前契约允许的标准化事实；已取消的 project_role 不参与输出，
    # 也不会因为角色文本变化而制造一个契约外的 Metadata 字段。
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
    """清洗列表值并按首次出现顺序去重，供教育 Metadata 使用。

    去空和去重可避免同一学校/专业重复写入 Metadata；保留源表稳定排序中的首次出现，
    可以让相同四表快照始终产生相同数组顺序和 metadata_hash。
    """

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return tuple(result)


def _latest_source_time(profile: object, educations: Iterable[object]) -> datetime | None:
    """取主档和教育记录的最新业务时间，作为画像来源版本时间。

    Profile 是多表聚合结果，因此只使用 experts.updated_at 会漏掉教育表更新。取所有
    参与记录的最新时间，可以让 ``source_version`` 在任一教育记录变化后同步前进；
    全部没有审计时间时返回 ``None``，不伪造运行时间。
    """

    candidates = [
        getattr(profile, "updated_at", None) or getattr(profile, "created_at", None),
        *(
            getattr(item, "updated_at", None) or getattr(item, "created_at", None)
            for item in educations
        ),
    ]
    known = [item for item in candidates if item is not None]
    return max(known) if known else None
