"""把一条专家原始记录转换成可向量化的 profile chunk。

处理原则：

1. 先清洗和解析原始字段，再做任何抽取；
2. 行业、职能和能力只有命中证据关键词时才输出；
3. 代表性业绩必须同时包含数字证据和结果性词语；
4. metadata 只保留检索、过滤和溯源所需字段，不放联系方式；
5. 输出内容必须可重复生成，便于使用 content_hash 做增量同步。

当前是规则抽取版本。未来接入 LLM 时，应让 LLM 输出结构化 JSON，并要求每个
标签携带原文 evidence；规则抽取可以继续作为低成本降级方案。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Any

CHUNK_SCHEMA_VERSION = "1.0"

# 规则集中放在模块顶部，便于业务人员审阅和迭代；规则顺序只在命中数相同时
# 影响排序，不影响是否命中。
INDUSTRY_RULES: dict[str, tuple[str, ...]] = {
    "人工智能/AI应用": (
        "人工智能", "ai ", "ai应用", "ai创作", "ai聊天", "智能体", "大模型", "aigc",
    ),
    "互联网/移动应用": ("互联网", "app", "应用商店", "移动应用"),
    "电子商务": ("电商", "直播带货", "gmv", "千川"),
    "二手奢侈品": ("二奢", "红布林", "奢侈品"),
    "社交与内容": ("社交产品", "互动小说", "阅读产品", "内容平台"),
}

FUNCTION_RULES: dict[str, tuple[str, ...]] = {
    "用户增长": ("用户增长", "用增", "获客", "增长操盘"),
    "广告投放": ("广告投放", "信息流", "kol", "sem", "cpm", "cpa", "cac", "星图", "巨量"),
    "产品运营": ("产品运营", "运营负责人", "运营体系"),
    "用户运营": ("用户运营", "生命周期管理", "精细化运营"),
    "内容营销": ("内容分发", "社媒", "达人合作", "种草"),
    "直播电商": ("直播", "主播", "gmv"),
    "私域运营": ("私域", "koc"),
    "商业化": ("商业化", "变现", "销售", "毛利"),
    "数据分析": ("数据分析", "看数据", "数据控", "指标"),
}

CAPABILITY_RULES: dict[str, tuple[str, ...]] = {
    "多渠道精准获客与投放优化": ("精准投放", "信息流", "kol", "应用商店", "sem", "千川"),
    "0-1用户增长体系搭建": ("0-1", "用增起盘", "用户增长"),
    "直播业务及主播孵化": ("主播孵化", "自孵化主播", "直播带货主播", "官播矩阵"),
    "达人/KOL合作": ("达人合作", "达人投放", "kol"),
    "全生命周期精细化运营": ("全生命周期", "精细化运营"),
    "公域与私域联动": ("私域承接", "引流到私域", "公域"),
    "数据驱动的增长优化": ("数据控", "看数据", "评估指标", "数据分析"),
    "AI应用产品增长": ("ai应用", "ai 创作", "ai聊天", "智能体社交", "互动小说"),
}

# 单独出现日期不代表业绩，因此抽取业绩时要求同时出现下面的结果词。
RESULT_TERMS = (
    "gmv", "cpm", "cpa", "cac", "转粉率", "点击率", "爆款率", "降低", "优化",
    "增长", "突破", "高于", "稳定", "孵化", "销售额", "投放", "获客", "利润", "毛利",
)

# 覆盖 20%、500万、10亿、50人、4-6个月等常见数字证据。
NUMBER_EVIDENCE_RE = re.compile(
    r"(?:\d+(?:\.\d+)?(?:\s*[-~至]\s*\d+(?:\.\d+)?)?\s*(?:%|万|亿|元|人|位|个?月|年|倍|家|场|天))",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    """将数据库中的任意值清洗为适合检索的单行文本。"""

    if value is None:
        return ""
    # 统一换行、制表符和连续空格，避免数据库中的排版差异影响抽取。
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_json_text_list(value: Any) -> list[str]:
    """解析项目经历等 JSON 数组字段，并对历史脏数据安全降级。

    数据库中可能同时存在真正的 JSON 数组、空字符串、普通字符串和 NULL。
    非法 JSON 不会让整批任务失败，而会作为一段普通履历文本继续处理。
    """

    if value in (None, "", []):
        return []

    parsed: Any = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [normalize_text(value)]

    if isinstance(parsed, list):
        return [text for item in parsed if (text := normalize_text(item))]
    return [normalize_text(parsed)]


def build_source_text(person: dict[str, Any]) -> str:
    """合并可用于抽取的字段，明确排除邮箱、电话等联系方式。"""

    ordinary_fields = (
        "company", "job_name", "background", "work_experience",
        "personal_intro", "self_appraisal", "certifications",
    )
    parts = [normalize_text(person.get(field)) for field in ordinary_fields]
    parts.extend(parse_json_text_list(person.get("project_experience")))
    return "\n".join(part for part in parts if part)


def _match_labels(source_text: str, rules: dict[str, tuple[str, ...]], limit: int) -> list[str]:
    """根据关键词命中次数排序，结果顺序稳定且最多返回 ``limit`` 项。"""

    lowered = source_text.lower()
    scored: list[tuple[int, int, str]] = []
    for rule_order, (label, keywords) in enumerate(rules.items()):
        matched_count = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if matched_count:
            # 第二个分值使用负的规则序号，保证同分时遵循词典定义顺序。
            scored.append((matched_count, -rule_order, label))

    scored.sort(reverse=True)
    return [label for _, _, label in scored[:limit]]


def _candidate_sentences(source_text: str) -> list[str]:
    """切分候选业绩句，去掉太短的项目标题和纯日期片段。"""

    raw_parts = re.split(r"[。！？；;\n]+|(?=第[一二三四五六七八九十]+阶段)", source_text)
    candidates: list[str] = []
    seen: set[str] = set()

    for raw_part in raw_parts:
        sentence = normalize_text(raw_part).strip("，,：:、- ")
        if len(sentence) < 12 or sentence in seen:
            continue
        seen.add(sentence)
        candidates.append(sentence)
    return candidates


def extract_verified_highlights(source_text: str, limit: int = 3) -> list[str]:
    """抽取带数字证据和结果词的原文业绩句。

    ``verified`` 在当前版本仅表示“原始记录存在直接文字证据”，不代表已经过
    第三方核验；不对原文数字做推导，也不生成原文中不存在的结论。
    """

    ranked: list[tuple[int, int, str]] = []
    for order, sentence in enumerate(_candidate_sentences(source_text)):
        lowered = sentence.lower()
        numbers = NUMBER_EVIDENCE_RE.findall(lowered)
        result_hits = sum(1 for term in RESULT_TERMS if term in lowered)
        if not numbers or not result_hits:
            continue

        score = min(len(numbers), 3) * 2 + min(result_hits, 3)
        clipped = sentence if len(sentence) <= 180 else sentence[:177].rstrip() + "…"
        ranked.append((score, -order, clipped))

    ranked.sort(reverse=True)
    return [sentence for _, _, sentence in ranked[:limit]]


def _display(items: list[str]) -> str:
    """将抽取列表渲染成模板文本；没有证据时显式标记。"""

    return "、".join(items) if items else "暂未提取"


def _json_safe(value: Any) -> Any:
    """将 MySQL 的 date/datetime 转为标准 JSON 字符串。"""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def build_expert_chunk(person: dict[str, Any]) -> dict[str, Any]:
    """构建一条包含 ``id``、``text`` 和 ``metadata`` 的标准 chunk。"""

    if person.get("id") is None:
        raise ValueError("专家记录缺少必需字段 id")

    source_text = build_source_text(person)
    industries = _match_labels(source_text, INDUSTRY_RULES, limit=4)
    functions = _match_labels(source_text, FUNCTION_RULES, limit=5)
    capabilities = _match_labels(source_text, CAPABILITY_RULES, limit=5)
    highlights = extract_verified_highlights(source_text, limit=3)

    company = normalize_text(person.get("company")) or "暂未提供"
    role = normalize_text(person.get("job_name")) or "暂未提供"
    text = "\n".join(
        (
            f"当前/最近任职：{company} · {role}",
            f"核心行业：{_display(industries)}",
            f"核心职能：{_display(functions)}",
            f"代表性能力：{_display(capabilities)}",
            f"代表性业绩：{_display(highlights)}",
        )
    )

    # chunk 文本不变时哈希不变，后续可以跳过重复 embedding。
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    expert_id = int(person["id"])
    metadata = {
        "expert_id": expert_id,
        "serial_number": normalize_text(person.get("serial_number")),
        "name": normalize_text(person.get("name")),
        "company": company,
        "role": role,
        "country": normalize_text(person.get("country")),
        "province": normalize_text(person.get("province")),
        "city": normalize_text(person.get("city")),
        "job_status": person.get("job_status"),
        "source_type": person.get("source_type"),
        "updated_at": _json_safe(person.get("updated_at")),
        "chunk_type": "expert_profile",
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "industries": industries,
        "functions": functions,
        "content_hash": content_hash,
    }

    return {
        # 稳定业务 ID 方便向量库 upsert，避免重复插入同一个专家画像。
        "id": f"expert:{expert_id}:profile:v{CHUNK_SCHEMA_VERSION}",
        "text": text,
        "metadata": metadata,
    }
