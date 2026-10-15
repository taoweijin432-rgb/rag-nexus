"""正式 Expert V16 Analyzer 与词典身份；只读本地文件，不依赖 POC。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import MilvusSettings


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class V16AnalyzerConfig:
    """构造期绑定不可变快照，日志中不展开词条或文件路径。"""

    profile: str
    _params_json: str = field(repr=False)
    dictionary_version: str | None = None
    dictionary_sha256: str | None = None
    term_count: int = 0

    @property
    def params(self) -> dict[str, Any]:
        # 每次返回新对象，Schema/SDK 的修改不会污染校验所用快照。
        return json.loads(self._params_json)

    def summary(self) -> dict[str, Any]:
        """可放入报告/断点指纹的脱敏身份，不包含完整词典。"""

        identity = {
            "profile": self.profile,
            "dictionary_version": self.dictionary_version,
            "dictionary_sha256": self.dictionary_sha256,
            "term_count": self.term_count,
            "analyzer_sha256": hashlib.sha256(self._params_json.encode("utf-8")).hexdigest(),
        }
        if self.profile == "jieba_targeted":
            identity.update(
                {
                    "mode": "exact",
                    "hmm": False,
                    "filter": ["cnalphanumonly"],
                    "dictionary_delivery": "inline",
                    "analyzer_json_bytes": len(self._params_json.encode("utf-8")),
                }
            )
        return identity


def resolve_v16_analyzer(settings: MilvusSettings) -> V16AnalyzerConfig:
    """验证固定词典版本/文件 SHA256；失败不回退到 Chinese 或默认 Jieba。"""

    if settings.expert_v16_analyzer == "chinese":
        return V16AnalyzerConfig("chinese", _canonical_json({"type": "chinese"}))

    # model_copy(update=...) 不运行 Pydantic 校验，因此边界仍须检查必要身份。
    path = settings.expert_v16_dictionary_path
    version = settings.expert_v16_dictionary_version
    expected_sha256 = settings.expert_v16_dictionary_sha256
    if not path or not version or not expected_sha256:
        raise ValueError("Jieba 大词典必须配置本地路径、版本及 SHA256")
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("Jieba 大词典文件不可读取") from exc
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Jieba 大词典 SHA256 与配置不一致")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Jieba 大词典必须为 UTF-8 文本") from exc

    terms: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        term = line.strip()
        if not term:
            continue
        if any(char.isspace() for char in term) or term == "_default_":
            raise ValueError(f"Jieba 大词典第 {line_number} 行不是独立词条")
        if term not in seen:
            seen.add(term)
            terms.append(term)
    if not terms:
        raise ValueError("Jieba 大词典没有有效词条")

    params = {
        "tokenizer": {
            "type": "jieba",
            "dict": ["_default_", *terms],
            "mode": "exact",
            "hmm": False,
        },
        "filter": ["cnalphanumonly"],
    }
    return V16AnalyzerConfig(
        profile="jieba_targeted",
        _params_json=_canonical_json(params),
        dictionary_version=version,
        dictionary_sha256=actual_sha256,
        term_count=len(terms),
    )
