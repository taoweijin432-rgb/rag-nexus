"""统一的 V16 BM25 文本过滤与规范化。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

FILTER_PROFILE = "v16_bm25_text_v2"
_ALLOWED_CONTEXT_SYMBOLS = frozenset(r"+#&/%.\-~")
_HTML_MARKUP = re.compile(r"<[^>]*>")
_MARKDOWN_MARKUP = re.compile(r"(^|\s)#{1,6}\s*|[*_`~]+")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_REPEAT_SEPARATOR = re.compile(r" {2,}")
_NUMBER_RANGE = re.compile(r"(?<=\d)[~\-](?=\d)")
_LANGUAGE_NAME = re.compile(r"(?<![A-Za-z0-9])[Cc](?:\+\+|#)")


@dataclass(frozen=True)
class FilterResult:
    text: str
    removed_counts: dict[str, int]
    input_chars: int
    output_chars: int


def _is_word_char(char: str) -> bool:
    return char.isalnum() or "\u4e00" <= char <= "\u9fff"


def _keep_symbol(text: str, index: int) -> bool:
    char = text[index]
    if char not in _ALLOWED_CONTEXT_SYMBOLS:
        return False
    before = text[index - 1] if index else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    if char == "~":
        return before.isdigit() and after.isdigit()
    if char == "%":
        return before.isdigit()
    if char == ".":
        return before.isalnum() and after.isalnum()
    return (
        bool(before)
        and bool(after)
        and (_is_word_char(before) or before in _ALLOWED_CONTEXT_SYMBOLS)
        and (_is_word_char(after) or after in _ALLOWED_CONTEXT_SYMBOLS)
        and (
            any(_is_word_char(candidate) for candidate in text[max(0, index - 3) : index])
            or any(_is_word_char(candidate) for candidate in text[index + 1 : index + 4])
        )
    )


def normalize_bm25_text(text: str) -> FilterResult:
    """规范化 BM25 专用文本；不改变原始 chunk_text。"""

    if not isinstance(text, str):
        raise TypeError("BM25 文本必须是字符串")
    original_length = len(text)
    text = unicodedata.normalize("NFKC", text)
    removed: dict[str, int] = {}

    def replace(pattern: re.Pattern[str], value: str, reason: str) -> None:
        nonlocal text
        count = len(pattern.findall(text))
        if count:
            removed[reason] = removed.get(reason, 0) + count
            text = pattern.sub(value, text)

    replace(_HTML_MARKUP, " ", "html_markup")
    replace(_MARKDOWN_MARKUP, " ", "markdown_markup")
    replace(_ZERO_WIDTH, "", "zero_width")
    replace(_CONTROL, " ", "control")

    # 语言名称的词尾也是词义的一部分；同一规则用于整段正文及独立 token。
    language_symbols = {
        index
        for match in _LANGUAGE_NAME.finditer(text)
        for index in range(match.start() + 1, match.end())
    }

    output: list[str] = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        if char.isspace():
            output.append(" ")
        elif category.startswith("C"):
            removed["format_character"] = removed.get("format_character", 0) + 1
            output.append(" ")
        elif category.startswith("P"):
            if index in language_symbols or _keep_symbol(text, index):
                output.append(char)
            else:
                removed["punctuation"] = removed.get("punctuation", 0) + 1
                output.append(" ")
        elif category.startswith("S"):
            if index in language_symbols or _keep_symbol(text, index):
                output.append(char)
            else:
                removed["symbol"] = removed.get("symbol", 0) + 1
                output.append(" ")
        else:
            output.append(char)

    normalized = "".join(output)
    normalized = _NUMBER_RANGE.sub("-", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    normalized = _REPEAT_SEPARATOR.sub(" ", normalized)
    if not normalized:
        removed["empty_result"] = 1
    return FilterResult(normalized, removed, original_length, len(normalized))


def filter_tokens(tokens: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """对外部分词器输出进行二次清理，保留有中文/英文/数字的 token。"""

    result: list[str] = []
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("token 必须是字符串")
        cleaned = normalize_bm25_text(token).text
        if cleaned and any(_is_word_char(char) for char in cleaned):
            result.extend(part for part in cleaned.split(" ") if part)
    return tuple(result)


def filter_fingerprint() -> str:
    payload: dict[str, Any] = {
        "profile": FILTER_PROFILE,
        "unicode": "NFKC",
        "symbols": "contextual_allowlist",
        "structure": "strip_markup_keep_text",
        "pii": "audit_only",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "FILTER_PROFILE",
    "FilterResult",
    "filter_fingerprint",
    "filter_tokens",
    "normalize_bm25_text",
]
