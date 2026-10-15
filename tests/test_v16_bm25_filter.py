from __future__ import annotations

from rag_vector_store.tokenization.bm25_text import (
    FILTER_PROFILE,
    filter_tokens,
    normalize_bm25_text,
)


def test_filter_keeps_business_symbols_and_removes_structure() -> None:
    result = normalize_bm25_text("[岗位] C++/C#研发，R&D；2024-2025，50%，A/B，***")
    assert result.text == "岗位 C++/C#研发 R&D 2024-2025 50% A/B"
    assert result.removed_counts["punctuation"] > 0


def test_filter_nfkc_whitespace_and_markup() -> None:
    result = normalize_bm25_text("ＡＰＰ\u00a0开发\n<b>GMV</b>\u200b")
    assert result.text == "APP 开发 GMV"
    assert result.output_chars < result.input_chars


def test_filter_keeps_single_chinese_char_and_removes_pure_symbols() -> None:
    assert filter_tokens(("A", "的", "***", "50%")) == ("A", "的", "50%")


def test_filter_profile_is_stable() -> None:
    assert FILTER_PROFILE == "v16_bm25_text_v2"


def test_language_names_survive_document_and_token_filtering() -> None:
    for text in ("C", "C++", "C#", "c++", "c#"):
        assert normalize_bm25_text(text).text == text
        assert filter_tokens((text,)) == (text,)
    assert normalize_bm25_text("Ｃ＋＋、Ｃ＃，C++17开发").text == "C++ C# C++17开发"
    assert filter_tokens(("++", "#", "+", "C++", "C#")) == ("C++", "C#")
