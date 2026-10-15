"""外部 Jieba 派生字段映射；不改变输入 Contract、正文或三个 Hash。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..tokenization.jieba_index import JiebaIndexTokenizer
from .contract_v16 import MilvusV16RowMapper


class MilvusV16ExternalRowMapper(MilvusV16RowMapper):
    def __init__(self, tokenizer: JiebaIndexTokenizer, adapter: Any | None = None) -> None:
        super().__init__(adapter)
        self.tokenizer = tokenizer

    def to_row(self, chunk: Any) -> dict[str, object]:
        row = super().to_row(chunk)
        row["chunk_text_bm25"] = self.tokenizer.encode(str(row["chunk_text"]))
        return row

    def to_rows(self, chunks: Sequence[Any]) -> list[dict[str, Any]]:
        rows = [self.to_row(chunk) for chunk in chunks]
        ids = [row["vector_doc_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("外部分词 mutation batch 不能包含重复 vector_doc_id")
        return rows

    def from_row(self, row: Mapping[str, object]) -> Any:
        chunk = super().from_row(row)
        if row.get("chunk_text_bm25") != self.tokenizer.encode(chunk.chunk_text):
            raise ValueError("已存 BM25 token 与绑定的分词配置不一致")
        return chunk
