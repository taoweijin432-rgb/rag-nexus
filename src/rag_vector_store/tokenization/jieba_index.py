"""文档与查询共用的进程内 Jieba 精确分词和不可变索引身份。"""

from __future__ import annotations

import hashlib
import json
import platform
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from ..config import EXPERT_V16_EXTERNAL_FORMAL_COLLECTION_NAME
from .bm25_text import FILTER_PROFILE, filter_fingerprint, filter_tokens, normalize_bm25_text

JIEBA_VERSION = "0.42.1"
BM25_TEXT_MAX_BYTES = 32768


@dataclass(frozen=True)
class JiebaIndexConfig:
    dictionary_path: Path
    dictionary_version: str
    dictionary_sha256: str
    hmm: bool

    @classmethod
    def from_file(cls, path: Path, *, hmm: bool = False) -> JiebaIndexConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            Path(data["dictionary_path"]), data["dictionary_version"],
            data["dictionary_sha256"], hmm,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.hmm, bool):
            raise TypeError("HMM 必须是布尔值")
        if not self.dictionary_version.strip():
            raise ValueError("词典版本不能为空")
        if len(self.dictionary_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.dictionary_sha256
        ):
            raise ValueError("词典 SHA256 必须是 64 位小写十六进制")

    def identity(self) -> dict[str, object]:
        return {
            "tokenizer_version": f"jieba:{JIEBA_VERSION}",
            "mode": "accurate",
            "hmm": self.hmm,
            "dictionary_version": self.dictionary_version,
            "dictionary_sha256": self.dictionary_sha256,
            "dictionary_status": "Candidate",
            "normalization_version": FILTER_PROFILE,
            "filter_version": FILTER_PROFILE,
            "filter_sha256": filter_fingerprint(),
            "python_version": platform.python_version(),
            "unicode_version": unicodedata.unidata_version,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def formal_collection_name(self) -> str:
        if self.hmm:
            return EXPERT_V16_EXTERNAL_FORMAL_COLLECTION_NAME
        return self.fingerprinted_formal_collection_name

    @property
    def fingerprinted_formal_collection_name(self) -> str:
        return f"expert_candidates_v16_extjieba_h{int(self.hmm)}_{self.fingerprint[:12]}_contract"


class JiebaIndexTokenizer:
    """每个实例加载一次词典，不在请求中切换 HMM 或词库。"""

    def __init__(self, config: JiebaIndexConfig) -> None:
        try:
            raw = config.dictionary_path.read_bytes()
        except OSError as exc:
            raise ValueError("外部 Jieba 词典不可读取") from exc
        if hashlib.sha256(raw).hexdigest() != config.dictionary_sha256:
            raise ValueError("外部 Jieba 词典 SHA256 不匹配")
        # 精确字节快照只读取一次，防止检查与实际加载之间切换词典。
        terms = raw.decode("utf-8-sig").splitlines()
        if not any(term.strip() for term in terms):
            raise ValueError("外部 Jieba 词典不能为空")
        import jieba

        if jieba.__version__ != JIEBA_VERSION:
            raise ValueError("Jieba 版本与索引身份不一致")
        import io

        self._config = config
        self._tokenizer = jieba.Tokenizer()
        self._tokenizer.load_userdict(io.BytesIO(raw))
        self._lock = Lock()
        self.term_count = len({term.strip() for term in terms if term.strip()})

    @property
    def config(self) -> JiebaIndexConfig:
        return self._config

    def encode(self, text: str) -> str:
        normalized = normalize_bm25_text(text).text
        with self._lock:
            tokens = self._tokenizer.lcut(normalized, cut_all=False, HMM=self.config.hmm)
        result = " ".join(filter_tokens(tokens))
        if len(result.encode("utf-8")) > BM25_TEXT_MAX_BYTES:
            raise ValueError("chunk_text_bm25 超过 UTF-8 字节容量")
        return result
