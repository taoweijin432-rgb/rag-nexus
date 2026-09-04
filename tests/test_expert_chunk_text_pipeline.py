from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from rag_vector_store import (
    FORMAL_EMBEDDING_DIMENSION,
    FORMAL_EMBEDDING_MODEL,
    FORMAL_EMBEDDING_MODEL_ALIAS,
    CompanyCloudEmbeddingModel,
)
from tests.support.expert_chunk_text_reader import (
    ExpertChunkTextTestReader,
    FlatTestChunk,
)
from tests.support.mysql_connection import database_config_from_test_env


@dataclass(frozen=True)
class FakeStoredChunk:
    """测试内存 Repository 保存的一行。"""

    vector_doc_id: str
    chunk_text: str
    embedding: list[float]
    metadata: dict[str, object]


class FakeRepository:
    """不连接 Milvus、只记录测试链写入结果的 Repository。"""

    def __init__(self) -> None:
        self.rows: list[FakeStoredChunk] = []

    def upsert_many(
        self,
        chunks: Sequence[FlatTestChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise AssertionError("Fake Repository 收到的 chunk 和 embedding 数量不一致")
        self.rows.extend(
            FakeStoredChunk(
                vector_doc_id=chunk.vector_doc_id,
                chunk_text=chunk.chunk_text,
                embedding=list(embedding),
                metadata=dict(chunk.metadata),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        )


class RecordingEmbeddingModel:
    """记录传给真实 Alibaba Bailian 模型的文本输入。"""

    def __init__(self, model: CompanyCloudEmbeddingModel) -> None:
        self.model = model
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self.model.dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        input_texts = list(texts)
        self.calls.append(input_texts)
        return self.model.embed(input_texts)


def _require_real_runtime_configuration() -> None:
    if os.environ.get("RUN_REAL_E2E", "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip("真实 E2E 默认关闭；设置 RUN_REAL_E2E=1 后显式运行")
    required = (
        "MYSQL_DATABASE",
        "MYSQL_USER",
        "EMBEDDING_API_URL",
        "EMBEDDING_API_KEY",
    )
    if any(not os.environ.get(name, "").strip() for name in required):
        pytest.skip("未配置本地 MySQL 或 Alibaba Bailian 真实验收环境")


def test_real_e2e_is_disabled_without_explicit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_REAL_E2E", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _require_real_runtime_configuration()


def test_expert_chunk_text_to_real_embedding_and_fake_repository() -> None:
    """验证临时测试表的五条数据只经 Flat DTO 进入真实 Embedding。"""

    _require_real_runtime_configuration()
    reader = ExpertChunkTextTestReader(database_config_from_test_env())
    chunks = reader.read()

    assert len(chunks) == 5
    assert Counter(chunk.metadata["chunk_type"] for chunk in chunks) == Counter(
        {"profile_summary": 1, "employment": 2, "project": 2}
    )
    assert len({chunk.vector_doc_id for chunk in chunks}) == 5
    assert all(chunk.vector_doc_id for chunk in chunks)

    excluded = {"id", "vector_doc_id", "chunk_text"}
    assert set(reader.last_columns) >= excluded
    expected_metadata_columns = set(reader.last_columns) - excluded
    for chunk in chunks:
        assert set(chunk.metadata) == expected_metadata_columns
        assert "id" not in chunk.metadata
        assert "vector_doc_id" not in chunk.metadata
        assert "chunk_text" not in chunk.metadata
        for field_name in ("school_names_raw", "major_names_raw"):
            if field_name in chunk.metadata:
                value = chunk.metadata[field_name]
                assert value is None or (
                    isinstance(value, list) and all(isinstance(item, str) for item in value)
                )

    real_model = CompanyCloudEmbeddingModel.from_env()
    assert real_model.settings.model == FORMAL_EMBEDDING_MODEL
    assert real_model.settings.model_alias == FORMAL_EMBEDDING_MODEL_ALIAS
    assert real_model.settings.expected_dimension == FORMAL_EMBEDDING_DIMENSION
    recording_model = RecordingEmbeddingModel(real_model)

    texts = [chunk.chunk_text for chunk in chunks]
    embeddings = recording_model.embed(texts)

    assert recording_model.calls == [texts]
    assert len(embeddings) == len(chunks) == 5
    assert all(len(vector) == FORMAL_EMBEDDING_DIMENSION for vector in embeddings)
    assert all(math.isfinite(value) for vector in embeddings for value in vector)

    repository = FakeRepository()
    repository.upsert_many(chunks, embeddings)

    assert len(repository.rows) == 5
    for stored, chunk, embedding in zip(repository.rows, chunks, embeddings, strict=True):
        assert stored.vector_doc_id == chunk.vector_doc_id
        assert stored.chunk_text == chunk.chunk_text
        assert stored.embedding == embedding
        assert stored.metadata == chunk.metadata
