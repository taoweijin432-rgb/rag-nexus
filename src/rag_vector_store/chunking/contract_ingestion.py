"""正式 Contract 的单条向量化与入库编排。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from ..models import MutationResult
from .contract_pipeline import (
    ContractEmbeddingResult,
    ContractEmbeddingWorker,
    SkippedContractChunk,
)
from .input_contract import EmbeddedContractPreparedChunk, ExpertChunkEmbeddingInput

__all__ = [
    "ContractIngestionError",
    "ContractIngestionResult",
    "ContractIngestionService",
    "ContractRepository",
]


class ContractRepository(Protocol):
    """正式 Contract 入库所需的最小 Repository 协议。"""

    def upsert_batch(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> MutationResult:
        """Upsert 已完成向量化的正式 Contract 记录。"""

        ...


class ContractIngestionError(RuntimeError):
    """单条 Contract 编排结果不符合内部一致性约束。"""


@dataclass(frozen=True)
class ContractIngestionResult:
    """单条 Contract ingestion 的稳定结果摘要。"""

    chunk_id: str
    status: Literal["embedded", "skipped"]
    vector_doc_id: str | None = None
    affected_count: int = 0
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        """确保 embedded 与 skipped 结果不会混合。"""

        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id 必须是非空字符串")
        if self.status not in ("embedded", "skipped"):
            raise ValueError("status 必须是 embedded 或 skipped")
        if (
            isinstance(self.affected_count, bool)
            or not isinstance(self.affected_count, int)
            or self.affected_count < 0
        ):
            raise ValueError("affected_count 必须是非负整数")

        if self.status == "embedded":
            if not isinstance(self.vector_doc_id, str) or not self.vector_doc_id.strip():
                raise ValueError("embedded 结果必须包含 vector_doc_id")
            if self.skip_reason is not None:
                raise ValueError("embedded 结果不能包含 skip_reason")
            if self.affected_count != 1:
                raise ValueError("单条 embedded 结果的 affected_count 必须是 1")
            return

        if self.vector_doc_id is not None:
            raise ValueError("skipped 结果不能包含 vector_doc_id")
        if self.affected_count != 0:
            raise ValueError("skipped 结果的 affected_count 必须是 0")
        if not isinstance(self.skip_reason, str) or not self.skip_reason.strip():
            raise ValueError("skipped 结果必须包含 skip_reason")


class ContractIngestionService:
    """编排一条正式 Contract 的校验、资格判断、向量化和 Upsert。"""

    def __init__(
        self,
        embedding_worker: ContractEmbeddingWorker,
        repository: ContractRepository,
    ) -> None:
        self.embedding_worker = embedding_worker
        self.repository = repository

    def ingest_one(
        self,
        record: ExpertChunkEmbeddingInput,
    ) -> ContractIngestionResult:
        """处理一条输入；skip 是正常结果，其它错误直接向上抛出。"""

        if not isinstance(record, ExpertChunkEmbeddingInput):
            raise TypeError("ingest_one 只接受 ExpertChunkEmbeddingInput")

        # Validator、Eligibility 和 Embedding 统一由 Worker 执行，避免这里重复调用。
        embedding_result = self.embedding_worker.embed([record])
        embedded, skipped = self._single_worker_result(record, embedding_result)
        if skipped is not None:
            return ContractIngestionResult(
                chunk_id=record.chunk_id,
                status="skipped",
                skip_reason=skipped.reason,
            )

        if embedded is None:  # 由 _single_worker_result 保证，仅用于保护未来修改。
            raise ContractIngestionError("Worker 未返回 embedded 或 skipped 结果")

        # 正式 Milvus Repository 内部负责 Schema gate、MilvusRowMapper 和 upsert。
        mutation = self.repository.upsert_batch([embedded])
        self._validate_repository_result(embedded, mutation)
        return ContractIngestionResult(
            chunk_id=record.chunk_id,
            status="embedded",
            vector_doc_id=embedded.vector_doc_id,
            affected_count=mutation.affected_count,
        )

    @staticmethod
    def _single_worker_result(
        record: ExpertChunkEmbeddingInput,
        result: ContractEmbeddingResult,
    ) -> tuple[EmbeddedContractPreparedChunk | None, SkippedContractChunk | None]:
        """拒绝 Worker 的空结果、多结果或记录身份错配。"""

        if not isinstance(result, ContractEmbeddingResult):
            raise ContractIngestionError("Worker 返回了非法 ContractEmbeddingResult")

        embedded_count = len(result.embedded_chunks)
        skipped_count = len(result.skipped_chunks)
        if embedded_count + skipped_count != 1:
            raise ContractIngestionError(
                "单条 ingestion 必须得到且只能得到一个 embedded 或 skipped 结果"
            )

        if embedded_count == 1:
            embedded = result.embedded_chunks[0]
            if not isinstance(embedded, EmbeddedContractPreparedChunk):
                raise ContractIngestionError("Worker 返回了非法 embedded Contract")
            if embedded.chunk_id != record.chunk_id:
                raise ContractIngestionError("Worker 返回的 chunk_id 与输入不一致")
            return embedded, None

        skipped = result.skipped_chunks[0]
        if not isinstance(skipped, SkippedContractChunk):
            raise ContractIngestionError("Worker 返回了非法 skipped Contract")
        if skipped.chunk_id != record.chunk_id:
            raise ContractIngestionError("Worker 返回的 skipped chunk_id 与输入不一致")
        return None, skipped

    @staticmethod
    def _validate_repository_result(
        embedded: EmbeddedContractPreparedChunk,
        mutation: MutationResult,
    ) -> None:
        """确保 Repository 的单条 mutation 结果与向量主键一致。"""

        if not isinstance(mutation, MutationResult):
            raise ContractIngestionError("Repository 返回了非法 MutationResult")
        if mutation.affected_count != 1:
            raise ContractIngestionError("单条 Contract upsert 的 affected_count 必须是 1")
        if mutation.primary_keys != [embedded.vector_doc_id]:
            raise ContractIngestionError("Repository 返回的 primary_keys 与 vector_doc_id 不一致")
