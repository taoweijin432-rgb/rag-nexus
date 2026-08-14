"""正式 Contract 4.0.0 的单条与批量向量化、入库编排。

编排层所有身份统一使用 ``vector_doc_id``（幂等键），彻底移除对 ``chunk_id``
的依赖。Eligibility 与向量化统一由 ``ContractEmbeddingWorker`` 执行，本模块只负责
结果对齐、身份一致性与 Repository 写入，不重复实现 Validator / Eligibility。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from expert_contracts.models import (
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
)

from ..models import MutationResult
from .contract_pipeline import (
    ContractEmbeddingResult,
    ContractEmbeddingWorker,
    SkippedContractChunk,
)

__all__ = [
    "ContractBatchIngestionResult",
    "ContractBatchIngestionService",
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
    """Contract 编排结果不符合内部一致性约束。"""


@dataclass(frozen=True)
class ContractIngestionResult:
    """单条 Contract ingestion 的稳定结果摘要，以 ``vector_doc_id`` 定位。

    embedded 与 skipped 都携带 ``vector_doc_id``（skipped 指向对应输入主键），
    便于调用方按幂等键对齐结果。
    """

    status: Literal["embedded", "skipped"]
    vector_doc_id: str
    affected_count: int = 0
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        """确保 embedded 与 skipped 结果不会混合，且固定以 vector_doc_id 定位。"""

        if self.status not in ("embedded", "skipped"):
            raise ValueError("status 必须是 embedded 或 skipped")
        if not isinstance(self.vector_doc_id, str) or not self.vector_doc_id.strip():
            raise ValueError("vector_doc_id 必须是非空字符串")
        if (
            isinstance(self.affected_count, bool)
            or not isinstance(self.affected_count, int)
            or self.affected_count < 0
        ):
            raise ValueError("affected_count 必须是非负整数")

        if self.status == "embedded":
            if self.skip_reason is not None:
                raise ValueError("embedded 结果不能包含 skip_reason")
            if self.affected_count != 1:
                raise ValueError("单条 embedded 结果的 affected_count 必须是 1")
            return

        if self.affected_count != 0:
            raise ValueError("skipped 结果的 affected_count 必须是 0")
        if not isinstance(self.skip_reason, str) or not self.skip_reason.strip():
            raise ValueError("skipped 结果必须包含 skip_reason")


@dataclass(frozen=True)
class ContractBatchIngestionResult:
    """多条 Contract ingestion 的汇总结果。"""

    results: list[ContractIngestionResult]
    embedded_count: int
    skipped_count: int
    affected_count: int

    def __post_init__(self) -> None:
        """确保批量汇总与逐条状态及 Repository 结果一致。"""

        if any(not isinstance(item, ContractIngestionResult) for item in self.results):
            raise ValueError("results 必须只包含 ContractIngestionResult")
        for name, value in (
            ("embedded_count", self.embedded_count),
            ("skipped_count", self.skipped_count),
            ("affected_count", self.affected_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")

        embedded_count = sum(item.status == "embedded" for item in self.results)
        skipped_count = sum(item.status == "skipped" for item in self.results)
        if self.embedded_count != embedded_count:
            raise ValueError("embedded_count 与 results 不一致")
        if self.skipped_count != skipped_count:
            raise ValueError("skipped_count 与 results 不一致")
        if self.embedded_count + self.skipped_count != len(self.results):
            raise ValueError("批量汇总数量与 results 不一致")
        if self.affected_count != self.embedded_count:
            raise ValueError("affected_count 必须等于 embedded_count")

    @property
    def total_count(self) -> int:
        """返回本批输入总数。"""

        return len(self.results)

    @property
    def item_results(self) -> list[ContractIngestionResult]:
        """返回按输入顺序排列的逐条结果。"""

        return self.results


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
                status="skipped",
                vector_doc_id=record.vector_doc_id,
                skip_reason=skipped.reason,
            )

        if embedded is None:  # 由 _single_worker_result 保证，仅用于保护未来修改。
            raise ContractIngestionError("Worker 未返回 embedded 或 skipped 结果")

        # 正式 Milvus Repository 内部负责 Schema gate、MilvusRowMapper 和 upsert。
        mutation = self.repository.upsert_batch([embedded])
        self._validate_repository_result(embedded, mutation)
        return ContractIngestionResult(
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
            if embedded.vector_doc_id != record.vector_doc_id:
                raise ContractIngestionError("Worker 返回的 vector_doc_id 与输入不一致")
            return embedded, None

        skipped = result.skipped_chunks[0]
        if not isinstance(skipped, SkippedContractChunk):
            raise ContractIngestionError("Worker 返回了非法 skipped Contract")
        if skipped.vector_doc_id != record.vector_doc_id:
            raise ContractIngestionError(
                "Worker 返回的 skipped vector_doc_id 与输入不一致"
            )
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


class ContractBatchIngestionService:
    """编排多条正式 Contract 的校验、向量化和单次批量 Upsert。"""

    def __init__(
        self,
        embedding_worker: ContractEmbeddingWorker,
        repository: ContractRepository,
    ) -> None:
        self.embedding_worker = embedding_worker
        self.repository = repository

    def ingest_batch(
        self,
        records: Iterable[ExpertChunkEmbeddingInput],
    ) -> ContractBatchIngestionResult:
        """处理一批输入；任何校验、Embedding 或写入错误都使整批失败。

        保持真正批量：Worker 一次向量化全部 eligible 记录，再对全部 embedded
        做单次 ``Repository.upsert_batch``，不循环调用单条入口。
        """

        items = list(records)
        self._validate_input_records(items)
        embedding_result = self.embedding_worker.embed(items)
        embedded, skipped = self._validate_worker_result(items, embedding_result)

        affected_count = 0
        if embedded:
            mutation = self.repository.upsert_batch(embedded)
            self._validate_repository_result(embedded, mutation)
            affected_count = mutation.affected_count

        embedded_by_vid = {item.vector_doc_id: item for item in embedded}
        skipped_by_vid = {item.vector_doc_id: item for item in skipped}
        results: list[ContractIngestionResult] = []
        for item in items:
            embedded_item = embedded_by_vid.get(item.vector_doc_id)
            if embedded_item is not None:
                results.append(
                    ContractIngestionResult(
                        status="embedded",
                        vector_doc_id=item.vector_doc_id,
                        affected_count=1,
                    )
                )
                continue

            skipped_item = skipped_by_vid.get(item.vector_doc_id)
            if skipped_item is None:
                # _validate_worker_result 已保证全覆盖，此处仅防御未来修改。
                raise ContractIngestionError(
                    f"Worker 结果缺少输入 vector_doc_id: {item.vector_doc_id}"
                )
            results.append(
                ContractIngestionResult(
                    status="skipped",
                    vector_doc_id=item.vector_doc_id,
                    skip_reason=skipped_item.reason,
                )
            )

        return ContractBatchIngestionResult(
            results=results,
            embedded_count=len(embedded),
            skipped_count=len(skipped),
            affected_count=affected_count,
        )

    @staticmethod
    def _validate_input_records(
        records: Sequence[ExpertChunkEmbeddingInput],
    ) -> None:
        """在调用 Worker 前拒绝非 Contract 和重复批次主键（vector_doc_id）。"""

        if not all(isinstance(item, ExpertChunkEmbeddingInput) for item in records):
            raise TypeError("ingest_batch 只接受 ExpertChunkEmbeddingInput")

        vector_doc_ids = [item.vector_doc_id for item in records]
        if len(vector_doc_ids) != len(set(vector_doc_ids)):
            raise ContractIngestionError("一个批次不能包含重复 vector_doc_id")

    @staticmethod
    def _validate_worker_result(
        records: Sequence[ExpertChunkEmbeddingInput],
        result: ContractEmbeddingResult,
    ) -> tuple[
        list[EmbeddedContractPreparedChunk],
        list[SkippedContractChunk],
    ]:
        """确保 Worker 返回的结果完整覆盖输入且不发生身份错配。"""

        if not isinstance(result, ContractEmbeddingResult):
            raise ContractIngestionError("Worker 返回了非法 ContractEmbeddingResult")

        embedded = result.embedded_chunks
        skipped = result.skipped_chunks
        if len(embedded) + len(skipped) != len(records):
            raise ContractIngestionError("批量 ingestion 返回结果数量与输入不一致")
        if any(
            not isinstance(item, EmbeddedContractPreparedChunk) for item in embedded
        ):
            raise ContractIngestionError("Worker 返回了非法 embedded Contract")
        if any(not isinstance(item, SkippedContractChunk) for item in skipped):
            raise ContractIngestionError("Worker 返回了非法 skipped Contract")

        input_vids = {item.vector_doc_id for item in records}
        result_vids = [item.vector_doc_id for item in embedded] + [
            item.vector_doc_id for item in skipped
        ]
        if len(result_vids) != len(set(result_vids)):
            raise ContractIngestionError("Worker 返回了重复 vector_doc_id")
        if set(result_vids) != input_vids:
            missing = sorted(input_vids - set(result_vids))
            extra = sorted(set(result_vids) - input_vids)
            raise ContractIngestionError(
                f"Worker 返回的 vector_doc_id 与输入不一致 "
                f"(missing={missing}, extra={extra})"
            )
        return embedded, skipped

    @staticmethod
    def _validate_repository_result(
        embedded: Sequence[EmbeddedContractPreparedChunk],
        mutation: MutationResult,
    ) -> None:
        """确保一次批量 mutation 完整覆盖所有已向量化记录。"""

        if not isinstance(mutation, MutationResult):
            raise ContractIngestionError("Repository 返回了非法 MutationResult")
        if mutation.affected_count != len(embedded):
            raise ContractIngestionError(
                "批量 Contract upsert 的 affected_count 与 embedded 数量不一致"
            )
        expected_ids = [item.vector_doc_id for item in embedded]
        if mutation.primary_keys != expected_ids:
            raise ContractIngestionError(
                "Repository 返回的 primary_keys 与 embedded 记录不一致"
            )