"""正式 Contract 4.1.0 的单条与批量向量化、入库编排。

编排层所有身份统一使用 ``vector_doc_id``（幂等键），彻底移除对 ``chunk_id``
的依赖。Eligibility 与向量化统一由 ``ContractEmbeddingWorker`` 执行，本模块只负责
结果对齐、身份一致性与 Repository 写入，不重复实现 Validator / Eligibility。
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Protocol, TypeAlias

from expert_contracts.models import (
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
)

from ..core.ingestion import (
    DomainBatchIngestionResult,
    DomainIngestionError,
    DomainIngestionResult,
    DomainIngestionService,
)
from ..domains import ExpertContractAdapter
from ..models import MutationResult
from .contract_pipeline import (
    ContractEmbeddingWorker,
)

__all__ = [
    "ContractBatchIngestionResult",
    "ContractBatchIngestionService",
    "ContractIngestionError",
    "ContractIngestionResult",
    "ContractIngestionService",
    "ContractRepository",
]


def _expert_adapter(worker: object) -> ExpertContractAdapter:
    """从专家 Worker 取出其校验器，兼容编排层测试替身。"""

    validator = getattr(worker, "validator", None)
    eligibility = getattr(worker, "eligibility_validator", None)
    if validator is None:
        from .contract_pipeline import ContractValidator

        validator = ContractValidator()
    if eligibility is None:
        from .contract_pipeline import EmbeddingEligibilityValidator

        eligibility = EmbeddingEligibilityValidator()
    return ExpertContractAdapter(validator, eligibility)


class ContractRepository(Protocol):
    """正式 Contract 入库所需的最小 Repository 协议。

    调用时机：
        ``ContractIngestionService.ingest_one`` / ``ContractBatchIngestionService.ingest_batch``
        在 Worker 完成 Eligibility 与向量化后，用这里写入 Milvus。

    当前正式实现：
        ``ContractMilvusV15VectorRepository``（见 ``milvus/repository_v15.py``），
        继承通用基类 ``ContractMilvusRepositoryBase``。

    输入：
        Sequence[EmbeddedContractPreparedChunk] —— 已通过 Contract 校验、Eligibility
        并携带完整 embedding 的正式记录。

    输出：
        MutationResult —— affected_count 与 primary_keys；调用方要求两者与输入完全一致。
    """

    def upsert_batch(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> MutationResult:
        """Upsert 已完成向量化的正式 Contract 记录。"""

        ...


ContractIngestionError = DomainIngestionError
ContractIngestionResult = DomainIngestionResult
ContractBatchIngestionResult = DomainBatchIngestionResult

_ExpertDomainIngestion: TypeAlias = DomainIngestionService[
    ExpertChunkEmbeddingInput,
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
]


class ContractBatchIngestionService(_ExpertDomainIngestion):
    """专家 Contract 的批量兼容 Facade，行为完全委托通用 ingestion 核心。"""

    def __init__(
        self,
        embedding_worker: ContractEmbeddingWorker,
        repository: ContractRepository,
    ) -> None:
        super().__init__(
            embedding_worker,
            repository,
            _expert_adapter(embedding_worker),
        )


class ContractIngestionService(ContractBatchIngestionService):
    """编排一条正式 Contract 的校验、资格判断、向量化和 Upsert。

    调用时机：
        需要单条处理一条 ``ExpertChunkEmbeddingInput``（如单条重试、单条回放）时使用；
        批量场景请使用 ``ContractBatchIngestionService``（批量不通过循环本类实现）。

    输入：
        ``ExpertChunkEmbeddingInput`` —— 已完成结构解析但尚未经过消费端校验/向量化的记录。

    主要调用：
        1. ``ContractEmbeddingWorker.embed([record])``（校验 + Eligibility + Embedding）
        2. ``ContractRepository.upsert_batch([embedded])``（写入 Milvus）

    输出：
        ``ContractIngestionResult`` —— status=embedded（affected_count=1）或
        status=skipped（携带 skip_reason）；skip 是正常结果，其余错误直接向上抛出。

    职责边界：
        本类只做单条结果对齐与身份一致性校验，不实现 Validator / Eligibility /
        Embedding 本身；是否跳过由 Worker 的 Eligibility 决定。
    """

    def __init__(
        self,
        embedding_worker: ContractEmbeddingWorker,
        repository: ContractRepository,
    ) -> None:
        super().__init__(embedding_worker, repository)

    def ingest_one(
        self,
        record: ExpertChunkEmbeddingInput,
    ) -> ContractIngestionResult:
        """处理一条输入；skip 是正常结果，其它错误直接向上抛出。

        调用链：
            ingest_one(record)
            -> embedding_worker.embed([record])            # 校验 + Eligibility + 向量化
            -> DomainIngestionService                   # 结果对齐与一致性校验
            -> repository.upsert_batch([embedded])         # 单条写入 Milvus
            -> ContractIngestionResult

        分支：
            - skipped：Eligibility 未通过（is_active=false 或 rejected），直接返回
              skipped 结果，不调用 Repository。
            - embedded：Eligibility 通过且向量化成功，进入 Repository upsert。
            - fail-closed：Worker/Repository 结果身份或数量不一致时抛异常。
        """

        warnings.warn(
            "ContractIngestionService.ingest_one 已弃用；"
            "请使用 ContractBatchIngestionService.ingest_batch",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().ingest_one(record)
