"""全量向量 dry-run 对账 Service（只读，不执行任何写入/删除/Embedding）。

本服务把上游传入的「完整 Chunk 快照」与当前 Milvus 现有向量做只读对账，
给出五类差异计数与少量 stale 采样。它是未来真实 Full Sync 的前置 dry-run，
因此：

    - 只使用 ``ContractValidator`` / ``EmbeddingEligibilityValidator`` 判断
      desired state；
    - 只用 ``list_all_vector_states()`` 读取 Milvus 现有向量，**不调用**
      Embedding、upsert、update_metadata、delete 或
      assert_destructive_maintenance_allowed；
    - fail-closed：Contract 校验失败、Eligibility 异常、desired 出现重复
      vector_doc_id、Milvus 状态不可信或 list 异常，任一情况直接抛异常，
      不返回一个看似完整但实际不可信的 diff。
"""

from __future__ import annotations

from collections.abc import Sequence

from expert_contracts.models import ContractPreparedChunk, ExpertChunkEmbeddingInput

from ..chunking.contract_pipeline import (
    ContractValidator,
    EmbeddingEligibilityValidator,
)
from ..milvus.repository import ContractMilvusVectorRepository
from .models import FullVectorReconciliationResult

_STALE_SAMPLE_LIMIT = 20


class FullVectorReconciliationService:
    """对传入的完整 Chunk 快照做只读 dry-run 对账。"""

    def __init__(
        self,
        repository: ContractMilvusVectorRepository,
        *,
        validator: ContractValidator | None = None,
        eligibility_validator: EmbeddingEligibilityValidator | None = None,
    ) -> None:
        self.repository = repository
        self.validator = validator or ContractValidator()
        self.eligibility_validator = (
            eligibility_validator or EmbeddingEligibilityValidator()
        )

    def reconcile_all(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> FullVectorReconciliationResult:
        """对给定完整 Chunk 快照执行只读 dry-run 对账。

        Args:
            chunks: 完整的 Chunk 快照（当前应存在的全部记录），不是增量变化。

        Returns:
            FullVectorReconciliationResult：五类差异计数与 stale 采样。

        Raises:
            任一对账步骤失败即抛异常（fail-closed），不执行任何删除。
        """

        # 1. 先完整校验 + Eligibility，构建可信的 desired state；
        #    任何 invalid / eligibility 异常 / desired 重复在此阶段 fail-closed，
        #    且尚未触碰 Milvus。
        eligible: list[ContractPreparedChunk] = []
        skipped_count = 0
        desired_ids: set[str] = set()

        for chunk in chunks:
            record = self.validator.validate(chunk)
            reason = self.eligibility_validator.get_skip_reason(record)
            if reason is not None:
                skipped_count += 1
                continue
            vector_doc_id = record.vector_doc_id
            if vector_doc_id in desired_ids:
                raise ValueError(
                    "reconcile_all 输入的 eligible 快照包含重复 vector_doc_id: "
                    f"{vector_doc_id!r}"
                )
            desired_ids.add(vector_doc_id)
            eligible.append(record)

        # 2. 读取 Milvus 现有向量状态（只读；异常直接 fail-closed）。
        existing_rows = self.repository.list_all_vector_states()

        existing: dict[str, tuple[str, str]] = {}
        for row in existing_rows:
            vector_doc_id = row.get("vector_doc_id")
            content_hash = row.get("content_hash")
            metadata_hash = row.get("metadata_hash")
            if not (
                isinstance(vector_doc_id, str)
                and bool(vector_doc_id.strip())
                and isinstance(content_hash, str)
                and bool(content_hash.strip())
                and isinstance(metadata_hash, str)
                and bool(metadata_hash.strip())
            ):
                raise RuntimeError(
                    "Milvus 现有向量状态字段不可信：vector_doc_id / content_hash / "
                    "metadata_hash 必须是非空字符串"
                )
            existing[vector_doc_id] = (content_hash, metadata_hash)

        # 3. 五类差异分类。
        missing: list[str] = []
        content_changed: list[str] = []
        metadata_changed: list[str] = []
        unchanged: list[str] = []
        for record in eligible:
            vector_doc_id = record.vector_doc_id
            current = existing.get(vector_doc_id)
            if current is None:
                missing.append(vector_doc_id)
            elif record.content_hash != current[0]:
                content_changed.append(vector_doc_id)
            elif record.metadata_hash != current[1]:
                metadata_changed.append(vector_doc_id)
            else:
                unchanged.append(vector_doc_id)

        stale_ids = [
            vector_doc_id
            for vector_doc_id in existing
            if vector_doc_id not in desired_ids
        ]

        return FullVectorReconciliationResult(
            input_count=len(chunks),
            eligible_count=len(eligible),
            skipped_count=skipped_count,
            desired_count=len(eligible),
            existing_count=len(existing_rows),
            missing_count=len(missing),
            content_changed_count=len(content_changed),
            metadata_changed_count=len(metadata_changed),
            unchanged_count=len(unchanged),
            stale_count=len(stale_ids),
            stale_sample=tuple(stale_ids[:_STALE_SAMPLE_LIMIT]),
        )