"""全量向量对账 Service：dry-run + 正向同步（只读对账，完成正向写入）。

本服务把上游传入的「完整 Chunk 快照」与当前 Milvus 现有向量做对账：

    - ``reconcile_all``：只读 dry-run，给出五类差异计数与少量 stale 采样，
      不执行任何写入 / 删除 / Embedding。
    - ``apply_positive_changes``：在 dry-run 基础上真实执行正向变化
      （missing / content_changed → upsert，metadata_changed → update_metadata），
      **绝不删除 stale**。

两个入口共用同一套私有 ``_build_plan`` diff 算法，避免复制两套分类逻辑。

fail-closed：Contract 校验失败、Eligibility 异常、desired 重复
vector_doc_id、Milvus 状态不可信、list 异常、safety preflight 失败、
upsert / update_metadata 失败、结果中出现 failed、实际成功数量与计划不一致，
任一情况直接抛异常，且绝不会因为正向同步失败而执行 stale delete。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from expert_contracts.models import ContractPreparedChunk, ExpertChunkEmbeddingInput

from ..chunking.contract_pipeline import (
    ContractValidator,
    EmbeddingEligibilityValidator,
)
from ..ingestion.vector_ingestion import VectorIngestionService
from ..milvus.repository import ContractMilvusVectorRepository
from .models import FullVectorReconciliationResult

_STALE_SAMPLE_LIMIT = 20


@dataclass(frozen=True)
class _FullSyncPlan:
    """私有对账计划：dry-run 与正向同步共用的中间结果。

    完整 ``stale_ids`` 只保存在这里，**不进入公共 Result**（公共结果仅返回
    统计与有限 sample），供后续 stale 删除阶段使用。
    """

    input_count: int
    skipped_count: int
    desired: list[ContractPreparedChunk]
    missing: list[ContractPreparedChunk]
    content_changed: list[ContractPreparedChunk]
    metadata_changed: list[ContractPreparedChunk]
    unchanged_ids: tuple[str, ...]
    stale_ids: tuple[str, ...]
    existing_count: int


class FullVectorReconciliationService:
    """对传入的完整 Chunk 快照做对账（dry-run 或正向同步）。"""

    def __init__(
        self,
        repository: ContractMilvusVectorRepository,
        *,
        ingestion: VectorIngestionService | None = None,
        validator: ContractValidator | None = None,
        eligibility_validator: EmbeddingEligibilityValidator | None = None,
    ) -> None:
        self.repository = repository
        self.ingestion = ingestion
        self.validator = validator or ContractValidator()
        self.eligibility_validator = (
            eligibility_validator or EmbeddingEligibilityValidator()
        )

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def reconcile_all(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> FullVectorReconciliationResult:
        """只读 dry-run：计算五类差异，不执行任何写入 / 删除 / Embedding。

        Returns:
            FullVectorReconciliationResult：差异计数与 stale 采样
              （embedded_count / metadata_updated_count 恒为 0）。
        """

        plan = self._build_plan(chunks)
        return self._result_from_plan(plan, embedded=0, metadata_updated=0)

    def apply_positive_changes(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> FullVectorReconciliationResult:
        """在 dry-run 基础上真实执行正向变化。

        在写任何东西之前先对当前 Collection 执行
        ``assert_destructive_maintenance_allowed``，确保本次 Full Reconciliation
        运行在明确授权的维护 Collection 上；只执行：

            - missing / content_changed → ``VectorIngestionService.upsert_chunks``
            - metadata_changed → ``VectorIngestionService.update_metadata``
            - unchanged → 不操作
            - stale → 本方法只统计，绝不删除

        任何失败（含 safety preflight 失败）立即 fail-closed，且**绝不**因失败
        执行 stale delete。

        Returns:
            FullVectorReconciliationResult：差异计数 + 实际执行的
              ``embedded_count`` / ``metadata_updated_count``。
        """

        if self.ingestion is None:
            raise RuntimeError(
                "apply_positive_changes 需要注入 VectorIngestionService（ingestion）"
            )

        plan = self._build_plan(chunks)

        # 任何写操作前先通过维护 Collection 安全守卫（fail-closed）。
        self.repository.assert_destructive_maintenance_allowed()

        embedded = 0
        metadata_updated = 0

        upsert_chunks = list(plan.missing) + list(plan.content_changed)
        if upsert_chunks:
            upsert_result = self.ingestion.upsert_chunks(upsert_chunks)
            if upsert_result.failed_count != 0:
                raise RuntimeError(
                    "reconcile 正向 upsert 出现失败项，已停止（不执行 stale delete）"
                )
            if upsert_result.embedded_count != len(upsert_chunks):
                raise RuntimeError(
                    "reconcile 正向 upsert 实际成功数量与计划不一致："
                    f"期望 {len(upsert_chunks)}，实际 {upsert_result.embedded_count}"
                )
            embedded = upsert_result.embedded_count

        if plan.metadata_changed:
            metadata_result = self.ingestion.update_metadata(plan.metadata_changed)
            if metadata_result.failed_count != 0:
                raise RuntimeError(
                    "reconcile update_metadata 出现失败项，已停止（不执行 stale delete）"
                )
            if metadata_result.metadata_updated_count != len(plan.metadata_changed):
                raise RuntimeError(
                    "reconcile update_metadata 实际成功数量与计划不一致："
                    f"期望 {len(plan.metadata_changed)}，"
                    f"实际 {metadata_result.metadata_updated_count}"
                )
            metadata_updated = metadata_result.metadata_updated_count

        return self._result_from_plan(plan, embedded=embedded, metadata_updated=metadata_updated)

    # ------------------------------------------------------------------
    # 私有 diff 算法（dry-run 与正向同步共用）
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> _FullSyncPlan:
        """校验 + Eligibility + 读取 existing + 五类分类，构造私有计划。

        任一步 fail-closed 即抛异常；本方法只读，不调用任何写 / Embedding。
        """

        # 1. 先完整校验 + Eligibility，构建可信的 desired state；
        #    任何 invalid / eligibility 异常 / desired 重复在此阶段 fail-closed，
        #    且尚未触碰 Milvus。
        desired: list[ContractPreparedChunk] = []
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
                    "reconcile 输入的 eligible 快照包含重复 vector_doc_id: "
                    f"{vector_doc_id!r}"
                )
            desired_ids.add(vector_doc_id)
            desired.append(record)

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
        missing: list[ContractPreparedChunk] = []
        content_changed: list[ContractPreparedChunk] = []
        metadata_changed: list[ContractPreparedChunk] = []
        unchanged: list[str] = []
        for record in desired:
            vector_doc_id = record.vector_doc_id
            current = existing.get(vector_doc_id)
            if current is None:
                missing.append(record)
            elif record.content_hash != current[0]:
                content_changed.append(record)
            elif record.metadata_hash != current[1]:
                metadata_changed.append(record)
            else:
                unchanged.append(vector_doc_id)

        stale_ids = [
            vector_doc_id
            for vector_doc_id in existing
            if vector_doc_id not in desired_ids
        ]

        return _FullSyncPlan(
            input_count=len(chunks),
            skipped_count=skipped_count,
            desired=desired,
            missing=missing,
            content_changed=content_changed,
            metadata_changed=metadata_changed,
            unchanged_ids=tuple(unchanged),
            stale_ids=tuple(stale_ids),
            existing_count=len(existing_rows),
        )

    @staticmethod
    def _result_from_plan(
        plan: _FullSyncPlan,
        *,
        embedded: int,
        metadata_updated: int,
    ) -> FullVectorReconciliationResult:
        """由私有计划映射为公共 Result（不带完整 stale_ids，仅有限 sample）。"""

        return FullVectorReconciliationResult(
            input_count=plan.input_count,
            eligible_count=len(plan.desired),
            skipped_count=plan.skipped_count,
            desired_count=len(plan.desired),
            existing_count=plan.existing_count,
            missing_count=len(plan.missing),
            content_changed_count=len(plan.content_changed),
            metadata_changed_count=len(plan.metadata_changed),
            unchanged_count=len(plan.unchanged_ids),
            stale_count=len(plan.stale_ids),
            stale_sample=plan.stale_ids[:_STALE_SAMPLE_LIMIT],
            embedded_count=embedded,
            metadata_updated_count=metadata_updated,
        )