"""全量向量对账 Service：dry-run + 正向同步（只读对账，完成正向写入）。

本服务把上游传入的「完整 Chunk 快照」与当前 Milvus 现有向量做对账：

    - ``reconcile_all``：只读 dry-run，给出五类差异计数与少量 stale 采样，
      不执行任何写入 / 逻辑下线 / Embedding。
    - ``apply_positive_changes``：在 dry-run 基础上真实执行正向变化
      （missing / content_changed → upsert，metadata_changed → update_metadata），
      **绝不逻辑下线 stale**。
    - ``apply_full_reconciliation``：全库维护入口，在正向同步全部成功后
      重新读库，逻辑下线全部 active 多余实体，再做最终一致性验证
      （只比较当前有效集合：MySQL 当前有效 Chunk ↔ Milvus 中
      ``is_active=true`` 的 Entity；inactive 实体保留在 Collection 中，
      只作统计审计，永不判为多余数据或导致终验失败）。

三个入口共用同一套私有 ``_build_plan`` diff 算法，避免复制分类逻辑；
``apply_positive_changes`` 与 ``apply_full_reconciliation`` 共用以
``_apply_positive`` 封装的正向写入。

fail-closed：Contract 校验失败、Eligibility 异常、desired 重复
vector_doc_id、Milvus 状态不可信、list 异常、safety preflight 失败、
upsert / update_metadata 失败、结果中出现 failed、实际成功数量与计划不一致、
逻辑下线批失败、最终一致性校验失败，任一情况直接抛异常；尤其任何正向同步
失败都会**立即停止且不执行 stale 下线**。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pymilvus.exceptions import MilvusException

from expert_contracts.models import ContractPreparedChunk, ExpertChunkEmbeddingInput

from .._utils import safe_error_message
from ..chunking.contract_pipeline import (
    ContractValidationError,
    ContractValidator,
    EmbeddingEligibilityValidator,
)
from ..ingestion.errors import (
    VectorContractError,
    VectorInconsistentStateError,
    VectorIngestionError,
    VectorMilvusError,
)
from ..ingestion.vector_ingestion import VectorIngestionService
from ..milvus.repository import ContractMilvusRepositoryBase
from .models import FullVectorReconciliationResult

_STALE_SAMPLE_LIMIT = 20
_DELETE_BATCH_SIZE = 200

# Milvus 边界（Repository 扫描 / schema / guard / SDK）会抛出的底层异常类型。
# 这些在具体调用边界统一转换为 ``VectorMilvusError``；不包含 AttributeError /
# AssertionError 等程序 bug（避免掩盖真实代码缺陷）。
_MILVUS_BOUNDARY_TYPES = (ValueError, TypeError, RuntimeError, MilvusException)


@dataclass(frozen=True)
class _FullSyncPlan:
    """私有对账计划：dry-run 与正向同步共用的中间结果。

    完整 ``stale_ids`` 只保存在这里，**不进入公共 Result**（公共结果仅返回
    统计与有限 sample），供后续 stale 下线阶段使用。
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
    active_count: int
    inactive_count: int


class FullVectorReconciliationService:
    """对传入的完整 Chunk 快照做对账（dry-run 或正向同步）。"""

    def __init__(
        self,
        repository: ContractMilvusRepositoryBase,
        *,
        ingestion: VectorIngestionService | None = None,
        validator: ContractValidator | None = None,
        eligibility_validator: EmbeddingEligibilityValidator | None = None,
        delete_batch_size: int = _DELETE_BATCH_SIZE,
    ) -> None:
        if isinstance(delete_batch_size, bool) or not isinstance(
            delete_batch_size, int
        ) or delete_batch_size < 1:
            raise ValueError("delete_batch_size 必须是正整数")
        self.repository = repository
        self.ingestion = ingestion
        self.validator = validator or ContractValidator()
        self.eligibility_validator = (
            eligibility_validator or EmbeddingEligibilityValidator()
        )
        self.delete_batch_size = delete_batch_size

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def reconcile_all(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> FullVectorReconciliationResult:
        """只读 dry-run：计算五类差异，不执行任何写入 / 逻辑下线 / Embedding。

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
            - stale → 本方法只统计，绝不逻辑下线

        任何失败（含 safety preflight 失败）立即 fail-closed，且**绝不**因失败
        执行 stale 逻辑下线。

        Returns:
            FullVectorReconciliationResult：差异计数 + 实际执行的
              ``embedded_count`` / ``metadata_updated_count``。
        """

        if self.ingestion is None:
            raise VectorIngestionError(
                "apply_positive_changes 需要注入 VectorIngestionService（ingestion）"
            )

        plan = self._build_plan(chunks)
        embedded, metadata_updated = self._apply_positive(plan)
        return self._result_from_plan(
            plan, embedded=embedded, metadata_updated=metadata_updated
        )

    def apply_full_reconciliation(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> FullVectorReconciliationResult:
        """全库维护入口：正向同步全部成功后清理全部 stale 并做最终一致性验证。

        严格顺序（任一失败立即 fail-closed）：
            1. 构建可信的全库快照 plan；
            2. 原始输入 snapshot 本身为空 + Collection 非空时禁止清空整个
               Collection，直接 fail（not 基于 desired_ids 为空）；
            3. 任何 mutation 前 ``assert_destructive_maintenance_allowed``；
            4. 执行正向 change（missing / content_changed → upsert，
               metadata_changed → update_metadata）；
            5. 重新读取 Milvus，在 active 集合内计算 ``stale_after_write``
               （inactive 实体不参与，不会被重复下线）；
            6. 分批逻辑下线 stale；
            7. 再次读取 Milvus，验证 active ID 集 == desired ID 集，
               且逐条 content_hash / metadata_hash 一致。

        本方法是全 Collection 维护能力，并非日常专家同步路径（后者由
        ``expert_index`` 的 ``ExpertVectorSynchronizer`` 负责）。

        Returns:
            FullVectorReconciliationResult：差异计数 + 实际执行的
              ``embedded_count`` / ``metadata_updated_count`` / ``deleted_count``，
              ``final_existing_count`` 与 ``consistent``。
        """

        if self.ingestion is None:
            raise VectorIngestionError(
                "apply_full_reconciliation 需要注入 VectorIngestionService（ingestion）"
            )

        plan = self._build_plan(chunks)
        desired_ids = {record.vector_doc_id for record in plan.desired}

        # 原始输入 snapshot 本身为空 + 非空 Collection：禁止清空整个 Collection（fail-closed）。
        # 注意：基于原始输入是否为空，而非 desired_ids 是否为空——
        # chunks 非空但全部被 Eligibility 跳过不属于「空 snapshot」，走正常 reconciliation。
        if len(chunks) == 0 and plan.existing_count > 0:
            raise VectorInconsistentStateError(
                "apply_full_reconciliation 空快照 + 非空 Collection：禁止清空整个 "
                "Collection，已停止（不执行任何 mutation）"
            )

        # 4. 正向 change（内部先执行维护守卫）。
        embedded, metadata_updated = self._apply_positive(plan)

        # 5. 写全部成功后重新读取，在 active 集合内计算 stale_after_write
        #    （不使用写前 stale_ids；inactive 实体不参与、不重复下线）。
        active_after, _, _ = self._scan_existing()
        stale_after_write = [
            vector_doc_id
            for vector_doc_id in active_after
            if vector_doc_id not in desired_ids
        ]

        # 6. 分批逻辑下线 stale；某一批失败即停止后续批次（已完成下线不回滚）。
        disabled = self._disable_stale(stale_after_write)

        # 7. 下线后再读取并做最终一致性验证（只比较当前有效集合：
        #    active ID 集 == desired ID 集，且逐条 hash 一致；
        #    inactive 实体只作审计计数，永不导致失败）。
        final_active, _final_total, final_inactive_count = self._scan_existing()
        final_active_count = len(final_active)
        extra_active = sorted(set(final_active) - desired_ids)
        missing_active = sorted(desired_ids - set(final_active))
        hash_mismatch: list[str] = []
        if not extra_active and not missing_active:
            for record in plan.desired:
                current = final_active[record.vector_doc_id]
                if (
                    record.content_hash != current[0]
                    or record.metadata_hash != current[1]
                ):
                    hash_mismatch.append(record.vector_doc_id)
        if extra_active or missing_active or hash_mismatch:
            raise VectorInconsistentStateError(
                "apply_full_reconciliation 最终一致性验证失败："
                f"final_active({final_active_count}) != desired({len(desired_ids)})；"
                f"多余={len(extra_active)} 缺失={len(missing_active)} "
                f"hash不一致={len(hash_mismatch)}；"
                f"inactive={final_inactive_count}（不参与判定）"
            )

        base = self._result_from_plan(
            plan, embedded=embedded, metadata_updated=metadata_updated
        )
        return base.model_copy(
            update={
                "deleted_count": disabled,
                "final_existing_count": final_active_count,
                "active_mismatch_count": 0,
                "consistent": True,
            }
        )

    # ------------------------------------------------------------------
    # 私有执行（dry-run 与正向/全库同步共用）
    # ------------------------------------------------------------------

    def _apply_positive(
        self,
        plan: _FullSyncPlan,
    ) -> tuple[int, int]:
        """在任何写操作前过维护守卫并执行正向变化，返回 (embedded, metadata_updated)。

        任一写入失败立即抛异常（fail-closed），不执行 stale 逻辑下线。
        """

        # 任何写操作前先通过维护 Collection 安全守卫（fail-closed）。
        # Milvus 边界（未授权 / 不存在 / schema 不兼容 / SDK）统一转
        # VectorMilvusError，保留原始异常作为 cause。
        assert self.ingestion is not None
        try:
            self.repository.assert_destructive_maintenance_allowed()
        except _MILVUS_BOUNDARY_TYPES as error:
            raise VectorMilvusError(safe_error_message(error)) from error

        embedded = 0
        metadata_updated = 0

        upsert_chunks = list(plan.missing) + list(plan.content_changed)
        if upsert_chunks:
            # upsert_chunks / update_metadata 属于 VectorIngestionService 边界：
            # 其已抛出的 Vector*Error 原样保留，不重复包装。
            upsert_result = self.ingestion.upsert_chunks(upsert_chunks)
            if upsert_result.failed_count != 0:
                raise VectorInconsistentStateError(
                    "reconcile 正向 upsert 出现失败项，已停止（不执行 stale 下线）"
                )
            if upsert_result.embedded_count != len(upsert_chunks):
                raise VectorInconsistentStateError(
                    "reconcile 正向 upsert 实际成功数量与计划不一致："
                    f"期望 {len(upsert_chunks)}，实际 {upsert_result.embedded_count}"
                )
            embedded = upsert_result.embedded_count

        if plan.metadata_changed:
            metadata_result = self.ingestion.update_metadata(plan.metadata_changed)
            if metadata_result.failed_count != 0:
                raise VectorInconsistentStateError(
                    "reconcile update_metadata 出现失败项，已停止（不执行 stale 下线）"
                )
            if metadata_result.metadata_updated_count != len(plan.metadata_changed):
                raise VectorInconsistentStateError(
                    "reconcile update_metadata 实际成功数量与计划不一致："
                    f"期望 {len(plan.metadata_changed)}，"
                    f"实际 {metadata_result.metadata_updated_count}"
                )
            metadata_updated = metadata_result.metadata_updated_count

        return embedded, metadata_updated

    def _disable_stale(self, stale_ids: Sequence[str]) -> int:
        """分批逻辑下线 stale；返回累计下线数。某批失败即停止，不回滚已完成下线。"""

        assert self.ingestion is not None
        total_disabled = 0
        size = self.delete_batch_size
        for start in range(0, len(stale_ids), size):
            batch = stale_ids[start : start + size]
            result = self.ingestion.disable_vectors(batch)
            total_disabled += result.deleted_count
        return total_disabled

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
            # Validator 边界：Contract 校验失败 / 输入类型错误 -> VectorContractError。
            # 保留原始异常作为 cause。
            try:
                record = self.validator.validate(chunk)
            except (ContractValidationError, TypeError) as error:
                raise VectorContractError(safe_error_message(error)) from error
            reason = self.eligibility_validator.get_skip_reason(record)
            if reason is not None:
                skipped_count += 1
                continue
            vector_doc_id = record.vector_doc_id
            if vector_doc_id in desired_ids:
                raise VectorInconsistentStateError(
                    "reconcile 输入的 eligible 快照包含重复 vector_doc_id: "
                    f"{vector_doc_id!r}"
                )
            desired_ids.add(vector_doc_id)
            desired.append(record)

        # 2. 读取 Milvus 现有向量活动状态（只读；异常直接 fail-closed）。
        #    只有 is_active=true 的实体参与差异分类；inactive 实体只作统计。
        existing, total_entities, inactive_count = self._scan_existing()

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
            existing_count=total_entities,
            active_count=len(existing),
            inactive_count=inactive_count,
        )

    def _scan_existing(
        self,
    ) -> tuple[dict[str, tuple[str, str]], int, int]:
        """读取 Milvus 活动状态并分区，返回 (active, 全部实体数, inactive 数)。

        只有 ``is_active=true`` 的实体进入差异分类字典；``is_active=false``
        的实体（逻辑下线保留）只计入审计数量，永不判为多余数据。
        Milvus 边界（扫描失败 / SDK 异常）统一转 ``VectorMilvusError``；
        读取到的行字段不可信则 fail-closed，转 ``VectorInconsistentStateError``。
        """

        try:
            existing_rows = self.repository.list_all_vector_activity_states()
        except _MILVUS_BOUNDARY_TYPES as error:
            raise VectorMilvusError(safe_error_message(error)) from error

        active: dict[str, tuple[str, str]] = {}
        inactive_count = 0
        for row in existing_rows:
            vector_doc_id = row.get("vector_doc_id")
            content_hash = row.get("content_hash")
            metadata_hash = row.get("metadata_hash")
            is_active = row.get("is_active")
            if not (
                isinstance(vector_doc_id, str)
                and bool(vector_doc_id.strip())
                and isinstance(content_hash, str)
                and bool(content_hash.strip())
                and isinstance(metadata_hash, str)
                and bool(metadata_hash.strip())
            ):
                raise VectorInconsistentStateError(
                    "Milvus 现有向量状态字段不可信：vector_doc_id / content_hash / "
                    "metadata_hash 必须是非空字符串"
                )
            if not isinstance(is_active, bool):
                raise VectorInconsistentStateError(
                    "Milvus 现有向量状态字段不可信：is_active 必须是布尔值"
                )
            if is_active:
                active[vector_doc_id] = (content_hash, metadata_hash)
            else:
                inactive_count += 1
        return active, len(existing_rows), inactive_count

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
            active_entity_count=plan.active_count,
            inactive_entity_count=plan.inactive_count,
            missing_count=len(plan.missing),
            content_changed_count=len(plan.content_changed),
            metadata_changed_count=len(plan.metadata_changed),
            unchanged_count=len(plan.unchanged_ids),
            stale_count=len(plan.stale_ids),
            stale_sample=plan.stale_ids[:_STALE_SAMPLE_LIMIT],
            active_mismatch_count=len(plan.missing) + len(plan.stale_ids),
            embedded_count=embedded,
            metadata_updated_count=metadata_updated,
        )
