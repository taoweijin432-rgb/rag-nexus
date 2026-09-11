"""单人物构建批次的创建、执行和终态收口。"""

from __future__ import annotations

from .helpers import (
    _generate_single_batch_number,
    _validate_data_version,
    _vector_sync_persistence_fields,
)
from .results import BuildResult, VectorSyncSummary
from .state import _ExpertProgress


class _SingleBuildMixin:
    """实现带全局构建锁的单人物构建入口。

    使用方需要提供构建锁、批次 Repository、人物处理和向量同步方法。领域异常会先
    将批次收口为 failed，然后保持原异常类型继续抛出。
    """

    def build_single_expert(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_no: str | None = None,
    ) -> BuildResult:
        """在全局加工锁内创建 ``single`` 批次并完整重建一位专家。

        方法先校验专家 ID 和数据版本，再取得 single/full 共用的互斥锁。取得锁后创建
        批次并执行完整专家处理；已有任务持锁时立即失败，不会并发改写批次或向量。
        返回成功批次的数量摘要，处理异常由内部批次收口后继续向调用方抛出。

        Args:
            expert_id: 待重建的专家主键，必须是大于 0 的整数。
            data_version: 本次构建使用的数据版本，必须是非空字符串。
            batch_no: 可选业务批次号；省略时根据专家 ID 和当前时间生成。

        Returns:
            已完成批次的标识、处理数量和向量同步摘要。

        Raises:
            ValueError: 专家 ID 或数据版本不符合要求。
            Exception: 获取锁、来源读取、批次持久化或向量同步失败。
        """

        _validate_data_version(data_version)
        if expert_id <= 0:
            raise ValueError("expert_id 必须大于 0")
        with self.build_lock_repository.build_lock("single"):
            return self._build_single_expert_locked(
                expert_id,
                data_version=data_version,
                batch_no=batch_no,
            )

    def _build_single_expert_locked(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_no: str | None,
    ) -> BuildResult:
        """创建并收口一个已经取得互斥锁的单专家批次。

        成功时把处理数量写入批次主表并返回 ``BuildResult``；异常时把已完成数量和安全
        错误摘要写入 failed 批次，再原样抛出异常。领域处理统一委托给
        ``_process_active_expert``。

        Args:
            expert_id: 当前批次唯一处理的专家主键。
            data_version: 传给标准化和 Contract 构建链路的数据版本。
            batch_no: 调用方指定的业务批次号；为 None 时自动生成。

        Returns:
            单专家批次的最终安全统计。
        """

        actual_batch_no = batch_no or _generate_single_batch_number(expert_id)
        batch_id = self.batch_repository.start_batch(actual_batch_no, "single")
        progress = _ExpertProgress(expert_id=expert_id)
        vector_sync = VectorSyncSummary()
        try:
            if self._single_person_is_deleted(expert_id):
                # 数据集显式声明人物已删除时，不读取正文；空快照负责停用历史向量。
                progress.vector_sync = self._sync_expert_vector_snapshot(expert_id, [])
                outcome = "deactivated"
            else:
                outcome = self._process_active_expert(
                    expert_id,
                    data_version=data_version,
                    batch_id=batch_id,
                    progress=progress,
                )
            assert progress.vector_sync is not None
            vector_sync = vector_sync.add(progress.vector_sync)
            # 当前只生成完整 Chunk；返回结构中的 partial 数量固定为 0。
            partial_count = 0
            self.batch_repository.finish_batch(
                batch_id,
                status="success",
                extracted_count=progress.extracted_count,
                stage_count=0,
                chunk_count=progress.chunk_count,
                rejected_count=1 if outcome == "rejected" else 0,
                **_vector_sync_persistence_fields(vector_sync),
            )
            return BuildResult(
                batch_id=batch_id,
                batch_no=actual_batch_no,
                expert_id=expert_id,
                extracted_count=progress.extracted_count,
                stage_count=0,
                chunk_count=progress.chunk_count,
                partial_chunk_count=partial_count,
                vector_sync=vector_sync,
            )
        except Exception as exc:
            # 批次错误摘要只记录异常类型和截断消息，不保存专家正文或 Chunk Text。
            self.batch_repository.finish_batch(
                batch_id,
                status="failed",
                extracted_count=progress.extracted_count,
                stage_count=0,
                chunk_count=progress.chunk_count,
                rejected_count=1,
                error_summary={"error_type": type(exc).__name__, "message": str(exc)[:400]},
                **_vector_sync_persistence_fields(vector_sync),
            )
            raise
