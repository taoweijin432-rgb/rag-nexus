"""单人物与批量构建命令的返回值模型。

本模块中的对象只携带批次标识、处理状态和安全计数，不包含来源正文、
Chunk Text 或向量内容。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..vector_sync import ExpertVectorSyncResult


@dataclass(frozen=True)
class VectorSyncSummary:
    """汇总当前命令中已完整返回的专家向量同步结果。"""

    expert_count: int = 0  # 已完成向量同步并返回可靠结果的专家数。
    current_count: int = 0  # 上游本次提交的当前 Chunk 总数。
    eligible_count: int = 0  # 满足激活和审核条件、应保留向量的 Chunk 总数。
    embedded_count: int = 0  # 正文新增或变化并实际重新向量化的 Chunk 总数。
    metadata_updated_count: int = 0  # 未重新向量化、只更新 Metadata 的 Chunk 总数。
    unchanged_count: int = 0  # 正文和 Metadata 均未变化、无需写入的 Chunk 总数。
    deleted_count: int = 0  # 当前快照已不存在并成功删除的旧向量总数。
    delete_not_found_count: int = 0  # 请求删除但下游未找到的向量总数。

    def add(self, result: ExpertVectorSyncResult) -> VectorSyncSummary:
        """返回累加一位专家同步结果后的新汇总，不修改已有对象。"""

        return VectorSyncSummary(
            expert_count=self.expert_count + 1,
            current_count=self.current_count + result.current_count,
            eligible_count=self.eligible_count + result.eligible_count,
            embedded_count=self.embedded_count + result.embedded_count,
            metadata_updated_count=(self.metadata_updated_count + result.metadata_updated_count),
            unchanged_count=self.unchanged_count + result.unchanged_count,
            deleted_count=self.deleted_count + result.deleted_count,
            delete_not_found_count=(self.delete_not_found_count + result.delete_not_found_count),
        )


@dataclass(frozen=True)
class BuildResult:
    """单专家任务完成后返回的批次标识和处理数量。

    该结果不包含源数据或 Chunk 正文。``vector_sync`` 是当前命令的同步摘要，且相同
    数量会累计写入批次主表。
    """

    batch_id: int  # expert_index_batch.id，用于回查本次运行。
    batch_no: str  # 单专家业务批次号；调用方不传时自动生成。
    expert_id: int  # 本次重建的 experts.id。
    extracted_count: int  # 1 条主档加工作、项目、教育来源记录数。
    stage_count: int  # 不再持久化 Stage，因此新任务恒为 0。
    chunk_count: int  # 成功同步到下游的最终 Chunk 行数。
    partial_chunk_count: int  # 当前流程不生成 partial Chunk，因此恒为 0。
    vector_sync: VectorSyncSummary  # 本次单专家命令的向量同步数量摘要。


@dataclass(frozen=True)
class BatchBuildResult:
    """批量任务完成后返回的扫描范围、最终游标和累计数量。

    该对象由新批次、续跑和失败重试入口共同返回，只包含任务状态和计数，不包含专家
    原始字段。返回的 ``vector_sync`` 只统计当前命令；批次主表保存跨续跑的累计值。

    ``rejected_count`` 表示存在但无可生成内容的专家数；``deactivated_count`` 表示
    主档已逻辑删除且历史 Chunk 已完成停用的专家数。两者都不是程序失败。
    """

    batch_id: int  # expert_index_batch.id。
    batch_no: str  # 全量业务批次号。
    status: str  # success/partial；批次级异常会抛出而不返回该对象。
    upper_bound_id: int  # 启动时固定的最大 experts.id。
    cursor_id: int  # 本轮内存中最后完成调度的专家 ID。
    candidate_count: int  # 实际遍历的专家候选数，包括逻辑删除候选。
    success_count: int  # 正常完成读取、生成、校验和向量同步的专家数。
    failed_count: int  # 本轮发生异常但未阻止后续处理的专家数。
    rejected_count: int  # 存在但没有可生成内容，并已停用旧 Chunk 的专家数。
    deactivated_count: int  # 主档逻辑删除，并已停用历史 Chunk 的专家数。
    extracted_count: int  # 所有已推进专家读取到的四表来源记录总数。
    stage_count: int  # 纯内存构建新增量为 0；续跑旧批次时保留其历史值。
    chunk_count: int  # 已生成的 Chunk 总数；同步失败专家可能已产生 Chunk。
    vector_sync: VectorSyncSummary  # 只统计本次命令实际处理并完成同步的专家。
