"""构建流程内部使用的进度状态。"""

from __future__ import annotations

from dataclasses import dataclass

from ..vector_sync import ExpertVectorSyncResult

# 批次主表只保存有限数量的失败样本；完整机器终态由批次明细表记录。
MAX_BATCH_ERROR_SUMMARIES = 20


@dataclass
class _ExpertProgress:
    """记录一位专家在当前调用中已完成阶段的数量。

    ``_process_active_expert`` 在来源读取和 Chunk 构建完成后更新对应字段，使外层在
    成功或异常时都能记录已经实际发生的数量。该对象只存在于内存。
    """

    expert_id: int
    extracted_count: int = 0
    chunk_count: int = 0
    vector_sync: ExpertVectorSyncResult | None = None  # 同步成功返回后才设置。
