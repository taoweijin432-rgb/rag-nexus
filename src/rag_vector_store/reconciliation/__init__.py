"""全量向量 dry-run 对账（只读）。

本包目前只提供 ``FullVectorReconciliationService``，对上游传入的完整
Chunk 快照与 Milvus 当前向量做只读 diff，不执行任何写入、删除或 Embedding。
"""

from .full_reconciliation import FullVectorReconciliationService
from .models import FullVectorReconciliationResult

__all__ = [
    "FullVectorReconciliationResult",
    "FullVectorReconciliationService",
]