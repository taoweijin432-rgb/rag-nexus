"""全量向量对账：dry-run + 正向同步 + 全库 stale 清理 + 生产工厂。

本包提供 ``FullVectorReconciliationService``，对上游传入的完整 Chunk 快照与
Milvus 当前向量做对账（只读 dry-run / 正向同步 / 全库 stale 清理），以及
``create_full_vector_reconciliation_service`` 从配置组装服务的生产工厂。
"""

from .factory import create_full_vector_reconciliation_service
from .full_reconciliation import FullVectorReconciliationService
from .models import FullVectorReconciliationResult

__all__ = [
    "FullVectorReconciliationResult",
    "FullVectorReconciliationService",
    "create_full_vector_reconciliation_service",
]