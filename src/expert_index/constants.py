"""Expert Index 跨模块共享常量。"""

from __future__ import annotations

# Poll/全量批次继续使用 MySQL 自增低位空间；CDC 使用无符号 BIGINT 高位空间，
# 避免未写入 expert_index_batch 的 CDC 本地批次号与后续 Poll 自增 ID 撞号。
CDC_LOCAL_BATCH_ID_BASE = 1 << 63
MAX_UNSIGNED_BIGINT = (1 << 64) - 1


__all__ = ["CDC_LOCAL_BATCH_ID_BASE", "MAX_UNSIGNED_BIGINT"]
