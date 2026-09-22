"""Expert Index 跨模块共享常量。"""

from __future__ import annotations

# 所有全量、单条、Poll 和 CDC 批次共享同一数据库序列，不能各自推算下一个编号。
BATCH_ID_SEQUENCE_TABLE = "expert_index_batch_id_sequence"
BATCH_ID_SEQUENCE_NAME_COLUMN = "sequence_name"
BATCH_ID_SEQUENCE_VALUE_COLUMN = "current_value"
GLOBAL_BATCH_ID_SEQUENCE_NAME = "global"
MAX_UNSIGNED_BIGINT = (1 << 64) - 1


__all__ = [
    "BATCH_ID_SEQUENCE_NAME_COLUMN",
    "BATCH_ID_SEQUENCE_TABLE",
    "BATCH_ID_SEQUENCE_VALUE_COLUMN",
    "GLOBAL_BATCH_ID_SEQUENCE_NAME",
    "MAX_UNSIGNED_BIGINT",
]
