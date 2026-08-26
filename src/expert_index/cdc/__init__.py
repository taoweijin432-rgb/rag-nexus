"""Canal CDC 边界模型与客户端端口。

本包只定义上游可依赖的协议边界，不依赖 Canal SDK，也不执行消费或确认。
"""

from .protocol import (
    CANAL_DESTINATION,
    CANAL_SUBSCRIPTION,
    CanalBatch,
    CanalClientPort,
    CanalEntry,
    CanalProtocolBaseline,
    CanalProtocolNotFrozenError,
)
from .replay import (
    CanalReplaySafetyError,
    FakeCanalClient,
    ReplayDelete,
    ReplayResult,
    replay_batch,
    replay_next_batch,
)

__all__ = [
    "CANAL_DESTINATION",
    "CANAL_SUBSCRIPTION",
    "CanalBatch",
    "CanalClientPort",
    "CanalEntry",
    "CanalProtocolBaseline",
    "CanalProtocolNotFrozenError",
    "CanalReplaySafetyError",
    "FakeCanalClient",
    "ReplayDelete",
    "ReplayResult",
    "replay_batch",
    "replay_next_batch",
]
