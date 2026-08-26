"""Canal P7.0/P7.1 协议边界。

领域层只依赖这些无 SDK 类型的模型和端口。真实 Offset、DELETE Before Image
和 ack 重投行为未确认前，生产契约保持 ``production_ready=False``，禁止调用确认接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

CANAL_SERVER_VERSION = "1.1.8"
CANAL_DESTINATION = "company_expert"
CANAL_SUBSCRIPTION = (
    r"expert\.(experts|experts_work_experience|experts_project_experience|"
    r"experts_education_experience)"
)


class CanalProtocolNotFrozenError(RuntimeError):
    """在真实 Canal 协议未冻结时拒绝执行生产 CDC 操作。"""


@dataclass(frozen=True)
class CanalEntry:
    """已脱敏的 Canal Entry 边界摘要，不保存行值或完整 Payload。"""

    entry_type: str
    source_schema: str
    source_table: str
    operation_type: str | None
    logfile_name: str | None
    logfile_offset: int | None
    server_id: int | None
    execute_time: int | None
    before_columns: tuple[str, ...] = ()
    after_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanalBatch:
    """一次 Canal 拉取结果的无 PII 表达。"""

    batch_id: int
    entries: tuple[CanalEntry, ...]

    @property
    def is_empty(self) -> bool:
        """判断本次拉取是否没有待确认事件。"""

        return self.batch_id == -1 or not self.entries


class CanalClientPort(Protocol):
    """Canal Client 的最小端口，隔离 Canal SDK 和领域编排。"""

    def fetch_without_ack(self, batch_size: int) -> CanalBatch:
        """读取一个 batch，但不改变 Canal 确认位点。"""

    def rollback(self, batch_id: int) -> None:
        """放弃当前 batch，使其可以按 Canal 语义重新投递。"""

    def ack(self, batch_id: int) -> None:
        """确认已完成的 batch；生产调用须通过协议冻结门禁。"""


@dataclass(frozen=True)
class CanalProtocolBaseline:
    """记录本机 Canal 已确认事实和仍待确认的生产协议项。"""

    server_version: str
    server_mode: str
    tcp_port: int
    destination: str
    subscription: str
    source_connected: bool
    probe_batch_id: int
    probe_entry_count: int
    ack_called: bool
    production_ready: bool
    unresolved: tuple[str, ...]

    def require_production_ready(self) -> None:
        """未完成真实协议核对时拒绝启用生产 CDC。"""

        if not self.production_ready:
            unresolved = ", ".join(self.unresolved) or "unknown_protocol"
            raise CanalProtocolNotFrozenError(
                f"Canal 生产协议尚未冻结: {unresolved}"
            )


__all__ = [
    "CANAL_DESTINATION",
    "CANAL_SERVER_VERSION",
    "CANAL_SUBSCRIPTION",
    "CanalBatch",
    "CanalClientPort",
    "CanalEntry",
    "CanalProtocolBaseline",
    "CanalProtocolNotFrozenError",
]
