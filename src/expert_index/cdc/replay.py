"""P7.2 Canal 固定 Fixture 重放边界。

本模块只做诊断级事件分类和 Fake Client 行为验证，不创建 CDC Task、不写 Index
MySQL，也不调用真实 Canal ACK。DELETE 的列值提取和业务影响范围留给 P7.3/P7.5。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .protocol import CanalBatch, CanalClientPort, CanalEntry

_ROW_DATA = "ROWDATA"
_TRANSACTION_ENTRIES = frozenset({"TRANSACTIONBEGIN", "TRANSACTIONEND"})
_DELETE_COLUMNS = {
    "experts": frozenset({"id"}),
    "experts_work_experience": frozenset({"id", "expert_id"}),
    "experts_project_experience": frozenset({"id", "expert_id"}),
    "experts_education_experience": frozenset({"id", "expert_id"}),
}


class CanalReplaySafetyError(ValueError):
    """表示固定重放消息不满足 fail-closed 结构约束。"""


@dataclass(frozen=True)
class ReplayDelete:
    """已通过结构校验的 DELETE 定位摘要，不包含被删行列值。"""

    source_table: str
    replay_key: str


@dataclass(frozen=True)
class ReplayResult:
    """一次 Fixture 重放的分类结果和是否必须回滚的决定。"""

    accepted_deletes: tuple[ReplayDelete, ...]
    ignored_count: int
    duplicate_count: int
    malformed_count: int
    rollback_required: bool


class FakeCanalClient(CanalClientPort):
    """按固定顺序提供 Canal batch，并记录 rollback/ack 调用。"""

    def __init__(self, batches: Iterable[CanalBatch]) -> None:
        """初始化不可变 batch 队列和诊断计数器。"""

        self._batches = tuple(batches)
        self._index = 0
        self.rollback_count = 0
        self.ack_count = 0

    def fetch_without_ack(self, batch_size: int) -> CanalBatch:
        """返回当前 batch；空队列返回 Canal 的空 batch 表达。"""

        if batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        if self._index >= len(self._batches):
            return CanalBatch(batch_id=-1, entries=())
        batch = self._batches[self._index]
        return CanalBatch(batch_id=batch.batch_id, entries=batch.entries[:batch_size])

    def rollback(self, batch_id: int) -> None:
        """记录回滚而不推进队列，模拟失败后的重复投递。"""

        if self._index >= len(self._batches) or self._batches[self._index].batch_id != batch_id:
            raise CanalReplaySafetyError(f"不能回滚未知 batch: {batch_id}")
        self.rollback_count += 1

    def ack(self, batch_id: int) -> None:
        """在 Fake Client 中推进队列，生产重放流程不会调用此方法。"""

        if self._index >= len(self._batches) or self._batches[self._index].batch_id != batch_id:
            raise CanalReplaySafetyError(f"不能确认未知 batch: {batch_id}")
        self.ack_count += 1
        self._index += 1


def replay_next_batch(
    client: CanalClientPort,
    *,
    batch_size: int = 100,
    seen_replay_keys: Iterable[str] = (),
) -> ReplayResult:
    """读取并分类一个 batch；畸形事件只回滚，不确认位点。"""

    batch = client.fetch_without_ack(batch_size)
    result = replay_batch(batch, seen_replay_keys=seen_replay_keys)
    if result.rollback_required and not batch.is_empty:
        client.rollback(batch.batch_id)
    return result


def replay_batch(
    batch: CanalBatch,
    *,
    seen_replay_keys: Iterable[str] = (),
) -> ReplayResult:
    """验证固定 batch 的结构，忽略 INSERT/UPDATE，仅收集 DELETE 定位摘要。"""

    if batch.is_empty:
        return ReplayResult((), 0, 0, 0, False)

    seen = set(seen_replay_keys)
    accepted: list[ReplayDelete] = []
    ignored_count = 0
    duplicate_count = 0
    malformed_count = 0

    for entry in batch.entries:
        if entry.entry_type in _TRANSACTION_ENTRIES:
            continue
        if entry.entry_type != _ROW_DATA:
            malformed_count += 1
            continue

        operation = (entry.operation_type or "").upper()
        if operation in {"INSERT", "UPDATE"}:
            ignored_count += 1
            continue
        if operation != "DELETE" or not _valid_delete_entry(entry):
            malformed_count += 1
            continue

        replay_key = _replay_key(entry)
        if replay_key in seen:
            duplicate_count += 1
            continue
        seen.add(replay_key)
        accepted.append(ReplayDelete(source_table=entry.source_table, replay_key=replay_key))

    return ReplayResult(
        accepted_deletes=tuple(accepted),
        ignored_count=ignored_count,
        duplicate_count=duplicate_count,
        malformed_count=malformed_count,
        rollback_required=malformed_count > 0,
    )


def _valid_delete_entry(entry: CanalEntry) -> bool:
    """检查 DELETE 的来源、Offset、Before Image 列和空 After Image。"""

    required_columns = _DELETE_COLUMNS.get(entry.source_table)
    if entry.source_schema != "expert" or required_columns is None:
        return False
    if not entry.logfile_name or not isinstance(entry.logfile_offset, int):
        return False
    if entry.logfile_offset < 0 or entry.after_columns:
        return False
    return required_columns.issubset(set(entry.before_columns))


def _replay_key(entry: CanalEntry) -> str:
    """生成仅供 Fixture 去重的稳定键，不宣称为生产 cdc_event_id。"""

    return f"{entry.logfile_name}:{entry.logfile_offset}:{entry.source_table}"


__all__ = [
    "CanalReplaySafetyError",
    "FakeCanalClient",
    "ReplayDelete",
    "ReplayResult",
    "replay_batch",
    "replay_next_batch",
]
