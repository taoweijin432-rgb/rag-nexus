"""Resume Poll 四表初始 Checkpoint 的显式初始化入口。

新建控制表只有结构，没有安全水位。部署人员必须明确选择从历史起点补扫，或在来源写入
暂停的受控切换窗口中从当前四表上界开始。初始化器只插入不存在的 Checkpoint，不覆盖
任何已有水位。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from expert_index.models import SourceCheckpoint, SourceScanCursor
from expert_index.repositories import MySQLSourceStateRepository

from .config import ResumeSettings
from .dataset import RESUME_SPEC
from .source import MySQLResumeSourceAdapter

_BEGINNING_CURSOR = SourceScanCursor(
    updated_at=datetime(1970, 1, 1),  # noqa: DTZ001 - Source DATETIME 不做时区换算。
    record_id=0,
)
_BOOTSTRAP_POSITIONS = frozenset({"beginning", "current"})


class ResumeSourceStatePort(Protocol):
    """Checkpoint 初始化所需的最小 Resume Index Repository 接口。"""

    def get_source_checkpoint(self, source_table: str) -> SourceCheckpoint | None: ...

    def initialize_source_checkpoint(self, checkpoint: SourceCheckpoint) -> None: ...


class ResumeCheckpointSourcePort(Protocol):
    """读取当前四表上界所需的最小只读 Source 接口。"""

    def get_source_scan_upper_bound(
        self,
        source_table: str,
    ) -> SourceScanCursor | None: ...


@dataclass(frozen=True)
class ResumeCheckpointInitializationResult:
    """不包含游标值或业务数据的 Checkpoint 初始化摘要。"""

    position: str
    initialized_tables: tuple[str, ...]
    existing_tables: tuple[str, ...]


def initialize_resume_checkpoints(
    settings: ResumeSettings,
    *,
    position: str,
    source: ResumeCheckpointSourcePort | None = None,
    state_repository: ResumeSourceStatePort | None = None,
) -> ResumeCheckpointInitializationResult:
    """在 Resume Index Schema 中初始化四张来源表的安全水位。

    Parameters
    ----------
    position:
        ``beginning`` 从 ``1970-01-01, id=0`` 开始补扫全部历史，是默认安全选择；
        ``current`` 读取各表当前最大 ``(updated_at, id)``，仅适用于来源写入已暂停、
        全量已完成且保持暂停到初始化结束的受控切换，否则可能跳过全量期间发生的变化。
    source:
        只读 Resume Source；仅 ``current`` 模式会调用。
    state_repository:
        Resume Index Checkpoint Repository。已有任一表水位时只记录为 existing，绝不覆盖。
    """

    if not isinstance(settings, ResumeSettings):
        raise TypeError("settings 必须是 ResumeSettings")
    settings.validate_isolation_boundaries()
    if position not in _BOOTSTRAP_POSITIONS:
        raise ValueError("position 必须是 beginning 或 current")

    repository = state_repository or MySQLSourceStateRepository(settings.index_db)
    resolved_source = source
    initialized: list[str] = []
    existing: list[str] = []

    for source_table in RESUME_SPEC.source_tables:
        if repository.get_source_checkpoint(source_table) is not None:
            existing.append(source_table)
            continue

        cursor = _BEGINNING_CURSOR
        if position == "current":
            if resolved_source is None:
                resolved_source = MySQLResumeSourceAdapter(settings.source_db)
            cursor = resolved_source.get_source_scan_upper_bound(source_table) or _BEGINNING_CURSOR

        repository.initialize_source_checkpoint(
            SourceCheckpoint(
                source_table=source_table,
                last_updated_at=cursor.updated_at,
                last_id=cursor.record_id,
            )
        )
        initialized.append(source_table)

    return ResumeCheckpointInitializationResult(
        position=position,
        initialized_tables=tuple(initialized),
        existing_tables=tuple(existing),
    )


__all__ = [
    "ResumeCheckpointInitializationResult",
    "ResumeCheckpointSourcePort",
    "ResumeSourceStatePort",
    "initialize_resume_checkpoints",
]
