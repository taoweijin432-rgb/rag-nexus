"""Canal Client 适配端口和四表统一变更 Parser。

适配器负责隔离 Canal SDK；Parser 只接受已转换的领域模型，校验来源、事务 Entry、Offset
和四表定位字段。行值只在 Parser 边界提取主键，绝不进入任务或持久化模型。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .protocol import (
    CANAL_DESTINATION,
    CANAL_SUBSCRIPTION,
    CanalBatch,
    CanalClientPort,
    CanalEntry,
)

_ROW_DATA = "ROWDATA"
_LOCATOR_COLUMNS = {
    "experts": frozenset({"id"}),
    "experts_work_experience": frozenset({"id", "expert_id"}),
    "experts_project_experience": frozenset({"id", "expert_id"}),
    "experts_education_experience": frozenset({"id", "expert_id"}),
}


class CanalParserError(ValueError):
    """表示事件违反严格 CDC 输入契约。"""


class CanalConnectorPort(Protocol):
    """Canal SDK 连接器所需的最小方法集合。"""

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def subscribe(self, subscription: str) -> None: ...

    def get_without_ack(self, batch_size: int) -> object: ...

    def rollback(self, batch_id: int) -> None: ...

    def ack(self, batch_id: int) -> None: ...


CanalBatchDecoder = Callable[[object, str], CanalBatch]


@dataclass
class CanalClientAdapter(CanalClientPort):
    """将 SDK 连接器转换为领域层的最小 Canal Client 端口。"""

    connector: CanalConnectorPort
    decoder: CanalBatchDecoder
    destination: str = CANAL_DESTINATION
    subscription: str = CANAL_SUBSCRIPTION
    _connected: bool = field(default=False, init=False)

    def connect(self) -> None:
        """连接 Canal 并注册冻结的订阅表达式。"""

        self.connector.connect()
        if self.subscription:
            self.connector.subscribe(self.subscription)
        self._connected = True

    def disconnect(self) -> None:
        """断开底层连接并清除连接状态。"""

        self.connector.disconnect()
        self._connected = False

    def fetch_without_ack(self, batch_size: int) -> CanalBatch:
        """读取并解码一个 batch，禁止在适配器内自动确认。"""

        if not self._connected:
            raise CanalParserError("Canal Client 尚未连接")
        if batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        batch = self.decoder(self.connector.get_without_ack(batch_size), self.destination)
        if not isinstance(batch, CanalBatch):
            raise CanalParserError("Canal decoder 未返回 CanalBatch")
        if batch.destination != self.destination:
            raise CanalParserError("Canal destination 与配置不一致")
        return batch

    def rollback(self, batch_id: int) -> None:
        """将 batch 回滚到 Canal，交由上层决定何时重试。"""

        _require_batch_id(batch_id)
        self.connector.rollback(batch_id)

    def ack(self, batch_id: int) -> None:
        """透传确认能力；生产调用仍需通过 P7.7 门禁。"""

        _require_batch_id(batch_id)
        self.connector.ack(batch_id)


@dataclass(frozen=True)
class CanalChangeEvent:
    """从一行 INSERT/UPDATE/DELETE Image 提取的最小定位事件。"""

    source_schema: str
    source_table: str
    source_record_id: int
    expert_id: int | None
    logfile_name: str
    logfile_offset: int
    server_id: int | None
    execute_time: int | None
    entry_index: int
    row_index: int
    operation_type: str = "delete"

    @property
    def is_delete(self) -> bool:
        """判断事件是否为物理删除。"""

        return self.operation_type == "delete"


# 保留旧名称，兼容 P7.4/P7.5 调用方；新代码应使用 CanalChangeEvent。
CanalDeleteEvent = CanalChangeEvent


@dataclass(frozen=True)
class CanalParseIssue:
    """不含原始 Payload 的安全解析错误摘要。"""

    code: str
    entry_index: int
    row_index: int | None = None


@dataclass(frozen=True)
class CanalParseResult:
    """一个 batch 的统一变更事件、兼容 DELETE 视图和安全失败摘要。"""

    delete_events: tuple[CanalChangeEvent, ...]
    ignored_count: int
    issues: tuple[CanalParseIssue, ...]
    change_events: tuple[CanalChangeEvent, ...] = ()

    @property
    def events(self) -> tuple[CanalChangeEvent, ...]:
        """返回所有可进入任务编排的 INSERT/UPDATE/DELETE 事件。"""

        return self.change_events or self.delete_events

    @property
    def rollback_required(self) -> bool:
        """存在任一结构错误时要求整个 batch rollback/not-ack。"""

        return bool(self.issues)


def parse_canal_batch(batch: CanalBatch) -> CanalParseResult:
    """严格解析 batch，提取四表 INSERT/UPDATE/DELETE 的 id/expert_id。"""

    if batch.is_empty:
        return CanalParseResult((), 0, ())
    if batch.destination != CANAL_DESTINATION:
        raise CanalParserError("Canal destination 缺失或不匹配")

    delete_events: list[CanalChangeEvent] = []
    change_events: list[CanalChangeEvent] = []
    issues: list[CanalParseIssue] = []
    ignored_count = 0
    transaction_open = False
    for entry_index, entry in enumerate(batch.entries):
        if entry.entry_type == "TRANSACTIONBEGIN":
            if not _valid_offset(entry):
                issues.append(CanalParseIssue("invalid_transaction_offset", entry_index))
            if transaction_open:
                issues.append(CanalParseIssue("unexpected_transaction_begin", entry_index))
            transaction_open = True
            continue
        if entry.entry_type == "TRANSACTIONEND":
            if not _valid_offset(entry):
                issues.append(CanalParseIssue("invalid_transaction_offset", entry_index))
            if not transaction_open:
                issues.append(CanalParseIssue("unexpected_transaction_end", entry_index))
            transaction_open = False
            continue
        if entry.entry_type != _ROW_DATA:
            issues.append(CanalParseIssue("unsupported_entry_type", entry_index))
            continue
        if not transaction_open:
            issues.append(CanalParseIssue("row_outside_transaction", entry_index))
            continue
        if not _valid_offset(entry):
            issues.append(CanalParseIssue("invalid_row_offset", entry_index))
            continue

        operation = (entry.operation_type or "").upper()
        if operation not in {"INSERT", "UPDATE", "DELETE"}:
            issues.append(CanalParseIssue("unsupported_operation", entry_index))
            continue
        if entry.source_schema != "expert" or entry.source_table not in _LOCATOR_COLUMNS:
            issues.append(CanalParseIssue("unsupported_source", entry_index))
            continue
        if not entry.rows:
            # 三种操作都必须携带 RowData；缺失时回滚，避免把真实变更静默当作已处理。
            issues.append(CanalParseIssue("missing_row_data", entry_index))
            continue

        for row_index, row in enumerate(entry.rows):
            before = _columns_to_values(row.before)
            after = _columns_to_values(row.after)
            if before is None:
                issues.append(CanalParseIssue("duplicate_before_column", entry_index, row_index))
                continue
            if after is None:
                issues.append(CanalParseIssue("duplicate_after_column", entry_index, row_index))
                continue
            if operation == "DELETE" and after:
                issues.append(CanalParseIssue("delete_after_image_not_empty", entry_index, row_index))
                continue
            image = before if operation == "DELETE" else {**before, **after}
            required = _LOCATOR_COLUMNS[entry.source_table]
            if not required.issubset(image):
                issue_code = (
                    "missing_delete_locator" if operation == "DELETE" else "missing_change_locator"
                )
                issues.append(CanalParseIssue(issue_code, entry_index, row_index))
                continue
            source_record_id = _positive_int(image.get("id"))
            if source_record_id is None:
                issues.append(CanalParseIssue("invalid_source_record_id", entry_index, row_index))
                continue
            expert_id = None
            if entry.source_table == "experts":
                expert_id = source_record_id
            else:
                expert_id = _positive_int(image.get("expert_id"))
                if expert_id is None:
                    issues.append(CanalParseIssue("invalid_expert_id", entry_index, row_index))
                    continue
            event = CanalChangeEvent(
                source_schema=entry.source_schema,
                source_table=entry.source_table,
                source_record_id=source_record_id,
                expert_id=expert_id,
                logfile_name=entry.logfile_name or "",
                logfile_offset=entry.logfile_offset or 0,
                server_id=entry.server_id,
                execute_time=entry.execute_time,
                entry_index=entry_index,
                row_index=row_index,
                operation_type=operation.lower(),
            )
            change_events.append(event)
            if event.is_delete:
                delete_events.append(event)

    if transaction_open:
        issues.append(CanalParseIssue("unclosed_transaction", len(batch.entries) - 1))
    return CanalParseResult(
        tuple(delete_events), ignored_count, tuple(issues), tuple(change_events)
    )


def _columns_to_values(columns: tuple[Any, ...]) -> dict[str, Any] | None:
    """将临时列对象转为字典，并拒绝重复列名。"""

    values: dict[str, Any] = {}
    for column in columns:
        name = getattr(column, "name", None)
        if not isinstance(name, str) or not name or name in values:
            return None
        values[name] = getattr(column, "value", None)
    return values


def _valid_offset(entry: CanalEntry) -> bool:
    """校验 Parser 依赖的真实 binlog 文件名和非负位置。"""

    return (
        isinstance(entry.logfile_name, str)
        and bool(entry.logfile_name)
        and isinstance(entry.logfile_offset, int)
        and not isinstance(entry.logfile_offset, bool)
        and entry.logfile_offset >= 0
        and isinstance(entry.server_id, int)
        and not isinstance(entry.server_id, bool)
        and entry.server_id >= 0
        and isinstance(entry.execute_time, int)
        and not isinstance(entry.execute_time, bool)
        and entry.execute_time >= 0
    )


def _positive_int(value: Any) -> int | None:
    """把 Canal 字符串/数值列严格归一为正整数定位字段。"""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value > 0 else None
    if isinstance(value, str) and value and value.isdecimal():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _require_batch_id(batch_id: int) -> None:
    """拒绝空 batch 或非法 batch ID 的 ack/rollback。"""

    if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 0:
        raise CanalParserError("batch_id 必须是非负整数")


__all__ = [
    "CanalBatchDecoder",
    "CanalChangeEvent",
    "CanalClientAdapter",
    "CanalConnectorPort",
    "CanalDeleteEvent",
    "CanalParseIssue",
    "CanalParseResult",
    "CanalParserError",
    "parse_canal_batch",
]
