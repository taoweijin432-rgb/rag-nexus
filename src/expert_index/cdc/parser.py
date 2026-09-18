"""Canal Client 适配端口和四表物理删除 Parser。

适配器负责隔离 Canal SDK；Parser 忽略 INSERT/UPDATE 和事务边界，只从四张来源表
的 DELETE Before Image 提取主键与专家定位。行值不进入任务或持久化模型。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..dataset import EXPERT_SPEC, DatasetSpec
from .protocol import (
    CANAL_DESTINATION,
    CANAL_SUBSCRIPTION,
    CanalBatch,
    CanalClientPort,
    CanalEntry,
)

_ROW_DATA = "ROWDATA"
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
        raw = self.connector.get_without_ack(batch_size)
        try:
            batch = self.decoder(raw, self.destination)
        except Exception:
            batch_id = _extract_batch_id(raw)
            if batch_id is not None and batch_id >= 0:
                try:
                    self.connector.rollback(batch_id)
                except Exception as rollback_error:
                    raise CanalParserError("Canal batch 解码失败且 rollback 失败") from rollback_error
            raise
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


def _extract_batch_id(raw: object) -> int | None:
    """从 SDK 原始响应提取 batch id，供解码失败时安全回滚。"""

    if isinstance(raw, Mapping):
        value = raw.get("id", raw.get("batch_id"))
    else:
        value = getattr(raw, "id", getattr(raw, "batch_id", None))
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CanalChangeEvent:
    """从一行 DELETE Before Image 提取的最小定位事件。"""

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
    """一个 batch 的 DELETE 事件、忽略数和安全失败摘要。"""

    delete_events: tuple[CanalChangeEvent, ...]
    ignored_count: int
    issues: tuple[CanalParseIssue, ...]

    @property
    def rollback_required(self) -> bool:
        """存在任一结构错误时要求整个 batch rollback/not-ack。"""

        return bool(self.issues)


def parse_canal_batch(
    batch: CanalBatch,
    *,
    destination: str = CANAL_DESTINATION,
    source_schema: str = "expert",
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> CanalParseResult:
    """解析 batch，忽略非删除事件并提取当前数据集的物理行及人物身份。

    默认参数完整保留 Expert 既有行为。Resume 调用方必须显式传入自己的
    ``destination``、来源 Schema 和 :class:`DatasetSpec`；主表 ``id`` 与
    ``resume_id`` 因而可以分别承担物理行定位和人物身份。
    """

    _validate_parser_contract(destination, source_schema, dataset_spec)

    if batch.is_empty:
        return CanalParseResult((), 0, ())
    if batch.destination != destination:
        raise CanalParserError("Canal destination 缺失或不匹配")

    delete_events: list[CanalChangeEvent] = []
    issues: list[CanalParseIssue] = []
    ignored_count = 0
    for entry_index, entry in enumerate(batch.entries):
        if entry.entry_type in {"TRANSACTIONBEGIN", "TRANSACTIONEND"}:
            continue
        if entry.entry_type != _ROW_DATA:
            issues.append(CanalParseIssue("unsupported_entry_type", entry_index))
            continue
        if not _valid_offset(entry):
            issues.append(CanalParseIssue("invalid_row_offset", entry_index))
            continue

        operation = (entry.operation_type or "").upper()
        if operation in {"INSERT", "UPDATE"}:
            ignored_count += max(1, len(entry.rows))
            continue
        if operation != "DELETE":
            issues.append(CanalParseIssue("unsupported_operation", entry_index))
            continue
        if (
            entry.source_schema != source_schema
            or entry.source_table not in dataset_spec.checkpoint_source_tables
        ):
            issues.append(CanalParseIssue("unsupported_source", entry_index))
            continue
        if not entry.rows:
            # DELETE 必须携带 Before Image，否则无法安全确定删除范围。
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
            image = before
            identity_column = _identity_column(dataset_spec, entry.source_table)
            required = frozenset({"id", identity_column})
            if not required.issubset(image):
                issues.append(CanalParseIssue("missing_delete_locator", entry_index, row_index))
                continue
            source_record_id = _positive_int(image.get("id"))
            if source_record_id is None:
                issues.append(CanalParseIssue("invalid_source_record_id", entry_index, row_index))
                continue
            expert_id = _positive_int(image.get(identity_column))
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
                operation_type="delete",
            )
            delete_events.append(event)

    return CanalParseResult(tuple(delete_events), ignored_count, tuple(issues))


def _validate_parser_contract(
    destination: str,
    source_schema: str,
    dataset_spec: DatasetSpec,
) -> None:
    """在读取 Canal Payload 前验证调用方提供的隔离契约。"""

    if not isinstance(destination, str) or not destination.strip():
        raise ValueError("destination 必须是非空字符串")
    if not isinstance(source_schema, str) or not source_schema.strip():
        raise ValueError("source_schema 必须是非空字符串")
    if not isinstance(dataset_spec, DatasetSpec):
        raise TypeError("dataset_spec 必须是 DatasetSpec")


def _identity_column(dataset_spec: DatasetSpec, source_table: str) -> str:
    """返回 CDC Before Image 中承载全链路人物身份的列名。"""

    if source_table == dataset_spec.profile_table:
        return dataset_spec.profile_identity_column
    if source_table in dataset_spec.checkpoint_source_tables:
        return dataset_spec.child_identity_column
    raise ValueError(f"{dataset_spec.name} 不支持来源表: {source_table}")


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
