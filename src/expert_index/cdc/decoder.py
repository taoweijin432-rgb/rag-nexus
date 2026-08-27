"""将 canal-python Message 解码为项目的无 SDK Canal 领域模型。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .protocol import CANAL_DESTINATION, CanalBatch, CanalColumn, CanalEntry, CanalRowData


class CanalDecodeError(ValueError):
    """表示 SDK Message 缺少统一 Parser 所需的结构字段。"""


def decode_canal_message(raw: object, destination: str = CANAL_DESTINATION) -> CanalBatch:
    """解码 dict/对象/Protobuf Message，不改变 ACK 状态。"""

    batch_id = _field(raw, "id", "batch_id")
    entries = _field(raw, "entries", default=())
    if batch_id is None:
        raise CanalDecodeError("Canal Message 缺少 batch id")
    try:
        parsed_batch_id = int(batch_id)
    except (TypeError, ValueError) as exc:
        raise CanalDecodeError("Canal batch id 非法") from exc
    return CanalBatch(
        batch_id=parsed_batch_id,
        entries=tuple(_decode_entry(item) for item in _as_iterable(entries)),
        destination=destination,
    )


def _decode_entry(raw: object) -> CanalEntry:
    """把单个 SDK Entry 转换为脱敏领域 Entry。"""

    raw = _parse_entry(raw)
    entry_type = _enum_name(
        _field(raw, "entry_type", "entryType", "getEntryType"),
        raw,
        ("entry_type", "entryType"),
    )
    header = _field(raw, "header", "getHeader", default=raw)
    source_schema = str(
        _field(header, "schema_name", "schemaName", "getSchemaName", default="") or ""
    )
    source_table = str(
        _field(header, "table_name", "tableName", "getTableName", default="") or ""
    )
    operation_value = _field(
        raw,
        "operation_type",
        "event_type",
        "eventType",
        "getEventType",
        default=None,
    )
    operation = _enum_name(
        operation_value,
        raw,
        ("operation_type", "event_type", "eventType"),
    )
    if not operation:
        store_value = _field(raw, "store_value", "storeValue", "getStoreValue", default=None)
        change = _parse_row_change(store_value)
        operation = _enum_name(
            _field(change, "event_type", "eventType", "getEventType", default=None),
            change,
            ("event_type", "eventType"),
        )
    else:
        change = _parse_row_change(_field(raw, "store_value", "storeValue", "getStoreValue", default=None))
    rows = tuple(
        _decode_row(row)
        for row in _as_iterable(
            _field(change, "row_datas", "rowDatas", "getRowDatasList", default=())
        )
    )
    return CanalEntry(
        entry_type=entry_type,
        source_schema=source_schema,
        source_table=source_table,
        operation_type=operation or None,
        logfile_name=_text_or_none(_field(header, "logfile_name", "logfileName", "getLogfileName")),
        logfile_offset=_int_or_none(_field(header, "logfile_offset", "logfileOffset", "getLogfileOffset")),
        server_id=_int_or_none(_field(header, "server_id", "serverId", "getServerId")),
        execute_time=_int_or_none(_field(header, "execute_time", "executeTime", "getExecuteTime")),
        rows=rows,
    )


def _decode_row(raw: object) -> CanalRowData:
    """提取 Before/After 列值，供 Parser 立即丢弃非定位字段。"""

    return CanalRowData(
        before=tuple(
            _decode_column(item)
            for item in _as_iterable(
                _field(
                    raw,
                    "before",
                    "beforeColumns",
                    "before_columns",
                    "getBeforeColumnsList",
                    default=(),
                )
            )
        ),
        after=tuple(
            _decode_column(item)
            for item in _as_iterable(
                _field(
                    raw,
                    "after",
                    "afterColumns",
                    "after_columns",
                    "getAfterColumnsList",
                    default=(),
                )
            )
        ),
    )


def _decode_column(raw: object) -> CanalColumn:
    """转换列名和值；值只在 Parser 边界短暂存在。"""

    name = _field(raw, "name", "getName")
    if not isinstance(name, str) or not name:
        raise CanalDecodeError("Canal Column 缺少列名")
    return CanalColumn(name=name, value=_field(raw, "value", "getValue"))


def _parse_row_change(raw: object) -> object:
    """解析 ROWDATA 的 protobuf bytes，已是对象时直接返回。"""

    if raw is None or not isinstance(raw, (bytes, bytearray, memoryview)):
        return raw or {}
    try:
        from canal.protocol import CanalEntry as SdkCanalEntry  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CanalDecodeError("当前 SDK 无法导入 CanalEntry protobuf") from exc
    change = SdkCanalEntry.RowChange()
    try:
        change.ParseFromString(bytes(raw))
    except Exception as exc:
        raise CanalDecodeError("Canal RowChange protobuf 解码失败") from exc
    return change


def _parse_entry(raw: object) -> object:
    """解析 canal-python 返回的 Entry protobuf bytes。"""

    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return raw
    try:
        from canal.protocol import CanalEntry as SdkCanalEntry  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CanalDecodeError("当前 SDK 无法导入 CanalEntry protobuf") from exc
    entry = SdkCanalEntry.Entry()
    try:
        entry.ParseFromString(bytes(raw))
    except Exception as exc:
        raise CanalDecodeError("Canal Entry protobuf 解码失败") from exc
    return entry


def _field(obj: object, *names: str, default: object = None) -> object:
    """兼容 dict、Python protobuf 属性和 Java 风格 getter。"""

    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        candidate = getattr(obj, name, None)
        if callable(candidate):
            try:
                return candidate()
            except TypeError:
                continue
        if candidate is not None:
            return candidate
    return default


def _enum_name(
    value: object,
    owner: object | None = None,
    field_names: tuple[str, ...] = (),
) -> str:
    """把枚举、字符串或 protobuf 名称规范化为大写文本。"""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.upper()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    if callable(name):
        try:
            resolved_name = name()
        except TypeError:
            resolved_name = None
        if isinstance(resolved_name, str):
            return resolved_name.upper()
    if isinstance(value, int) and not isinstance(value, bool) and owner is not None:
        descriptor = getattr(owner, "DESCRIPTOR", None)
        fields = getattr(descriptor, "fields_by_name", {})
        for field_name in field_names:
            field = fields.get(field_name)
            enum_type = getattr(field, "enum_type", None)
            values = getattr(enum_type, "values_by_number", {})
            enum_value = values.get(value)
            if enum_value is not None:
                return str(enum_value.name).upper()
    return str(value).upper().split(".")[-1]


def _as_iterable(value: object) -> Iterable[object]:
    """将缺失字段统一为空序列，拒绝把字符串拆成列。"""

    if value is None or isinstance(value, (str, bytes, bytearray, memoryview)):
        return ()
    try:
        return iter(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise CanalDecodeError("Canal 列表字段不是可迭代对象") from exc


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError) as exc:
        raise CanalDecodeError("Canal 数值字段非法") from exc


__all__ = ["CanalDecodeError", "decode_canal_message"]
