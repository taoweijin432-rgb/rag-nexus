"""验证 P7.3 Canal 适配器和严格 DELETE Parser 的最小契约。"""

from __future__ import annotations

import pytest

from expert_index.cdc import (
    CANAL_DESTINATION,
    CanalBatch,
    CanalClientAdapter,
    CanalColumn,
    CanalEntry,
    CanalRowData,
    parse_canal_batch,
)


def _entry(table: str, *, record_id: int = 11, expert_id: int | None = 7) -> CanalEntry:
    """构造只带定位列值的脱敏 DELETE Entry。"""

    columns = [CanalColumn("id", record_id)]
    if table != "experts":
        columns.append(CanalColumn("expert_id", expert_id))
    return CanalEntry(
        entry_type="ROWDATA",
        source_schema="expert",
        source_table=table,
        operation_type="DELETE",
        logfile_name="binlog.000012",
        logfile_offset=100 + record_id,
        server_id=1,
        execute_time=1,
        rows=(CanalRowData(before=tuple(columns)),),
    )


def _transaction_marker(entry_type: str, offset: int) -> CanalEntry:
    """构造真实 batch 中的事务边界 Entry。"""

    return CanalEntry(
        entry_type,
        "",
        "",
        None,
        "binlog.000012",
        offset,
        1,
        1,
    )


@pytest.mark.parametrize(
    ("table", "record_id", "expert_id"),
    (
        ("experts", 7, 7),
        ("experts_work_experience", 11, 7),
        ("experts_project_experience", 12, 7),
        ("experts_education_experience", 13, 7),
    ),
)
def test_parser_extracts_only_delete_locator(table: str, record_id: int, expert_id: int) -> None:
    """四张来源表均按固定列提取定位信息，不携带其他列值。"""

    batch = CanalBatch(
        9,
        (
            _transaction_marker("TRANSACTIONBEGIN", 90),
            _entry(table, record_id=record_id, expert_id=expert_id),
            _transaction_marker("TRANSACTIONEND", 200),
        ),
        CANAL_DESTINATION,
    )

    result = parse_canal_batch(batch)

    assert len(result.delete_events) == 1
    event = result.delete_events[0]
    assert (event.source_table, event.source_record_id, event.expert_id) == (
        table,
        record_id,
        expert_id,
    )
    assert result.ignored_count == 0
    assert result.rollback_required is False


def test_parser_ignores_update_but_fails_closed_on_missing_delete_locator() -> None:
    """UPDATE 不进入 CDC；DELETE 缺少定位列时整批要求 rollback。"""

    update = CanalEntry(
        "ROWDATA", "expert", "experts_work_experience", "UPDATE", "binlog.000012", 201, 1, 1
    )
    malformed = CanalEntry(
        "ROWDATA",
        "expert",
        "experts_work_experience",
        "DELETE",
        "binlog.000012",
        202,
        1,
        1,
        rows=(CanalRowData(before=(CanalColumn("id", None),)),),
    )

    result = parse_canal_batch(
        CanalBatch(
            10,
            (
                _transaction_marker("TRANSACTIONBEGIN", 190),
                update,
                malformed,
                _transaction_marker("TRANSACTIONEND", 290),
            ),
            CANAL_DESTINATION,
        )
    )

    assert result.ignored_count == 1
    assert [issue.code for issue in result.issues] == ["missing_delete_locator"]
    assert result.rollback_required is True


def test_parser_ignores_insert_and_update() -> None:
    """INSERT/UPDATE 由 Poll 负责，CDC 只计数后推进 Canal 位点。"""

    insert = CanalEntry(
        "ROWDATA", "expert", "experts_work_experience", "INSERT",
        "binlog.000012", 301, 1, 1,
        rows=(CanalRowData(after=(CanalColumn("id", "21"), CanalColumn("expert_id", "7"))),),
    )
    update = CanalEntry(
        "ROWDATA", "expert", "experts", "UPDATE",
        "binlog.000012", 302, 1, 1,
        rows=(CanalRowData(
            before=(CanalColumn("id", 8),),
            after=(CanalColumn("id", 8), CanalColumn("job_status", "active")),
        ),),
    )
    batch = CanalBatch(
        11,
        (_transaction_marker("TRANSACTIONBEGIN", 300), insert, update,
         _transaction_marker("TRANSACTIONEND", 303)),
        CANAL_DESTINATION,
    )

    result = parse_canal_batch(batch)

    assert result.delete_events == ()
    assert result.ignored_count == 2
    assert result.rollback_required is False


def test_parser_accepts_delete_without_transaction_markers() -> None:
    """DELETE 解析不依赖 Canal 是否返回事务边界 Entry。"""

    result = parse_canal_batch(CanalBatch(12, (_entry("experts", record_id=9),), CANAL_DESTINATION))

    assert len(result.delete_events) == 1
    assert result.delete_events[0].expert_id == 9
    assert result.rollback_required is False


class _Connector:
    """记录适配器转发的连接、订阅、读取和确认调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def connect(self) -> None:
        self.calls.append(("connect", None))

    def disconnect(self) -> None:
        self.calls.append(("disconnect", None))

    def subscribe(self, subscription: str) -> None:
        self.calls.append(("subscribe", subscription))

    def get_without_ack(self, batch_size: int) -> object:
        self.calls.append(("get_without_ack", batch_size))
        return "raw-batch"

    def rollback(self, batch_id: int) -> None:
        self.calls.append(("rollback", batch_id))

    def ack(self, batch_id: int) -> None:
        self.calls.append(("ack", batch_id))


def test_client_adapter_forwards_only_minimal_transport_calls() -> None:
    """适配器隔离 SDK 类型，读取时不自动 ACK。"""

    connector = _Connector()
    batch = CanalBatch(-1, (), CANAL_DESTINATION)
    adapter = CanalClientAdapter(
        connector,
        decoder=lambda raw, destination: batch,
        subscription="expert\\.(experts)",
    )

    adapter.connect()
    assert adapter.fetch_without_ack(10).is_empty
    adapter.rollback(4)
    adapter.ack(4)
    adapter.disconnect()

    assert connector.calls == [
        ("connect", None),
        ("subscribe", "expert\\.(experts)"),
        ("get_without_ack", 10),
        ("rollback", 4),
        ("ack", 4),
        ("disconnect", None),
    ]
