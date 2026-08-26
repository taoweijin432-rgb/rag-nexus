"""验证 P7.4 CDC Event/Task 的 Index 持久化边界与重放幂等。"""

from __future__ import annotations

from dataclasses import replace
from typing import Self

import pytest

from expert_index.cdc import (
    CanalDeleteEvent,
    CdcPersistenceConflictError,
    CdcTaskPlan,
    CdcTaskReceipt,
)
from expert_index.config import DatabaseBoundaryError, DatabaseConfig, DatabaseRole
from expert_index.models import NewSourceSyncTask
from expert_index.repositories.cdc_event import MySQLCdcBatchPersistence


def _plan() -> CdcTaskPlan:
    """构造不含正文的固定 DELETE 计划。"""

    event = CanalDeleteEvent(
        source_schema="expert",
        source_table="experts_work_experience",
        source_record_id=901,
        expert_id=101,
        logfile_name="binlog.000012",
        logfile_offset=76094823,
        server_id=1,
        execute_time=1787747052000,
        entry_index=1,
        row_index=0,
    )
    task = NewSourceSyncTask(
        dedupe_key="a" * 64,
        batch_id=88,
        event_source="cdc",
        source_table=event.source_table,
        source_record_id=event.source_record_id,
        source_updated_at=None,
        expert_id=event.expert_id,
        operation_type="delete",
        target_type="employment",
        policy_version="p7.4-test",
    )
    return CdcTaskPlan("event-1", event, task)


class _Cursor:
    """只模拟本 Repository 使用的 SQL 分支，不连接真实数据库。"""

    def __init__(self) -> None:
        self.event: dict[str, object] | None = None
        self.task: dict[str, object] | None = None
        self.result: dict[str, object] | None = None
        self.lastrowid = 0
        self.rowcount = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        """按 SQL 目的更新最小状态，模拟 FOR UPDATE 的读取结果。"""

        normalized = " ".join(sql.split()).upper()
        self.result = None
        self.rowcount = 0
        if normalized.startswith("SELECT") and "EXPERT_INDEX_CDC_EVENT" in normalized:
            if self.event is not None and (
                ("WHERE DESTINATION" in normalized and self.event["binlog_file"] == params[1])
                or ("WHERE CDC_EVENT_ID" in normalized and self.event["cdc_event_id"] == params[0])
                or ("WHERE ID=" in normalized and self.event["id"] == params[0])
            ):
                self.result = self.event
        elif normalized.startswith("INSERT INTO EXPERT_INDEX_CDC_EVENT"):
            self.event = {
                "id": 11,
                "cdc_event_id": params[0],
                "local_batch_id": params[1],
                "task_id": None,
                "status": "pending",
                "destination": params[3],
                "binlog_file": params[4],
                "binlog_position": params[5],
                "row_index": params[6],
                "source_schema": params[9],
                "source_table": params[10],
                "operation_type": "delete",
                "source_record_id": params[11],
                "expert_id": params[12],
                "binlog_server_id": params[7],
            }
            self.lastrowid = 11
        elif normalized.startswith("SELECT") and "EXPERT_INDEX_SYNC_TASK" in normalized:
            if self.task is not None and self.task["dedupe_key"] == params[0]:
                self.result = self.task
        elif normalized.startswith("INSERT INTO EXPERT_INDEX_SYNC_TASK"):
            if self.task is None:
                self.task = {
                    "id": 501,
                    "dedupe_key": params[0],
                    "batch_id": params[1],
                    "event_source": "cdc",
                    "source_table": params[2],
                    "source_record_id": params[3],
                    "source_updated_at": None,
                    "expert_id": params[4],
                    "operation_type": "delete",
                    "target_type": params[5],
                    "policy_version": params[6],
                }
                self.lastrowid = 501
        elif (
            normalized.startswith("UPDATE EXPERT_INDEX_CDC_EVENT")
            and "SET TASK_ID" in normalized
            and self.event is not None
            and self.event["task_id"] is None
        ):
            self.event["task_id"] = params[0]
            self.rowcount = 1

    def fetchone(self) -> dict[str, object] | None:
        """返回最近一次 SELECT 的单行结果。"""

        return self.result


class _Connection:
    """提供事务调用记录，确保 Repository 使用单连接提交。"""

    def __init__(self) -> None:
        self.cursor_obj = _Cursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _Cursor:
        return self.cursor_obj

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        return None


def _index_config() -> DatabaseConfig:
    """返回合法的 Index 角色配置。"""

    return DatabaseConfig("127.0.0.1", 3306, "index", "", "expert_index", DatabaseRole.INDEX)


def test_cdc_persistence_reuses_event_and_task_on_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """首次落库创建 Event/Task，重复投递复用相同任务并安全提交。"""

    connection = _Connection()
    monkeypatch.setattr(
        "expert_index.repositories.cdc_event.pymysql.connect",
        lambda **_kwargs: connection,
    )
    repository = MySQLCdcBatchPersistence(_index_config())
    first = repository.persist_batch([_plan()])
    second = repository.persist_batch([_plan()])

    assert first == (CdcTaskReceipt("event-1", 501, "created"),)
    assert second == (CdcTaskReceipt("event-1", 501, "reused_pending"),)
    assert connection.commits == 2
    assert connection.rollbacks == 0


def test_cdc_persistence_rejects_same_position_with_different_locator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相同 position 若对应不同业务定位必须回滚，不能覆盖既有 Event。"""

    connection = _Connection()
    monkeypatch.setattr(
        "expert_index.repositories.cdc_event.pymysql.connect",
        lambda **_kwargs: connection,
    )
    repository = MySQLCdcBatchPersistence(_index_config())
    original = _plan()
    repository.persist_batch([original])
    conflicting = replace(
        original,
        event=replace(original.event, source_record_id=902),
        task=replace(original.task, source_record_id=902),
    )

    with pytest.raises(CdcPersistenceConflictError):
        repository.persist_batch([conflicting])

    assert connection.commits == 1
    assert connection.rollbacks == 1


def test_cdc_persistence_rejects_non_index_role() -> None:
    """CDC 状态持久化拒绝 Source 或未声明角色的数据库配置。"""

    source = DatabaseConfig("127.0.0.1", 3306, "source", "", "expert", DatabaseRole.SOURCE)
    with pytest.raises(DatabaseBoundaryError):
        MySQLCdcBatchPersistence(source)
