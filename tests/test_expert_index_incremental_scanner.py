"""P3.4 Poll 候选确定性任务映射测试。"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from expert_index.models import (
    BatchSourceScan,
    NewSourceSyncTask,
    SourceChangeCandidate,
    SourceCheckpoint,
    SourceScanCursor,
)
from expert_index.scanner import (
    CrossBatchTaskConflictError,
    IncrementalScanner,
    IncrementalSourceScanService,
    build_poll_task_dedupe_key,
    map_poll_candidate_to_task,
)

SOURCE_TIME = datetime(2026, 8, 24, 8, 9, 10, 123456, tzinfo=timezone.utc)
DELETED_TIME = datetime(2026, 8, 24, 8, 10, tzinfo=timezone.utc)
POLICY_VERSION = "1.3"

SOURCE_TARGET_CASES = (
    ("experts", "expert"),
    ("experts_work_experience", "employment"),
    ("experts_project_experience", "project"),
    ("experts_education_experience", "education"),
)


def _candidate(
    source_table: str = "experts_work_experience",
    **changes: object,
) -> SourceChangeCandidate:
    """构造只含调度字段的固定 Poll 候选。"""

    values: dict[str, object] = {
        "source_table": source_table,
        "record_id": 901,
        "expert_id": 101,
        "updated_at": SOURCE_TIME,
        "deleted_at": None,
    }
    if source_table == "experts":
        values["expert_id"] = values["record_id"]
    values.update(changes)
    return SourceChangeCandidate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(("source_table", "target_type"), SOURCE_TARGET_CASES)
def test_poll_candidate_maps_frozen_source_target_and_update_operation(
    source_table: str,
    target_type: str,
) -> None:
    """四张来源表的非删除候选应映射到冻结的粗粒度目标和 update。"""

    candidate = _candidate(source_table)
    task = map_poll_candidate_to_task(
        candidate,
        batch_id=701,
        policy_version=POLICY_VERSION,
    )

    assert task.event_source == "poll"
    assert task.source_table == source_table
    assert task.source_record_id == candidate.record_id
    assert task.source_updated_at == SOURCE_TIME
    assert task.expert_id == candidate.expert_id
    assert task.operation_type == "update"
    assert task.target_type == target_type
    assert task.policy_version == POLICY_VERSION


@pytest.mark.parametrize(("source_table", "target_type"), SOURCE_TARGET_CASES)
def test_logically_deleted_poll_candidate_maps_to_delete(
    source_table: str,
    target_type: str,
) -> None:
    """逻辑删除只改变 operation_type，不扩大来源表级影响范围。"""

    task = map_poll_candidate_to_task(
        _candidate(source_table, deleted_at=DELETED_TIME),
        batch_id=701,
        policy_version=POLICY_VERSION,
    )

    assert task.operation_type == "delete"
    assert task.target_type == target_type


def test_poll_task_dedupe_key_matches_exact_golden_fixture() -> None:
    """固定候选必须持续生成同一个 canonical JSON SHA-256。"""

    candidate = _candidate()

    assert build_poll_task_dedupe_key(
        candidate,
        policy_version=POLICY_VERSION,
    ) == "1b038c4aa71e3fffaa417455f9b71996819389d10ca10cf519e2fcfae1a9c7c0"


def test_repeated_candidate_maps_to_identical_task() -> None:
    """相同批次中的同一候选重复映射必须逐字段一致。"""

    scanner = IncrementalScanner(policy_version=POLICY_VERSION)
    candidate = _candidate()

    assert scanner.map_candidate(candidate, batch_id=701) == scanner.map_candidate(
        candidate,
        batch_id=701,
    )


def test_new_source_updated_at_creates_new_dedupe_identity() -> None:
    """同一来源记录出现新的 updated_at 时必须形成新的任务身份。"""

    candidate = _candidate()
    changed = replace(candidate, updated_at=SOURCE_TIME + timedelta(seconds=1))

    assert build_poll_task_dedupe_key(
        candidate,
        policy_version=POLICY_VERSION,
    ) != build_poll_task_dedupe_key(
        changed,
        policy_version=POLICY_VERSION,
    )


def test_dedupe_key_excludes_batch_expert_and_deleted_timestamp() -> None:
    """非冻结定位值变化不能污染相同删除事件的跨批次幂等键。"""

    first_candidate = _candidate(deleted_at=DELETED_TIME)
    second_candidate = replace(
        first_candidate,
        expert_id=202,
        deleted_at=DELETED_TIME + timedelta(seconds=30),
    )
    first_task = map_poll_candidate_to_task(
        first_candidate,
        batch_id=701,
        policy_version=POLICY_VERSION,
    )
    second_task = map_poll_candidate_to_task(
        second_candidate,
        batch_id=702,
        policy_version=POLICY_VERSION,
    )

    assert first_task.dedupe_key == second_task.dedupe_key
    assert first_task.batch_id != second_task.batch_id
    assert first_task.expert_id != second_task.expert_id


def test_operation_target_and_policy_version_participate_in_dedupe_key() -> None:
    """操作、目标和策略变化必须形成不同的任务身份。"""

    candidate = _candidate()
    active_key = build_poll_task_dedupe_key(candidate, policy_version="1.3")
    deleted_key = build_poll_task_dedupe_key(
        replace(candidate, deleted_at=DELETED_TIME),
        policy_version="1.3",
    )
    policy_changed_key = build_poll_task_dedupe_key(candidate, policy_version="1.4")
    other_target_key = build_poll_task_dedupe_key(
        _candidate("experts_project_experience"),
        policy_version="1.3",
    )

    assert len({active_key, deleted_key, policy_changed_key, other_target_key}) == 4


def test_source_datetime_uses_literal_microseconds_without_timezone_offset() -> None:
    """相同 DATETIME 字面值不能因 Python tzinfo 表示差异产生不同幂等键。"""

    utc_candidate = _candidate()
    offset_candidate = replace(
        utc_candidate,
        updated_at=SOURCE_TIME.astimezone(timezone(timedelta(hours=8))).replace(
            hour=SOURCE_TIME.hour
        ),
    )

    assert offset_candidate.updated_at.utcoffset() == timedelta(hours=8)
    assert offset_candidate.updated_at.replace(tzinfo=None) == SOURCE_TIME.replace(tzinfo=None)
    assert build_poll_task_dedupe_key(
        utc_candidate,
        policy_version=POLICY_VERSION,
    ) == build_poll_task_dedupe_key(
        offset_candidate,
        policy_version=POLICY_VERSION,
    )


def test_scanner_maps_page_in_order_without_adding_payload_fields() -> None:
    """批量映射应保持复合游标顺序，任务只包含冻结的调度契约字段。"""

    candidates = (
        _candidate(record_id=901),
        _candidate(record_id=902, updated_at=SOURCE_TIME + timedelta(seconds=1)),
    )

    tasks = IncrementalScanner(POLICY_VERSION).map_candidates(candidates, batch_id=701)

    assert [task.source_record_id for task in tasks] == [901, 902]
    assert {field.name for field in fields(NewSourceSyncTask)} == {
        "dedupe_key",
        "batch_id",
        "event_source",
        "source_table",
        "source_record_id",
        "source_updated_at",
        "expert_id",
        "operation_type",
        "target_type",
        "policy_version",
    }


@pytest.mark.parametrize("expert_id", [None, 900, 902])
def test_experts_candidate_requires_matching_expert_id(expert_id: int | None) -> None:
    """experts.id 与 expert_id 不一致时必须在任务持久化前停止。"""

    with pytest.raises(ValueError, match="expert_id 必须等于"):
        map_poll_candidate_to_task(
            _candidate("experts", expert_id=expert_id),
            batch_id=701,
            policy_version=POLICY_VERSION,
        )


def test_unmapped_experience_candidate_remains_schedulable() -> None:
    """经历表缺少 expert_id 时仍保留候选，供后续事实读取阶段安全处置。"""

    task = map_poll_candidate_to_task(
        _candidate(expert_id=None),
        batch_id=701,
        policy_version=POLICY_VERSION,
    )

    assert task.expert_id is None
    assert task.target_type == "employment"


@pytest.mark.parametrize(
    ("call", "expected_error"),
    [
        (
            lambda: map_poll_candidate_to_task(  # type: ignore[arg-type]
                object(), batch_id=701, policy_version=POLICY_VERSION
            ),
            TypeError,
        ),
        (
            lambda: map_poll_candidate_to_task(
                _candidate(), batch_id=0, policy_version=POLICY_VERSION
            ),
            ValueError,
        ),
        (
            lambda: IncrementalScanner(policy_version=" "),
            ValueError,
        ),
        (
            lambda: IncrementalScanner(POLICY_VERSION).map_candidates((), batch_id=True),
            ValueError,
        ),
    ],
)
def test_scanner_rejects_invalid_contract_before_mapping(
    call: object,
    expected_error: type[Exception],
) -> None:
    """非法候选、批次和策略版本不能生成半合法任务。"""

    with pytest.raises(expected_error):
        call()  # type: ignore[operator]


SCAN_START = SourceScanCursor(
    datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc),
    0,
)
SCAN_UPPER_BOUND = SourceScanCursor(
    datetime(2026, 8, 24, 9, 0, tzinfo=timezone.utc),
    999,
)


class FakeSourceAdapter:
    """按调用顺序返回固定上界和候选页，不读取真实 Source MySQL。"""

    def __init__(
        self,
        *,
        upper_bound: SourceScanCursor | None,
        pages: list[list[SourceChangeCandidate]],
        events: list[str] | None = None,
        reject_upper_bound_read: bool = False,
    ) -> None:
        self.upper_bound = upper_bound
        self.pages = list(pages)
        self.events = events if events is not None else []
        self.reject_upper_bound_read = reject_upper_bound_read
        self.upper_bound_calls = 0
        self.page_calls: list[tuple[SourceScanCursor, SourceScanCursor, int]] = []

    def get_source_scan_upper_bound(self, source_table: str) -> SourceScanCursor | None:
        self.events.append(f"source:upper:{source_table}")
        self.upper_bound_calls += 1
        if self.reject_upper_bound_read:
            raise AssertionError("恢复批次不能重新读取 fixed upper bound")
        return self.upper_bound

    def load_source_change_candidates(
        self,
        source_table: str,
        *,
        after_cursor: SourceScanCursor,
        upper_bound: SourceScanCursor,
        limit: int,
    ) -> list[SourceChangeCandidate]:
        self.events.append(f"source:page:{source_table}")
        self.page_calls.append((after_cursor, upper_bound, limit))
        if not self.pages:
            raise AssertionError("测试未提供下一页候选")
        return self.pages.pop(0)


class FakeSourceStateRepository:
    """在内存中模拟 lease、固定范围创建和临时游标 CAS。"""

    def __init__(
        self,
        *,
        scan: BatchSourceScan | None = None,
        events: list[str] | None = None,
        fail_advance_once: bool = False,
    ) -> None:
        self.scan = scan
        self.events = events if events is not None else []
        self.fail_advance_once = fail_advance_once
        self.created_batch_nos: list[str] = []
        self.advance_calls: list[tuple[SourceScanCursor, SourceScanCursor, int]] = []
        self.release_count = 0
        self.renew_count = 0

    def acquire_source_lease(
        self,
        source_table: str,
        lease_owner: str,
        *,
        lease_seconds: int,
    ) -> SourceCheckpoint:
        self.events.append(f"state:acquire:{source_table}:{lease_owner}")
        return SourceCheckpoint(
            source_table=source_table,
            last_updated_at=SCAN_START.updated_at,
            last_id=SCAN_START.record_id,
            lease_owner=lease_owner,
            lease_until=SOURCE_TIME + timedelta(seconds=lease_seconds),
        )

    def get_recoverable_batch_source_scan(
        self,
        source_table: str,
        lease_owner: str,
    ) -> BatchSourceScan | None:
        self.events.append(f"state:recover:{source_table}:{lease_owner}")
        return self.scan

    def release_source_lease(
        self,
        source_table: str,
        lease_owner: str,
    ) -> SourceCheckpoint:
        self.events.append(f"state:release:{source_table}:{lease_owner}")
        self.release_count += 1
        return SourceCheckpoint(
            source_table=source_table,
            last_updated_at=SCAN_START.updated_at,
            last_id=SCAN_START.record_id,
        )

    def create_incremental_batch_source_scan(
        self,
        source_table: str,
        lease_owner: str,
        *,
        batch_no: str,
        upper_bound: SourceScanCursor,
    ) -> BatchSourceScan:
        self.events.append(f"state:create:{source_table}:{lease_owner}")
        self.created_batch_nos.append(batch_no)
        self.scan = BatchSourceScan(
            batch_id=701,
            source_table=source_table,
            start_cursor=SCAN_START,
            current_cursor=SCAN_START,
            upper_bound=upper_bound,
            checkpoint_lock_version=0,
        )
        return self.scan

    def renew_source_lease(
        self,
        source_table: str,
        lease_owner: str,
        *,
        lease_seconds: int,
    ) -> SourceCheckpoint:
        self.events.append(f"state:renew:{source_table}:{lease_owner}")
        self.renew_count += 1
        return SourceCheckpoint(
            source_table=source_table,
            last_updated_at=SCAN_START.updated_at,
            last_id=SCAN_START.record_id,
            lease_owner=lease_owner,
            lease_until=SOURCE_TIME + timedelta(seconds=lease_seconds),
        )

    def advance_batch_source_scan_cursor(
        self,
        source_table: str,
        batch_id: int,
        lease_owner: str,
        *,
        checkpoint_lock_version: int,
        expected_cursor: SourceScanCursor,
        next_cursor: SourceScanCursor,
        candidate_increment: int,
    ) -> BatchSourceScan:
        del lease_owner, checkpoint_lock_version
        self.events.append(f"state:advance:{source_table}:{candidate_increment}")
        self.advance_calls.append(
            (expected_cursor, next_cursor, candidate_increment)
        )
        if self.fail_advance_once:
            self.fail_advance_once = False
            raise RuntimeError("simulated cursor commit crash")
        assert self.scan is not None
        assert self.scan.batch_id == batch_id
        assert self.scan.current_cursor == expected_cursor
        self.scan = replace(
            self.scan,
            current_cursor=next_cursor,
            candidate_count=self.scan.candidate_count + candidate_increment,
            status="running",
        )
        return self.scan


class FakeTaskRepository:
    """按 dedupe_key 模拟幂等任务持久化和跨批次命中。"""

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        fail_create_number: int | None = None,
        forced_existing_batch_id: int | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.fail_create_number = fail_create_number
        self.forced_existing_batch_id = forced_existing_batch_id
        self.create_count = 0
        self.tasks_by_key: dict[str, tuple[int, int]] = {}
        self.batch_by_id: dict[int, int] = {}

    def create_source_sync_task(self, task: NewSourceSyncTask) -> int:
        self.create_count += 1
        self.events.append(f"task:create:{task.source_record_id}")
        if self.fail_create_number == self.create_count:
            raise RuntimeError("simulated task persistence failure")
        existing = self.tasks_by_key.get(task.dedupe_key)
        if existing is not None:
            return existing[0]
        task_id = 800 + len(self.tasks_by_key) + 1
        persisted_batch_id = self.forced_existing_batch_id or task.batch_id
        self.tasks_by_key[task.dedupe_key] = (task_id, persisted_batch_id)
        self.batch_by_id[task_id] = persisted_batch_id
        return task_id

    def get_source_sync_task(self, task_id: int) -> Any:
        self.events.append(f"task:get:{task_id}")
        return SimpleNamespace(batch_id=self.batch_by_id[task_id])


def _scan_service(
    source: FakeSourceAdapter,
    state: FakeSourceStateRepository,
    tasks: FakeTaskRepository,
) -> IncrementalSourceScanService:
    """使用内存 Fake 构造一张表扫描服务。"""

    return IncrementalSourceScanService(
        source_adapter=source,  # type: ignore[arg-type]
        source_state_repository=state,  # type: ignore[arg-type]
        task_repository=tasks,  # type: ignore[arg-type]
        policy_version=POLICY_VERSION,
        page_size=2,
        lease_seconds=120,
    )


def test_source_scan_creates_fixed_range_and_persists_tasks_before_cursor() -> None:
    """新范围应固定上界，并严格先落任务再推进每页临时游标。"""

    events: list[str] = []
    candidates = [
        _candidate(record_id=901),
        _candidate(record_id=902, updated_at=SOURCE_TIME + timedelta(seconds=1)),
    ]
    source = FakeSourceAdapter(
        upper_bound=SCAN_UPPER_BOUND,
        pages=[candidates, []],
        events=events,
    )
    state = FakeSourceStateRepository(events=events)
    tasks = FakeTaskRepository(events=events)

    result = _scan_service(source, state, tasks).scan_source(
        "experts_work_experience",
        "scanner-p3-5-attempt-1",
        batch_no="incremental-work-20260825-001",
    )

    assert result.batch_source is not None
    assert result.batch_source.current_cursor == SCAN_UPPER_BOUND
    assert result.batch_source.candidate_count == 2
    assert result.recovered is False
    assert result.page_count == 1
    assert result.scheduled_candidate_count == 2
    assert result.lease_released is False
    assert state.created_batch_nos == ["incremental-work-20260825-001"]
    assert state.advance_calls == [
        (SCAN_START, candidates[-1].cursor, 2),
        (candidates[-1].cursor, SCAN_UPPER_BOUND, 0),
    ]
    first_advance = events.index("state:advance:experts_work_experience:2")
    assert events.index("task:get:801") < first_advance
    assert events.index("task:get:802") < first_advance


def test_source_scan_handles_same_timestamp_pages_and_later_low_record_id() -> None:
    """复合游标应跨同时间戳分页，并发现更新时间较新但 ID 较小的记录。"""

    next_time = SOURCE_TIME + timedelta(seconds=1)
    upper_bound = SourceScanCursor(next_time, 1)
    pages = [
        [
            _candidate(record_id=901, updated_at=SOURCE_TIME),
            _candidate(record_id=902, updated_at=SOURCE_TIME),
        ],
        [
            _candidate(record_id=903, updated_at=SOURCE_TIME),
            _candidate(record_id=1, updated_at=next_time),
        ],
        [],
    ]
    source = FakeSourceAdapter(upper_bound=upper_bound, pages=pages)
    state = FakeSourceStateRepository()
    tasks = FakeTaskRepository()

    result = _scan_service(source, state, tasks).scan_source(
        "experts_work_experience",
        "scanner-p3-8-composite-cursor",
        batch_no="incremental-work-20260825-p3-8",
    )

    assert result.page_count == 2
    assert result.scheduled_candidate_count == 4
    assert result.batch_source is not None
    assert result.batch_source.current_cursor == upper_bound
    assert state.advance_calls == [
        (SCAN_START, SourceScanCursor(SOURCE_TIME, 902), 2),
        (SourceScanCursor(SOURCE_TIME, 902), upper_bound, 2),
    ]
    assert len(tasks.tasks_by_key) == 4


def test_source_scan_releases_lease_without_new_range() -> None:
    """Source 上界不大于安全水位时不能伪造批次或推进任何位置。"""

    source = FakeSourceAdapter(upper_bound=SCAN_START, pages=[])
    state = FakeSourceStateRepository()
    tasks = FakeTaskRepository()

    result = _scan_service(source, state, tasks).scan_source(
        "experts",
        "scanner-p3-5-attempt-2",
        batch_no="incremental-experts-20260825-001",
    )

    assert result.batch_source is None
    assert result.lease_released is True
    assert state.release_count == 1
    assert state.created_batch_nos == []
    assert tasks.create_count == 0


def test_source_scan_recovers_saved_range_without_recomputing_upper_bound() -> None:
    """接管未完成批次时必须复用原 start/current/upper_bound。"""

    current_cursor = SourceScanCursor(SOURCE_TIME, 901)
    recovered_scan = BatchSourceScan(
        batch_id=701,
        source_table="experts_work_experience",
        start_cursor=SCAN_START,
        current_cursor=current_cursor,
        upper_bound=SCAN_UPPER_BOUND,
        checkpoint_lock_version=0,
        status="running",
        candidate_count=1,
    )
    source = FakeSourceAdapter(
        upper_bound=None,
        pages=[[]],
        reject_upper_bound_read=True,
    )
    state = FakeSourceStateRepository(scan=recovered_scan)
    tasks = FakeTaskRepository()

    result = _scan_service(source, state, tasks).scan_source(
        "experts_work_experience",
        "scanner-p3-5-attempt-3",
        batch_no="ignored-during-recovery",
    )

    assert result.recovered is True
    assert result.batch_source is not None
    assert result.batch_source.current_cursor == SCAN_UPPER_BOUND
    assert source.upper_bound_calls == 0
    assert state.created_batch_nos == []


def test_task_failure_does_not_advance_page_cursor() -> None:
    """任一任务创建失败时整页临时游标必须保持不变。"""

    candidates = [_candidate(record_id=901), _candidate(record_id=902)]
    source = FakeSourceAdapter(
        upper_bound=SCAN_UPPER_BOUND,
        pages=[candidates],
    )
    state = FakeSourceStateRepository()
    tasks = FakeTaskRepository(fail_create_number=2)

    with pytest.raises(RuntimeError, match="task persistence failure"):
        _scan_service(source, state, tasks).scan_source(
            "experts_work_experience",
            "scanner-p3-5-attempt-4",
            batch_no="incremental-work-20260825-002",
        )

    assert state.scan is not None
    assert state.scan.current_cursor == SCAN_START
    assert state.advance_calls == []


def test_tasks_are_replayed_idempotently_after_cursor_commit_crash() -> None:
    """任务已落库但页游标提交失败时，接管重放应复用原任务并继续。"""

    candidate = _candidate(record_id=901)
    source = FakeSourceAdapter(
        upper_bound=SCAN_UPPER_BOUND,
        pages=[[candidate], [candidate], []],
    )
    state = FakeSourceStateRepository(fail_advance_once=True)
    tasks = FakeTaskRepository()
    service = _scan_service(source, state, tasks)

    with pytest.raises(RuntimeError, match="cursor commit crash"):
        service.scan_source(
            "experts_work_experience",
            "scanner-p3-5-attempt-5",
            batch_no="incremental-work-20260825-003",
        )

    result = service.scan_source(
        "experts_work_experience",
        "scanner-p3-5-attempt-6",
        batch_no="ignored-during-recovery",
    )

    assert result.recovered is True
    assert result.batch_source is not None
    assert result.batch_source.current_cursor == SCAN_UPPER_BOUND
    assert tasks.create_count == 2
    assert len(tasks.tasks_by_key) == 1
    assert source.upper_bound_calls == 1


def test_cross_batch_dedupe_hit_stops_before_cursor_advance() -> None:
    """任务幂等键命中其他 batch_id 时必须 fail-closed。"""

    source = FakeSourceAdapter(
        upper_bound=SCAN_UPPER_BOUND,
        pages=[[_candidate(record_id=901)]],
    )
    state = FakeSourceStateRepository()
    tasks = FakeTaskRepository(forced_existing_batch_id=999)

    with pytest.raises(CrossBatchTaskConflictError, match="其他批次"):
        _scan_service(source, state, tasks).scan_source(
            "experts_work_experience",
            "scanner-p3-5-attempt-7",
            batch_no="incremental-work-20260825-004",
        )

    assert state.advance_calls == []


def test_recovered_sealed_range_does_not_read_source_or_create_tasks() -> None:
    """已封口的 failed 范围只恢复状态，不重复读取 Source 或创建任务。"""

    sealed_scan = BatchSourceScan(
        batch_id=701,
        source_table="experts_work_experience",
        start_cursor=SCAN_START,
        current_cursor=SCAN_UPPER_BOUND,
        upper_bound=SCAN_UPPER_BOUND,
        checkpoint_lock_version=0,
        status="failed",
        candidate_count=2,
    )
    source = FakeSourceAdapter(
        upper_bound=None,
        pages=[],
        reject_upper_bound_read=True,
    )
    state = FakeSourceStateRepository(scan=sealed_scan)
    tasks = FakeTaskRepository()

    result = _scan_service(source, state, tasks).scan_source(
        "experts_work_experience",
        "scanner-p3-5-attempt-8",
        batch_no="ignored-during-recovery",
    )

    assert result.recovered is True
    assert result.page_count == 0
    assert result.batch_source is not None
    assert result.batch_source.status == "running"
    assert source.page_calls == []
    assert tasks.create_count == 0
    assert state.advance_calls == [
        (SCAN_UPPER_BOUND, SCAN_UPPER_BOUND, 0),
    ]
