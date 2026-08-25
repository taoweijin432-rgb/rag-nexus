"""P3.4 Poll 候选确定性任务映射测试。"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone

import pytest

from expert_index.models import NewSourceSyncTask, SourceChangeCandidate
from expert_index.scanner import (
    IncrementalScanner,
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
