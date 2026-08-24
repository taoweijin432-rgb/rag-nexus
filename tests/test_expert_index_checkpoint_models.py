"""来源 Checkpoint 领域模型的最小契约测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from expert_index.models import (
    BATCH_SOURCE_SCAN_STATUSES,
    CHECKPOINT_SOURCE_TABLES,
    BatchSourceScan,
    SourceChangeCandidate,
    SourceCheckpoint,
    SourceScanCursor,
)

CHECKPOINT_TIME = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)


def test_batch_source_scan_records_fixed_range_and_temporary_progress() -> None:
    """来源批次模型应明确区分扫描起点、临时进度和固定上界。"""

    start_cursor = SourceScanCursor(updated_at=CHECKPOINT_TIME, record_id=0)
    current_cursor = SourceScanCursor(updated_at=CHECKPOINT_TIME, record_id=10)
    upper_bound = SourceScanCursor(updated_at=CHECKPOINT_TIME, record_id=20)

    scan = BatchSourceScan(
        batch_id=7,
        source_table="experts_work_experience",
        start_cursor=start_cursor,
        current_cursor=current_cursor,
        upper_bound=upper_bound,
        checkpoint_lock_version=3,
        status="running",
        candidate_count=12,
        success_count=8,
        failed_count=1,
        skipped_count=3,
    )

    assert scan.start_cursor == start_cursor
    assert scan.current_cursor == current_cursor
    assert scan.upper_bound == upper_bound
    assert scan.status in BATCH_SOURCE_SCAN_STATUSES
    assert (
        scan.candidate_count,
        scan.success_count,
        scan.failed_count,
        scan.skipped_count,
    ) == (12, 8, 1, 3)


@pytest.mark.parametrize(
    ("start_id", "current_id", "upper_bound_id"),
    [(1, 0, 20), (0, 21, 20)],
)
def test_batch_source_scan_rejects_cursor_outside_fixed_range(
    start_id: int,
    current_id: int,
    upper_bound_id: int,
) -> None:
    """临时游标不能落在批次扫描起点之前或固定上界之后。"""

    with pytest.raises(ValueError, match="扫描游标必须满足"):
        BatchSourceScan(
            batch_id=7,
            source_table="experts",
            start_cursor=SourceScanCursor(
                updated_at=CHECKPOINT_TIME,
                record_id=start_id,
            ),
            current_cursor=SourceScanCursor(
                updated_at=CHECKPOINT_TIME,
                record_id=current_id,
            ),
            upper_bound=SourceScanCursor(
                updated_at=CHECKPOINT_TIME,
                record_id=upper_bound_id,
            ),
            checkpoint_lock_version=0,
        )


def test_batch_source_scan_rejects_invalid_status_and_lock_version() -> None:
    """扫描状态和 Checkpoint 版本必须符合批次来源表约束。"""

    cursor = SourceScanCursor(updated_at=CHECKPOINT_TIME, record_id=0)
    with pytest.raises(ValueError, match="不支持的批次来源扫描状态"):
        BatchSourceScan(
            batch_id=1,
            source_table="experts",
            start_cursor=cursor,
            current_cursor=cursor,
            upper_bound=cursor,
            checkpoint_lock_version=0,
            status="partial",
        )
    with pytest.raises(ValueError, match="checkpoint_lock_version 必须是非负整数"):
        BatchSourceScan(
            batch_id=1,
            source_table="experts",
            start_cursor=cursor,
            current_cursor=cursor,
            upper_bound=cursor,
            checkpoint_lock_version=-1,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_count": -1},
        {"success_count": True},
        {"failed_count": 1.5},
        {"skipped_count": -1},
    ],
)
def test_batch_source_scan_rejects_invalid_processing_counts(
    changes: dict[str, object],
) -> None:
    """来源表级处理计数必须与 MySQL 无符号计数字段一致。"""

    cursor = SourceScanCursor(updated_at=CHECKPOINT_TIME, record_id=0)
    values: dict[str, object] = {
        "batch_id": 1,
        "source_table": "experts",
        "start_cursor": cursor,
        "current_cursor": cursor,
        "upper_bound": cursor,
        "checkpoint_lock_version": 0,
    }
    values.update(changes)

    with pytest.raises(ValueError, match="必须是非负整数"):
        BatchSourceScan(**values)  # type: ignore[arg-type]


def test_source_scan_cursor_orders_by_time_then_record_id() -> None:
    """复合游标顺序必须与增量 SQL 的 updated_at、id 排序一致。"""

    earlier = SourceScanCursor(updated_at=CHECKPOINT_TIME, record_id=1)
    later = SourceScanCursor(updated_at=CHECKPOINT_TIME, record_id=2)

    assert earlier < later


def test_source_scan_cursor_accepts_initial_zero_record_id() -> None:
    """首次增量扫描可以从指定时间的 ID 0 起点开始。"""

    cursor = SourceScanCursor(updated_at=CHECKPOINT_TIME, record_id=0)

    assert cursor.record_id == 0


def test_source_change_candidate_exposes_cursor_and_delete_state() -> None:
    """最小变化候选应能提供下一页位置和逻辑删除状态。"""

    candidate = SourceChangeCandidate(
        source_table="experts_project_experience",
        record_id=9,
        expert_id=None,
        updated_at=CHECKPOINT_TIME,
        deleted_at=CHECKPOINT_TIME,
    )

    assert candidate.cursor == SourceScanCursor(
        updated_at=CHECKPOINT_TIME,
        record_id=9,
    )
    assert candidate.is_deleted is True


def test_source_checkpoint_accepts_each_supported_source_table() -> None:
    """四张物理来源表都能构造独立且未推进的安全位置。"""

    for source_table in CHECKPOINT_SOURCE_TABLES:
        checkpoint = SourceCheckpoint(
            source_table=source_table,
            last_updated_at=CHECKPOINT_TIME,
            last_id=0,
        )

        assert checkpoint.source_table == source_table
        assert checkpoint.last_batch_id is None
        assert checkpoint.last_success_time is None
        assert checkpoint.lock_version == 0
        assert checkpoint.lease_owner is None
        assert checkpoint.lease_until is None


def test_source_checkpoint_rejects_unknown_source_table() -> None:
    """表名拼写错误不能形成不会被扫描的孤立 Checkpoint。"""

    with pytest.raises(ValueError, match="不支持的 Checkpoint 来源表"):
        SourceCheckpoint(
            source_table="expert_work_experience",
            last_updated_at=CHECKPOINT_TIME,
            last_id=0,
        )


@pytest.mark.parametrize("last_id", [-1, True, 1.5])
def test_source_checkpoint_rejects_invalid_last_id(last_id: object) -> None:
    """来源 ID 必须能安全用于非负主键游标比较。"""

    with pytest.raises(ValueError, match="last_id 必须是非负整数"):
        SourceCheckpoint(
            source_table="experts",
            last_updated_at=CHECKPOINT_TIME,
            last_id=last_id,  # type: ignore[arg-type]
        )


def test_source_checkpoint_rejects_invalid_version_and_batch_id() -> None:
    """并发版本和批次 ID 必须符合数据库字段约束。"""

    with pytest.raises(ValueError, match="last_batch_id 必须是正整数"):
        SourceCheckpoint(
            source_table="experts",
            last_updated_at=CHECKPOINT_TIME,
            last_id=1,
            last_batch_id=0,
        )
    with pytest.raises(ValueError, match="lock_version 必须是非负整数"):
        SourceCheckpoint(
            source_table="experts",
            last_updated_at=CHECKPOINT_TIME,
            last_id=1,
            lock_version=-1,
        )


def test_source_checkpoint_requires_complete_valid_lease() -> None:
    """Checkpoint 租约持有者和到期时间必须成对并符合字段类型。"""

    with pytest.raises(ValueError, match="lease_owner 和 lease_until 必须同时"):
        SourceCheckpoint(
            source_table="experts",
            last_updated_at=CHECKPOINT_TIME,
            last_id=1,
            lease_owner="scanner-1",
        )
    with pytest.raises(TypeError, match="lease_until 必须是 datetime 或 None"):
        SourceCheckpoint(
            source_table="experts",
            last_updated_at=CHECKPOINT_TIME,
            last_id=1,
            lease_owner="scanner-1",
            lease_until="2026-08-21",  # type: ignore[arg-type]
        )
