"""来源 Checkpoint 领域模型的最小契约测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from expert_index.models import CHECKPOINT_SOURCE_TABLES, SourceCheckpoint

CHECKPOINT_TIME = datetime(2026, 8, 13, 22, 0, tzinfo=timezone.utc)


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
