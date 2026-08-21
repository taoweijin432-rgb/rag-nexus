"""来源记录级增量同步任务领域模型的最小契约测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from expert_index.models import (
    SOURCE_SYNC_ERROR_STAGES,
    SOURCE_SYNC_OPERATION_TYPES,
    SOURCE_SYNC_TASK_STATUSES,
    SourceSyncTask,
)

CREATED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def _task(**changes: object) -> SourceSyncTask:
    """构造可用任务，并允许单个测试覆盖指定字段。"""

    values: dict[str, object] = {
        "task_id": 1,
        "batch_id": 8,
        "source_table": "experts_work_experience",
        "source_record_id": 100,
        "expert_id": 20,
        "operation_type": "update",
        "sync_status": "pending",
        "retry_count": 0,
        "created_at": CREATED_AT,
    }
    values.update(changes)
    return SourceSyncTask(**values)  # type: ignore[arg-type]


def test_source_sync_task_defines_frozen_status_operation_and_stage_sets() -> None:
    """任务模型与数据库应共享一套封闭的状态、操作和失败阶段术语。"""

    assert SOURCE_SYNC_TASK_STATUSES == {
        "pending",
        "processing",
        "success",
        "failed",
        "retrying",
        "dead",
    }
    assert SOURCE_SYNC_OPERATION_TYPES == {"insert", "update", "delete"}
    assert SOURCE_SYNC_ERROR_STAGES == {
        "source",
        "parse",
        "pii",
        "chunk_builder",
        "generation",
        "validation",
        "embedding",
        "milvus",
        "checkpoint",
    }


def test_source_sync_task_accepts_unmapped_expert_and_safe_failure_summary() -> None:
    """无法映射专家的来源变化仍可保存，等待后续处置而不携带业务正文。"""

    task = _task(
        expert_id=None,
        sync_status="failed",
        retry_count=1,
        error_stage="source",
        error_type="ExpertMappingError",
        error_message="expert_id unavailable",
        result_code="expert_mapping_failed",
        started_at=CREATED_AT,
    )

    assert task.expert_id is None
    assert task.error_stage == "source"
    assert task.retry_count == 1


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("task_id", 0, "task_id 必须是正整数"),
        ("batch_id", True, "batch_id 必须是正整数"),
        ("source_record_id", -1, "source_record_id 必须是正整数"),
        ("expert_id", 0, "expert_id 必须是正整数或 None"),
    ],
)
def test_source_sync_task_rejects_invalid_identifiers(
    field_name: str,
    value: object,
    message: str,
) -> None:
    """任务和来源标识必须能够安全映射到无符号数据库主键。"""

    with pytest.raises(ValueError, match=message):
        _task(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("source_table", "expert_work", "不支持的同步任务来源表"),
        ("operation_type", "upsert", "不支持的同步任务操作类型"),
        ("sync_status", "running", "不支持的同步任务状态"),
        ("error_stage", "vector_store", "不支持的同步任务失败阶段"),
    ],
)
def test_source_sync_task_rejects_unknown_contract_values(
    field_name: str,
    value: object,
    message: str,
) -> None:
    """未知表名、操作、状态和失败阶段不能进入持久化任务。"""

    with pytest.raises(ValueError, match=message):
        _task(**{field_name: value})


@pytest.mark.parametrize("retry_count", [-1, True, 1.5])
def test_source_sync_task_rejects_invalid_retry_count(retry_count: object) -> None:
    """重试次数必须与数据库无符号整数约束一致。"""

    with pytest.raises(ValueError, match="retry_count 必须是非负整数"):
        _task(retry_count=retry_count)


def test_source_sync_task_rejects_unsafe_summary_lengths_and_time_types() -> None:
    """错误摘要长度和时间类型必须符合任务表字段约束。"""

    with pytest.raises(ValueError, match="error_message 必须是长度不超过 512"):
        _task(error_message="x" * 513)
    with pytest.raises(ValueError, match="result_code 必须是长度不超过 64"):
        _task(result_code="")
    with pytest.raises(TypeError, match="completed_at 必须是 datetime 或 None"):
        _task(completed_at="2026-08-21")
