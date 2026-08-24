"""来源记录级增量同步任务领域模型的最小契约测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from expert_index.models import (
    SOURCE_SYNC_ERROR_STAGES,
    SOURCE_SYNC_EVENT_SOURCES,
    SOURCE_SYNC_OPERATION_TYPES,
    SOURCE_SYNC_PROCESSING_STAGES,
    SOURCE_SYNC_TARGET_TYPES,
    SOURCE_SYNC_TASK_STATUSES,
    NewSourceSyncTask,
    SourceSyncTask,
)

CREATED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
DEDUPE_KEY = "a" * 64


def _task(**changes: object) -> SourceSyncTask:
    """构造可用任务，并允许单个测试覆盖指定字段。"""

    values: dict[str, object] = {
        "task_id": 1,
        "dedupe_key": DEDUPE_KEY,
        "batch_id": 8,
        "event_source": "poll",
        "source_table": "experts_work_experience",
        "source_record_id": 100,
        "source_updated_at": CREATED_AT,
        "expert_id": 20,
        "operation_type": "update",
        "target_type": "employment",
        "processing_stage": "candidate",
        "sync_status": "pending",
        "retry_count": 0,
        "policy_version": "1.3",
        "created_at": CREATED_AT,
        "updated_at": CREATED_AT,
    }
    values.update(changes)
    return SourceSyncTask(**values)  # type: ignore[arg-type]


def test_source_sync_task_defines_frozen_identity_and_lifecycle_sets() -> None:
    """任务模型应与 012 共享事件来源、目标、阶段、状态和操作术语。"""

    assert SOURCE_SYNC_TASK_STATUSES == {
        "pending",
        "processing",
        "success",
        "failed",
        "retrying",
        "dead",
    }
    assert SOURCE_SYNC_OPERATION_TYPES == {"insert", "update", "delete"}
    assert SOURCE_SYNC_EVENT_SOURCES == {"poll", "cdc", "reconcile", "replay"}
    assert SOURCE_SYNC_TARGET_TYPES == {
        "expert",
        "profile",
        "employment",
        "project",
        "education",
    }
    assert SOURCE_SYNC_PROCESSING_STAGES == {
        "candidate",
        "source_read",
        "normalized",
        "chunk_built",
        "validated",
        "contract_ready",
        "handed_off",
        "completed",
    }
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


def test_new_source_sync_task_requires_complete_deduplicated_identity() -> None:
    """待创建任务应显式携带跨扫描幂等所需的已冻结身份字段。"""

    task = NewSourceSyncTask(
        dedupe_key=DEDUPE_KEY,
        batch_id=8,
        event_source="poll",
        source_table="experts_education_experience",
        source_record_id=100,
        source_updated_at=CREATED_AT,
        expert_id=20,
        operation_type="update",
        target_type="education",
        policy_version="1.3",
    )

    assert task.dedupe_key == DEDUPE_KEY
    assert task.target_type == "education"


@pytest.mark.parametrize("dedupe_key", ["A" * 64, "a" * 63, "g" * 64, ""])
def test_new_source_sync_task_rejects_noncanonical_dedupe_key(dedupe_key: str) -> None:
    """幂等键必须是数据库 ascii_bin 约束对应的小写 SHA-256。"""

    with pytest.raises(ValueError, match="dedupe_key 必须是 64 位小写 SHA-256"):
        NewSourceSyncTask(
            dedupe_key=dedupe_key,
            batch_id=8,
            event_source="poll",
            source_table="experts",
            source_record_id=100,
            source_updated_at=CREATED_AT,
            expert_id=20,
            operation_type="update",
            target_type="profile",
            policy_version="1.3",
        )


def test_new_source_sync_task_rejects_target_larger_than_source_grain() -> None:
    """经历表候选必须保留同来源记录的最小影响类型。"""

    with pytest.raises(ValueError, match="experts_work_experience 不能定位到 expert"):
        NewSourceSyncTask(
            dedupe_key=DEDUPE_KEY,
            batch_id=8,
            event_source="poll",
            source_table="experts_work_experience",
            source_record_id=100,
            source_updated_at=CREATED_AT,
            expert_id=20,
            operation_type="update",
            target_type="expert",
            policy_version="1.3",
        )


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
        ("event_source", "timer", "不支持的同步任务事件来源"),
        ("operation_type", "upsert", "不支持的同步任务操作类型"),
        ("target_type", "all", "不支持的同步任务目标类型"),
        ("processing_stage", "embedding", "不支持的同步任务处理阶段"),
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


def test_source_sync_task_requires_attempt_token_and_lease_as_a_pair() -> None:
    """任务认领令牌与租约必须成对，避免无法判断所有者的残留租约。"""

    with pytest.raises(ValueError, match="attempt_token 和 lease_until 必须同时"):
        _task(attempt_token="attempt-1")
    with pytest.raises(ValueError, match="attempt_token 和 lease_until 必须同时"):
        _task(lease_until=CREATED_AT)
