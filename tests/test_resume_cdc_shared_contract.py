"""验证公共 CDC 组件可承载 Resume 契约且不改变 Expert 默认值。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from expert_index.cdc import (
    CANAL_DESTINATION,
    CanalBatch,
    CanalChangeEvent,
    CanalColumn,
    CanalEntry,
    CanalRowData,
    CdcBatchReceiver,
    CdcTaskPlan,
    CdcTaskReceipt,
    build_cdc_event_id,
    parse_canal_batch,
    prepare_cdc_delete_task,
)
from expert_index.models import SourceSyncTask
from resume_index.dataset import RESUME_SPEC

RESUME_DESTINATION = "company_resume"


class _Persistence:
    """只记录 Receiver 生成的计划；当前测试不需要真实 Index Repository。"""

    def __init__(self) -> None:
        self.plans: tuple[CdcTaskPlan, ...] = ()

    def persist_batch(
        self,
        plans: Sequence[CdcTaskPlan],
    ) -> tuple[CdcTaskReceipt, ...]:
        self.plans = tuple(plans)
        return tuple(
            CdcTaskReceipt(plan.cdc_event_id, index, "created")
            for index, plan in enumerate(self.plans, 1)
        )


def _resume_delete(table: str, *, row_id: int, resume_id: int) -> CanalBatch:
    """构造只包含物理行 ID 和 Resume 人物身份的 DELETE batch。"""

    return CanalBatch(
        7,
        (
            CanalEntry(
                entry_type="ROWDATA",
                source_schema="expert",
                source_table=table,
                operation_type="DELETE",
                logfile_name="binlog.000021",
                logfile_offset=100,
                server_id=1,
                execute_time=1,
                rows=(
                    CanalRowData(
                        before=(
                            CanalColumn("id", row_id),
                            CanalColumn("resume_id", resume_id),
                        )
                    ),
                ),
            ),
        ),
        RESUME_DESTINATION,
    )


def test_resume_parser_keeps_physical_id_separate_from_resume_identity() -> None:
    """Resume 主表 id 只定位物理行，人物范围必须使用独立 resume_id。"""

    result = parse_canal_batch(
        _resume_delete("resume", row_id=501, resume_id=9001),
        destination=RESUME_DESTINATION,
        source_schema="expert",
        dataset_spec=RESUME_SPEC,
    )

    assert len(result.delete_events) == 1
    event = result.delete_events[0]
    assert event.source_record_id == 501
    assert event.expert_id == 9001
    assert result.rollback_required is False


def test_resume_receiver_maps_profile_delete_to_expert_scope() -> None:
    """Resume 主档物理删除生成 expert target，供下游停用人物全部 Chunk。"""

    persistence = _Persistence()
    receiver = CdcBatchReceiver(
        identity_provider=lambda _event: "resume-event-1",
        policy_version="1.3",
        destination=RESUME_DESTINATION,
        source_schema="expert",
        dataset_spec=RESUME_SPEC,
    )

    result = receiver.receive(
        _resume_delete("resume", row_id=501, resume_id=9001),
        local_batch_id=88,
        persistence=persistence,
    )

    task = result.plans[0].task
    assert task.source_table == "resume"
    assert task.source_record_id == 501
    assert task.expert_id == 9001
    assert task.target_type == "expert"


def test_resume_profile_delete_does_not_require_row_id_to_equal_resume_id() -> None:
    """DELETE 范围校验允许 Resume 主档的物理 ID 与人物 ID 不同。"""

    task = SourceSyncTask(
        task_id=1,
        dedupe_key="a" * 64,
        batch_id=88,
        event_source="cdc",
        source_table="resume",
        source_record_id=501,
        source_updated_at=None,
        expert_id=9001,
        operation_type="delete",
        target_type="expert",
        processing_stage="candidate",
        sync_status="pending",
        retry_count=0,
        policy_version="1.3",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    prepared = prepare_cdc_delete_task(
        task,
        object(),  # 主档删除不读取 Source。
        object(),  # 主档删除不调用 ChunkFactory。
        dataset_spec=RESUME_SPEC,
    )

    assert prepared.impact_scope.scope_type == "expert"
    assert prepared.impact_scope.expert_id == 9001
    assert prepared.is_empty is True


def test_event_identity_namespace_can_be_isolated_by_destination() -> None:
    """同一 Binlog 位点在 Expert/Resume destination 下不会得到相同事件身份。"""

    event = CanalChangeEvent(
        "expert",
        "resume_work_experience",
        501,
        9001,
        "binlog.000021",
        100,
        1,
        1,
        0,
        0,
    )

    assert build_cdc_event_id(event, destination=RESUME_DESTINATION) != build_cdc_event_id(
        event,
        destination=CANAL_DESTINATION,
    )
