"""验证 P7.5 四表物理 DELETE 的最小影响范围。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from expert_index.cdc import prepare_cdc_delete_batch, prepare_cdc_delete_task
from expert_index.models import (
    EducationSource,
    ExpertSource,
    PreparedChunkSet,
    ScopedCurrentFacts,
    SourceSyncTask,
    impact_scope_from_target,
)

NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)


def _task(
    task_id: int,
    source_table: str,
    source_record_id: int,
    target_type: str,
    *,
    expert_id: int = 101,
) -> SourceSyncTask:
    """构造不含来源正文的 CDC DELETE Task。"""

    return SourceSyncTask(
        task_id=task_id,
        dedupe_key=f"{task_id:064x}",
        batch_id=88,
        event_source="cdc",
        source_table=source_table,
        source_record_id=source_record_id,
        source_updated_at=None,
        expert_id=expert_id,
        operation_type="delete",
        target_type=target_type,
        processing_stage="candidate",
        sync_status="pending",
        retry_count=0,
        policy_version="1.3",
        created_at=NOW,
        updated_at=NOW,
    )


class _Source:
    """只记录教育删除当前事实读取，不提供普通触发行读取。"""

    def __init__(self, facts: ScopedCurrentFacts | None = None) -> None:
        self.facts = facts
        self.calls: list[int] = []

    def read_cdc_education_delete_facts(self, expert_id: int) -> ScopedCurrentFacts | None:
        self.calls.append(expert_id)
        return self.facts


def _active_profile_facts(expert_id: int = 101) -> ScopedCurrentFacts:
    """构造专家与删除后剩余教育的 Profile 当前事实。"""

    expert = ExpertSource(
        id=expert_id,
        serial_number=f"S{expert_id}",
        company="ACME",
        job_status="active",
        job_name="Engineer",
        contact_status="normal",
        doubtful_flag="normal",
        abroad_flag=1,
        background="AI researcher",
        work_years=8.0,
        updated_at=NOW,
        created_at=NOW,
        deleted_at=None,
    )
    education = EducationSource(
        id=702,
        expert_id=expert_id,
        school_name="MIT",
        major="CS",
        updated_at=NOW,
        created_at=NOW,
        deleted_at=None,
    )
    return ScopedCurrentFacts(
        scope=impact_scope_from_target(target_type="profile", expert_id=expert_id),
        expert=expert,
        educations=(education,),
    )


@pytest.mark.parametrize(
    ("source_table", "source_record_id", "target_type", "scope_type"),
    (
        ("experts", 101, "expert", "expert"),
        ("experts_work_experience", 501, "employment", "employment"),
        ("experts_project_experience", 601, "project", "project"),
    ),
)
def test_non_education_delete_builds_empty_scope_without_source_read(
    source_table: str,
    source_record_id: int,
    target_type: str,
    scope_type: str,
) -> None:
    """专家、工作、项目物理删除不重读已删除行，直接交付空 desired。"""

    source = _Source()
    prepared = prepare_cdc_delete_task(
        _task(1, source_table, source_record_id, target_type),
        source,
        object(),  # type: ignore[arg-type]
    )

    assert prepared.impact_scope.scope_type == scope_type
    assert prepared.desired_chunks == ()
    assert source.calls == []


def test_parent_delete_in_same_batch_upgrades_child_scopes() -> None:
    """同批专家 DELETE 覆盖子表范围，且不读取教育剩余事实。"""

    source = _Source()
    tasks = (
        _task(1, "experts", 101, "expert"),
        _task(2, "experts_work_experience", 501, "employment"),
        _task(3, "experts_education_experience", 701, "education"),
    )

    prepared = prepare_cdc_delete_batch(tasks, source, object())  # type: ignore[arg-type]

    assert [item.task_id for item in prepared] == [1, 2, 3]
    assert {item.impact_scope.scope_type for item in prepared} == {"expert"}
    assert all(item.is_empty for item in prepared)
    assert source.calls == []


def test_education_delete_rebuilds_profile_from_remaining_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """教育删除跳过已删除触发行，复用 Profile Handler 处理剩余事实。"""

    facts = _active_profile_facts()
    source = _Source(facts)
    expected = PreparedChunkSet(
        task_id=4,
        batch_id=88,
        data_version="88",
        impact_scope=facts.scope,
        desired_chunks=(),
    )
    calls: list[ScopedCurrentFacts] = []

    def handle(current: ScopedCurrentFacts, _factory: object, **_kwargs: object) -> PreparedChunkSet:
        calls.append(current)
        return expected

    monkeypatch.setattr("expert_index.cdc.delete_scope.handle_profile_scope", handle)
    prepared = prepare_cdc_delete_task(
        _task(4, "experts_education_experience", 701, "education"),
        source,
        object(),  # type: ignore[arg-type]
    )

    assert prepared is expected
    assert source.calls == [101]
    assert calls == [facts]


def test_education_delete_with_missing_parent_clears_expert_scope() -> None:
    """教育删除后父专家物理不存在时升级为专家生命周期清空。"""

    source = _Source(None)
    prepared = prepare_cdc_delete_task(
        _task(5, "experts_education_experience", 701, "education"),
        source,
        object(),  # type: ignore[arg-type]
    )

    assert prepared.impact_scope.scope_type == "expert"
    assert prepared.is_empty is True
    assert source.calls == [101]


def test_p7_5_rejects_non_cdc_delete_task() -> None:
    """Poll 或非 DELETE 任务不能误入物理删除准备器。"""

    task = _task(6, "experts_work_experience", 501, "employment")
    invalid = SourceSyncTask(**{**task.__dict__, "event_source": "poll"})
    with pytest.raises(ValueError, match="event_source=cdc"):
        prepare_cdc_delete_task(invalid, _Source(), object())  # type: ignore[arg-type]
