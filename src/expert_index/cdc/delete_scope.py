"""P7.5 物理 DELETE 到最小影响范围和 desired 集合的映射。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from ..chunk_builder import ChunkFactory
from ..dataset import EXPERT_SPEC, DatasetSpec
from ..models import (
    PreparedChunkSet,
    ScopedCurrentFacts,
    SourceSyncTask,
    impact_scope_from_target,
)
from ..scope_handlers import handle_profile_scope


class CdcDeleteSourcePort(Protocol):
    """教育物理删除所需的最小 Source 当前事实接口。"""

    def read_cdc_education_delete_facts(
        self,
        expert_id: int,
    ) -> ScopedCurrentFacts | None:
        """返回专家与剩余教育；父专家物理不存在时返回 ``None``。"""


def prepare_cdc_delete_batch(
    tasks: Sequence[SourceSyncTask],
    source: CdcDeleteSourcePort,
    factory: ChunkFactory,
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> tuple[PreparedChunkSet, ...]:
    """按输入顺序准备同一 Canal batch 的物理删除任务。

    同一批次包含专家 DELETE 时，该专家的所有子表 DELETE 都提升为 expert scope 空
    desired，避免级联删除顺序造成较小范围覆盖生命周期结果。每个任务仍保留自己的
    ``task_id``，供 P7.6 独立完成状态与回执。
    """

    normalized = tuple(tasks)
    if not normalized:
        return ()
    for task in normalized:
        _validate_cdc_delete_task(task, dataset_spec=dataset_spec)
    batch_ids = {task.batch_id for task in normalized}
    if len(batch_ids) != 1:
        raise ValueError("CDC DELETE 批次准备不能混入不同 batch_id")
    expert_delete_ids = frozenset(
        task.expert_id
        for task in normalized
        if task.source_table == dataset_spec.profile_table and task.expert_id is not None
    )
    return tuple(
        prepare_cdc_delete_task(
            task,
            source,
            factory,
            parent_delete_expert_ids=expert_delete_ids,
            dataset_spec=dataset_spec,
        )
        for task in normalized
    )


def prepare_cdc_delete_task(
    task: SourceSyncTask,
    source: CdcDeleteSourcePort,
    factory: ChunkFactory,
    *,
    parent_delete_expert_ids: Iterable[int] = (),
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> PreparedChunkSet:
    """把一个 CDC DELETE Task 转为 P5 可消费的 ``PreparedChunkSet``。

    专家、工作和项目删除直接交付空 desired，不读取已删除来源行。教育删除只重读父
    专家和当前剩余教育，并复用 Profile Handler 构建当前 Profile；父专家物理/逻辑删除
    时统一提升为 expert scope 空 desired。此处不比较 Hash、不调用 Embedding/Milvus。
    """

    _validate_cdc_delete_task(task, dataset_spec=dataset_spec)
    expert_id = task.expert_id
    if expert_id is None:  # 防御后续模型放宽校验时静默构造错误范围。
        raise ValueError("CDC DELETE Task 缺少 expert_id")
    parent_deletes = frozenset(parent_delete_expert_ids)
    if expert_id in parent_deletes or task.source_table == dataset_spec.profile_table:
        return _empty_prepared_set(
            task,
            target_type="expert",
            dataset_spec=dataset_spec,
        )
    if task.source_table == dataset_spec.employment_table:
        return _empty_prepared_set(
            task,
            target_type="employment",
            dataset_spec=dataset_spec,
        )
    if task.source_table == dataset_spec.project_table:
        return _empty_prepared_set(
            task,
            target_type="project",
            dataset_spec=dataset_spec,
        )

    facts = source.read_cdc_education_delete_facts(expert_id)
    if facts is None:
        return _empty_prepared_set(
            task,
            target_type="expert",
            dataset_spec=dataset_spec,
        )
    expected_scope = impact_scope_from_target(
        target_type="profile",
        expert_id=expert_id,
        dataset_spec=dataset_spec,
    )
    if facts.scope != expected_scope or facts.expert.id != expert_id:
        raise ValueError("教育删除事实的专家或 Profile 范围与 CDC Task 不一致")
    return handle_profile_scope(
        facts,
        factory,
        task_id=task.task_id,
        batch_id=task.batch_id,
        data_version=str(task.batch_id),
    )


def _empty_prepared_set(
    task: SourceSyncTask,
    *,
    target_type: str,
    dataset_spec: DatasetSpec,
) -> PreparedChunkSet:
    """按当前数据集规格为物理删除构造明确的空 desired 集合。

    ``impact_scope_from_target`` 默认使用 Expert 规格，因此 Resume 调用必须把
    ``dataset_spec`` 继续向下传递，避免子表删除范围回退为 ``experts_*`` 标签。
    """

    expert_id = task.expert_id
    if expert_id is None:
        raise ValueError("CDC DELETE Task 缺少 expert_id")
    scope = impact_scope_from_target(
        target_type=target_type,
        expert_id=expert_id,
        source_field_id=(
            str(task.source_record_id) if target_type in {"employment", "project"} else None
        ),
        dataset_spec=dataset_spec,
    )
    return PreparedChunkSet(
        task_id=task.task_id,
        batch_id=task.batch_id,
        data_version=str(task.batch_id),
        impact_scope=scope,
        desired_chunks=(),
    )


def _validate_cdc_delete_task(
    task: SourceSyncTask,
    *,
    dataset_spec: DatasetSpec,
) -> None:
    """校验任务确实来自四表 CDC 物理删除且身份字段相互一致。"""

    if not isinstance(task, SourceSyncTask):
        raise TypeError("task 必须是 SourceSyncTask")
    if not isinstance(dataset_spec, DatasetSpec):
        raise TypeError("dataset_spec 必须是 DatasetSpec")
    if task.source_table not in dataset_spec.checkpoint_source_tables:
        raise ValueError(f"不支持的 CDC DELETE 来源表: {task.source_table!r}")
    expected_target = (
        "expert"
        if task.source_table == dataset_spec.profile_table
        else dataset_spec.target_type_for_source_table(task.source_table)
    )
    if task.event_source != "cdc" or task.operation_type != "delete":
        raise ValueError("P7.5 只接受 event_source=cdc 的物理 delete 任务")
    if task.source_updated_at is not None:
        raise ValueError("CDC 物理 DELETE 的 source_updated_at 必须为空")
    if task.target_type != expected_target:
        raise ValueError("CDC DELETE 的 source_table 与 target_type 不一致")
    if task.expert_id is None:
        raise ValueError("CDC DELETE Task 缺少 expert_id")
    if (
        task.source_table == dataset_spec.profile_table
        and dataset_spec.profile_identity_column == "id"
        and task.source_record_id != task.expert_id
    ):
        raise ValueError("experts DELETE 的 source_record_id 必须等于 expert_id")


__all__ = [
    "CdcDeleteSourcePort",
    "prepare_cdc_delete_batch",
    "prepare_cdc_delete_task",
]
