"""P4.5~P4.7 Scope Handler：把定向读取的当前事实转为期望 Chunk 集。

本模块只负责生成当前 scope 的完整 desired Chunk 集合，不查询 Milvus、不计算
previous state/stale ID、不调用下游 Consumer、不修改任务状态/checkpoint/scanner/
lease/retry，也不触碰 rag_vector_store。它复用 P4.4 单对象 Normalize、现有
ChunkFactory 和 validate_chunks，不复制清洗、Hash、ID 或 Metadata 组装逻辑。

四类范围的处理语义（V2.2 方案 P4.5~P4.7）：

* ``expert`` 范围
  - 专家有效：生成 profile + 全部 employment + 全部 project 的完整 desired；
  - 专家已删除（``experts_deleted_at`` 非空）：返回 expert scope + 空 desired。
* ``profile`` 范围
  - 使用专家主档和全部当前有效教育重建 profile_summary；education 触发同样走该路径；
  - 父专家已删除：升级为 expert scope + 空 desired。
* ``employment``/``project`` 范围
  - 当前记录有效：只生成该 ``source_field_id`` 的 ``0..M-1`` 分片；
  - 当前记录逻辑删除：返回对应 scope + 空 desired；
  - 父专家已删除：升级为 expert scope + 空 desired。

空 desired 是合法清空语义（不是 Skip），下游据此停用历史 Chunk。验证失败一律
fail-closed，``ChunkValidationError`` 直接向上传播，不在本层吞异常。
"""

from __future__ import annotations

from collections.abc import Iterable

from .chunk_builder import ChunkFactory
from .models import (
    ChunkRecord,
    EmploymentSource,
    ExpertAggregate,
    PreparedChunkSet,
    ProjectSource,
    ScopedCurrentFacts,
    impact_scope_from_target,
)
from .normalize import (
    clean_text,
    normalize_aggregate,
    normalize_employment,
    normalize_profile,
    normalize_project,
)
from .validators import validate_chunks

__all__ = [
    "handle_employment_scope",
    "handle_expert_scope",
    "handle_profile_scope",
    "handle_project_scope",
]


# ---------------------------------------------------------------------------
# 内部公共辅助
# ---------------------------------------------------------------------------


def _require_scope(facts: ScopedCurrentFacts, expected_type: str) -> None:
    """校验当前事实的范围类型与专家身份一致性。

    Handler 必须只处理自己声明的范围；范围类型不匹配说明调用方错误路由，必须
    fail-closed 停止，避免把一个 scope 的事实写给另一个 scope。
    """

    if facts.scope.scope_type != expected_type:
        raise ValueError(
            f"期望范围 {expected_type}，实际收到 {facts.scope.scope_type}"
        )
    if facts.expert.id != facts.scope.expert_id:
        raise ValueError(
            f"事实专家 id={facts.expert.id} 与范围 expert_id={facts.scope.expert_id} 不一致"
        )


def _require_owned_children(facts: ScopedCurrentFacts) -> None:
    """校验所有子表行均归属当前 scope 专家。

    防止畸形 `ScopedCurrentFacts` 把另一位专家的子表行混入当前范围：这会污染
    `source_hash`，或当该行不产生 Chunk 时静默消失。P4.3 读取层已保证归属，
    但本层 fail-closed 二次校验，不信任调用方传入完全正确。
    """

    expected = facts.scope.expert_id
    for row in facts.employments:
        if row.expert_id != expected:
            raise ValueError(
                f"工作行 expert_id={row.expert_id} 与范围 expert_id={expected} 不一致"
            )
    for row in facts.projects:
        if row.expert_id != expected:
            raise ValueError(
                f"项目行 expert_id={row.expert_id} 与范围 expert_id={expected} 不一致"
            )
    for row in facts.educations:
        if row.expert_id != expected:
            raise ValueError(
                f"教育行 expert_id={row.expert_id} 与范围 expert_id={expected} 不一致"
            )


def _expert_empty_result(
    facts: ScopedCurrentFacts, *, task_id: int, batch_id: int, data_version: str
) -> PreparedChunkSet:
    """父专家已删除时升级为 expert scope + 空 desired。

    重新构造一个 expert 类型的影响范围，保证下游按专家级别执行逻辑清理，
    而不是只清空触发的那一条记录的 family。空 desired 是合法清空语义。
    """

    expert_scope = impact_scope_from_target(target_type="expert", expert_id=facts.expert.id)
    return PreparedChunkSet(
        task_id=task_id,
        batch_id=batch_id,
        data_version=data_version,
        impact_scope=expert_scope,
        desired_chunks=(),
    )


def _prepared_result(
    facts: ScopedCurrentFacts,
    chunks: Iterable[ChunkRecord],
    *,
    task_id: int,
    batch_id: int,
    data_version: str,
) -> PreparedChunkSet:
    """对生成结果执行 fail-closed 校验后构造期望 Chunk 集。

    ``PreparedChunkSet`` 会再次校验每个 Chunk 归属当前范围并做稳定排序去重，
    但本层仍先调用 ``validate_chunks`` 做完整契约校验（vector_doc_id 唯一性、
    字段格式、Metadata 一致性等），失败直接向上传播，不捕获、不降级。
    """

    materialized = tuple(chunks)
    validate_chunks(list(materialized))
    return PreparedChunkSet(
        task_id=task_id,
        batch_id=batch_id,
        data_version=data_version,
        impact_scope=facts.scope,
        desired_chunks=materialized,
    )


def _scope_empty_result(
    facts: ScopedCurrentFacts, *, task_id: int, batch_id: int, data_version: str
) -> PreparedChunkSet:
    """当前 scope 的空 desired（合法清空），保留原范围不变。"""

    return PreparedChunkSet(
        task_id=task_id,
        batch_id=batch_id,
        data_version=data_version,
        impact_scope=facts.scope,
        desired_chunks=(),
    )


# ---------------------------------------------------------------------------
# P4.7 Expert 生命周期 Handler
# ---------------------------------------------------------------------------


def handle_expert_scope(
    facts: ScopedCurrentFacts,
    factory: ChunkFactory,
    *,
    task_id: int,
    batch_id: int,
    data_version: str,
) -> PreparedChunkSet:
    """生成 expert 范围的完整 desired Chunk 集。

    专家已删除（``experts_deleted_at`` 非空）返回 expert scope 空 desired，不读取
    三张经历正文、不生成 Disable Action、不调用下游。专家有效时按 P4.1 冻结一致性
    四表快照复用 ``normalize_aggregate -> ChunkFactory.build_all -> validate_chunks``
    生成完整当前集合。恢复不读取或激活历史 inactive 行，只交付当前有效事实。
    """

    _require_scope(facts, "expert")
    if facts.expert_deleted:
        return _expert_empty_result(
            facts, task_id=task_id, batch_id=batch_id, data_version=data_version
        )
    _require_owned_children(facts)
    aggregate = ExpertAggregate(
        profile=facts.expert,
        employments=facts.employments,
        projects=facts.projects,
        educations=facts.educations,
    )
    profile_stage, employment_stages, project_stages = normalize_aggregate(
        aggregate, data_version=data_version, build_batch_id=batch_id
    )
    chunks = factory.build_all(profile_stage, employment_stages, project_stages)
    return _prepared_result(
        facts, chunks, task_id=task_id, batch_id=batch_id, data_version=data_version
    )


# ---------------------------------------------------------------------------
# P4.5 Profile 与 Education Handler
# ---------------------------------------------------------------------------


def handle_profile_scope(
    facts: ScopedCurrentFacts,
    factory: ChunkFactory,
    *,
    task_id: int,
    batch_id: int,
    data_version: str,
) -> PreparedChunkSet:
    """生成 profile 范围的 desired Chunk 集。

    使用专家主档和全部当前有效教育重建 profile_summary；education 触发同样走该
    路径，不查看事件行的学校/专业。父专家已删除时升级为 expert scope 空 desired。
    教育全删或正文不可构建时返回 profile scope 空 desired，不伪造空正文 Chunk。
    """

    _require_scope(facts, "profile")
    if facts.expert_deleted:
        return _expert_empty_result(
            facts, task_id=task_id, batch_id=batch_id, data_version=data_version
        )
    _require_owned_children(facts)
    profile_stage = normalize_profile(
        facts.expert,
        facts.educations,
        data_version=data_version,
        build_batch_id=batch_id,
    )
    chunks = factory.build_profile(profile_stage)
    return _prepared_result(
        facts, chunks, task_id=task_id, batch_id=batch_id, data_version=data_version
    )


# ---------------------------------------------------------------------------
# P4.6 Employment 与 Project Handler
# ---------------------------------------------------------------------------


def handle_employment_scope(
    facts: ScopedCurrentFacts,
    factory: ChunkFactory,
    *,
    task_id: int,
    batch_id: int,
    data_version: str,
) -> PreparedChunkSet:
    """生成 employment 范围的 desired Chunk 集。

    当前工作记录有效：只生成该 ``source_field_id`` 的 ``0..M-1`` 分片；记录逻辑
    删除：返回 employment scope 空 desired；父专家已删除：升级为 expert scope 空
    desired。``serial_number`` 复用 ``clean_text`` 规则（与 normalize_profile 一致），
    不为取得该字段虚构完整 Profile 聚合，也不复制清洗逻辑。
    """

    _require_scope(facts, "employment")
    employment_row = facts.employments[0]
    _require_child_identity(employment_row, facts, expected_source_field_id=facts.scope.source_field_id)
    # 父专家删除升级为 expert scope 空 desired，但子行身份必须先通过校验，
    # 避免一个错误的删除信号被父删除快路径掩盖而错清到其他 family。
    if facts.expert_deleted:
        return _expert_empty_result(
            facts, task_id=task_id, batch_id=batch_id, data_version=data_version
        )
    if employment_row.deleted_at is not None:
        return _scope_empty_result(
            facts, task_id=task_id, batch_id=batch_id, data_version=data_version
        )
    stage = normalize_employment(
        employment_row, data_version=data_version, build_batch_id=batch_id
    )
    serial_number = clean_text(facts.expert.serial_number) or None
    chunks = factory.build_employment(serial_number, stage)
    return _prepared_result(
        facts, chunks, task_id=task_id, batch_id=batch_id, data_version=data_version
    )


def handle_project_scope(
    facts: ScopedCurrentFacts,
    factory: ChunkFactory,
    *,
    task_id: int,
    batch_id: int,
    data_version: str,
) -> PreparedChunkSet:
    """生成 project 范围的 desired Chunk 集。

    与 employment 对称：当前项目记录有效时只生成该 ``source_field_id`` 的
    ``0..M-1`` 分片；逻辑删除返回 project scope 空 desired；父专家已删除升级为
    expert scope 空 desired。
    """

    _require_scope(facts, "project")
    project_row = facts.projects[0]
    _require_child_identity(project_row, facts, expected_source_field_id=facts.scope.source_field_id)
    if facts.expert_deleted:
        return _expert_empty_result(
            facts, task_id=task_id, batch_id=batch_id, data_version=data_version
        )
    if project_row.deleted_at is not None:
        return _scope_empty_result(
            facts, task_id=task_id, batch_id=batch_id, data_version=data_version
        )
    stage = normalize_project(
        project_row, data_version=data_version, build_batch_id=batch_id
    )
    serial_number = clean_text(facts.expert.serial_number) or None
    chunks = factory.build_project(serial_number, stage)
    return _prepared_result(
        facts, chunks, task_id=task_id, batch_id=batch_id, data_version=data_version
    )


def _require_child_identity(
    child_row: EmploymentSource | ProjectSource,
    facts: ScopedCurrentFacts,
    *,
    expected_source_field_id: str,
) -> None:
    """校验子表触发行身份与 scope 完全一致。

    子表行 ``expert_id`` 必须等于 scope 的专家 ID，行主键字符串必须等于 scope 的
    ``source_field_id``。逻辑删除行也必须先通过身份校验，避免把一个删除信号错误
    清空到另一位专家或另一条经历的 family。当前行 ``expert_id`` 与任务
    ``expert_id`` 不一致时停止，不把记录写给错误专家。
    """

    if child_row.expert_id != facts.scope.expert_id:
        raise ValueError(
            f"子表行 expert_id={child_row.expert_id} 与范围 expert_id="
            f"{facts.scope.expert_id} 不一致"
        )
    if str(child_row.id) != expected_source_field_id:
        raise ValueError(
            f"子表行 id={child_row.id} 与范围 source_field_id="
            f"{expected_source_field_id} 不一致"
        )
