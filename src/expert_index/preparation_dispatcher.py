"""P4.8 单任务 Preparation Dispatcher。

把"已由 P3 领取且持有有效 attempt token 的任务"路由到对应 scope Handler，
产出内存 :class:`PreparedChunkSet`，完成 P4 闭环。

职责边界（严格）：

- 只做内存层面的状态/token 一致性检查：确认任务处于 ``processing`` 且 token 匹配，
  但**不写状态、不续租、不推进阶段、不完成任务、不解读 lease 有效性**。
  真正的 lease 有效性由外层 Coordinator 调用 Repository 续租时以 MySQL
  ``CURRENT_TIMESTAMP(6)`` + ``attempt_token`` + rowcount fencing 权威判定。
- 只读取一次当前事实（复用 P4.3 ``MySQLSourceAdapter.read_current_facts``），只调用
  一个目标 Handler。时序校验由 P4.3 在 ``read_current_facts`` 内部完成，本层不重复实现。
- 异常 fail-closed：missing、归属错误、时序错误、``ChunkValidationError`` 全部向上传播，
  Dispatcher 不吞异常、不包装、不返回部分集合。
- 不调用 Consumer、Embedding、Milvus；不更新任务状态、checkpoint、lease/retry。
- ``task_id``/``batch_id`` 使用正式任务值，``data_version`` 固定为 ``str(batch_id)``。
- ``scope.scope_type == 'profile'`` 的 education 任务经 P4.3 ``source_sync_task_to_read_request``
  已映射为 profile scope，本层无需特判。

本 Dispatcher 是纯 preparation 入口；P3 task lease/stage 协调由外层
:class:`PreparationCoordinator`（``preparation_coordinator.py``）负责。
下游 Consumer 接线属 P5，本阶段只返回 :class:`PreparedChunkSet`。
"""

from __future__ import annotations

from . import scope_handlers
from .chunk_builder import ChunkFactory
from .models import (
    PreparedChunkSet,
    ScopedCurrentFacts,
    SourceSyncTask,
    source_sync_task_to_read_request,
)
from .source import MySQLSourceAdapter

__all__ = [
    "build_prepared_chunk_set",
    "prepare_claimed_source_sync_task",
    "read_scoped_facts",
]


def _validate_token(token: str) -> None:
    """校验调用方传入的 attempt token 可精确比对且满足 VARCHAR(64) 约束。"""

    if (
        not isinstance(token, str)
        or not token.strip()
        or token != token.strip()
        or len(token) > 64
    ):
        raise ValueError("attempt_token 必须是长度不超过 64 的非空字符串")


def prepare_claimed_source_sync_task(
    task: SourceSyncTask,
    attempt_token: str,
    source: MySQLSourceAdapter,
    factory: ChunkFactory,
) -> PreparedChunkSet:
    """把一个已领取的来源同步任务转换为 :class:`PreparedChunkSet`。

    :param task: P3 已领取任务模型（``sync_status == 'processing'``）。
    :param attempt_token: 调用方持有的本次认领令牌，必须与 ``task.attempt_token`` 完全一致。
    :param source: P4.3 定向只读事实适配器。
    :param factory: ChunkFactory，注入 ``task_id/batch_id/data_version`` 并提供策略版本。
    :return: 目标 scope Handler 产出的 :class:`PreparedChunkSet`，原样返回不改写。

    :raises ValueError: 任务非 :class:`SourceSyncTask`、状态非 processing、token 不匹配，
        或策略版本缺失/不一致。
    :raises Exception: 上游 mapping、source 读取、归属、时序、ChunkFactory、
        ``validate_chunks`` 等任何异常原样向上传播，不吞不包装。

    lease 有效性不在本层判定：外层 :class:`PreparationCoordinator` 在调用本函数前已通过
    Repository 续租以 MySQL ``CURRENT_TIMESTAMP(6)`` 权威校验。本层不解读
    ``task.lease_until`` 的时区或与本机时间比较。
    """

    if not isinstance(task, SourceSyncTask):
        raise TypeError("task 必须是 SourceSyncTask")
    if task.sync_status != "processing":
        raise ValueError(
            f"任务状态非 processing，实际为 {task.sync_status!r}，无法准备"
        )

    _validate_token(attempt_token)
    if task.attempt_token is None or task.attempt_token != attempt_token:
        raise ValueError("attempt_token 与任务当前令牌不一致，调用方不再持有有效 lease")

    # 策略版本校验：以注入 Factory 的权威配置为准，不引入第二个策略常量。
    # 历史迁移任务的 policy_version 可能为 None，但未知策略任务无法在当前策略下安全准备。
    expected_policy = factory.settings.chunk_policy_version
    if task.policy_version is None:
        raise ValueError("已领取任务缺少 policy_version，无法在当前策略下准备")
    if task.policy_version != expected_policy:
        raise ValueError(
            f"任务 policy_version={task.policy_version!r} "
            f"与当前 ChunkFactory policy_version={expected_policy!r} 不一致"
        )

    # 复用 P4.3 mapping（education→profile 已在此完成）+ 防御性 scope 一致性校验。
    # 复用 read_scoped_facts，保证“整函数路径”与“拆分路径”走同一套防御。
    facts = read_scoped_facts(task, source)
    return build_prepared_chunk_set(task, facts, factory)


def read_scoped_facts(
    task: SourceSyncTask,
    source: MySQLSourceAdapter,
) -> ScopedCurrentFacts:
    """读取当前事实但不构建 Chunk。

    供外层 :class:`PreparationCoordinator` 在阶段推进之间拆分使用：
    读取事实后推进 ``source_read``，再由 :func:`build_prepared_chunk_set` 完成
    Normalize/Build/Validate。复用 P4.3 mapping（education→profile）与
    ``read_current_facts`` 内部的时序校验。
    """

    request = source_sync_task_to_read_request(task)
    facts: ScopedCurrentFacts = source.read_current_facts(request)
    # 防御畸形 source：返回事实的范围必须与请求范围一致，否则可能路由到错误 Handler。
    if facts.scope != request.impact_scope:
        raise ValueError("source 返回的 scope 与请求 scope 不一致，拒绝分发")
    return facts


def build_prepared_chunk_set(
    task: SourceSyncTask,
    facts: ScopedCurrentFacts,
    factory: ChunkFactory,
) -> PreparedChunkSet:
    """按 scope 分发到目标 Handler，复用单对象 Normalize + ChunkFactory + Validator。

    供外层 :class:`PreparationCoordinator` 在 ``normalized`` 与 ``chunk_built`` 之间
    拆分使用；Handler 内部 Normalize/Build/Validate 失败一律 fail-closed 向上传播，
    不返回部分集合。本函数不读取事实、不写状态、不调 Consumer/Embedding/Milvus。
    """

    # 注入正式任务定位参数，data_version 固定为 str(batch_id)。
    task_id = task.task_id
    batch_id = task.batch_id
    data_version = str(batch_id)

    # if/elif 显式分发，便于测试 monkeypatch 证明恰好调用一个 Handler。
    scope_type = facts.scope.scope_type
    if scope_type == "expert":
        return scope_handlers.handle_expert_scope(
            facts, factory, task_id=task_id, batch_id=batch_id, data_version=data_version,
        )
    if scope_type == "profile":
        return scope_handlers.handle_profile_scope(
            facts, factory, task_id=task_id, batch_id=batch_id, data_version=data_version,
        )
    if scope_type == "employment":
        return scope_handlers.handle_employment_scope(
            facts, factory, task_id=task_id, batch_id=batch_id, data_version=data_version,
        )
    if scope_type == "project":
        return scope_handlers.handle_project_scope(
            facts, factory, task_id=task_id, batch_id=batch_id, data_version=data_version,
        )
    # ScopedCurrentFacts 构造时已约束 scope_type，此处防御性 fail-closed。
    raise ValueError(f"不支持的影响范围类型: {scope_type!r}")
