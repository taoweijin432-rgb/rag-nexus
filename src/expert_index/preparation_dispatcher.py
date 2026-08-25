"""P4.8 单任务 Preparation Dispatcher。

把"已由 P3 领取且持有有效 attempt token 的任务"路由到对应 scope Handler，
产出内存 :class:`PreparedChunkSet`，完成 P4 闭环。

职责边界（严格）：

- 只校验和透传 attempt/fencing token 上下文：确认任务处于 ``processing``、token 匹配、
  lease 未过期，但**不写状态、不续租、不推进阶段、不完成任务**。
- 只读取一次当前事实（复用 P4.3 ``MySQLSourceAdapter.read_current_facts``），只调用
  一个目标 Handler。时序校验由 P4.3 在 ``read_current_facts`` 内部完成，本层不重复实现。
- 异常 fail-closed：missing、归属错误、时序错误、``ChunkValidationError`` 全部向上传播，
  Dispatcher 不吞异常、不包装、不返回部分集合。
- 不调用 Consumer、Embedding、Milvus；不更新任务状态、checkpoint、lease/retry。
- ``task_id``/``batch_id`` 使用正式任务值，``data_version`` 固定为 ``str(batch_id)``。
- ``scope.scope_type == 'profile'`` 的 education 任务经 P4.3 ``source_sync_task_to_read_request``
  已映射为 profile scope，本层无需特判。

下游 Consumer 接线属 P5，本阶段只返回 :class:`PreparedChunkSet`。
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import scope_handlers
from .chunk_builder import ChunkFactory
from .models import (
    PreparedChunkSet,
    ScopedCurrentFacts,
    SourceSyncTask,
    source_sync_task_to_read_request,
)
from .source import MySQLSourceAdapter

__all__ = ["prepare_claimed_source_sync_task"]


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
    *,
    now: datetime | None = None,
) -> PreparedChunkSet:
    """把一个已领取的来源同步任务转换为 :class:`PreparedChunkSet`。

    :param task: P3 已领取任务模型（``sync_status == 'processing'``）。
    :param attempt_token: 调用方持有的本次认领令牌，必须与 ``task.attempt_token`` 完全一致。
    :param source: P4.3 定向只读事实适配器。
    :param factory: ChunkFactory，注入 ``task_id/batch_id/data_version`` 并提供策略版本。
    :param now: 当前时间，用于 lease 过期判定；默认 ``datetime.now(timezone.utc)``。
    :return: 目标 scope Handler 产出的 :class:`PreparedChunkSet`，原样返回不改写。

    :raises ValueError: 任务非 :class:`SourceSyncTask`、状态非 processing、token 不匹配、
        lease 缺失或过期、时区不匹配，或策略版本缺失/不一致。
    :raises Exception: 上游 mapping、source 读取、归属、时序、ChunkFactory、
        ``validate_chunks`` 等任何异常原样向上传播，不吞不包装。
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

    # lease 校验：双方必须时区感知，不静默强制转换。
    if task.lease_until is None:
        raise ValueError("processing 任务缺少 lease_until")
    when = now if now is not None else datetime.now(timezone.utc)
    if not isinstance(when, datetime):
        raise TypeError("now 必须是 datetime")
    if when.tzinfo is None or task.lease_until.tzinfo is None:
        raise ValueError("lease_until 与 now 必须均为时区感知时间，拒绝静默强制转换")
    if task.lease_until <= when:
        raise ValueError("任务 lease 已过期，调用方不再持有有效 lease")

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

    # 复用 P4.3 mapping（education→profile 已在此完成）与时序校验（read_current_facts 内部）。
    request = source_sync_task_to_read_request(task)
    facts: ScopedCurrentFacts = source.read_current_facts(request)

    # 防御畸形 source：返回事实的范围必须与请求范围一致，否则可能路由到错误 Handler。
    if facts.scope != request.impact_scope:
        raise ValueError(
            "source 返回的 scope 与请求 scope 不一致，拒绝分发"
        )

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
