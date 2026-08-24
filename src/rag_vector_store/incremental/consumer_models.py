"""通用 Work Item Consumer 的纯业务执行结果模型（Phase 3A）。

本模块只定义 Consumer 核心的结构化返回契约，刻意保持“无状态”：

- 不依赖任何 MySQL 状态表、不定义 StateStore 端口、不携带任何持久化句柄；
- ``ExecutionResult.status`` 只表达本次业务执行的终态分类，后续由最终确认的
  唯一状态表适配层把该结果映射为 Work Item 状态流转（committed / skipped /
  rejected 落库，failed 进入重试队列）；
- ``rejected`` 表示确定性不可重试（Contract 失败、缺字段、hash 不一致、
  目标缺失、旧事件等）；``failed`` 表示可重试的基础设施失败
  （Embedding / Milvus / 网络）；二者不混用。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import V15WorkItemOperation

# Consumer 业务执行的终态分类；planned 仅出现在 plan_only 只读规划模式。
ConsumerExecutionStatus = Literal[
    "committed",
    "skipped",
    "rejected",
    "failed",
    "planned",
]

# 执行计划中的动作名；actions 按发生顺序记录，plan_only 模式下为“将要执行”的动作。
CONSUMER_ACTION_ADAPT_UPSERT = "adapt_upsert"
CONSUMER_ACTION_ADAPT_DISABLE = "adapt_disable"
CONSUMER_ACTION_FETCH_EXISTING = "fetch_existing"
CONSUMER_ACTION_SYNTHESIZE_FROM_STORED = "synthesize_from_stored"
CONSUMER_ACTION_EMBEDDING_UPSERT = "embedding_upsert"
CONSUMER_ACTION_METADATA_ONLY_UPDATE = "metadata_only_update"
CONSUMER_ACTION_SKIP_UNCHANGED = "skip_unchanged"
CONSUMER_ACTION_DISABLE_TARGETS = "disable_targets"
CONSUMER_ACTION_DISABLE_STALE_TARGETS = "disable_stale_targets"

_CONSUMER_ACTIONS: tuple[str, ...] = (
    CONSUMER_ACTION_ADAPT_UPSERT,
    CONSUMER_ACTION_ADAPT_DISABLE,
    CONSUMER_ACTION_FETCH_EXISTING,
    CONSUMER_ACTION_SYNTHESIZE_FROM_STORED,
    CONSUMER_ACTION_EMBEDDING_UPSERT,
    CONSUMER_ACTION_METADATA_ONLY_UPDATE,
    CONSUMER_ACTION_SKIP_UNCHANGED,
    CONSUMER_ACTION_DISABLE_TARGETS,
    CONSUMER_ACTION_DISABLE_STALE_TARGETS,
)


class ExecutionResult(BaseModel):
    """单条 Work Item 的业务执行结果。

    字段语义：

    - ``status``：终态分类，见模块 docstring；
    - ``reason``：人类可读的结果说明（拒绝原因、跳过原因、降级说明等）；
    - ``actions``：实际执行（或 plan_only 下规划）的动作序列；
    - ``disabled_ids``：本次被逻辑下线的 vector_doc_id 集合（含主 id 与
      stale 子块 id；幂等重放的下线也计入）；
    - ``fallback_used``：metadata-only 更新安全门失败后是否降级为全量 upsert。

    本模型不包含任何状态回写语义：调用方（未来的状态表适配层）据此决定
    Work Item 的状态流转，Consumer 自身不写任何状态表。
    """

    model_config = ConfigDict(extra="forbid")

    vector_doc_id: str = Field(min_length=1)
    operation: V15WorkItemOperation
    status: ConsumerExecutionStatus
    reason: str | None = None
    actions: tuple[str, ...] = ()
    disabled_ids: tuple[str, ...] = ()
    fallback_used: bool = False

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """动作名必须属于冻结的动作集合（fail-closed，防止拼写漂移）。"""

        for item in value:
            if item not in _CONSUMER_ACTIONS:
                allowed = "、".join(_CONSUMER_ACTIONS)
                raise ValueError(f"未知动作 {item!r}（允许: {allowed}）")
        return value


__all__ = [
    "CONSUMER_ACTION_ADAPT_DISABLE",
    "CONSUMER_ACTION_ADAPT_UPSERT",
    "CONSUMER_ACTION_DISABLE_STALE_TARGETS",
    "CONSUMER_ACTION_DISABLE_TARGETS",
    "CONSUMER_ACTION_EMBEDDING_UPSERT",
    "CONSUMER_ACTION_FETCH_EXISTING",
    "CONSUMER_ACTION_METADATA_ONLY_UPDATE",
    "CONSUMER_ACTION_SKIP_UNCHANGED",
    "CONSUMER_ACTION_SYNTHESIZE_FROM_STORED",
    "ConsumerExecutionStatus",
    "ExecutionResult",
]
