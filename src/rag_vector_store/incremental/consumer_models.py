"""Consumer 结构化结果模型与稳定结果码（D2A 抽取后的共享层）。

本模块只承载 Consumer 对外的结构化产物定义：状态、动作、result_code、
error_code 与 ``ExecutionResult``。它不依赖任何历史 Work Item 行模型或仓储，
由正式 ``ContractConsumerCore`` 与范围 Consumer 共同复用。状态持久化语义
仍由上游唯一状态表适配层负责。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# 基础字面量
# ---------------------------------------------------------------------------
#: Consumer 支持的操作标签；本地定义以保持核心不依赖历史 Work Item 模块。
ConsumerOperation = Literal["upsert", "disable"]

#: Consumer 终态。
ConsumerExecutionStatus = Literal["committed", "skipped", "rejected", "failed"]

# ---------------------------------------------------------------------------
# 动作常量（执行轨迹，供审计与测试断言）
# ---------------------------------------------------------------------------
CONSUMER_ACTION_FETCH_EXISTING = "fetch_existing"
CONSUMER_ACTION_ADAPT_UPSERT = "adapt_upsert"
CONSUMER_ACTION_METADATA_ONLY_UPDATE = "metadata_only_update"
CONSUMER_ACTION_EMBEDDING_UPSERT = "embedding_upsert"
CONSUMER_ACTION_SKIP_UNCHANGED = "skip_unchanged"
CONSUMER_ACTION_DISABLE_TARGETS = "disable_targets"

_KNOWN_ACTIONS = frozenset(
    {
        CONSUMER_ACTION_FETCH_EXISTING,
        CONSUMER_ACTION_ADAPT_UPSERT,
        CONSUMER_ACTION_METADATA_ONLY_UPDATE,
        CONSUMER_ACTION_EMBEDDING_UPSERT,
        CONSUMER_ACTION_SKIP_UNCHANGED,
        CONSUMER_ACTION_DISABLE_TARGETS,
    }
)

# ---------------------------------------------------------------------------
# result_code / error_code 稳定枚举
# ---------------------------------------------------------------------------
#: 成功完成 Embedding 并写入新向量。
RESULT_CODE_EMBEDDED = "embedded"
#: 仅元数据更新成功（复用旧向量，零 Embedding）。
RESULT_CODE_METADATA_UPDATED = "metadata_updated"
#: 目标逻辑下线成功（显式 disable / Eligibility 路由 / skip 后 stale 补偿）。
RESULT_CODE_TARGETS_DISABLED = "targets_disabled"
#: 双 hash 全等且实体活跃，跳过写入。
RESULT_CODE_UNCHANGED_SKIPPED = "unchanged_skipped"
#: 防御分支：ingestion 层对本可嵌入输入返回 skipped（正常链路不应触达）。
RESULT_CODE_UPSERT_SKIPPED = "upsert_skipped"

ConsumerResultCode = Literal[
    "embedded",
    "metadata_updated",
    "targets_disabled",
    "unchanged_skipped",
    "upsert_skipped",
]

#: metadata_update 安全门失败：已存向量模型空间/维度与正式配置不一致。
ERROR_CODE_MODEL_SPACE_MISMATCH = "model_space_mismatch"
#: 已存记录状态提取失败（缺 hash 或字段形态非法）。
ERROR_CODE_EXISTING_STATE_INVALID = "existing_state_invalid"
#: metadata-only 一致性门抛出 VectorContractError。
ERROR_CODE_METADATA_ONLY_INCONSISTENT = "metadata_only_inconsistent"
#: upsert 阶段 Contract 校验失败。
ERROR_CODE_UPSERT_CONTRACT_FAILED = "upsert_contract_failed"

#: 可重试基础设施异常族（failed 状态专用）。
ERROR_CODE_EXISTING_QUERY_FAILED = "existing_query_failed"
ERROR_CODE_METADATA_UPDATE_INFRA_FAILED = "metadata_update_infra_failed"
ERROR_CODE_UPSERT_INFRA_FAILED = "upsert_infra_failed"
ERROR_CODE_ELIGIBILITY_DISABLE_FAILED = "eligibility_disable_failed"
ERROR_CODE_DISABLE_INFRA_FAILED = "disable_infra_failed"
# 下游返回数量、身份或查询集合与请求不一致（通常可重试）。
ERROR_CODE_DOWNSTREAM_RESULT_MISMATCH = "downstream_result_mismatch"

# ---------------------------------------------------------------------------
# P5.2 范围 Consumer 稳定错误码（scope 层收敛失败，不语义复用核心错误码）
# ---------------------------------------------------------------------------
#: scope 字段组合非法（不支持的范围类型或字段不一致）。
ERROR_CODE_SCOPE_UNSUPPORTED_TYPE = "scope_unsupported_type"
#: scope 当前 active 向量状态查询基础设施失败（可重试）。
ERROR_CODE_SCOPE_QUERY_FAILED = "scope_query_failed"
#: scope active 状态提取失败（metadata 缺字段/类型非法，确定性）。
ERROR_CODE_SCOPE_ACTIVE_STATE_INVALID = "scope_active_state_invalid"
#: scope stale 逻辑下线失败（desired 已完成，可重试）。
ERROR_CODE_SCOPE_STALE_DISABLE_FAILED = "scope_stale_disable_failed"


class ExecutionResult(BaseModel):
    """单条输入的 Consumer 处理结果（纯描述，无任何状态写回语义）。

    字段不变量由模型强制：

    - ``retryable=True`` 只允许出现在 ``failed``（可重试基础设施异常）；
      ``rejected`` 恒为不可重试。
    - ``rejected`` / ``failed`` 必须携带稳定 ``error_code`` 且不得携带
      ``result_code``、``embedding_performed`` 或 ``metadata_only``。
    - ``committed`` / ``skipped`` 必须携带 ``result_code`` 且不得携带
      ``error_code``。
    - ``embedding_performed=True`` 只能出现在实际发生 Embedding 的提交路径
      （收到 embedded 确认）；``metadata_only=True`` 只能出现在成功的
      metadata-only 提交路径；二者互斥。
    - ``fallback_used`` 为保留字段：当前核心不再自动降级重嵌
      （安全门失败一律拒绝），恒为 False；仅当未来出现安全降级语义时启用，
      且只允许伴随 committed 状态。
    """

    model_config = ConfigDict(extra="forbid")

    vector_doc_id: str
    operation: ConsumerOperation
    status: ConsumerExecutionStatus
    reason: str | None = None
    actions: tuple[str, ...] = Field(default=(), frozen=True)
    disabled_ids: tuple[str, ...] = ()
    #: 正向/中性结果的稳定分类码；rejected/failed 下必须为 None。
    result_code: ConsumerResultCode | None = None
    #: 失败类结果的稳定错误码；其余状态必须为 None。
    error_code: str | None = None
    #: 是否进入了真实 Embedding 路径并收到 embedded 确认。
    embedding_performed: bool = False
    #: 是否走了 metadata-only 更新路径（零 Embedding）。
    metadata_only: bool = False
    #: 保留字段：是否使用了降级路径。见类 docstring 说明。
    fallback_used: bool = False
    #: 是否属于可重试的基础设施异常；仅 failed 状态允许 True。
    retryable: bool = False

    @model_validator(mode="after")
    def _validate_field_invariants(self) -> ExecutionResult:
        if self.status == "failed":
            if not self.error_code:
                raise ValueError("failed 结果必须携带稳定的 error_code")
        elif self.status in ("rejected",):
            if not self.error_code:
                raise ValueError("rejected 结果必须携带稳定的 error_code")
            if self.retryable:
                raise ValueError("rejected 结果是确定性拒绝，不允许 retryable=True")
        else:
            # committed / skipped
            if self.result_code is None:
                raise ValueError(
                    f"{self.status} 结果必须携带 result_code 以供状态映射"
                )
        if self.error_code is not None and self.status not in ("rejected", "failed"):
            raise ValueError("只有 rejected/failed 允许携带 error_code")
        if self.result_code is not None and self.status not in ("committed", "skipped"):
            raise ValueError("只有 committed/skipped 允许携带 result_code")
        if self.embedding_performed and self.status != "committed":
            raise ValueError(
                "embedding_performed=True 只能出现在实际 Embedding 后提交的结果"
            )
        if self.metadata_only:
            if self.status != "committed":
                raise ValueError("metadata_only=True 只能出现在成功的 metadata-only 提交")
            if self.embedding_performed:
                raise ValueError("metadata_only 与 embedding_performed 互斥")
        if self.fallback_used and self.status != "committed":
            raise ValueError("fallback_used=True 只能伴随 committed 状态")
        for action in self.actions:
            if action not in _KNOWN_ACTIONS:
                raise ValueError(f"未知的动作常量：{action}")
        return self


__all__ = [
    "CONSUMER_ACTION_ADAPT_UPSERT",
    "CONSUMER_ACTION_DISABLE_TARGETS",
    "CONSUMER_ACTION_EMBEDDING_UPSERT",
    "CONSUMER_ACTION_FETCH_EXISTING",
    "CONSUMER_ACTION_METADATA_ONLY_UPDATE",
    "CONSUMER_ACTION_SKIP_UNCHANGED",
    "ERROR_CODE_DISABLE_INFRA_FAILED",
    "ERROR_CODE_DOWNSTREAM_RESULT_MISMATCH",
    "ERROR_CODE_ELIGIBILITY_DISABLE_FAILED",
    "ERROR_CODE_EXISTING_QUERY_FAILED",
    "ERROR_CODE_EXISTING_STATE_INVALID",
    "ERROR_CODE_METADATA_ONLY_INCONSISTENT",
    "ERROR_CODE_METADATA_UPDATE_INFRA_FAILED",
    "ERROR_CODE_MODEL_SPACE_MISMATCH",
    "ERROR_CODE_SCOPE_ACTIVE_STATE_INVALID",
    "ERROR_CODE_SCOPE_QUERY_FAILED",
    "ERROR_CODE_SCOPE_STALE_DISABLE_FAILED",
    "ERROR_CODE_SCOPE_UNSUPPORTED_TYPE",
    "ERROR_CODE_UPSERT_CONTRACT_FAILED",
    "ERROR_CODE_UPSERT_INFRA_FAILED",
    "RESULT_CODE_EMBEDDED",
    "RESULT_CODE_METADATA_UPDATED",
    "RESULT_CODE_TARGETS_DISABLED",
    "RESULT_CODE_UNCHANGED_SKIPPED",
    "RESULT_CODE_UPSERT_SKIPPED",
    "ConsumerExecutionStatus",
    "ConsumerOperation",
    "ConsumerResultCode",
    "ExecutionResult",
]
