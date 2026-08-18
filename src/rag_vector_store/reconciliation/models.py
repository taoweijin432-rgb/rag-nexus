"""全量向量 dry-run 对账的公共返回模型。

``FullVectorReconciliationResult`` 是本阶段只读对账（``reconcile_all``）的稳定
返回契约。它只承载计数与少量 stale 采样 ID，**不返回** chunk_text / embedding /
Milvus raw row，避免把大字段或内部结构泄漏给调用方。

设计约束：
    - 计数之间保持一致性（见 ``model_validator``），任一不满足都视为调用方
      可信度问题，fail-closed。
    - ``stale_sample`` 最多保留 20 个 ID，仅用于人工抽查，不代表完整 stale 集。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

_STALE_SAMPLE_LIMIT = 20


class FullVectorReconciliationResult(BaseModel):
    """全量向量 dry-run 对账结果。

    字段语义：
        - ``input_count``：传入的完整 Chunk 快照总数。
        - ``eligible_count``：校验通过且符合资格、进入 desired 的记录数。
        - ``skipped_count``：校验通过但被 Eligibility 跳过（is_active=false /
          verification_status=rejected）的记录数。
        - ``desired_count``：desired state 记录数（= eligible_count）。
        - ``existing_count``：Milvus 当前实际存在的向量数。
        - ``missing_count``：desired 有、Milvus 无。
        - ``content_changed_count``：两边都有但 content_hash 不同。
        - ``metadata_changed_count``：content_hash 相同但 metadata_hash 不同。
        - ``unchanged_count``：content_hash 与 metadata_hash 均相同。
        - ``stale_count``：Milvus 有、desired 无。
        - ``stale_sample``：stale 的最小采样 ID（最多 20 个，供抽查）。
        - ``embedded_count``：正向执行实际 upsert 成功数（dry-run 恒为 0）。
        - ``metadata_updated_count``：正向执行实际 update_metadata
          成功数（dry-run 恒为 0）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_count: int = Field(default=0, ge=0)
    eligible_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    desired_count: int = Field(default=0, ge=0)
    existing_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    content_changed_count: int = Field(default=0, ge=0)
    metadata_changed_count: int = Field(default=0, ge=0)
    unchanged_count: int = Field(default=0, ge=0)
    stale_count: int = Field(default=0, ge=0)
    stale_sample: tuple[str, ...] = ()
    embedded_count: int = Field(default=0, ge=0)
    metadata_updated_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_count_consistency(self) -> FullVectorReconciliationResult:
        if self.input_count != self.eligible_count + self.skipped_count:
            raise ValueError(
                "input_count 必须等于 eligible_count + skipped_count"
            )
        if self.desired_count != self.eligible_count:
            raise ValueError("desired_count 必须等于 eligible_count")
        if self.desired_count != (
            self.missing_count
            + self.content_changed_count
            + self.metadata_changed_count
            + self.unchanged_count
        ):
            raise ValueError("desired 分类计数之和必须等于 desired_count")
        if self.existing_count != (
            self.content_changed_count
            + self.metadata_changed_count
            + self.unchanged_count
            + self.stale_count
        ):
            raise ValueError("existing 分类计数之和必须等于 existing_count")
        if len(self.stale_sample) > min(self.stale_count, _STALE_SAMPLE_LIMIT):
            raise ValueError(
                "stale_sample 长度不能超过 stale_count 或 20"
            )
        if self.embedded_count + self.metadata_updated_count > self.desired_count:
            raise ValueError(
                "embedded_count + metadata_updated_count 不能超过 desired_count"
            )
        return self