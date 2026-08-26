"""P5.1 正式内存交付链上游 Handoff/Receipt 契约与构建器。

承接 P4 ``PreparedChunkSet``（已通过 Validator 的内存期望 Chunk 集合），经现有
``contract_bridge.chunk_records_to_contracts`` 无损转换为有序
``ExpertChunkEmbeddingInput``，构造确定性的范围级 ``ScopeHandoffRequest``，并定义
下游范围级回执 ``ScopeHandoffReceipt``。

正式链路边界（P5）：
- 只读 P3/P4 产物（``SourceSyncTask`` + ``attempt_token`` + ``PreparedChunkSet``），
  复用 ``contract_bridge`` 做无损转换，不复制 Bridge、不重排 Chunk、不重算三 Hash、
  不重新 Embedding、不读 previous state、不调 Consumer/Milvus/Embedding。
- 正式链路不复用本机兼容包装：``rag_vector_store/incremental`` 的 ``V15WorkItemConsumer``
  是 Work Item 兼容包装层；正式可复用的 Work-Item 无关核心是 ``ContractConsumerCore``。
  P5.1 上游 Handoff 不 import、不依赖下游任何 Consumer 类型，只定义上游交付契约；
  P5.2 范围 Consumer 必须复用/扩展 ``ContractConsumerCore``，不得复制三 Hash、
  metadata-only、重新激活等判定逻辑。
- 不携带 previous state、预判 action（Upsert/Metadata/Disable/Skip）、
  ``stale_vector_doc_ids``、embedding vector、原始 MySQL 行或额外敏感字段；
  不持久化 Artifact、不写公共状态表、不推进任务阶段或 Checkpoint。
- 空 desired set 合法（携带合法 scope 的显式空列表），表示该范围当前应为空，不当作 Skip。

本模块只提供模型与构建器；Artifact 持久化、阶段流转、真实范围 Consumer 均为后续工作。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from expert_contracts import ExpertChunkEmbeddingInput

from .contract_bridge import chunk_records_to_contracts
from .models import (
    IncrementalImpactScope,
    PreparedChunkSet,
    SourceSyncTask,
)

HandoffReceiptStatus = Literal["committed", "rejected", "failed"]
"""回执终态。committed 表示范围已收敛成功；rejected 表示确定性拒绝（不可重试）；
failed 表示可重试的下游/基础设施失败。"""

DesiredResultCode = Literal[
    "embedded",
    "metadata_updated",
    "unchanged_skipped",
    "targets_disabled",
]

_REQUEST_ID_PREFIX = "handoff"
_SHA256_HEX_LEN = 64
_MAX_ERROR_MESSAGE_LEN = 512
_MAX_RESULT_CODE_LEN = 128


def _sha256_hex(text: str) -> str:
    """返回 UTF-8 字节的 SHA-256 十六进制摘要。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_payload_hash(
    *,
    task_id: int,
    batch_id: int,
    impact_scope: IncrementalImpactScope,
    contracts: Sequence[ExpertChunkEmbeddingInput],
) -> str:
    """计算 payload_hash 的确定性 canonical 序列化。

    ``attempt_token`` 不参与 payload_hash（它用于 fencing，仅同任务同 attempt 绑定，
    不属于稳定 identity/scope/payload；同一任务仅 attempt_token 变化时 payload_hash
    必须相同）。``request_id`` 同样不参与（它派生自 task/batch/token，含 attempt_token）。
    payload_hash 只覆盖稳定 identity（task_id/batch_id）、scope、以及有序 Contract
    payload。Contract 用 ``model_dump(mode="json")`` 序列化；保留 Contract 列表原顺序
    （P4 已提供 canonical 排序，顺序变化必须改变 hash）。不对 Contract 列表内部排序、
    不剔除任何字段、不包含 ``payload_hash`` 自身。
    """

    scope_obj = {
        "scope_type": impact_scope.scope_type,
        "expert_id": impact_scope.expert_id,
        "chunk_type": impact_scope.chunk_type,
        "source_table": impact_scope.source_table,
        "source_field_id": impact_scope.source_field_id,
    }
    payload = {
        "task_id": task_id,
        "batch_id": batch_id,
        "impact_scope": scope_obj,
        "contracts": [c.model_dump(mode="json") for c in contracts],
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _sha256_hex(serialized)


@dataclass(frozen=True)
class ScopeHandoffRequest:
    """范围级正式交付请求。

    表达 request/task/batch identity、attempt/fencing token、影响范围、有序
    desired ``ExpertChunkEmbeddingInput`` 集合和确定性 ``payload_hash``。空 desired
    合法（携带合法 scope 的空列表）。不携带 previous state、预判 action、
    ``stale_vector_doc_ids``、embedding vector 或原始 MySQL 行。
    """

    request_id: str
    task_id: int
    batch_id: int
    attempt_token: str
    impact_scope: IncrementalImpactScope
    contracts: tuple[ExpertChunkEmbeddingInput, ...]
    payload_hash: str

    def __post_init__(self) -> None:
        """校验 identity、scope、Contract 归属与 payload_hash 完整性。"""

        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id 必须是非空字符串")
        if not isinstance(self.task_id, int) or isinstance(self.task_id, bool) or self.task_id < 1:
            raise ValueError("task_id 必须是正整数")
        if not isinstance(self.batch_id, int) or isinstance(self.batch_id, bool) or self.batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if not isinstance(self.attempt_token, str) or not self.attempt_token.strip():
            raise ValueError("attempt_token 必须是非空字符串")
        if not isinstance(self.impact_scope, IncrementalImpactScope):
            raise TypeError("impact_scope 必须是 IncrementalImpactScope")
        contracts = tuple(self.contracts)
        object.__setattr__(self, "contracts", contracts)
        self._validate_contracts(contracts)
        if not isinstance(self.payload_hash, str) or len(self.payload_hash) != _SHA256_HEX_LEN:
            raise ValueError("payload_hash 必须是 64 位 SHA-256 十六进制串")
        # payload_hash 必须与请求内容一致（构造后复算校验，防伪造）。
        # 注意：attempt_token 与 request_id 均不参与 payload_hash（仅用于 fencing）。
        recomputed = _canonical_payload_hash(
            task_id=self.task_id,
            batch_id=self.batch_id,
            impact_scope=self.impact_scope,
            contracts=self.contracts,
        )
        if recomputed != self.payload_hash:
            raise ValueError("payload_hash 与请求内容不一致")

    def _validate_contracts(self, contracts: tuple[ExpertChunkEmbeddingInput, ...]) -> None:
        """校验所有 Contract 归属当前范围，且 vector_doc_id 唯一。"""

        scope = self.impact_scope
        seen: set[str] = set()
        for contract in contracts:
            if not isinstance(contract, ExpertChunkEmbeddingInput):
                raise TypeError("contracts 必须全部是 ExpertChunkEmbeddingInput")
            if contract.expert_id != scope.expert_id:
                raise ValueError(
                    f"Contract expert_id={contract.expert_id} 不属于范围 "
                    f"expert_id={scope.expert_id}"
                )
            if scope.scope_type == "expert":
                # expert 范围跨三类，chunk_type 必须合法即可
                if contract.chunk_type not in ("profile_summary", "employment", "project"):
                    raise ValueError(
                        f"expert 范围不允许 chunk_type={contract.chunk_type}"
                    )
            else:
                if contract.chunk_type != scope.chunk_type:
                    raise ValueError(
                        f"{scope.scope_type} 范围不允许 chunk_type={contract.chunk_type}"
                    )
                if contract.source.source_table != scope.source_table:
                    raise ValueError(
                        f"{scope.scope_type} 范围 Contract source_table="
                        f"{contract.source.source_table} 与范围不一致"
                    )
                if contract.source.source_field_id != scope.source_field_id:
                    raise ValueError(
                        f"{scope.scope_type} 范围 Contract source_field_id="
                        f"{contract.source.source_field_id} 与范围不一致"
                    )
            if contract.vector_doc_id in seen:
                raise ValueError(
                    f"Contract vector_doc_id={contract.vector_doc_id} 重复"
                )
            seen.add(contract.vector_doc_id)


@dataclass(frozen=True)
class ScopeDesiredItemResult:
    """单个 desired vector_doc_id 的确定性收敛结果。"""

    vector_doc_id: str
    result_code: DesiredResultCode

    def __post_init__(self) -> None:
        if not isinstance(self.vector_doc_id, str) or not self.vector_doc_id.strip():
            raise ValueError("desired result vector_doc_id 必须是非空字符串")
        if self.result_code not in {
            "embedded",
            "metadata_updated",
            "unchanged_skipped",
            "targets_disabled",
        }:
            raise ValueError(f"不支持的 desired result_code: {self.result_code!r}")


@dataclass(frozen=True)
class ScopeHandoffReceipt:
    """范围级下游回执。

    对应 request/task/batch/token、scope、committed/rejected/failed 终态、稳定
    result_code，以及 embedding/metadata-only/disable/unchanged 计数与可重试
    标记/错误信息。上游只依赖这些公开语义核验范围收敛，不复算动作。
    """

    request_id: str
    task_id: int
    batch_id: int
    attempt_token: str
    payload_hash: str
    impact_scope: IncrementalImpactScope
    status: HandoffReceiptStatus
    result_code: str
    desired_results: tuple[ScopeDesiredItemResult, ...] = ()
    stale_vector_doc_ids: tuple[str, ...] = ()
    desired_count: int = 0
    embedding_count: int = 0
    metadata_only_count: int = 0
    disable_count: int = 0
    unchanged_count: int = 0
    retryable: bool = False
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        """校验回执不变式（状态、计数、错误字段）。"""

        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id 必须是非空字符串")
        if not isinstance(self.task_id, int) or isinstance(self.task_id, bool) or self.task_id < 1:
            raise ValueError("task_id 必须是正整数")
        if not isinstance(self.batch_id, int) or isinstance(self.batch_id, bool) or self.batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if not isinstance(self.attempt_token, str) or not self.attempt_token.strip():
            raise ValueError("attempt_token 必须是非空字符串")
        if not isinstance(self.payload_hash, str) or len(self.payload_hash) != _SHA256_HEX_LEN:
            raise ValueError("payload_hash 必须是 64 位 SHA-256 十六进制串")
        if not isinstance(self.impact_scope, IncrementalImpactScope):
            raise TypeError("impact_scope 必须是 IncrementalImpactScope")
        if self.status not in ("committed", "rejected", "failed"):
            raise ValueError(f"未知回执状态: {self.status!r}")
        if not isinstance(self.result_code, str) or not self.result_code.strip():
            raise ValueError("result_code 必须是非空字符串")
        if len(self.result_code) > _MAX_RESULT_CODE_LEN:
            raise ValueError("result_code 过长")
        self._normalize_evidence()
        self._validate_counts()
        self._validate_error_fields()

    def _normalize_evidence(self) -> None:
        """冻结逐项 desired 结果和 stale 下线 ID，并拒绝重复证据。"""

        normalized: list[ScopeDesiredItemResult] = []
        for item in tuple(self.desired_results):
            if isinstance(item, ScopeDesiredItemResult):
                normalized.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                normalized.append(ScopeDesiredItemResult(item[0], item[1]))
            else:
                raise TypeError("desired_results 必须包含 ScopeDesiredItemResult")
        desired_results = tuple(normalized)
        stale_ids = tuple(self.stale_vector_doc_ids)
        if len({item.vector_doc_id for item in desired_results}) != len(desired_results):
            raise ValueError("desired_results 包含重复 vector_doc_id")
        if any(not isinstance(value, str) or not value.strip() for value in stale_ids):
            raise ValueError("stale_vector_doc_ids 必须全部是非空字符串")
        if len(set(stale_ids)) != len(stale_ids):
            raise ValueError("stale_vector_doc_ids 包含重复值")
        if {item.vector_doc_id for item in desired_results} & set(stale_ids):
            raise ValueError("desired 与 stale 证据不得包含相同 vector_doc_id")
        object.__setattr__(self, "desired_results", desired_results)
        object.__setattr__(self, "stale_vector_doc_ids", stale_ids)

    def _validate_counts(self) -> None:
        """计数必须是非负整数（拒绝 bool），disable 独立不参与收敛等式。"""

        for name in (
            "desired_count",
            "embedding_count",
            "metadata_only_count",
            "disable_count",
            "unchanged_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")

    def _validate_error_fields(self) -> None:
        """committed: 无错误；rejected: 必有稳定错误；failed: 必有错误。"""

        def _check_single_line_short(value: str, label: str) -> None:
            if not value.strip():
                raise ValueError(f"{label} 必须是非空字符串")
            if "\n" in value or "\r" in value:
                raise ValueError(f"{label} 必须是单行")
            if len(value) > _MAX_ERROR_MESSAGE_LEN:
                raise ValueError(f"{label} 过长")

        has_error_type = self.error_type is not None
        has_error_message = self.error_message is not None
        if has_error_type != has_error_message:
            raise ValueError("error_type 与 error_message 必须同时提供或同时为空")
        if self.status == "committed":
            if has_error_type or has_error_message:
                raise ValueError("committed 回执不得携带错误字段")
            if self.retryable:
                raise ValueError("committed 回执 retryable 必须为 False")
            # committed 收敛不变式（P5.2 放宽以涵盖 Eligibility 逻辑下线 desired）：
            # 成功收敛的 embedding/metadata-only/unchanged 不得超过 desired；
            # 未被这三类覆盖的 desired（如 ineligible 逻辑下线）必须由 disable_count 兜底。
            resolved = self.embedding_count + self.metadata_only_count + self.unchanged_count
            if resolved > self.desired_count:
                raise ValueError(
                    "committed 回执收敛计数超过 desired_count"
                )
            if self.desired_count - resolved > self.disable_count:
                raise ValueError(
                    "committed 回执存在未收敛且未下线的 desired"
                )
        elif self.status == "rejected":
            if self.retryable:
                raise ValueError("rejected 回执 retryable 必须为 False")
            if not has_error_type:
                raise ValueError("rejected 回执必须携带 error_type/error_message")
            _check_single_line_short(self.error_type or "", "error_type")  # type: ignore[arg-type]
            _check_single_line_short(self.error_message or "", "error_message")  # type: ignore[arg-type]
        elif self.status == "failed":
            if not has_error_type:
                raise ValueError("failed 回执必须携带 error_type/error_message")
            _check_single_line_short(self.error_type or "", "error_type")  # type: ignore[arg-type]
            _check_single_line_short(self.error_message or "", "error_message")  # type: ignore[arg-type]


@dataclass(frozen=True)
class ScopeHandoffBuilder:
    """范围级 Handoff 内存构建器。

    生产环境复用 ``contract_bridge.chunk_records_to_contracts`` 做无损转换；不调
    时钟、不读 previous state、不预判动作、不持久化 Artifact。``updated_at`` 取
    ``task.updated_at``（稳定持久值），全部 Contract 共享同一交付时间。
    """

    timezone_name: str

    def build(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        prepared: PreparedChunkSet,
    ) -> ScopeHandoffRequest:
        """把已领取任务 + PreparedChunkSet 转为确定性 Handoff Request。"""

        if not isinstance(task, SourceSyncTask):
            raise TypeError("task 必须是 SourceSyncTask")
        if task.sync_status != "processing":
            raise ValueError(
                f"只有 processing 任务可构建 Handoff，当前 sync_status={task.sync_status!r}"
            )
        if not isinstance(attempt_token, str) or not attempt_token.strip():
            raise ValueError("attempt_token 必须是非空字符串")
        if task.attempt_token is None or task.attempt_token != attempt_token:
            raise ValueError("attempt_token 与任务认领 token 不一致")
        if not isinstance(prepared, PreparedChunkSet):
            raise TypeError("prepared 必须是 PreparedChunkSet")
        if prepared.task_id != task.task_id:
            raise ValueError(
                f"prepared.task_id={prepared.task_id} 与 task.task_id={task.task_id} 不一致"
            )
        if prepared.batch_id != task.batch_id:
            raise ValueError(
                f"prepared.batch_id={prepared.batch_id} 与 task.batch_id={task.batch_id} 不一致"
            )
        if prepared.data_version != str(task.batch_id):
            raise ValueError(
                f"prepared.data_version={prepared.data_version!r} 必须等于 str(batch_id)={str(task.batch_id)!r}"
            )
        # P5.1 是正式 fail-closed 边界，不依赖 P4.8 已做的校验：任务专家身份必须非空且与范围一致。
        if task.expert_id is None:
            raise ValueError("P5.1 要求 task.expert_id 非空，None 任务无法安全交付")
        if task.expert_id != prepared.impact_scope.expert_id:
            raise ValueError(
                f"task.expert_id={task.expert_id} 与范围 expert_id="
                f"{prepared.impact_scope.expert_id} 不一致"
            )
        # policy 版本必须是非空字符串（历史迁移任务 policy_version=None 由 P4.8 已拒绝；
        # P5.1 同样拒绝，确保正式交付带明确策略版本）。
        if not isinstance(task.policy_version, str) or not task.policy_version.strip():
            raise ValueError("P5.1 要求 task.policy_version 为非空字符串")
        # 转换前用 Chunk 的 chunk_policy_version 做 policy 一致性校验
        policies = {chunk.chunk_policy_version for chunk in prepared.desired_chunks}
        if len(policies) > 1:
            raise ValueError("PreparedChunkSet 内 Chunk policy_version 不一致")
        if policies and task.policy_version not in policies:
            raise ValueError(
                f"task.policy_version={task.policy_version} 与 Chunk policy_version 不一致"
            )

        # 复用 contract_bridge 无损转换；空 desired 直接返回空列表，不伪造空 Chunk。
        # task.updated_at 是稳定持久时间，全部 Contract 共享；不调时钟。
        contracts_list = chunk_records_to_contracts(
            prepared.desired_chunks,
            updated_at=task.updated_at,
            timezone_name=self.timezone_name,
        )
        contracts = tuple(contracts_list)

        # 转换后复核：Contract versions.chunk_policy_version 必须全等于 task.policy_version
        # （Bridge 无损转换不改 policy，但 P5.1 是正式边界，转换后必须复核）。
        if contracts:
            contract_policies = {c.versions.chunk_policy_version for c in contracts}
            if len(contract_policies) > 1:
                raise ValueError("转换后 Contract policy_version 不一致")
            if task.policy_version not in contract_policies:
                raise ValueError(
                    f"转换后 Contract policy_version={contract_policies} 与 "
                    f"task.policy_version={task.policy_version} 不一致"
                )

        # 构造确定性 request_id：prefix + 任务/批次/token 身份的 SHA-256 摘要。
        request_id = f"{_REQUEST_ID_PREFIX}:{_sha256_hex(f'{task.task_id}:{task.batch_id}:{attempt_token}')}"

        payload_hash = _canonical_payload_hash(
            task_id=task.task_id,
            batch_id=task.batch_id,
            impact_scope=prepared.impact_scope,
            contracts=contracts,
        )
        return ScopeHandoffRequest(
            request_id=request_id,
            task_id=task.task_id,
            batch_id=task.batch_id,
            attempt_token=attempt_token,
            impact_scope=prepared.impact_scope,
            contracts=contracts,
            payload_hash=payload_hash,
        )


__all__ = [
    "DesiredResultCode",
    "HandoffReceiptStatus",
    "ScopeDesiredItemResult",
    "ScopeHandoffBuilder",
    "ScopeHandoffReceipt",
    "ScopeHandoffRequest",
]
