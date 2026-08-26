"""P6.4 最小指标与交付门禁。

记录器只接受任务/来源/阶段等安全标签，不接受正文、Payload、向量、PII 或凭据；门禁
使用调用方提供的事实结果，不在本模块猜测外部系统状态。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock

_SAFE_LABELS = frozenset(
    {
        "source_table",
        "processing_stage",
        "sync_status",
        "error_stage",
        "error_code",
        "result_code",
        "artifact_reused",
        "cdc_status",
    }
)


class MetricsInputError(ValueError):
    """表示指标名称、标签或数值不满足安全契约。"""


class IncrementalMetrics:
    """进程内安全计数器，不负责日志输出或远端指标发送。"""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self._lock = Lock()

    def increment(
        self,
        metric_name: str,
        value: int = 1,
        *,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        """增加一个带安全标签的计数指标。"""

        if not isinstance(metric_name, str) or not metric_name.strip():
            raise MetricsInputError("metric_name 必须是非空字符串")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MetricsInputError("指标增量必须是非负整数")
        normalized_labels = _normalize_labels(labels or {})
        with self._lock:
            self._counts[(metric_name, normalized_labels)] += value

    def snapshot(self) -> dict[str, int]:
        """返回不含原始标签的稳定指标快照，键为 ``name{label=value}``。"""

        with self._lock:
            return {
                _metric_key(metric_name, labels): count
                for (metric_name, labels), count in sorted(self._counts.items())
            }

    def record_round(self, round_result) -> None:
        """从 P6.2 轮次摘要记录任务、来源和 Checkpoint 计数。"""

        for metric_name, value in (
            ("task_claimed", round_result.claimed_count),
            ("task_success", round_result.success_count),
            ("task_retry", round_result.retryable_count),
            ("task_dead", round_result.rejected_count),
            ("task_lease_conflict", round_result.lease_conflict_count),
            ("source_round_failed", round_result.failed_source_count),
            ("checkpoint_committed", round_result.checkpoint_committed_count),
            ("checkpoint_already_committed", round_result.checkpoint_already_committed_count),
        ):
            self.increment(metric_name, value)

    def record_cdc_batch(self, batch_result) -> None:
        """记录 CDC 单轮终态和任务数，不采集 batch、Event 或 Before Image。"""

        status = str(batch_result.status)
        if status not in {"empty", "rolled_back", "not_ready", "acked"}:
            raise MetricsInputError("未知 CDC batch 状态")
        labels = {"cdc_status": status}
        self.increment("cdc_batch", labels=labels)
        self.increment("cdc_task", batch_result.task_count, labels=labels)
        checkpoint = batch_result.checkpoint
        if checkpoint is not None:
            metric_name = (
                "cdc_checkpoint_replayed"
                if checkpoint.already_applied
                else "cdc_checkpoint_committed"
            )
            self.increment(metric_name)


@dataclass(frozen=True)
class DeliveryGateReport:
    """P6 交付门禁的只读结果。"""

    ready: bool
    blockers: tuple[str, ...]


def evaluate_delivery_gate(
    *,
    required_migrations: tuple[str, ...],
    applied_migrations: tuple[str, ...],
    source_index_isolated: bool,
    contract_frozen: bool,
    targeted_tests_passed: bool,
    incremental_enabled: bool,
    retry_policy_frozen: bool = False,
    runtime_budgets_frozen: bool = False,
) -> DeliveryGateReport:
    """检查 P6 上线前置事实；未满足任一项即保持关闭。

    重试上限/退避和下游超时/lease 预算必须由运行负责人显式冻结，方法不提供默认值。
    """

    blockers: list[str] = []
    applied = set(applied_migrations)
    missing = tuple(migration for migration in required_migrations if migration not in applied)
    if missing:
        blockers.append("missing_migrations:" + ",".join(missing))
    if not source_index_isolated:
        blockers.append("source_index_not_isolated")
    if not contract_frozen:
        blockers.append("downstream_contract_not_frozen")
    if not targeted_tests_passed:
        blockers.append("targeted_tests_not_passed")
    if not retry_policy_frozen:
        blockers.append("retry_policy_not_frozen")
    if not runtime_budgets_frozen:
        blockers.append("runtime_budgets_not_frozen")
    # P6 只提供门禁，不替调用方打开正式增量开关。
    if incremental_enabled:
        blockers.append("incremental_must_remain_disabled_until_P9")
    return DeliveryGateReport(ready=not blockers, blockers=tuple(blockers))


def evaluate_cdc_delivery_gate(
    *,
    required_migrations: tuple[str, ...],
    applied_migrations: tuple[str, ...],
    source_index_isolated: bool,
    fixtures_passed: bool,
    failure_recovery_passed: bool,
    protocol_frozen: bool,
    cdc_enabled: bool,
) -> DeliveryGateReport:
    """检查 P7 代码交付和真实 CDC 启用条件，并要求交付时保持关闭。"""

    blockers: list[str] = []
    applied = set(applied_migrations)
    missing = tuple(migration for migration in required_migrations if migration not in applied)
    if missing:
        blockers.append("missing_migrations:" + ",".join(missing))
    if not source_index_isolated:
        blockers.append("source_index_not_isolated")
    if not fixtures_passed:
        blockers.append("cdc_fixtures_not_passed")
    if not failure_recovery_passed:
        blockers.append("cdc_failure_recovery_not_passed")
    if not protocol_frozen:
        blockers.append("canal_production_protocol_not_frozen")
    # P7 交付只证明实现可回放；正式启用必须由 P9 独立授权。
    if cdc_enabled:
        blockers.append("cdc_must_remain_disabled_until_P9")
    return DeliveryGateReport(ready=not blockers, blockers=tuple(blockers))


def _normalize_labels(labels: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """限制指标标签集合和值为无正文安全标识。"""

    normalized: list[tuple[str, str]] = []
    for key, value in labels.items():
        if key not in _SAFE_LABELS:
            raise MetricsInputError(f"不允许的指标标签: {key!r}")
        text = str(value)
        if not text or len(text) > 128 or any(char in text for char in "\r\n\x00{}"):
            raise MetricsInputError(f"指标标签 {key!r} 不是安全单行值")
        normalized.append((key, text))
    return tuple(sorted(normalized))


def _metric_key(metric_name: str, labels: tuple[tuple[str, str], ...]) -> str:
    """生成稳定、可读且不含敏感正文的指标键。"""

    if not labels:
        return metric_name
    return metric_name + "{" + ",".join(f"{key}={value}" for key, value in labels) + "}"


__all__ = [
    "DeliveryGateReport",
    "IncrementalMetrics",
    "MetricsInputError",
    "evaluate_cdc_delivery_gate",
    "evaluate_delivery_gate",
]
