"""把 Poll 增量候选确定性映射为来源同步任务。

本模块只处理调度身份，不读取数据库、不解释业务正文，也不执行 Hash 对比或下游动作。
后续分页编排可以复用 :class:`IncrementalScanner`，同时保持单条映射函数可独立测试。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from .models import NewSourceSyncTask, SourceChangeCandidate
from .normalize import canonical_json, sha256_value

POLL_EVENT_SOURCE = "poll"

_POLL_TARGET_TYPE_BY_SOURCE_TABLE = {
    "experts": "expert",
    "experts_work_experience": "employment",
    "experts_project_experience": "project",
    "experts_education_experience": "education",
}


def build_poll_task_dedupe_key(
    candidate: SourceChangeCandidate,
    *,
    policy_version: str,
) -> str:
    """按冻结的 Poll 任务身份字段生成跨批次 SHA-256 幂等键。

    Source MySQL 的 ``DATETIME`` 按原始字面值固定到六位微秒，不执行时区换算，也不
    输出时区偏移。``batch_id``、``expert_id``、``deleted_at`` 原值和业务正文不参与。
    """

    _validate_candidate(candidate)
    _validate_policy_version(policy_version)
    operation_type = "delete" if candidate.is_deleted else "update"
    target_type = _target_type(candidate.source_table)
    identity = {
        "event_source": POLL_EVENT_SOURCE,
        "source_table": candidate.source_table,
        "source_record_id": candidate.record_id,
        "source_updated_at": _source_datetime_literal(candidate.updated_at),
        "operation_type": operation_type,
        "target_type": target_type,
        "policy_version": policy_version,
    }
    return sha256_value(canonical_json(identity))


def map_poll_candidate_to_task(
    candidate: SourceChangeCandidate,
    *,
    batch_id: int,
    policy_version: str,
) -> NewSourceSyncTask:
    """把一个最小 Poll 候选映射为可幂等持久化的 pending 任务参数。

    ``source_updated_at`` 始终取候选扫描使用的中间表 ``updated_at``；对 ``experts`` 也
    不替换成正式来源的 ``experts_updated_at``。
    """

    _validate_candidate(candidate)
    _validate_batch_id(batch_id)
    _validate_policy_version(policy_version)
    if candidate.source_table == "experts" and candidate.expert_id != candidate.record_id:
        raise ValueError("experts 候选的 expert_id 必须等于来源记录 id")

    operation_type = "delete" if candidate.is_deleted else "update"
    target_type = _target_type(candidate.source_table)
    return NewSourceSyncTask(
        dedupe_key=build_poll_task_dedupe_key(
            candidate,
            policy_version=policy_version,
        ),
        batch_id=batch_id,
        event_source=POLL_EVENT_SOURCE,
        source_table=candidate.source_table,
        source_record_id=candidate.record_id,
        source_updated_at=candidate.updated_at,
        expert_id=candidate.expert_id,
        operation_type=operation_type,
        target_type=target_type,
        policy_version=policy_version,
    )


@dataclass(frozen=True)
class IncrementalScanner:
    """使用固定策略版本批量映射已读取的 Poll 候选。

    P3.4 的服务没有 Source 或 Index 依赖；它只保持输入顺序并调用单条纯映射函数。
    固定上界读取、任务落库和临时游标推进由 P3.5 在此边界之外接入。
    """

    policy_version: str

    def __post_init__(self) -> None:
        """在处理任何候选前校验策略版本可写入任务契约。"""

        _validate_policy_version(self.policy_version)

    def map_candidate(
        self,
        candidate: SourceChangeCandidate,
        *,
        batch_id: int,
    ) -> NewSourceSyncTask:
        """使用服务固定的策略版本映射一个候选。"""

        return map_poll_candidate_to_task(
            candidate,
            batch_id=batch_id,
            policy_version=self.policy_version,
        )

    def map_candidates(
        self,
        candidates: Iterable[SourceChangeCandidate],
        *,
        batch_id: int,
    ) -> tuple[NewSourceSyncTask, ...]:
        """保持候选顺序，返回不可变的任务参数序列。"""

        _validate_batch_id(batch_id)
        return tuple(
            self.map_candidate(candidate, batch_id=batch_id) for candidate in candidates
        )


def _source_datetime_literal(value: datetime) -> str:
    """把 Source DATETIME 格式化为无时区偏移的六位微秒字面值。"""

    return value.replace(tzinfo=None).isoformat(timespec="microseconds")


def _target_type(source_table: str) -> str:
    """返回 P3 冻结的来源表级最小影响类型。"""

    try:
        return _POLL_TARGET_TYPE_BY_SOURCE_TABLE[source_table]
    except KeyError as exc:
        raise ValueError(f"不支持的 Poll 来源表: {source_table}") from exc


def _validate_candidate(candidate: SourceChangeCandidate) -> None:
    """校验映射入口收到显式的最小来源候选模型。"""

    if not isinstance(candidate, SourceChangeCandidate):
        raise TypeError("candidate 必须是 SourceChangeCandidate")


def _validate_batch_id(batch_id: int) -> None:
    """校验任务所属批次可映射到 MySQL 无符号主键。"""

    if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
        raise ValueError("batch_id 必须是正整数")


def _validate_policy_version(policy_version: str) -> None:
    """校验策略版本是任务表能够保存的非空稳定文本。"""

    if (
        not isinstance(policy_version, str)
        or not policy_version.strip()
        or policy_version != policy_version.strip()
        or len(policy_version) > 32
    ):
        raise ValueError("policy_version 必须是长度不超过 32 的非空字符串")
