"""把 Poll 增量候选确定性映射为来源同步任务。

本模块只处理调度身份，不读取数据库、不解释业务正文，也不执行 Hash 对比或下游动作。
后续分页编排可以复用 :class:`IncrementalScanner`，同时保持单条映射函数可独立测试。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from .config import (
    DEFAULT_INCREMENTAL_BATCH_SIZE,
    DEFAULT_INCREMENTAL_LEASE_SECONDS,
    Settings,
)
from .dataset import EXPERT_SPEC, DatasetSpec
from .models import (
    BatchSourceScan,
    NewSourceSyncTask,
    SourceChangeCandidate,
    SourceScanCursor,
)
from .normalize import canonical_json, sha256_value
from .repositories import MySQLSourceStateRepository, MySQLSourceSyncTaskRepository
from .source import MAX_SOURCE_SCAN_PAGE_SIZE, MySQLSourceAdapter

POLL_EVENT_SOURCE = "poll"
DEFAULT_INCREMENTAL_PAGE_SIZE = DEFAULT_INCREMENTAL_BATCH_SIZE
DEFAULT_SOURCE_LEASE_SECONDS = DEFAULT_INCREMENTAL_LEASE_SECONDS

# 四表顺序是一次扫描周期的稳定契约，不能从无序白名单集合推导。
INCREMENTAL_SOURCE_TABLE_ORDER = (
    "experts",
    "experts_work_experience",
    "experts_project_experience",
    "experts_education_experience",
)

_SAFE_ORCHESTRATION_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_BATCH_SOURCE_ALIAS = {
    "experts": "experts",
    "experts_work_experience": "work",
    "experts_project_experience": "project",
    "experts_education_experience": "education",
}


def build_poll_task_dedupe_key(
    candidate: SourceChangeCandidate,
    *,
    policy_version: str,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> str:
    """按冻结的 Poll 任务身份字段生成跨批次 SHA-256 幂等键。

    Source MySQL 的 ``DATETIME`` 按原始字面值固定到六位微秒，不执行时区换算，也不
    输出时区偏移。``batch_id``、``expert_id``、``deleted_at`` 原值和业务正文不参与。
    """

    _validate_candidate(candidate, dataset_spec=dataset_spec)
    _validate_policy_version(policy_version)
    operation_type = "delete" if candidate.is_deleted else "update"
    target_type = _target_type(candidate.source_table, dataset_spec=dataset_spec)
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
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> NewSourceSyncTask:
    """把一个最小 Poll 候选映射为可幂等持久化的 pending 任务参数。

    ``source_updated_at`` 始终取候选扫描使用的中间表 ``updated_at``；对 ``experts`` 也
    不替换成正式来源的 ``experts_updated_at``。
    """

    _validate_candidate(candidate, dataset_spec=dataset_spec)
    _validate_batch_id(batch_id)
    _validate_policy_version(policy_version)
    if (
        dataset_spec.profile_identity_column == "id"
        and candidate.source_table == dataset_spec.profile_table
        and candidate.expert_id != candidate.record_id
    ):
        raise ValueError("experts 候选的 expert_id 必须等于来源记录 id")

    operation_type = "delete" if candidate.is_deleted else "update"
    target_type = _target_type(candidate.source_table, dataset_spec=dataset_spec)
    return NewSourceSyncTask(
        dedupe_key=build_poll_task_dedupe_key(
            candidate,
            policy_version=policy_version,
            dataset_spec=dataset_spec,
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
    dataset_spec: DatasetSpec = EXPERT_SPEC

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
            dataset_spec=self.dataset_spec,
        )

    def map_candidates(
        self,
        candidates: Iterable[SourceChangeCandidate],
        *,
        batch_id: int,
    ) -> tuple[NewSourceSyncTask, ...]:
        """保持候选顺序，返回不可变的任务参数序列。"""

        _validate_batch_id(batch_id)
        return tuple(self.map_candidate(candidate, batch_id=batch_id) for candidate in candidates)


class CrossBatchTaskConflictError(RuntimeError):
    """表示候选幂等键意外命中其他逻辑批次的已有任务。"""


class IncrementalScanDisabledError(RuntimeError):
    """表示环境配置未显式允许创建增量候选任务。"""


@dataclass(frozen=True)
class IncrementalScanResult:
    """一张来源表完成候选扫描后的固定范围结果。"""

    source_table: str
    batch_source: BatchSourceScan | None
    recovered: bool
    page_count: int
    scheduled_candidate_count: int
    lease_released: bool
    # 保留扫描期间的 owner，供总编排入口继续使用同一来源 lease 执行批次。
    lease_owner: str | None = None


@dataclass(frozen=True)
class IncrementalSourceScanOutcome:
    """保存一张来源表的扫描结果或不含原始异常文本的失败类型。"""

    source_table: str
    result: IncrementalScanResult | None = None
    failure_type: str | None = None

    def __post_init__(self) -> None:
        """确保成功结果和失败类型二选一，且结果属于当前来源表。"""

        if (self.result is None) == (self.failure_type is None):
            raise ValueError("扫描结果必须在 result 和 failure_type 之间二选一")
        if self.result is not None and self.result.source_table != self.source_table:
            raise ValueError("扫描结果的来源表与 outcome 不一致")

    @property
    def succeeded(self) -> bool:
        """返回当前来源表是否完成候选扫描。"""

        return self.result is not None


@dataclass(frozen=True)
class IncrementalScanCycleResult:
    """按固定四表顺序保存一次候选扫描周期的隔离结果。"""

    cycle_id: str
    outcomes: tuple[IncrementalSourceScanOutcome, ...]

    @property
    def succeeded_source_count(self) -> int:
        """返回本周期成功完成候选扫描的来源表数量。"""

        return sum(outcome.succeeded for outcome in self.outcomes)

    @property
    def failed_source_count(self) -> int:
        """返回本周期发生隔离失败的来源表数量。"""

        return len(self.outcomes) - self.succeeded_source_count


@dataclass(frozen=True)
class IncrementalSourceScanService:
    """按固定上界把一张来源表的候选可靠落成 pending 任务。

    服务只完成候选发现和任务调度，不读取业务完整事实、不比较三 Hash，也不处理任务或
    推进安全 Checkpoint。有固定范围时来源 lease 保持有效，供后续任务处理与提交阶段
    继续使用；零候选范围会立即释放 lease。
    """

    source_adapter: object
    source_state_repository: MySQLSourceStateRepository
    task_repository: MySQLSourceSyncTaskRepository
    policy_version: str
    page_size: int = DEFAULT_INCREMENTAL_PAGE_SIZE
    lease_seconds: int = DEFAULT_SOURCE_LEASE_SECONDS
    dataset_spec: DatasetSpec = EXPERT_SPEC

    def __post_init__(self) -> None:
        """校验分页、租约和策略参数，避免扫描启动后才发现配置错误。"""

        _validate_policy_version(self.policy_version)
        if not isinstance(self.dataset_spec, DatasetSpec):
            raise TypeError("dataset_spec 必须是 DatasetSpec")
        if (
            isinstance(self.page_size, bool)
            or not isinstance(self.page_size, int)
            or not 1 <= self.page_size <= MAX_SOURCE_SCAN_PAGE_SIZE
        ):
            raise ValueError(f"page_size 必须位于 1 到 {MAX_SOURCE_SCAN_PAGE_SIZE} 之间")
        if (
            isinstance(self.lease_seconds, bool)
            or not isinstance(self.lease_seconds, int)
            or self.lease_seconds < 1
        ):
            raise ValueError("lease_seconds 必须是正整数")

    def scan_source(
        self,
        source_table: str,
        lease_owner: str,
        *,
        batch_no: str,
    ) -> IncrementalScanResult:
        """恢复或创建固定范围，并按“先任务、后游标”扫描到上界。"""

        _validate_scan_request(
            source_table,
            lease_owner,
            batch_no,
            dataset_spec=self.dataset_spec,
        )
        checkpoint = self.source_state_repository.acquire_source_lease(
            source_table,
            lease_owner,
            lease_seconds=self.lease_seconds,
        )
        scan = self.source_state_repository.get_recoverable_batch_source_scan(
            source_table,
            lease_owner,
        )
        recovered = scan is not None
        if scan is None:
            upper_bound = self.source_adapter.get_source_scan_upper_bound(source_table)
            start_cursor = SourceScanCursor(
                updated_at=checkpoint.last_updated_at,
                record_id=checkpoint.last_id,
            )
            if not _has_new_source_range(start_cursor, upper_bound):
                self.source_state_repository.release_source_lease(
                    source_table,
                    lease_owner,
                )
                return IncrementalScanResult(
                    source_table=source_table,
                    batch_source=None,
                    recovered=False,
                    page_count=0,
                    scheduled_candidate_count=0,
                    lease_released=True,
                    lease_owner=lease_owner,
                )
            assert upper_bound is not None
            scan = self.source_state_repository.create_incremental_batch_source_scan(
                source_table,
                lease_owner,
                batch_no=batch_no,
                upper_bound=upper_bound,
            )

        mapper = IncrementalScanner(
            policy_version=self.policy_version,
            dataset_spec=self.dataset_spec,
        )
        page_count = 0
        scheduled_candidate_count = 0
        if scan.current_cursor == scan.upper_bound and scan.status == "failed":
            scan = self.source_state_repository.advance_batch_source_scan_cursor(
                source_table,
                scan.batch_id,
                lease_owner,
                checkpoint_lock_version=scan.checkpoint_lock_version,
                expected_cursor=scan.current_cursor,
                next_cursor=scan.upper_bound,
                candidate_increment=0,
            )
        while scan.current_cursor < scan.upper_bound:
            self.source_state_repository.renew_source_lease(
                source_table,
                lease_owner,
                lease_seconds=self.lease_seconds,
            )
            candidates = self.source_adapter.load_source_change_candidates(
                source_table,
                after_cursor=scan.current_cursor,
                upper_bound=scan.upper_bound,
                limit=self.page_size,
            )
            _validate_candidate_page(candidates, scan, page_size=self.page_size)
            if not candidates:
                scan = self.source_state_repository.advance_batch_source_scan_cursor(
                    source_table,
                    scan.batch_id,
                    lease_owner,
                    checkpoint_lock_version=scan.checkpoint_lock_version,
                    expected_cursor=scan.current_cursor,
                    next_cursor=scan.upper_bound,
                    candidate_increment=0,
                )
                break

            tasks = mapper.map_candidates(candidates, batch_id=scan.batch_id)
            for task in tasks:
                task_id = self.task_repository.create_source_sync_task(task)
                persisted_task = self.task_repository.get_source_sync_task(task_id)
                if persisted_task is None:
                    raise RuntimeError(f"任务 {task_id} 创建后无法读取持久化状态")
                if persisted_task.batch_id != scan.batch_id:
                    raise CrossBatchTaskConflictError(
                        f"dedupe_key 命中其他批次任务: "
                        f"current={scan.batch_id}, existing={persisted_task.batch_id}"
                    )

            next_cursor = candidates[-1].cursor
            scan = self.source_state_repository.advance_batch_source_scan_cursor(
                source_table,
                scan.batch_id,
                lease_owner,
                checkpoint_lock_version=scan.checkpoint_lock_version,
                expected_cursor=scan.current_cursor,
                next_cursor=next_cursor,
                candidate_increment=len(candidates),
            )
            page_count += 1
            scheduled_candidate_count += len(candidates)

        return IncrementalScanResult(
            source_table=source_table,
            batch_source=scan,
            recovered=recovered,
            page_count=page_count,
            scheduled_candidate_count=scheduled_candidate_count,
            lease_released=False,
            lease_owner=lease_owner,
        )


class IncrementalScanCoordinator:
    """按固定顺序独立调用四张来源表的候选扫描服务。

    每张表使用独立批次号并由单表服务获取各自 lease。单表异常只记录稳定异常类型，
    协调器继续扫描后续表；本类不处理 pending 任务，也不提交安全 Checkpoint。
    """

    def __init__(
        self,
        scan_service: IncrementalSourceScanService,
        *,
        dataset_spec: DatasetSpec = EXPERT_SPEC,
    ) -> None:
        """绑定单表扫描服务和固定四表顺序，不暴露额外业务执行方法。"""

        if not isinstance(dataset_spec, DatasetSpec):
            raise TypeError("dataset_spec 必须是 DatasetSpec")
        self.scan_service = scan_service
        self._dataset_spec = dataset_spec

    def scan_all_sources(
        self,
        worker_id: str,
        *,
        cycle_id: str,
    ) -> IncrementalScanCycleResult:
        """执行一次四表候选扫描并返回逐表成功或失败结果。

        ``cycle_id`` 必须由调用方为一次新扫描周期生成唯一值；恢复既有未完成范围时，
        单表服务会继续使用数据库中保存的原批次，不会创建重复范围。
        """

        _validate_orchestration_identifier(
            "worker_id",
            worker_id,
            maximum_length=48,
        )
        _validate_orchestration_identifier(
            "cycle_id",
            cycle_id,
            maximum_length=48,
        )
        batch_numbers = {
            source_table: _incremental_batch_no(
                cycle_id,
                source_table,
                dataset_spec=self._dataset_spec,
            )
            for source_table in self._dataset_spec.source_tables
        }
        lease_owners = {
            source_table: _source_lease_owner(
                worker_id,
                cycle_id,
                source_table,
                dataset_spec=self._dataset_spec,
            )
            for source_table in self._dataset_spec.source_tables
        }

        outcomes: list[IncrementalSourceScanOutcome] = []
        for source_table in self._dataset_spec.source_tables:
            try:
                result = self.scan_service.scan_source(
                    source_table,
                    lease_owners[source_table],
                    batch_no=batch_numbers[source_table],
                )
            except Exception as exc:  # noqa: BLE001 - 单表故障必须与后续来源表隔离。
                # 原始异常文本可能包含数据库值；周期摘要只暴露可聚合的异常类名。
                outcomes.append(
                    IncrementalSourceScanOutcome(
                        source_table=source_table,
                        failure_type=type(exc).__name__,
                    )
                )
                continue
            outcomes.append(
                IncrementalSourceScanOutcome(
                    source_table=source_table,
                    result=result,
                )
            )

        return IncrementalScanCycleResult(
            cycle_id=cycle_id,
            outcomes=tuple(outcomes),
        )


def build_incremental_scan_coordinator(
    settings: Settings,
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
    source_adapter: object | None = None,
) -> IncrementalScanCoordinator:
    """从正式 Source/Index 配置组装默认关闭的候选扫描协调器。

    构造阶段先验证所有状态写入明确指向独立 Index Schema，再检查显式启用开关。
    返回的协调器只创建 pending 候选任务，不领取任务、不写 success，也不推进 Checkpoint。
    """

    if not isinstance(settings, Settings):
        raise TypeError("settings 必须是 Settings")
    settings.validate_index_write_boundary()
    if not settings.incremental_scan.enabled:
        raise IncrementalScanDisabledError("INCREMENTAL_ENABLED 未开启，候选扫描入口保持禁用")
    scan_config = settings.incremental_scan
    resolved_source = source_adapter or MySQLSourceAdapter(settings.source_db)
    return IncrementalScanCoordinator(
        scan_service=IncrementalSourceScanService(
            source_adapter=resolved_source,
            source_state_repository=MySQLSourceStateRepository(settings.index_db),
            task_repository=MySQLSourceSyncTaskRepository(settings.index_db),
            policy_version=settings.chunk_policy_version,
            page_size=scan_config.batch_size,
            lease_seconds=scan_config.lease_seconds,
            dataset_spec=dataset_spec,
        ),
        dataset_spec=dataset_spec,
    )


def _source_datetime_literal(value: datetime) -> str:
    """把 Source DATETIME 格式化为无时区偏移的六位微秒字面值。"""

    return value.replace(tzinfo=None).isoformat(timespec="microseconds")


def _incremental_batch_no(
    cycle_id: str,
    source_table: str,
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> str:
    """生成同一周期内按来源表隔离且满足数据库长度约束的批次号。"""

    batch_no = f"inc-{cycle_id}-{_source_alias(source_table, dataset_spec=dataset_spec)}"
    if len(batch_no) > 64:
        raise ValueError("生成的 incremental batch_no 超过 64 个字符")
    return batch_no


def _source_lease_owner(
    worker_id: str,
    cycle_id: str,
    source_table: str,
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> str:
    """生成一次扫描周期内按来源表隔离且不可跨表复用的 lease owner。"""

    lease_owner = f"{worker_id}.{cycle_id}.{_source_alias(source_table, dataset_spec=dataset_spec)}"
    if len(lease_owner) > 128:
        raise ValueError("生成的 source lease_owner 超过 128 个字符")
    return lease_owner


def _validate_orchestration_identifier(
    name: str,
    value: str,
    *,
    maximum_length: int,
) -> None:
    """校验编排标识可安全用于 lease owner 和批次号。"""

    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum_length
        or _SAFE_ORCHESTRATION_IDENTIFIER.fullmatch(value) is None
    ):
        raise ValueError(f"{name} 必须是长度不超过 {maximum_length} 的安全非空标识")


def _has_new_source_range(
    start_cursor: SourceScanCursor,
    upper_bound: SourceScanCursor | None,
) -> bool:
    """返回 Source 当前上界是否严格位于安全 Checkpoint 之后。"""

    if upper_bound is None:
        return False
    try:
        return upper_bound > start_cursor
    except TypeError as exc:
        raise ValueError("Source 上界与 Checkpoint 必须使用一致的时间类型") from exc


def _validate_candidate_page(
    candidates: list[SourceChangeCandidate],
    scan: BatchSourceScan,
    *,
    page_size: int,
) -> None:
    """校验 Source 返回页严格位于当前游标和固定上界之间并保持升序。"""

    if len(candidates) > page_size:
        raise RuntimeError("Source 返回候选数超过请求页大小")
    previous_cursor = scan.current_cursor
    for candidate in candidates:
        if not isinstance(candidate, SourceChangeCandidate):
            raise TypeError("Source 候选页包含非 SourceChangeCandidate 对象")
        if candidate.source_table != scan.source_table:
            raise RuntimeError("Source 候选页包含其他来源表记录")
        try:
            in_range = previous_cursor < candidate.cursor <= scan.upper_bound
        except TypeError as exc:
            raise ValueError("候选和固定范围必须使用一致的时间类型") from exc
        if not in_range:
            raise RuntimeError("Source 候选页未严格升序或越过固定范围")
        previous_cursor = candidate.cursor


def _validate_scan_request(
    source_table: str,
    lease_owner: str,
    batch_no: str,
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> None:
    """校验一张来源表扫描请求的表名、owner 和批次号。"""

    if source_table not in dataset_spec.checkpoint_source_tables:
        raise ValueError(f"不支持的 Checkpoint 来源表: {source_table}")
    if (
        not isinstance(lease_owner, str)
        or not lease_owner.strip()
        or lease_owner != lease_owner.strip()
        or len(lease_owner) > 128
    ):
        raise ValueError("lease_owner 必须是长度不超过 128 的非空字符串")
    if (
        not isinstance(batch_no, str)
        or not batch_no.strip()
        or batch_no != batch_no.strip()
        or len(batch_no) > 64
    ):
        raise ValueError("batch_no 必须是长度不超过 64 的非空字符串")


def _target_type(
    source_table: str,
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> str:
    """返回 P3 冻结的来源表级最小影响类型。"""

    if source_table == dataset_spec.profile_table:
        # 主档变化（尤其人物删除/恢复）必须收敛人物完整范围。
        return "expert"
    return dataset_spec.target_type_for_source_table(source_table)


def _validate_candidate(
    candidate: SourceChangeCandidate,
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> None:
    """校验映射入口收到显式的最小来源候选模型。"""

    if not isinstance(candidate, SourceChangeCandidate):
        raise TypeError("candidate 必须是 SourceChangeCandidate")
    if candidate.source_table not in dataset_spec.checkpoint_source_tables:
        raise ValueError(f"候选来源表 {candidate.source_table} 不属于 {dataset_spec.name} 数据集")


def _source_alias(source_table: str, *, dataset_spec: DatasetSpec) -> str:
    """返回批次号和租约使用的短别名，不把物理表名直接拼入标识。"""

    if dataset_spec == EXPERT_SPEC:
        try:
            return _BATCH_SOURCE_ALIAS[source_table]
        except KeyError as exc:  # pragma: no cover - 前置白名单已经拒绝。
            raise ValueError(f"不支持的 Poll 来源表: {source_table}") from exc
    mapping = {
        dataset_spec.profile_table: "profile",
        dataset_spec.employment_table: "work",
        dataset_spec.project_table: "project",
        dataset_spec.education_table: "education",
    }
    try:
        return mapping[source_table]
    except KeyError as exc:
        raise ValueError(f"{dataset_spec.name} 不支持来源表: {source_table}") from exc


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
