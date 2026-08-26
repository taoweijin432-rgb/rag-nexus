"""P4.8 PreparationCoordinator 测试。

Coordinator 是 P4 验收修复新增的外层状态协调器：把"已由 P3 领取且持有有效
attempt token 的任务"经 Repository 续租和单调阶段推进驱动到 ``validated``，
产出内存 :class:`PreparedChunkSet`。真正的 lease 有效性由 Repository 以 Index MySQL
``CURRENT_TIMESTAMP(6)`` + ``attempt_token`` + rowcount fencing 权威判定，
Coordinator **不解读** ``task.lease_until`` 的时区或与本机时间比较。

覆盖验收要求：
- 阶段严格单调相邻：``candidate → source_read → normalized → chunk_built → validated``。
- 每次 renew/stage 都携带 task_id + attempt_token；token/lease 冲突立即停止。
- 读取失败不得推进 ``source_read``。
- Handler/Validation 失败不得推进 ``chunk_built`` 或 ``validated``。
- 不调用 Consumer、Artifact、Checkpoint、Milvus、Embedding。
- MySQL naive DATETIME ``lease_until`` 不被应用层错误拒绝（Fake Repository 负责模拟
  数据库 CURRENT_TIMESTAMP(6) 权威判定）。

全部使用 Fake Repository + Fake Source + 真实 ChunkFactory，不连接真实 MySQL/Milvus。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.models import (
    CurrentFactReadRequest,
    ExpertSource,
    ScopedCurrentFacts,
    SourceSyncTask,
    impact_scope_from_target,
)
from expert_index.preparation_coordinator import PreparationCoordinator
from expert_index.validators import ChunkValidationError

CREATED = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
# 注意：MySQL DATETIME 无时区；这里用 naive datetime 模拟真实 Index MySQL lease_until。
# Coordinator 不解读它，Fake Repository 用自己的内部时钟模拟 CURRENT_TIMESTAMP(6)。
NAIVE_LEASE = datetime(2026, 8, 24, 13, 0)  # noqa: DTZ001 - 模拟 MySQL naive DATETIME
AWARE_LEASE = datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc)
TOKEN = "tok-claim-001"
TASK_TS = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
DEDUPE = "a" * 64


class LeaseConflict(Exception):
    """Fake Repository 模拟续租/阶段 token 或 lease 失效冲突。"""


def _settings() -> Settings:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _factory() -> ChunkFactory:
    return ChunkFactory(_settings())


def _expert(expert_id: int = 880001) -> ExpertSource:
    return ExpertSource(
        id=expert_id, serial_number="S", company="ACME", job_status=1,
        job_name="Engineer", contact_status=3, doubtful_flag=1, abroad_flag=2,
        background="bio", work_years=8.0, updated_at=UPDATED, created_at=CREATED,
        deleted_at=None,
    )


def _facts() -> ScopedCurrentFacts:
    return ScopedCurrentFacts(
        scope=impact_scope_from_target(target_type="expert", expert_id=880001),
        expert=_expert(), employments=(), projects=(), educations=(),
    )


def _task(
    *,
    processing_stage: str = "candidate",
    sync_status: str = "processing",
    attempt_token: str | None = TOKEN,
    lease_until: datetime | None = NAIVE_LEASE,
    policy_version: str | None = "1.3",
) -> SourceSyncTask:
    values: dict[str, object] = {
        "task_id": 701, "dedupe_key": DEDUPE, "batch_id": 8, "event_source": "poll",
        "source_table": "experts", "source_record_id": 880001, "source_updated_at": TASK_TS,
        "expert_id": 880001, "operation_type": "update", "target_type": "expert",
        "processing_stage": processing_stage, "sync_status": sync_status,
        "retry_count": 0, "policy_version": policy_version,
        "created_at": TASK_TS, "updated_at": TASK_TS,
    }
    if attempt_token is not None:
        values["attempt_token"] = attempt_token
        values["lease_until"] = lease_until
    return SourceSyncTask(**values)  # type: ignore[arg-type]


class FakeRepo:
    """记录 renew/advance 调用序列的 Fake Repository，模拟数据库权威判定。

    每次 renew/advance 都接受 task_id + attempt_token；模拟 token 失配或 lease
    已变（数据库 CURRENT_TIMESTAMP 判定已过期）时抛 :class:`LeaseConflict`。
    不连接真实 MySQL；不解读传入任务的 lease_until 时区（naive/aware 都接受）。
    """

    def __init__(self, *, fail_at_step: int | None = None) -> None:
        # step 计数（build_prepared_chunk_set 不 bump）：
        # 1=renew0 2=source_read 3=renew1 4=renew2 5=normalized 6=chunk_built 7=validated
        self._step = 0
        self._fail_at = fail_at_step
        self.calls: list[tuple[str, str, str | None]] = []
        self.stages_reached: list[str] = []

    def _bump(self, action: str, token: str) -> None:
        self._step += 1
        self.calls.append((action, token, None))
        if self._fail_at is not None and self._step == self._fail_at:
            raise LeaseConflict(f"step {self._step} conflict (模拟 DB 拒绝)")

    def renew_source_sync_task_lease(
        self, task_id: int, attempt_token: str, *, lease_seconds: int,
    ) -> SourceSyncTask:
        self._bump("renew", attempt_token)
        return _task(attempt_token=attempt_token)

    def advance_source_sync_task_stage(
        self, task_id: int, attempt_token: str, *, next_stage: str,
    ) -> SourceSyncTask:
        self._bump(f"advance:{next_stage}", attempt_token)
        self.stages_reached.append(next_stage)
        return _task(processing_stage=next_stage, attempt_token=attempt_token)


class _ThrowingSource:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.read_calls = 0

    def read_current_facts(self, request: CurrentFactReadRequest) -> ScopedCurrentFacts:
        self.read_calls += 1
        raise self._exc


def _coordinator(repo: FakeRepo, *, source=None) -> PreparationCoordinator:
    return PreparationCoordinator(
        task_repository=repo,  # type: ignore[arg-type]
        source=source if source is not None else _FakeSource(_facts()),
        factory=_factory(),
        lease_seconds=120,
    )


class _FakeSource:
    def __init__(self, facts: ScopedCurrentFacts) -> None:
        self._facts = facts
        self.read_calls: list[CurrentFactReadRequest] = []

    def read_current_facts(self, request: CurrentFactReadRequest) -> ScopedCurrentFacts:
        self.read_calls.append(request)
        return self._facts


# ---------------------------------------------------------------------------
# 成功路径：严格阶段单调推进
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_strict_stage_sequence(self):
        repo = FakeRepo()
        coord = _coordinator(repo)
        result = coord.prepare(_task(), TOKEN)
        assert result.impact_scope.scope_type == "expert"
        # 阶段顺序严格为 source_read → normalized → chunk_built → validated
        assert repo.stages_reached == [
            "source_read", "normalized", "chunk_built", "validated",
        ]
        # 调用序列：renew0, source_read, renew1, (build), renew2, normalized, chunk_built, validated
        assert [c[0] for c in repo.calls] == [
            "renew", "advance:source_read",
            "renew",
            "renew",
            "advance:normalized",
            "advance:chunk_built",
            "advance:validated",
        ]
        # 所有写路径都携带当前 token
        assert all(c[1] == TOKEN for c in repo.calls)

    def test_source_read_only_once(self):
        source = _FakeSource(_facts())
        coord = _coordinator(FakeRepo(), source=source)
        coord.prepare(_task(), TOKEN)
        assert len(source.read_calls) == 1


# ---------------------------------------------------------------------------
# 读取失败不得推进 source_read
# ---------------------------------------------------------------------------


class TestReadFailureStopsBeforeSourceRead:
    def test_read_exception_does_not_advance_stage(self):
        source = _ThrowingSource(RuntimeError("source boom"))
        repo = FakeRepo()
        coord = _coordinator(repo, source=source)
        with pytest.raises(RuntimeError, match="source boom"):
            coord.prepare(_task(), TOKEN)
        # 续租发生一次，但 source_read 未推进
        assert repo.stages_reached == []
        assert [c[0] for c in repo.calls] == ["renew"]


# ---------------------------------------------------------------------------
# Normalize/Build/Validate 失败最多停留在 source_read
# ---------------------------------------------------------------------------


class TestBuildFailureStopsAtSourceRead:
    def test_chunk_validation_error_stays_at_source_read(self, monkeypatch):
        # build_prepared_chunk_set 在 source_read 之后、任何 normalized 推进之前执行；
        # Validation 失败时阶段最多停留在 source_read。
        import expert_index.scope_handlers as sh

        def boom(facts, factory, *, task_id, batch_id, data_version):
            raise ChunkValidationError("v=demo", "validate 失败")

        monkeypatch.setattr(sh, "handle_expert_scope", boom)
        repo = FakeRepo()
        coord = _coordinator(repo)
        with pytest.raises(ChunkValidationError, match="validate 失败"):
            coord.prepare(_task(), TOKEN)
        # source_read 已推进，但 normalized/chunk_built/validated 均未推进
        assert repo.stages_reached == ["source_read"]
        # 最后一次 bump 是 build 前的 renew1（step 3）
        assert repo.calls[-1][0] == "renew"
        assert "advance:normalized" not in [c[0] for c in repo.calls]

    def test_build_returns_before_normalized_advanced(self, monkeypatch):
        # 增加调用顺序测试：证明 build_prepared_chunk_set 成功返回后才首次推进 normalized。
        # 用一个监听器记录 build 被调用时 stages_reached 的快照。
        import expert_index.preparation_coordinator as pc

        build_snapshot: list[list[str]] = []
        real_build = pc.build_prepared_chunk_set

        def spy_build(task, facts, factory):
            build_snapshot.append(list(repo_state))
            return real_build(task, facts, factory)

        repo = FakeRepo()
        repo_state = repo.stages_reached  # FakeRepo 内部可变 list
        monkeypatch.setattr(pc, "build_prepared_chunk_set", spy_build)
        coord = _coordinator(repo)
        coord.prepare(_task(), TOKEN)
        # build 被调用 1 次
        assert len(build_snapshot) == 1
        # build 被调用时阶段只有 source_read（尚未推进 normalized）
        assert build_snapshot[0] == ["source_read"]
        # build 成功返回后才首次推进 normalized
        normalized_idx = [c[0] for c in repo.calls].index("advance:normalized")
        # build 发生在 renew1 之后、renew2/normalized 之前；renew1 是 step 3，
        # renew2 是 step 4，normalized 是 step 5。故 normalized_idx >= 4（0-based）。
        assert normalized_idx >= 3

    def test_normalize_exception_stays_at_source_read(self, monkeypatch):
        # Normalize 阶段抛同样停在 source_read（build 内部 Normalize 失败也向上传播）
        import expert_index.scope_handlers as sh

        def boom(facts, factory, *, task_id, batch_id, data_version):
            raise RuntimeError("normalize 失败")

        monkeypatch.setattr(sh, "handle_expert_scope", boom)
        repo = FakeRepo()
        coord = _coordinator(repo)
        with pytest.raises(RuntimeError, match="normalize 失败"):
            coord.prepare(_task(), TOKEN)
        assert repo.stages_reached == ["source_read"]


# ---------------------------------------------------------------------------
# token/lease 冲突立即停止后续操作
# ---------------------------------------------------------------------------


class TestConflictStopsImmediately:
    @pytest.mark.parametrize("fail_step", [1, 2, 3, 4, 5, 6, 7])
    def test_conflict_stops_at_failure_point(self, fail_step):
        repo = FakeRepo(fail_at_step=fail_step)
        coord = _coordinator(repo)
        with pytest.raises(LeaseConflict):
            coord.prepare(_task(), TOKEN)
        # 冲突发生后，后续没有任何阶段推进超过失败点。
        # 调用序列共 7 个 bump（build_prepared_chunk_set 不 bump）：
        # 1=renew0 2=source_read 3=renew1 4=renew2 5=normalized 6=chunk_built 7=validated。
        # bump 在 append 前 raise，故该阶段未到达。
        if fail_step <= 2:
            expected_stages = []  # renew0 或 source_read 中失败：source_read 未到达
        elif fail_step <= 5:
            expected_stages = ["source_read"]  # renew1/renew2/normalized 中失败
        elif fail_step == 6:
            expected_stages = ["source_read", "normalized"]  # chunk_built 中失败
        else:
            expected_stages = ["source_read", "normalized", "chunk_built"]  # validated 中失败
        assert repo.stages_reached == expected_stages

    def test_wrong_token_rejected_in_memory_before_renew(self):
        # token 与任务不一致：内存层即拒绝，不调 Repository
        repo = FakeRepo()
        coord = _coordinator(repo)
        with pytest.raises(ValueError, match="不一致"):
            coord.prepare(_task(), "wrong-token")
        assert repo.calls == []


# ---------------------------------------------------------------------------
# naive MySQL lease_until 不被应用层错误拒绝
# ---------------------------------------------------------------------------


class TestNaiveLeaseNotRejected:
    def test_naive_lease_until_accepted(self):
        # task.lease_until 为 naive DATETIME（模拟真实 Index MySQL），Coordinator 不解读
        repo = FakeRepo()
        coord = _coordinator(repo)
        result = coord.prepare(_task(lease_until=NAIVE_LEASE), TOKEN)
        assert result.impact_scope.scope_type == "expert"
        assert repo.stages_reached == [
            "source_read", "normalized", "chunk_built", "validated",
        ]

    def test_aware_lease_until_also_accepted(self):
        repo = FakeRepo()
        coord = _coordinator(repo)
        result = coord.prepare(_task(lease_until=AWARE_LEASE), TOKEN)
        assert result.impact_scope.scope_type == "expert"


# ---------------------------------------------------------------------------
# 不推进 contract_ready/handed_off；不调用 Consumer/Artifact/Checkpoint/Milvus
# ---------------------------------------------------------------------------


class TestDoesNotAdvanceP5Stages:
    def test_stops_at_validated_not_contract_ready(self):
        repo = FakeRepo()
        coord = _coordinator(repo)
        coord.prepare(_task(), TOKEN)
        # P4 终态是 validated，不推进 contract_ready/handed_off/completed
        assert "contract_ready" not in repo.stages_reached
        assert "handed_off" not in repo.stages_reached
        assert "completed" not in repo.stages_reached

    def test_coordinator_module_has_no_downstream_imports(self):
        import ast
        from pathlib import Path

        import expert_index.preparation_coordinator as mod

        tree = ast.parse(Path(mod.__file__).read_text())
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                for n in node.names:
                    imported_modules.add(n.name)
        forbidden = {
            "expert_index.checkpoint", "expert_index.contract_bridge",
            "expert_index.vector_sync", "expert_index.handoff",
            "expert_index.handoff_orchestrator", "pymilvus", "requests",
            "urllib", "rag_vector_store",
        }
        leaked = {m for f in forbidden for m in imported_modules if f in m}
        assert not leaked, f"Coordinator 不得 import Consumer/Artifact/Checkpoint/Milvus: {leaked}"


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_source_sync_task_rejected(self):
        coord = _coordinator(FakeRepo())
        with pytest.raises(TypeError, match="SourceSyncTask"):
            coord.prepare("not a task", TOKEN)  # type: ignore[arg-type]

    def test_non_processing_status_rejected(self):
        coord = _coordinator(FakeRepo())
        # sync_status 非 processing（例如 pending，其 token/lease 必须为 None）
        with pytest.raises(ValueError, match="非 processing"):
            coord.prepare(_task(sync_status="pending"), TOKEN)

    def test_blank_token_rejected(self):
        coord = _coordinator(FakeRepo())
        with pytest.raises(ValueError, match="attempt_token"):
            coord.prepare(_task(), "   ")

    def test_token_mismatch_rejected(self):
        coord = _coordinator(FakeRepo())
        with pytest.raises(ValueError, match="不一致"):
            coord.prepare(_task(), "wrong-token")
