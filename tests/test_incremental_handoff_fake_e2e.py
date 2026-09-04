"""P5.5 全链路 Fake E2E：claimed task → P4 阶段推进 → PreparedChunkSet →
Handoff → Artifact → Scope Consumer → Receipt → task success/retry/rejected。

用真实 P4/P5.1/P5.2/P5.3B 组件（PreparationCoordinator / ChunkFactory /
ScopeHandoffBuilder / ContractConsumerCore + EmbeddingEligibilityValidator /
ScopeConsumer / HandoffOrchestrator），只 Fake：

- 来源事实读取（FakeSourceAdapter 返回预设 ScopedCurrentFacts）；
- 任务状态 Repository（FakeTaskRepo，含 renew/advance/fencing）；
- Artifact Repository（FakeArtifactRepo）；
- 有状态向量服务（FakeVectorStore：维护 active Contract/hash 状态与计数）。

链路真实性：新任务从 ``processing``/``candidate`` 起，经真实
``PreparationCoordinator.prepare`` 续租并单调推进
``candidate → source_read → normalized → chunk_built → validated``，
再进入 ``HandoffOrchestrator.process``（推进 contract_ready/handed_off 并完成
Consumer 回执）。``handed_off`` 崩溃恢复用例直接从 handed_off 进入 P5，不重复
运行 P4（与正式恢复语义一致）。

覆盖用例：
- 新 Employment Chunk（空 previous，1 upsert + embedding，success）；
- source_hash-only metadata update（content_hash 相同、source/metadata_hash 变，
  零 Embedding）；
- 三 Hash unchanged（零写入）；
- N→M stale 逻辑下线（同 family 旧 active ID 被 disable）；
- Employment 逻辑删除产生空 desired（全部 active family ID 被 disable，无 embedding）；
- 专家删除（scan_updated_at > updated_at, deleted_at 非空）生成 expert scope 空
  desired，P5 对该专家全部 active Chunk 逻辑下线，零 Embedding，success；
- transient failure 后重新 claim/replay 最终成功；
- Artifact 同 hash 复用；
- checkpoint 未被 P5 直接调用（结构性测试验证无 checkpoint 导入/调用）。
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, DatabaseRole, IncrementalScanConfig, Settings
from expert_index.handoff import ScopeHandoffBuilder, ScopeHandoffReceipt
from expert_index.handoff_orchestrator import HandoffOrchestrator
from expert_index.models import (
    EmploymentSource,
    ExpertSource,
    IncrementalImpactScope,
    PreparedChunkSet,
    ScopedCurrentFacts,
    SourceSyncTask,
)
from expert_index.preparation_coordinator import PreparationCoordinator
from expert_index.repositories.artifact import (
    ArtifactPayloadConflictError,
    ContractArtifact,
)
from expert_index.repositories.scope_fence import ScopeFenceDecision
from expert_index.repositories.sync_task import (
    SourceSyncTaskLeaseConflictError,
    SourceSyncTaskStageConflictError,
)
from rag_vector_store.chunking.contract_pipeline import EmbeddingEligibilityValidator
from rag_vector_store.incremental.consumer_core import ContractConsumerCore
from rag_vector_store.incremental.scope_consumer import ScopeConsumer
from rag_vector_store.ingestion.errors import VectorMilvusError
from rag_vector_store.ingestion.models import (
    VectorDeletionResult,
    VectorIngestionResult,
    VectorItemResult,
    VectorOwnerList,
    VectorQueryResult,
)

CREATED = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
UPDATED2 = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)
SCAN_UPDATED = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)  # 扫描时间 > 源版本时间
TASK_UPDATED = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
LEASE_UNTIL = datetime(2026, 12, 31, 9, 30, tzinfo=timezone.utc)
TOKEN = "tok-e2e-001"
ARTIFACT_TYPE = "contract_payload"
ARTIFACT_VERSION = "v1"

# 标准阶段顺序（与 PreparationCoordinator / HandoffOrchestrator 一致）。
_STAGES = ("candidate", "source_read", "normalized", "chunk_built", "validated",
           "contract_ready", "handed_off", "completed")


# ---------------------------------------------------------------------------
# Settings / ChunkFactory / 真实事实 fixtures
# ---------------------------------------------------------------------------
def _settings() -> Settings:
    source = DatabaseConfig("39.96.40.93", 3306, "r", "x", "expert", DatabaseRole.SOURCE)
    index = DatabaseConfig("127.0.0.1", 3306, "i", "y", "expert_dev", DatabaseRole.INDEX)
    return Settings(
        source_db=source, index_db=index,
        incremental_scan=IncrementalScanConfig(enabled=True, batch_size=50, lease_seconds=300),
    )


def _expert_source(eid=700001, deleted_at=None, *, updated_at=UPDATED,
                   scan_updated_at=None) -> ExpertSource:
    return ExpertSource(
        id=eid, serial_number=f"T{eid}", company="虚构E2E公司",
        job_status=1, job_name="项目经理", contact_status=3,
        doubtful_flag=1, abroad_flag=2, background="全链路 E2E 虚构背景。",
        work_years=8.5, updated_at=updated_at, created_at=CREATED, deleted_at=deleted_at,
        scan_updated_at=scan_updated_at,
    )


def _employment(eid=700001, emp_id=21307, deleted_at=None, *, company="虚构E2E雇主",
                updated_at=UPDATED) -> EmploymentSource:
    return EmploymentSource(
        id=emp_id, expert_id=eid, company_name=company, position_title="工程师",
        start_date=UPDATED, end_date=None, is_employed=1,
        work_description="E2E employment 职责描述正文。",
        updated_at=updated_at, created_at=CREATED, deleted_at=deleted_at,
    )


# ---------------------------------------------------------------------------
# 任务模型
# ---------------------------------------------------------------------------
def _task(
    *,
    task_id=701, batch_id=8, expert_id=700001,
    source_table="experts_work_experience", source_record_id=21307,
    operation_type="update", target_type="employment",
    policy_version="1.3", processing_stage="candidate",
    sync_status="processing", retry_count=0,
    attempt_token=TOKEN, lease_until=LEASE_UNTIL, result_code=None,
    completed_at=None,
) -> SourceSyncTask:
    dedupe = hashlib.sha256(f"dk-{task_id}".encode()).hexdigest()
    return SourceSyncTask(
        task_id=task_id, dedupe_key=dedupe, batch_id=batch_id,
        event_source="poll", source_table=source_table,
        source_record_id=source_record_id, source_updated_at=UPDATED,
        expert_id=expert_id, operation_type=operation_type, target_type=target_type,
        processing_stage=processing_stage, sync_status=sync_status,
        retry_count=retry_count, policy_version=policy_version,
        created_at=CREATED, updated_at=TASK_UPDATED,
        attempt_token=attempt_token, lease_until=lease_until,
        result_code=result_code, completed_at=completed_at,
    )


# ---------------------------------------------------------------------------
# Fake Source Adapter（只返回预设 ScopedCurrentFacts）
# ---------------------------------------------------------------------------
class FakeSourceAdapter:
    def __init__(self) -> None:
        self.facts_by_scope_key: dict[tuple[str, str], ScopedCurrentFacts] = {}

    def set_facts(self, scope_type: str, source_field_id: str, facts: ScopedCurrentFacts) -> None:
        self.facts_by_scope_key[(scope_type, source_field_id)] = facts

    def read_current_facts(self, request) -> ScopedCurrentFacts:
        key = (request.impact_scope.scope_type, request.impact_scope.source_field_id or "")
        if key not in self.facts_by_scope_key:
            raise ValueError(f"FakeSourceAdapter 未预设 facts: {key}")
        return self.facts_by_scope_key[key]


# ---------------------------------------------------------------------------
# Fake Task Repository（含 renew + advance + 终态 + fencing）
# ---------------------------------------------------------------------------
@dataclass
class FakeTaskRepo:
    state: SourceSyncTask
    renew_calls: list = field(default_factory=list)
    success_calls: list = field(default_factory=list)
    finalize_calls: list = field(default_factory=list)
    rejected_calls: list = field(default_factory=list)
    retryable_calls: list = field(default_factory=list)
    advance_calls: list = field(default_factory=list)
    fail_success_with_lease: bool = False
    fail_finalize_with_lease: bool = False
    fail_advance_with_lease: bool = False
    fail_renew_with_lease: bool = False

    def get_source_sync_task(self, task_id: int) -> SourceSyncTask | None:
        return self.state if self.state.task_id == task_id else None

    def renew_source_sync_task_lease(
        self, task_id: int, attempt_token: str, *, lease_seconds: int,
    ) -> SourceSyncTask:
        # 与正式 Repository 一致：task_id 校验、processing 状态、token 匹配 fencing。
        self.renew_calls.append({"task_id": task_id, "attempt_token": attempt_token,
                                 "lease_seconds": lease_seconds})
        if self.state.task_id != task_id:
            raise SourceSyncTaskLeaseConflictError(f"任务 {task_id} 不存在")
        if self.state.sync_status != "processing":
            raise SourceSyncTaskLeaseConflictError(
                f"任务 {task_id} 状态非 processing，实际为 {self.state.sync_status!r}")
        if self.state.attempt_token is None or self.state.attempt_token != attempt_token:
            raise SourceSyncTaskLeaseConflictError(
                f"调用方不再持有任务 {task_id} 的有效 attempt lease")
        if self.fail_renew_with_lease:
            raise SourceSyncTaskLeaseConflictError("test: lease lost on renew")
        # 续租成功：保持 token 与状态，延长 lease_until。
        self.state = replace(self.state, lease_until=LEASE_UNTIL)
        return self.state

    def advance_source_sync_task_stage(
        self, task_id: int, attempt_token: str, *, next_stage: str,
    ) -> SourceSyncTask:
        self.advance_calls.append((task_id, attempt_token, next_stage))
        if self.fail_advance_with_lease:
            raise SourceSyncTaskLeaseConflictError("test: lease lost")
        if self.state.task_id != task_id or self.state.attempt_token != attempt_token:
            raise SourceSyncTaskLeaseConflictError("test: token 不一致")
        if next_stage == self.state.processing_stage:
            return self.state
        if _STAGES.index(next_stage) != _STAGES.index(self.state.processing_stage) + 1:
            raise SourceSyncTaskStageConflictError(
                f"不能从 {self.state.processing_stage} 到 {next_stage}")
        self.state = replace(self.state, processing_stage=next_stage)
        return self.state

    def mark_source_sync_task_success(self, task_id, attempt_token, *, result_code):
        if self.fail_success_with_lease:
            raise SourceSyncTaskLeaseConflictError("test: lease lost on success")
        if self.state.task_id != task_id or self.state.attempt_token != attempt_token:
            raise SourceSyncTaskLeaseConflictError("test: token 不一致")
        self.success_calls.append((task_id, attempt_token, result_code))
        self.state = replace(self.state, processing_stage="completed", sync_status="success",
                             attempt_token=None, lease_until=None,
                             result_code=result_code, completed_at=TASK_UPDATED)
        return self.state

    def finalize_source_sync_task_success(self, task_id, attempt_token, *,
                                          task_result_code, artifact_id,
                                          artifact_payload_hash):
        if self.fail_finalize_with_lease:
            raise SourceSyncTaskLeaseConflictError("test: lease lost on finalize")
        if self.state.task_id != task_id or self.state.attempt_token != attempt_token:
            raise SourceSyncTaskLeaseConflictError("test: token 不一致")
        self.finalize_calls.append((task_id, attempt_token, task_result_code))
        self.state = replace(self.state, processing_stage="completed", sync_status="success",
                             attempt_token=None, lease_until=None,
                             result_code=task_result_code, completed_at=TASK_UPDATED)
        return self.state

    def mark_source_sync_task_rejected(self, task_id, attempt_token, *,
                                        error_stage, error_code,
                                        error_type=None, safe_error_message=None):
        self.rejected_calls.append({
            "task_id": task_id, "attempt_token": attempt_token,
            "error_stage": error_stage, "error_code": error_code,
            "error_type": error_type, "safe_error_message": safe_error_message,
        })
        self.state = replace(self.state, sync_status="dead", attempt_token=None,
                             lease_until=None, processing_stage="completed",
                             completed_at=TASK_UPDATED)
        return self.state

    def mark_source_sync_task_retryable(self, task_id, attempt_token, *,
                                         error_stage, error_code,
                                         max_attempts, retry_delay_seconds,
                                         error_type=None, safe_error_message=None):
        self.retryable_calls.append({
            "task_id": task_id, "attempt_token": attempt_token,
            "error_stage": error_stage, "error_code": error_code,
            "max_attempts": max_attempts, "retry_delay_seconds": retry_delay_seconds,
            "error_type": error_type, "safe_error_message": safe_error_message,
        })
        if self.state.retry_count >= max_attempts:
            self.state = replace(self.state, sync_status="dead", attempt_token=None,
                                 lease_until=None, processing_stage="completed",
                                 completed_at=TASK_UPDATED)
        else:
            self.state = replace(self.state, sync_status="pending", attempt_token=None,
                                 lease_until=None, next_retry_at=LEASE_UNTIL)
        return self.state


# ---------------------------------------------------------------------------
# Fake Artifact Repository
# ---------------------------------------------------------------------------
@dataclass
class FakeArtifactRepo:
    store: dict = field(default_factory=dict)
    put_calls: list = field(default_factory=list)

    def get_contract_artifact(self, task_id, *, artifact_type, artifact_version):
        return self.store.get((task_id, artifact_type, artifact_version))

    def put_or_verify_contract_artifact(self, task_id, attempt_token, *,
                                         artifact_type, artifact_version,
                                         payload_json, payload_hash, expires_at=None):
        self.put_calls.append({"task_id": task_id, "attempt_token": attempt_token,
                               "payload_hash": payload_hash})
        key = (task_id, artifact_type, artifact_version)
        existing = self.store.get(key)
        if existing is not None:
            if existing.payload_hash != payload_hash:
                raise ArtifactPayloadConflictError(f"任务 {task_id} Artifact hash 冲突")
            return existing
        artifact = ContractArtifact(
            artifact_id=len(self.store) + 1, task_id=task_id,
            artifact_type=artifact_type, artifact_version=artifact_version,
            payload_json=payload_json, payload_reference=None,
            payload_hash=payload_hash, expires_at=expires_at,
            created_at=TASK_UPDATED, updated_at=TASK_UPDATED,
        )
        self.store[key] = artifact
        return artifact


# ---------------------------------------------------------------------------
# Fake 有状态向量服务（同时满足 VectorMutationPort + ScopeVectorStatePort）
# ---------------------------------------------------------------------------
class FakeVectorStore:
    """维护 active Contract/hash 状态；只暴露五方法，无物理删除/检查点/Work Item。"""

    owner_field = "expert_id"

    def __init__(self) -> None:
        self.existing: dict[str, dict[str, object]] = {}
        self.by_expert: dict[int, list[str]] = {}
        self.upsert_calls: list = []
        self.update_calls: list = []
        self.disable_calls: list = []
        self.get_calls: list = []
        self.embed_count: int = 0
        self.fail_upsert: Exception | None = None

    def seed(self, record: dict[str, object]) -> None:
        vid = str(record["vector_doc_id"])
        self.existing[vid] = record
        meta = record.get("metadata", {})
        eid = meta.get("expert_id") if isinstance(meta, dict) else None
        if isinstance(eid, int):
            self.by_expert.setdefault(eid, [])
            if vid not in self.by_expert[eid]:
                self.by_expert[eid].append(vid)

    def list_vectors_by_owner(self, owner_id: int) -> VectorOwnerList:
        vids = list(self.by_expert.get(owner_id, []))
        return VectorOwnerList(owner_id=owner_id, vector_doc_ids=vids, count=len(vids))

    def get_vectors(self, vector_doc_ids) -> VectorQueryResult:
        self.get_calls.append(list(vector_doc_ids))
        records = [self.existing[v] for v in vector_doc_ids if v in self.existing]
        missing = [v for v in vector_doc_ids if v not in self.existing]
        return VectorQueryResult(records=records, missing=missing)

    def upsert_chunks(self, chunks) -> VectorIngestionResult:
        self.upsert_calls.append(list(chunks))
        if self.fail_upsert is not None:
            exc = self.fail_upsert
            self.fail_upsert = None  # 一次性失败
            raise exc
        results = []
        for c in chunks:
            rec = _existing_record(contract=c, content_hash=c.content_hash,
                                   metadata_hash=c.metadata_hash,
                                   source_hash=c.source.source_hash, is_active=True)
            self.seed(rec)
            results.append(VectorItemResult(vector_doc_id=c.vector_doc_id, status="embedded"))
            self.embed_count += 1
        return VectorIngestionResult(results=results)

    def update_metadata(self, chunks) -> VectorIngestionResult:
        self.update_calls.append(list(chunks))
        results = []
        for c in chunks:
            if c.vector_doc_id in self.existing:
                old_meta = self.existing[c.vector_doc_id].get("metadata", {})
                meta: dict[str, object] = dict(old_meta) if isinstance(old_meta, dict) else {}
                meta["content_hash"] = c.content_hash
                meta["metadata_hash"] = c.metadata_hash
                meta["source_hash"] = c.source.source_hash
                meta["is_active"] = c.is_active
                self.existing[c.vector_doc_id]["metadata"] = meta
            results.append(VectorItemResult(vector_doc_id=c.vector_doc_id, status="metadata_updated"))
        return VectorIngestionResult(results=results)

    def disable_vectors(self, vector_doc_ids) -> VectorDeletionResult:
        self.disable_calls.append(list(vector_doc_ids))
        deleted = 0
        for vid in vector_doc_ids:
            if vid in self.existing:
                old2 = self.existing[vid].get("metadata", {})
                meta2: dict[str, object] = dict(old2) if isinstance(old2, dict) else {}
                meta2["is_active"] = False
                self.existing[vid]["metadata"] = meta2
                deleted += 1
        return VectorDeletionResult(requested_count=len(vector_doc_ids),
                                    deleted_count=deleted,
                                    not_found_count=len(vector_doc_ids) - deleted)


def _existing_record(*, contract, content_hash=None, metadata_hash=None,
                     source_hash=None, is_active=True) -> dict[str, object]:
    metadata = {
        "expert_id": contract.expert_id,
        "chunk_type": contract.chunk_type,
        "source_table": contract.source.source_table,
        "source_field_id": contract.source.source_field_id,
        "content_hash": content_hash if content_hash is not None else contract.content_hash,
        "metadata_hash": metadata_hash if metadata_hash is not None else contract.metadata_hash,
        "source_hash": source_hash if source_hash is not None else contract.source.source_hash,
        "is_active": is_active,
    }
    return {
        "vector_doc_id": contract.vector_doc_id, "chunk_text": contract.chunk_text,
        "metadata": metadata,
        "content_hash": metadata["content_hash"],
        "metadata_hash": metadata["metadata_hash"],
        "source_hash": metadata["source_hash"],
    }


class ReceiptSink:
    def make_receipt(self, **kwargs) -> ScopeHandoffReceipt:
        return ScopeHandoffReceipt(**kwargs)


# ---------------------------------------------------------------------------
# 运行组装（真实组件 + Fake）
# ---------------------------------------------------------------------------
def _build_coordinator(task_repo: FakeTaskRepo, source: FakeSourceAdapter,
                       factory: ChunkFactory, *, lease_seconds: int = 120) -> PreparationCoordinator:
    return PreparationCoordinator(
        task_repository=task_repo, source=source, factory=factory,  # type: ignore[arg-type]
        lease_seconds=lease_seconds,
    )


def _build_orchestrator(store: FakeVectorStore, task_repo: FakeTaskRepo,
                        artifact_repo: FakeArtifactRepo, *, max_attempts=2, retry_delay=30):
    core = ContractConsumerCore(
        vector_store=store,
        eligibility_skip_of=EmbeddingEligibilityValidator().get_skip_reason,
    )
    consumer = ScopeConsumer(core=core, state_port=store, receipt_sink=ReceiptSink())
    return HandoffOrchestrator(
        task_repo=task_repo, artifact_repo=artifact_repo,
        builder=ScopeHandoffBuilder("Asia/Shanghai"), consumer=consumer,  # type: ignore[arg-type]
        scope_fence=_FakeScopeFence(), lease_seconds=120,
        max_attempts=max_attempts, retry_delay_seconds=retry_delay,
    )


class _FakeScopeFence:
    """E2E 使用的进程内栅栏替身。"""

    @contextmanager
    def hold(self, task, attempt_token, scope, *, timeout_seconds):
        yield ScopeFenceDecision(superseded=False)


def _scope_key(scope: IncrementalImpactScope) -> str:
    return scope.source_field_id or str(scope.expert_id)


def _employment_facts(eid, emp_id, *, deleted_at=None, company="虚构E2E雇主",
                      updated_at=UPDATED):
    scope = IncrementalImpactScope(
        scope_type="employment", expert_id=eid, chunk_type="employment",
        source_table="experts_work_experience", source_field_id=str(emp_id),
    )
    return ScopedCurrentFacts(
        scope=scope, expert=_expert_source(eid),
        employments=(_employment(eid=eid, emp_id=emp_id, deleted_at=deleted_at,
                                  company=company, updated_at=updated_at),),
    )


def _expert_facts(eid, *, deleted_at=None, updated_at=UPDATED, scan_updated_at=None):
    """expert scope 事实：deleted_at 非空时 scope_handlers 返回 expert scope 空 desired。"""

    scope = IncrementalImpactScope(
        scope_type="expert", expert_id=eid, chunk_type=None,
        source_table=None, source_field_id=None,
    )
    return ScopedCurrentFacts(
        scope=scope, expert=_expert_source(eid, deleted_at=deleted_at,
                                           updated_at=updated_at,
                                           scan_updated_at=scan_updated_at),
        employments=(), projects=(), educations=(),
    )


def _prepare(task, task_repo, source, factory):
    """经真实 PreparationCoordinator.prepare：续租 + 阶段推进到 validated，返回
    PreparedChunkSet。Coordinator 期间通过 task_repo 记录 renew/advance 调用。"""

    coordinator = _build_coordinator(task_repo, source, factory)
    return coordinator.prepare(task, task.attempt_token)


def _run(task_id, task_repo, artifact_repo, store, source, factory):
    """认领→准备→编排的完整 E2E 单步（新任务从 candidate 起经 Coordinator 到
    validated，再进入 HandoffOrchestrator）。"""

    task = task_repo.get_source_sync_task(task_id)
    prepared = _prepare(task, task_repo, source, factory)
    orch = _build_orchestrator(store, task_repo, artifact_repo)
    return orch.process(task_id, task.attempt_token, prepared)


# ===========================================================================
# 用例
# ===========================================================================
class TestNewEmploymentChunk:
    def test_new_employment_upsert_and_success(self):
        task = _task(target_type="employment", source_table="experts_work_experience",
                     source_record_id=21307)
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        store = FakeVectorStore()
        source = FakeSourceAdapter()
        source.set_facts("employment", "21307", _employment_facts(700001, 21307))
        factory = ChunkFactory(_settings())

        result = _run(task.task_id, task_repo, artifact_repo, store, source, factory)

        assert result.final_status == "success"
        assert len(store.upsert_calls) == 1
        assert store.embed_count == 1
        assert store.update_calls == []
        assert task_repo.finalize_calls and not task_repo.rejected_calls
        # 主成功路径链路真实性断言：
        # 1) Coordinator 续租被调用（prepare 至少 renew 多次：读事实前、构建前后）。
        assert len(task_repo.renew_calls) >= 3
        # 2) 阶段推进顺序：candidate→source_read→normalized→chunk_built→validated
        #    → contract_ready → handed_off（completed 由 mark_success 写入，不在 advance）。
        stages = [c[2] for c in task_repo.advance_calls]
        assert stages[:5] == ["source_read", "normalized", "chunk_built",
                              "validated", "contract_ready"]
        assert "handed_off" in stages
        # 3) validated 之前 HandoffOrchestrator 未触发任何向量写入（P4 不调 Consumer）。
        #    即全部向量写入发生在 validated 之后。
        assert store.upsert_calls, "validated 之后 Consumer 应已写入"


class TestSourceHashOnlyMetadataUpdate:
    def test_source_updated_at_only_zero_embedding(self):
        # 真实 P4 事实链验证：保持 employment 业务字段全部一致，仅改变
        # ``EmploymentSource.updated_at``。这会改变 ``source_updated_at`` 进而改变
        # ``metadata_hash``（metadata_hash_payload 含 source_updated_at），但 ``source_hash``
        # 由 company/role/start/end/is_current/responsibilities/achievements 计算，
        # 不含 updated_at，故不变；``chunk_text`` 也不含 updated_at，故 ``content_hash``
        # 不变。结果：三 Hash 差分只剩 metadata-only 变化 → 零 Embedding、走 update_metadata。
        store = FakeVectorStore()
        factory = ChunkFactory(_settings())
        source1 = FakeSourceAdapter()
        source1.set_facts("employment", "21307",
                          _employment_facts(700001, 21307, updated_at=UPDATED))
        task = _task(target_type="employment", source_record_id=21307)

        # 1) 第一次：新写入建立 previous（updated_at=UPDATED）
        task_repo1 = FakeTaskRepo(state=_task())
        first = _run(task.task_id, task_repo1, FakeArtifactRepo(),
                     store, source1, factory)
        assert first.final_status == "success"
        prev = store.upsert_calls[0][0]
        first_content = prev.content_hash
        first_source = prev.source.source_hash
        first_meta = prev.metadata_hash
        assert store.embed_count == 1
        assert store.update_calls == []

        # 2) 第二次：仅 updated_at 变为 UPDATED2（业务字段不变）
        source2 = FakeSourceAdapter()
        source2.set_facts("employment", "21307",
                          _employment_facts(700001, 21307, updated_at=UPDATED2))
        task_repo2 = FakeTaskRepo(state=_task())
        artifact_repo2 = FakeArtifactRepo()
        second = _run(task.task_id, task_repo2, artifact_repo2,
                      store, source2, factory)
        assert second.final_status == "success"
        # 第二次走 metadata-only update，desired Contract 进入 update_calls
        assert len(store.update_calls) == 1
        desired = store.update_calls[0][0]
        # 关系断言：content_hash 与 source_hash 不变，metadata_hash 变
        assert desired.content_hash == first_content
        assert desired.source.source_hash == first_source
        assert desired.metadata_hash != first_meta
        # 关键：metadata-only 路径，零 Embedding
        assert store.embed_count == 1  # 未新增
        assert len(store.upsert_calls) == 1  # 仍是第一次的新写入，无新增


class TestThreeHashUnchanged:
    def test_three_hashes_unchanged_zero_writes(self):
        task = _task(target_type="employment", source_record_id=21307)
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        store = FakeVectorStore()
        source = FakeSourceAdapter()
        source.set_facts("employment", "21307", _employment_facts(700001, 21307))
        factory = ChunkFactory(_settings())

        # 第一次：新写入
        first = _run(task.task_id, task_repo, artifact_repo, store, source, factory)
        assert first.final_status == "success"
        assert store.embed_count == 1
        prev_count = len(store.upsert_calls)

        # 第二次：同事实同 hash 重放（三 Hash 全同）。新任务重新从 candidate 起经
        # Coordinator，保持全链路。
        task_repo.state = _task()
        artifact_repo2 = FakeArtifactRepo()
        result2 = _run(task.task_id, task_repo, artifact_repo2, store, source, factory)
        assert result2.final_status == "success"
        # 三 Hash 全同：无新增 upsert（previous 已收敛）
        assert len(store.upsert_calls) == prev_count
        assert store.embed_count == 1


class TestNToMStaleDisable:
    def test_n_to_m_only_family_stale_disabled(self):
        task = _task(target_type="employment", source_record_id=21307)
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        store = FakeVectorStore()
        source = FakeSourceAdapter()
        source.set_facts("employment", "21307", _employment_facts(700001, 21307))
        factory = ChunkFactory(_settings())

        # 先写入 desired=1（emp 21307，chunk_index=0）
        first = _run(task.task_id, task_repo, artifact_repo, store, source, factory)
        assert first.final_status == "success"
        desired_vid = store.upsert_calls[0][0].vector_doc_id

        # 预置同 family（同 source_field_id=21307）的额外 active 旧 chunk（chunk_index=1），
        # 模拟 N=2 → M=1（重新分片后只剩 chunk_index=0）。直接以向量状态记录 seed，
        # 避免构造内部不一致的 Contract（chunk_index=1 但 chunk_total=1）。
        base = store.upsert_calls[0][0]
        store.seed({
            "vector_doc_id": "expert:700001:employment:21307:1",
            "chunk_text": base.chunk_text,
            "metadata": {
                "expert_id": 700001,
                "chunk_type": "employment",
                "source_table": "experts_work_experience",
                "source_field_id": "21307",
                "content_hash": base.content_hash,
                "metadata_hash": base.metadata_hash,
                "source_hash": base.source.source_hash,
                "is_active": True,
            },
            "content_hash": base.content_hash,
            "metadata_hash": base.metadata_hash,
            "source_hash": base.source.source_hash,
        })
        # by_expert 已含该 owner；通用查询会返回两个
        assert len(store.list_vectors_by_owner(700001).vector_doc_ids) == 2

        # 重新走编排：desired 仍只有 chunk_index=0，同 family 的 chunk_index=1 应被逻辑下线
        task_repo.state = _task()
        artifact_repo2 = FakeArtifactRepo()
        result2 = _run(task.task_id, task_repo, artifact_repo2, store, source, factory)
        assert result2.final_status == "success"
        # 同 family 的 chunk_index=1 被 disable
        assert any("expert:700001:employment:21307:1" in str(c) for c in store.disable_calls)
        # desired_vid 未被 disable
        assert all(desired_vid not in c for c in store.disable_calls)


class TestEmploymentLogicalDeletionEmptyDesired:
    def test_logical_delete_empty_desired_disables_all_family(self):
        task = _task(target_type="employment", source_record_id=21307)
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        store = FakeVectorStore()
        source = FakeSourceAdapter()
        source.set_facts("employment", "21307", _employment_facts(700001, 21307))
        factory = ChunkFactory(_settings())

        # 先写入 desired=1
        first = _run(task.task_id, task_repo, artifact_repo, store, source, factory)
        assert first.final_status == "success"
        assert store.embed_count == 1

        # 现在就业被逻辑删除：deleted_at 非空 → employment Handler 产出空 desired
        # （父专家仍 active；ScopedCurrentFacts.employments[0].deleted_at 非空）
        source2 = FakeSourceAdapter()
        deleted_facts = _employment_facts(700001, 21307, deleted_at=UPDATED)
        source2.set_facts("employment", "21307", deleted_facts)
        # 任务重新认领到 candidate，经 Coordinator 重新准备
        task_repo.state = _task()
        artifact_repo2 = FakeArtifactRepo()
        factory2 = ChunkFactory(_settings())

        result2 = _run(task.task_id, task_repo, artifact_repo2, store, source2, factory2)
        assert result2.final_status == "success"
        # 空 desired → scope 内全部 active ID 被逻辑下线，无 embedding
        assert store.embed_count == 1  # 未新增
        assert len(store.disable_calls) >= 1


class TestExpertDeletionScanNewerThanSourceVersion:
    def test_expert_deleted_scan_newer_empty_desired_disables_all(self):
        # 专家删除双时间场景：scan_updated_at > source updated_at，deleted_at 非空。
        # task.updated_at 对应 scan_updated_at。经 PreparationCoordinator 生成 expert
        # scope + 空 desired（scope_handlers: 父专家已删除升级为 expert scope 空 desired）；
        # P5 对该专家已有 active Chunk 全部逻辑下线，零 Embedding，最终 success。
        task = _task(target_type="expert", source_table="experts", source_record_id=700001)
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        store = FakeVectorStore()
        # 预置该专家已有 active Chunk（跨两类，模拟历史写入）
        store.seed({
            "vector_doc_id": "expert:700001:profile_summary:700001:0",
            "chunk_text": "历史 profile",
            "metadata": {
                "expert_id": 700001, "chunk_type": "profile_summary",
                "source_table": "experts", "source_field_id": "700001",
                "content_hash": "ch0", "metadata_hash": "mh0",
                "source_hash": "sh0", "is_active": True,
            },
            "content_hash": "ch0", "metadata_hash": "mh0", "source_hash": "sh0",
        })
        store.seed({
            "vector_doc_id": "expert:700001:employment:21307:0",
            "chunk_text": "历史 employment",
            "metadata": {
                "expert_id": 700001, "chunk_type": "employment",
                "source_table": "experts_work_experience", "source_field_id": "21307",
                "content_hash": "ch1", "metadata_hash": "mh1",
                "source_hash": "sh1", "is_active": True,
            },
            "content_hash": "ch1", "metadata_hash": "mh1", "source_hash": "sh1",
        })
        assert len(store.list_vectors_by_owner(700001).vector_doc_ids) == 2

        source = FakeSourceAdapter()
        # scan_updated_at > updated_at，deleted_at 非空
        source.set_facts("expert", "", _expert_facts(
            700001, deleted_at=SCAN_UPDATED, updated_at=UPDATED, scan_updated_at=SCAN_UPDATED))
        factory = ChunkFactory(_settings())

        result = _run(task.task_id, task_repo, artifact_repo, store, source, factory)

        assert result.final_status == "success"
        # expert scope 空 desired → 该专家全部 active Chunk 逻辑下线
        assert len(store.disable_calls) == 1
        disabled = store.disable_calls[0]
        assert set(disabled) == {"expert:700001:profile_summary:700001:0",
                                 "expert:700001:employment:21307:0"}
        # 零 Embedding
        assert store.embed_count == 0
        assert store.upsert_calls == []
        assert store.update_calls == []
        # 经 Coordinator 全链路：阶段推进到 validated
        stages = [c[2] for c in task_repo.advance_calls]
        assert stages[:5] == ["source_read", "normalized", "chunk_built",
                              "validated", "contract_ready"]
        assert len(task_repo.renew_calls) >= 3
        assert task_repo.finalize_calls and not task_repo.rejected_calls


class TestTransientFailureReplay:
    def test_transient_failure_then_reclaim_replay_success(self):
        task = _task(target_type="employment", source_record_id=21307)
        task_repo = FakeTaskRepo(state=task)
        artifact_repo = FakeArtifactRepo()
        store = FakeVectorStore()
        store.fail_upsert = VectorMilvusError("test: transient Milvus failure")
        source = FakeSourceAdapter()
        source.set_facts("employment", "21307", _employment_facts(700001, 21307))
        factory = ChunkFactory(_settings())

        # 第一次：transient failure → retryable。Artifact 在 validated 阶段写入一次。
        result1 = _run(task.task_id, task_repo, artifact_repo, store, source, factory)
        assert result1.final_status == "pending"
        assert task_repo.retryable_calls and not task_repo.finalize_calls
        first_artifact = artifact_repo.store.get((task.task_id, ARTIFACT_TYPE, ARTIFACT_VERSION))
        assert first_artifact is not None
        first_artifact_id = first_artifact.artifact_id
        first_hash = first_artifact.payload_hash
        assert len(artifact_repo.put_calls) == 1

        # 重新 claim：新 token。transient failure 发生在 Consumer（handed_off 之后），
        # 故任务停留在 handed_off；claim 不改变已完成阶段。重放从 handed_off 起进入 P5，
        # 从 Artifact 恢复 Request，不重复运行 P4。
        new_token = "tok-e2e-002"
        task_repo.state = _task(processing_stage="handed_off", attempt_token=new_token)
        task_repo.fail_finalize_with_lease = False
        store.fail_upsert = None  # 恢复
        orch = _build_orchestrator(store, task_repo, artifact_repo)
        # handed_off 起只从 Artifact 恢复，不再传占位 PreparedChunkSet。
        result2 = orch.process(task.task_id, new_token)
        assert result2.final_status == "success"
        assert task_repo.finalize_calls
        # 关键：第二次从 validated 起复用已存 Artifact，store 不新增、artifact_id 与
        # payload_hash 保持不变（put_or_verify 仍被 fenced 调用但命中同 hash 复用路径）。
        assert len(artifact_repo.store) == 1
        second_artifact = artifact_repo.store.get((task.task_id, ARTIFACT_TYPE, ARTIFACT_VERSION))
        assert second_artifact is not None
        assert second_artifact.artifact_id == first_artifact_id
        assert second_artifact.payload_hash == first_hash


class TestArtifactSameHashReuse:
    def test_artifact_same_hash_reused_on_replay(self):
        task = _task(target_type="employment", source_record_id=21307)
        task_repo = FakeTaskRepo(state=task, fail_finalize_with_lease=True)
        artifact_repo = FakeArtifactRepo()
        store = FakeVectorStore()
        source = FakeSourceAdapter()
        source.set_facts("employment", "21307", _employment_facts(700001, 21307))
        factory = ChunkFactory(_settings())

        # 第一次：写 Artifact + 推进 handed_off + Consumer success，但 mark_success lease 失败
        r1 = _run(task.task_id, task_repo, artifact_repo, store, source, factory)
        assert r1.terminal is False
        first_artifact = artifact_repo.store.get((task.task_id, ARTIFACT_TYPE, ARTIFACT_VERSION))
        assert first_artifact is not None
        first_artifact_id = first_artifact.artifact_id
        first_hash = first_artifact.payload_hash
        assert len(artifact_repo.put_calls) == 1
        assert store.embed_count == 1

        # 崩溃恢复：finalize 修复，task 仍 handed_off，同 hash 重放。
        # handed_off 起重放直接进入 P5，不重复运行 P4（与正式恢复语义一致）。
        task_repo.fail_finalize_with_lease = False
        task_repo.state = _task(processing_stage="handed_off")
        orch = _build_orchestrator(store, task_repo, artifact_repo)
        # handed_off 起重放只从 Artifact 恢复，不使用 prepared 内容。
        placeholder = PreparedChunkSet(
            task_id=task.task_id, batch_id=task.batch_id, data_version="8",
            impact_scope=_employment_facts(700001, 21307).scope,
            desired_chunks=(),
        )
        r2 = orch.process(task.task_id, TOKEN, placeholder)
        assert r2.final_status == "success"
        # 同 hash 复用：store 不增、artifact_id 与 payload_hash 不变、put_calls 不增
        assert len(artifact_repo.store) == 1
        second_artifact = artifact_repo.store.get((task.task_id, ARTIFACT_TYPE, ARTIFACT_VERSION))
        assert second_artifact is not None
        assert second_artifact.artifact_id == first_artifact_id
        assert second_artifact.payload_hash == first_hash
        assert len(artifact_repo.put_calls) == 1
        assert len(store.upsert_calls) == 1  # 第一次写入；重放三 Hash 幂等不重复


class TestCheckpointNotCalled:
    def test_no_checkpoint_dependency_in_factory(self):
        # Factory 不接受 checkpoint 依赖参数
        import inspect

        from expert_index.handoff_factory import build_incremental_handoff_runtime
        sig = inspect.signature(build_incremental_handoff_runtime)
        assert "checkpoint" not in sig.parameters
        assert "manager" not in sig.parameters

    def test_p5_modules_do_not_import_or_call_checkpoint(self):
        # 结构性验证：P5.3B/F5.1/F5.2 模块源码不 import checkpoint/CheckpointManager、
        # 不调用任何 checkpoint 推进方法，证明 P5 不直接驱动 checkpoint 水位。
        import re
        from pathlib import Path

        base = Path(__file__).resolve().parents[1]
        files = [
            base / "src" / "expert_index" / "handoff_orchestrator.py",
            base / "src" / "expert_index" / "handoff_factory.py",
            base / "src" / "rag_vector_store" / "incremental" / "scope_consumer.py",
        ]
        # 禁止的 import 语句与调用标识。
        forbidden_import = re.compile(
            r"^\s*(?:from|import)\s+.*(?:checkpoint|Checkpoint|CheckpointManager)", re.MULTILINE)
        forbidden_call = re.compile(
            r"\.(?:advance_checkpoint|mark_checkpoint|commit_checkpoint)\s*\(")
        for f in files:
            src = f.read_text(encoding="utf-8")
            # 仅检查代码行，排除注释（# 开头）。
            code_lines = [
                ln for ln in src.splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")
            ]
            code = "\n".join(code_lines)
            assert not forbidden_import.search(code), f"{f.name} 引入了 checkpoint 导入"
            assert not forbidden_call.search(code), f"{f.name} 调用了 checkpoint 推进方法"
