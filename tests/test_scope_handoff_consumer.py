"""P5.2 正式范围 Consumer 测试。

全部使用 Fake：不连接真实 Milvus/MySQL，不创建 commit。

覆盖：
    - 四类 exact scope 过滤（expert/profile/employment/project，跨 family 与 inactive 不误判）；
    - 三 Hash 三种更新路径（embedding / metadata-only / unchanged）；
    - 空 desired（逻辑下线 scope 全部 active，零 Embedding）；
    - N→M stale 下线；
    - inactive 重新激活（交由 ContractConsumerCore）；
    - desired rejected/failed 时不执行 stale 下线；
    - disable 失败返回 retryable failed；
    - 重放幂等（首次写成功后重放不重复 Embedding）；
    - Receipt 计数与 identity；
    - 结构断言：无 expert_index/V15WorkItem/repository/MySQL/state/artifact/checkpoint、
      无物理删除。
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from expert_contracts import ExpertChunkEmbeddingInput
from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.handoff import ScopeHandoffReceipt, ScopeHandoffRequest
from expert_index.models import (
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
    impact_scope_from_target,
)
from expert_index.normalize import normalize_aggregate
from rag_vector_store.incremental.consumer_core import ContractConsumerCore
from rag_vector_store.incremental.scope_consumer import ScopeConsumer
from rag_vector_store.ingestion.errors import VectorIngestionError
from rag_vector_store.ingestion.models import (
    VectorDeletionResult,
    VectorIngestionResult,
    VectorItemResult,
    VectorOwnerList,
    VectorQueryResult,
)
from resume_index.dataset import RESUME_SPEC

CREATED = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
TASK_UPDATED = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
TOKEN = "tok-claim-001"


def _factory() -> ChunkFactory:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return ChunkFactory(Settings(source_db=database, index_db=database))


def _expert_source(eid: int = 700001, deleted_at=None) -> ExpertSource:
    return ExpertSource(
        id=eid,
        serial_number=f"TEST{eid}",
        company="虚构范围Consumer测试公司",
        job_status=1,
        job_name="项目经理",
        contact_status=3,
        doubtful_flag=1,
        abroad_flag=2,
        background="仅用于范围 Consumer 单元测试的虚构背景。",
        work_years=8.5,
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _employment(eid: int = 700001, emp_id: int = 21307, deleted_at=None) -> EmploymentSource:
    return EmploymentSource(
        id=emp_id,
        expert_id=eid,
        company_name="虚构范围Consumer测试公司",
        position_title="高级工程师",
        start_date=UPDATED,
        end_date=None,
        work_description="负责范围 Consumer 单元测试。",
        is_employed=1,
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _project(eid: int = 700001, proj_id: int = 161, deleted_at=None) -> ProjectSource:
    return ProjectSource(
        id=proj_id,
        expert_id=eid,
        project_title="范围 Consumer 测试项目",
        company_name="虚构范围Consumer测试公司",
        start_date=UPDATED,
        end_date=None,
        work_description="范围 Consumer 测试目标与动作。",
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _education(eid: int = 700001, edu_id: int = 1) -> EducationSource:
    return EducationSource(
        id=edu_id,
        expert_id=eid,
        school_name="范围 Consumer 测试大学",
        major="测试工程",
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=None,
    )


def _aggregate(eid=700001, *, employments=(), projects=(), educations=(), deleted_at=None):
    return ExpertAggregate(
        profile=_expert_source(eid, deleted_at=deleted_at),
        employments=tuple(employments),
        projects=tuple(projects),
        educations=tuple(educations),
    )


def _chunks(agg):
    profile, employments, projects = normalize_aggregate(
        agg, data_version="20260824_scope", build_batch_id=701
    )
    return _factory().build_all(profile, employments, projects)


def _make_request(scope, contracts: list[ExpertChunkEmbeddingInput]) -> ScopeHandoffRequest:
    """直接构造 ScopeHandoffRequest（不经 Builder，避免 SourceSyncTask 依赖）。"""

    import hashlib
    import json

    scope_obj = {
        "scope_type": scope.scope_type,
        "expert_id": scope.expert_id,
        "chunk_type": scope.chunk_type,
        "source_table": scope.source_table,
        "source_field_id": scope.source_field_id,
    }
    payload = {
        "task_id": 701,
        "batch_id": 8,
        "impact_scope": scope_obj,
        "contracts": [c.model_dump(mode="json") for c in contracts],
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    request_id = f"handoff:{hashlib.sha256(f'701:8:{TOKEN}'.encode()).hexdigest()}"
    return ScopeHandoffRequest(
        request_id=request_id,
        task_id=701,
        batch_id=8,
        attempt_token=TOKEN,
        impact_scope=scope,
        contracts=tuple(contracts),
        payload_hash=payload_hash,
    )


def _existing_record(
    *,
    contract: ExpertChunkEmbeddingInput,
    content_hash: str | None = None,
    metadata_hash: str | None = None,
    source_hash: str | None = None,
    is_active: bool = True,
    owner_field: str = "expert_id",
) -> dict[str, object]:
    """构造 FakeVectorStore 的已存记录（V15 flat metadata 形态）。"""

    metadata = {
        owner_field: contract.expert_id,
        "chunk_type": contract.chunk_type,
        "source_table": contract.source.source_table,
        "source_field_id": contract.source.source_field_id,
        "content_hash": content_hash if content_hash is not None else contract.content_hash,
        "metadata_hash": metadata_hash if metadata_hash is not None else contract.metadata_hash,
        "source_hash": source_hash if source_hash is not None else contract.source.source_hash,
        "is_active": is_active,
    }
    return {
        "vector_doc_id": contract.vector_doc_id,
        "chunk_text": contract.chunk_text,
        "metadata": metadata,
        "content_hash": metadata["content_hash"],
        "metadata_hash": metadata["metadata_hash"],
        "source_hash": metadata["source_hash"],
    }


# ---------------------------------------------------------------------------
# Fake: 向量能力 + scope 状态端口 + Receipt 汇
# ---------------------------------------------------------------------------
class FakeScopeVectorStore:
    """同时满足 VectorMutationPort（供 ContractConsumerCore）与 ScopeVectorStatePort。"""

    def __init__(self, *, owner_field: str = "expert_id") -> None:
        self.owner_field = owner_field
        self.existing: dict[str, dict[str, object]] = {}
        self.by_expert: dict[int, list[str]] = {}
        self.upsert_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.update_calls: list[list[ExpertChunkEmbeddingInput]] = []
        self.disable_calls: list[list[str]] = []
        self.get_calls: list[list[str]] = []
        self.raise_on_disable: Exception | None = None
        self.update_mode: str = "ok"

    def seed(self, record: dict[str, object]) -> None:
        vid = str(record["vector_doc_id"])
        self.existing[vid] = record
        metadata = record.get("metadata", {})
        expert_id = (
            metadata.get(self.owner_field)
            if isinstance(metadata, dict)
            else None
        )
        if isinstance(expert_id, int):
            self.by_expert.setdefault(expert_id, [])
            if vid not in self.by_expert[expert_id]:
                self.by_expert[expert_id].append(vid)

    def list_vectors_by_owner(self, owner_id: int) -> VectorOwnerList:
        return VectorOwnerList(
            owner_id=owner_id,
            vector_doc_ids=list(self.by_expert.get(owner_id, [])),
            count=len(self.by_expert.get(owner_id, [])),
        )

    def get_vectors(self, vector_doc_ids: Sequence[str]) -> VectorQueryResult:
        self.get_calls.append(list(vector_doc_ids))
        records = [self.existing[vid] for vid in vector_doc_ids if vid in self.existing]
        missing = [vid for vid in vector_doc_ids if vid not in self.existing]
        return VectorQueryResult(records=records, missing=missing)

    def upsert_chunks(self, chunks: Sequence[ExpertChunkEmbeddingInput]) -> VectorIngestionResult:
        self.upsert_calls.append(list(chunks))
        for c in chunks:
            # 写入后记录可见（含三 Hash 一致，active）
            rec = _existing_record(contract=c, content_hash=c.content_hash, metadata_hash=c.metadata_hash, source_hash=c.source.source_hash, is_active=True)
            self.seed(rec)
        return VectorIngestionResult(
            results=[
                VectorItemResult(vector_doc_id=c.vector_doc_id, status="embedded")
                for c in chunks
            ]
        )

    def update_metadata(self, chunks: Sequence[ExpertChunkEmbeddingInput]) -> VectorIngestionResult:
        self.update_calls.append(list(chunks))
        if self.update_mode == "raise_inconsistent":
            from rag_vector_store.ingestion.errors import VectorContractError

            raise VectorContractError("update_metadata 只允许 metadata-only")
        for c in chunks:
            if c.vector_doc_id in self.existing:
                meta = dict[str, object](self.existing[c.vector_doc_id].get("metadata", {}))  # type: ignore[arg-type]
                meta["content_hash"] = c.content_hash
                meta["metadata_hash"] = c.metadata_hash
                meta["source_hash"] = c.source.source_hash
                # 重新激活守卫：metadata-only 更新把 is_active 恢复为 Contract 的 is_active
                meta["is_active"] = c.is_active
                self.existing[c.vector_doc_id]["metadata"] = meta
        return VectorIngestionResult(
            results=[
                VectorItemResult(vector_doc_id=c.vector_doc_id, status="metadata_updated")
                for c in chunks
            ]
        )

    def disable_vectors(self, vector_doc_ids: Sequence[str]) -> VectorDeletionResult:
        self.disable_calls.append(list(vector_doc_ids))
        if self.raise_on_disable is not None:
            raise self.raise_on_disable
        deleted = 0
        for vid in vector_doc_ids:
            if vid in self.existing:
                meta = dict[str, object](self.existing[vid].get("metadata", {}))  # type: ignore[arg-type]
                meta["is_active"] = False
                self.existing[vid]["metadata"] = meta
                deleted += 1
        return VectorDeletionResult(
            requested_count=len(vector_doc_ids),
            deleted_count=deleted,
            not_found_count=len(vector_doc_ids) - deleted,
        )


class ReceiptSink:
    """接收字段并返回真实 ScopeHandoffReceipt。"""

    def make_receipt(self, **kwargs) -> ScopeHandoffReceipt:  # type: ignore[override]
        return ScopeHandoffReceipt(**kwargs)


def _build_consumer(
    store: FakeScopeVectorStore,
    *,
    scope_source_tables: dict[str, str] | None = None,
) -> tuple[ScopeConsumer, FakeScopeVectorStore]:
    from rag_vector_store.chunking.contract_pipeline import EmbeddingEligibilityValidator

    core = ContractConsumerCore(
        vector_store=store,
        eligibility_skip_of=EmbeddingEligibilityValidator().get_skip_reason,
    )
    consumer = ScopeConsumer(
        core=core,
        state_port=store,
        receipt_sink=ReceiptSink(),
        scope_source_tables=scope_source_tables,
    )
    return consumer, store


def _chunks_to_contracts(chunks) -> list[ExpertChunkEmbeddingInput]:
    """把 ChunkRecord 经 contract_bridge 转为 ExpertChunkEmbeddingInput。"""
    from expert_index.contract_bridge import chunk_records_to_contracts

    return chunk_records_to_contracts(
        chunks,
        updated_at=TASK_UPDATED,
        timezone_name="Asia/Shanghai",
    )


def _profile_chunks(eid=700001) -> list[ExpertChunkEmbeddingInput]:
    agg = _aggregate(eid=eid, educations=(_education(eid=eid),))
    chunks = _chunks(agg)
    profile = [c for c in chunks if c.chunk_type == "profile_summary"]
    return _chunks_to_contracts(profile)


def _employment_chunks(eid=700001, emp_id=21307) -> list[ExpertChunkEmbeddingInput]:
    agg = _aggregate(eid=eid, employments=(_employment(eid=eid, emp_id=emp_id),))
    chunks = _chunks(agg)
    employment = [c for c in chunks if c.chunk_type == "employment"]
    return _chunks_to_contracts(employment)


def _project_chunks(eid=700001, proj_id=161) -> list[ExpertChunkEmbeddingInput]:
    agg = _aggregate(eid=eid, projects=(_project(eid=eid, proj_id=proj_id),))
    chunks = _chunks(agg)
    project = [c for c in chunks if c.chunk_type == "project"]
    return _chunks_to_contracts(project)


def _all_chunks(eid=700001, *, emp_id=21307, proj_id=161) -> list[ExpertChunkEmbeddingInput]:
    agg = _aggregate(
        eid=eid,
        employments=(_employment(eid=eid, emp_id=emp_id),),
        projects=(_project(eid=eid, proj_id=proj_id),),
        educations=(_education(eid=eid),),
    )
    return _chunks_to_contracts(_chunks(agg))


# ---------------------------------------------------------------------------
# 1. 四类 exact scope 过滤
# ---------------------------------------------------------------------------
class TestExactScopeFilter:
    def test_expert_scope_disables_only_in_scope_active(self):
        """expert 范围下线该专家全部 active；其它专家不受影响。"""
        own = _all_chunks(eid=700001)
        other_expert_chunks = _all_chunks(eid=700002)
        store = FakeScopeVectorStore()
        for c in own + other_expert_chunks:
            store.seed(_existing_record(contract=c))
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        request = _make_request(scope, [])
        consumer, _ = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        disabled_ids = {vid for call in store.disable_calls for vid in call}
        # 只下线 700001 的全部 active，700002 全部保留
        assert all(vid.startswith("expert:700001:") for vid in disabled_ids)
        assert all(not vid.startswith("expert:700002:") for vid in disabled_ids)
        assert receipt.disable_count == len(own)

    def test_profile_scope_only_profile_summary(self):
        """profile 范围只下线 profile_summary，同专家 employment/project 保留。"""
        chunks = _all_chunks(eid=700001)
        store = FakeScopeVectorStore()
        for c in chunks:
            store.seed(_existing_record(contract=c))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [])
        consumer, _ = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        disabled = {vid for call in store.disable_calls for vid in call}
        assert all(":profile_summary:" in vid for vid in disabled)
        assert all(":employment:" not in vid for vid in disabled)
        assert all(":project:" not in vid for vid in disabled)

    def test_employment_scope_only_matching_source_field_id(self):
        """employment 范围只下线对应 source_field_id 的 family，其它任职 family 保留。"""
        chunks_a = _employment_chunks(eid=700001, emp_id=21307)
        chunks_b = _employment_chunks(eid=700001, emp_id=21308)
        store = FakeScopeVectorStore()
        for c in chunks_a + chunks_b:
            store.seed(_existing_record(contract=c))
        scope = impact_scope_from_target(
            target_type="employment", expert_id=700001, source_field_id="21307"
        )
        request = _make_request(scope, [])
        consumer, _ = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        disabled = {vid for call in store.disable_calls for vid in call}
        assert all(":21307:" in vid for vid in disabled)
        assert all(":21308:" not in vid for vid in disabled)

    def test_project_scope_only_matching_source_field_id(self):
        chunks_a = _project_chunks(eid=700001, proj_id=161)
        chunks_b = _project_chunks(eid=700001, proj_id=162)
        store = FakeScopeVectorStore()
        for c in chunks_a + chunks_b:
            store.seed(_existing_record(contract=c))
        scope = impact_scope_from_target(
            target_type="project", expert_id=700001, source_field_id="161"
        )
        request = _make_request(scope, [])
        consumer, _ = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        disabled = {vid for call in store.disable_calls for vid in call}
        assert all(":161:" in vid for vid in disabled)
        assert all(":162:" not in vid for vid in disabled)

    def test_inactive_records_not_treated_as_stale(self):
        """inactive 记录不进入 active set，不在空 desired 下被重复下线。"""
        chunks = _profile_chunks(eid=700001)
        store = FakeScopeVectorStore()
        for c in chunks:
            store.seed(_existing_record(contract=c, is_active=False))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [])
        consumer, _ = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        assert store.disable_calls == []  # active 为空，无 stale
        assert receipt.disable_count == 0


# ---------------------------------------------------------------------------
# 2. 三 Hash 三种更新路径
# ---------------------------------------------------------------------------
class TestThreeHashPaths:
    def test_content_hash_change_triggers_embedding(self):
        chunks = _profile_chunks(eid=700001)
        contract = chunks[0]
        store = FakeScopeVectorStore()
        # 已存 content_hash 不同 → embedding_upsert
        store.seed(_existing_record(contract=contract, content_hash="old-content-hash-different"))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])
        consumer, store = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        assert store.upsert_calls and len(store.upsert_calls[0]) == 1
        assert store.update_calls == []
        assert receipt.embedding_count == 1
        assert receipt.metadata_only_count == 0
        assert receipt.unchanged_count == 0

    def test_metadata_hash_change_metadata_only(self):
        chunks = _profile_chunks(eid=700001)
        contract = chunks[0]
        store = FakeScopeVectorStore()
        store.seed(_existing_record(contract=contract, metadata_hash="old-metadata-hash-different"))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])
        consumer, store = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        assert store.update_calls and not store.upsert_calls
        assert receipt.metadata_only_count == 1
        assert receipt.embedding_count == 0

    def test_three_hash_equal_active_unchanged(self):
        chunks = _profile_chunks(eid=700001)
        contract = chunks[0]
        store = FakeScopeVectorStore()
        store.seed(_existing_record(contract=contract))  # 三 Hash 全等 active
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])
        consumer, store = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        assert store.upsert_calls == [] and store.update_calls == []
        assert receipt.unchanged_count == 1


# ---------------------------------------------------------------------------
# 3. 空 desired
# ---------------------------------------------------------------------------
class TestEmptyDesired:
    def test_empty_desired_disables_all_active_no_embedding(self):
        chunks = _all_chunks(eid=700001)
        store = FakeScopeVectorStore()
        for c in chunks:
            store.seed(_existing_record(contract=c))
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        request = _make_request(scope, [])
        consumer, store = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        assert store.upsert_calls == []  # 零 Embedding
        assert store.update_calls == []
        disabled = [vid for call in store.disable_calls for vid in call]
        assert len(disabled) == len(chunks)
        assert receipt.disable_count == len(chunks)
        assert receipt.desired_count == 0


# ---------------------------------------------------------------------------
# 4. N→M stale 下线
# ---------------------------------------------------------------------------
class TestStaleDownsize:
    def test_n_to_m_stale_disable(self):
        """同 source_field_id 旧 N 片，desired 只给 M 片，stale=N−M 下线。"""
        chunks_v1 = _employment_chunks(eid=700001, emp_id=21307)
        assert len(chunks_v1) == 1
        base = chunks_v1[0]
        # 手构同 source_field_id 的第二片（chunk_index=1），作为历史 stale
        stale_vid = base.vector_doc_id.replace(":0", ":1")
        stale_contract = base.model_copy(
            update={"vector_doc_id": stale_vid, "chunk_index": 1, "chunk_total": 2}
        )
        stale_record = _existing_record(
            contract=stale_contract,
            content_hash="old-stale-content",
        )
        store = FakeScopeVectorStore()
        store.seed(_existing_record(contract=base, content_hash="old-base-content"))
        store.seed(stale_record)
        scope = impact_scope_from_target(
            target_type="employment", expert_id=700001, source_field_id="21307"
        )
        # desired 只给第 0 片
        request = _make_request(scope, [base])
        consumer, store = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        disabled = [vid for call in store.disable_calls for vid in call]
        assert base.vector_doc_id not in disabled
        assert stale_vid in disabled
        assert receipt.disable_count == 1
        assert receipt.stale_vector_doc_ids == (stale_vid,)
        assert [item.vector_doc_id for item in receipt.desired_results] == [
            base.vector_doc_id
        ]


class TestResumeOwnerIsolation:
    def test_resume_metadata_update_and_stale_disable_use_resume_id(self):
        resume_id = 700001
        scope = impact_scope_from_target(
            target_type="profile",
            expert_id=resume_id,
            dataset_spec=RESUME_SPEC,
        )
        expert_contract = _profile_chunks(eid=resume_id)[0]
        desired = expert_contract.model_copy(
            update={
                "vector_doc_id": (
                    f"resume:{resume_id}:profile_summary:{resume_id}:0"
                ),
                "metadata_hash": "d" * 64,
                "source": expert_contract.source.model_copy(
                    update={
                        "source_table": scope.source_table,
                        "source_field_id": scope.source_field_id,
                    }
                ),
            }
        )
        stale = desired.model_copy(
            update={
                "vector_doc_id": (
                    f"resume:{resume_id}:profile_summary:{resume_id}:1"
                ),
            }
        )
        store = FakeScopeVectorStore(owner_field="resume_id")
        store.seed(
            _existing_record(
                contract=desired,
                metadata_hash="c" * 64,
                owner_field="resume_id",
            )
        )
        store.seed(
            _existing_record(
                contract=stale,
                owner_field="resume_id",
            )
        )
        consumer, _ = _build_consumer(
            store,
            scope_source_tables={
                "profile": RESUME_SPEC.profile_source_label,
                "employment": RESUME_SPEC.employment_table,
                "project": RESUME_SPEC.project_table,
            },
        )

        receipt = consumer.process(_make_request(scope, [desired]))

        assert receipt.status == "committed"
        assert receipt.metadata_only_count == 1
        assert receipt.disable_count == 1
        assert store.update_calls == [[desired]]
        assert store.disable_calls == [[stale.vector_doc_id]]
        desired_metadata = store.existing[desired.vector_doc_id]["metadata"]
        assert desired_metadata["resume_id"] == resume_id
        assert "expert_id" not in desired_metadata


# ---------------------------------------------------------------------------
# 5. inactive 重新激活（交由 ContractConsumerCore）
# ---------------------------------------------------------------------------
class TestReactivation:
    def test_inactive_same_id_desired_reactivates_via_core(self):
        chunks = _profile_chunks(eid=700001)
        contract = chunks[0]
        store = FakeScopeVectorStore()
        # 已存三 Hash 全等但 inactive → 核心走 metadata-only 重新激活
        store.seed(_existing_record(contract=contract, is_active=False))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])
        consumer, store = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        assert store.update_calls and not store.upsert_calls
        assert receipt.metadata_only_count == 1
        # scope 内无其它 active，无 stale 下线
        assert store.disable_calls == []


# ---------------------------------------------------------------------------
# 6. desired rejected/failed 时不下线 stale（详见 TestDesiredRejectedNoStale）
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. disable 失败返回 retryable failed
# ---------------------------------------------------------------------------
class TestStaleDisableFailure:
    def test_stale_disable_failure_returns_retryable_failed(self):
        chunks = _all_chunks(eid=700001)
        store = FakeScopeVectorStore()
        for c in chunks:
            store.seed(_existing_record(contract=c))
        store.raise_on_disable = VectorIngestionError("milvus down")
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        request = _make_request(scope, [])
        consumer, store = _build_consumer(store)
        store.raise_on_disable = VectorIngestionError("milvus down")
        receipt = consumer.process(request)
        assert receipt.status == "failed"
        assert receipt.retryable is True
        assert receipt.error_type == "scope_stale_disable_failed"

    def test_stale_failure_retains_successful_desired_disable_count(self):
        """stale 下线失败时 disable_count 只保留已成功的 desired disable 数。"""
        contract = _profile_chunks(eid=700001)[0]
        inactive_desired = contract.model_copy(update={"is_active": False})
        extra = _employment_chunks(eid=700001, emp_id=21307)[0]
        store = FakeScopeVectorStore()
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        store.seed(_existing_record(contract=extra))  # scope stale
        # 让 disable 第二次调用（stale）失败，但第一次（desired ineligible）成功
        call_count = {"n": 0}
        original_disable = store.disable_vectors

        def disable_then_fail(vids):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return original_disable(vids)
            raise VectorIngestionError("milvus down")
        store.disable_vectors = disable_then_fail  # type: ignore[assignment]
        request = _make_request(scope, [inactive_desired])
        consumer, store = _build_consumer(store)
        store.disable_vectors = disable_then_fail  # type: ignore[assignment]
        receipt = consumer.process(request)
        assert receipt.status == "failed"  # stale 失败
        assert receipt.error_type == "scope_stale_disable_failed"
        assert receipt.disable_count == 1  # 只保留 desired ineligible disable 成功数


# ---------------------------------------------------------------------------
# 8. 重放幂等
# ---------------------------------------------------------------------------
class TestReplayIdempotent:
    def test_replay_after_success_no_duplicate_embedding(self):
        chunks = _profile_chunks(eid=700001)
        contract = chunks[0]
        store = FakeScopeVectorStore()
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])
        consumer, _ = _build_consumer(store)
        receipt1 = consumer.process(request)
        assert receipt1.status == "committed"
        assert receipt1.embedding_count == 1
        upsert_count_first = len(store.upsert_calls)
        # 重放：已存三 Hash 全等 active → unchanged
        receipt2 = consumer.process(request)
        assert receipt2.status == "committed"
        assert receipt2.unchanged_count == 1
        assert receipt2.embedding_count == 0
        assert len(store.upsert_calls) == upsert_count_first  # 无重复 embedding


# ---------------------------------------------------------------------------
# 9. Receipt 计数与 identity
# ---------------------------------------------------------------------------
class TestReceiptIdentityAndCounts:
    def test_receipt_identity_matches_request(self):
        chunks = _profile_chunks(eid=700001)
        contract = chunks[0]
        store = FakeScopeVectorStore()
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])
        consumer, _ = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.request_id == request.request_id
        assert receipt.task_id == request.task_id
        assert receipt.batch_id == request.batch_id
        assert receipt.attempt_token == request.attempt_token
        assert receipt.payload_hash == request.payload_hash
        assert receipt.impact_scope is request.impact_scope

    def test_committed_counts(self):
        chunks = _all_chunks(eid=700001)
        store = FakeScopeVectorStore()
        # 全部三 Hash 不等 → embedding
        for c in chunks:
            store.seed(_existing_record(contract=c, content_hash="old"))
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        request = _make_request(scope, chunks)
        consumer, _ = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        assert receipt.desired_count == len(chunks)
        assert receipt.embedding_count == len(chunks)
        assert receipt.metadata_only_count == 0
        assert receipt.unchanged_count == 0
        assert receipt.disable_count == 0  # 无 stale


# ---------------------------------------------------------------------------
# 11. Eligibility disable desired → committed（advisor 要求）
# ---------------------------------------------------------------------------
class TestEligibilityDisableDesired:
    def test_eligibility_inactive_desired_committed_and_disabled(self):
        """desired is_active=false → Eligibility skip → 核心逻辑下线目标集合，
        scope Receipt committed 且 disable_count 含该 desired。不下线 scope stale。"""
        contract = _profile_chunks(eid=700001)[0]
        # 构造 is_active=False 的 desired（仍合法 Contract）
        inactive_desired = contract.model_copy(update={"is_active": False})
        extra = _employment_chunks(eid=700001, emp_id=21307)[0]
        store = FakeScopeVectorStore()
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        store.seed(_existing_record(contract=extra))  # scope 内 stale
        request = _make_request(scope, [inactive_desired])
        consumer, _ = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        # disable_count 含 ineligible desired（1）+ scope stale（1）
        assert receipt.disable_count == 2
        disabled = [vid for call in store.disable_calls for vid in call]
        assert inactive_desired.vector_doc_id in disabled
        assert extra.vector_doc_id in disabled


# ---------------------------------------------------------------------------
# 12. desired rejected/failed 不下线 stale（advisor 要求）
# ---------------------------------------------------------------------------
class TestDesiredRejectedNoStale:
    def test_desired_contract_rejected_does_not_disable_stale(self):
        """desired Contract 确定性错误（VectorContractError）→ rejected，不下线 stale。"""
        from rag_vector_store.ingestion.errors import VectorContractError

        contract = _profile_chunks(eid=700001)[0]
        extra = _employment_chunks(eid=700001, emp_id=21307)[0]
        store = FakeScopeVectorStore()
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        # desired：content_hash 变化触发 embedding，但让 upsert 抛 VectorContractError
        store.seed(_existing_record(contract=contract, content_hash="old"))
        store.seed(_existing_record(contract=extra))  # scope 内 stale
        def rejecting_upsert(chunks):
            raise VectorContractError("contract invalid")
        store.upsert_chunks = rejecting_upsert  # type: ignore[assignment]
        request = _make_request(scope, [contract])
        consumer, store = _build_consumer(store)
        store.upsert_chunks = rejecting_upsert  # type: ignore[assignment]
        receipt = consumer.process(request)
        assert receipt.status == "rejected"
        assert store.disable_calls == []  # 未下线 stale

    def test_desired_failed_does_not_disable_stale(self):
        """desired infra 失败（VectorIngestionError）→ failed，不下线 stale。"""
        contract = _profile_chunks(eid=700001)[0]
        extra = _employment_chunks(eid=700001, emp_id=21307)[0]
        store = FakeScopeVectorStore()
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        store.seed(_existing_record(contract=contract, content_hash="old"))
        store.seed(_existing_record(contract=extra))
        def failing_upsert(chunks):
            raise VectorIngestionError("infra down")
        store.upsert_chunks = failing_upsert  # type: ignore[assignment]
        request = _make_request(scope, [contract])
        consumer, store = _build_consumer(store)
        store.upsert_chunks = failing_upsert  # type: ignore[assignment]
        receipt = consumer.process(request)
        assert receipt.status == "failed"
        assert receipt.retryable is True
        assert store.disable_calls == []

    def test_rejected_and_failed_same_batch_priority_rejected(self):
        """同批 desired 一条 rejected（VectorContractError）一条 failed（VectorIngestionError）
        → 整体 rejected 且不下线 stale。"""
        from rag_vector_store.ingestion.errors import VectorContractError

        c1 = _profile_chunks(eid=700001)[0]
        c2 = _employment_chunks(eid=700001, emp_id=21307)[0]
        store = FakeScopeVectorStore()
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        store.seed(_existing_record(contract=c1, content_hash="old"))  # c1 embedding → reject
        store.seed(_existing_record(contract=c2, content_hash="old"))  # c2 embedding → fail
        store.seed(_existing_record(contract=_project_chunks(eid=700001, proj_id=161)[0]))  # stale
        def routing_upsert(chunks):
            for c in chunks:
                if c.vector_doc_id == c1.vector_doc_id:
                    raise VectorContractError("contract invalid")
            raise VectorIngestionError("infra down")
        store.upsert_chunks = routing_upsert  # type: ignore[assignment]
        request = _make_request(scope, [c1, c2])
        consumer, store = _build_consumer(store)
        store.upsert_chunks = routing_upsert  # type: ignore[assignment]
        receipt = consumer.process(request)
        assert receipt.status == "rejected"  # rejected 优先于 failed
        assert store.disable_calls == []  # 不下线 stale


# ---------------------------------------------------------------------------
# 12. source_hash-only 变化 → metadata-only（advisor 要求）
# ---------------------------------------------------------------------------
class TestSourceHashOnlyMetadataOnly:
    def test_source_hash_only_change_metadata_only(self):
        contract = _profile_chunks(eid=700001)[0]
        store = FakeScopeVectorStore()
        # 已存 source_hash 不同，content/metadata_hash 相同 → metadata-only
        store.seed(_existing_record(contract=contract, source_hash="old-source-hash-different"))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])
        consumer, store = _build_consumer(store)
        receipt = consumer.process(request)
        assert receipt.status == "committed"
        assert store.update_calls and not store.upsert_calls
        assert receipt.metadata_only_count == 1
        assert receipt.embedding_count == 0


# ---------------------------------------------------------------------------
# 13. 重新激活两次重放（advisor 要求）
# ---------------------------------------------------------------------------
class TestReactivationReplay:
    def test_reactivation_then_replay_unchanged(self):
        """已存 inactive + 三 Hash 全等 → 第一次 metadata-only 恢复 active；
        第二次重放 → unchanged（不重复 update/embedding）。"""
        contract = _profile_chunks(eid=700001)[0]
        store = FakeScopeVectorStore()
        store.seed(_existing_record(contract=contract, is_active=False))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])
        consumer, _ = _build_consumer(store)
        receipt1 = consumer.process(request)
        assert receipt1.status == "committed"
        assert receipt1.metadata_only_count == 1
        update_count_1 = len(store.update_calls)
        # 第二次重放：is_active 已恢复 True，三 Hash 全等 → unchanged
        receipt2 = consumer.process(request)
        assert receipt2.status == "committed"
        assert receipt2.unchanged_count == 1
        assert len(store.update_calls) == update_count_1  # 无重复 update
        assert store.upsert_calls == []


# ---------------------------------------------------------------------------
# 14. get_vectors missing fail-closed（advisor 要求）
# ---------------------------------------------------------------------------
class TestMissingFailClosed:
    def test_get_vectors_missing_returns_retryable_failed(self):
        """listing 声明存在的 ID 但 get_vectors 未命中 → failed/retryable，不写 desired/stale。"""
        contract = _profile_chunks(eid=700001)[0]
        store = FakeScopeVectorStore()
        store.seed(_existing_record(contract=contract))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        # 伪造 missing：让 get_vectors 返回空 records + listing 的 vid 为 missing
        def get_with_missing(vids):
            from rag_vector_store.ingestion.models import VectorQueryResult
            return VectorQueryResult(records=[], missing=list(vids))
        store.get_vectors = get_with_missing  # type: ignore[assignment]
        request = _make_request(scope, [contract])
        consumer, store = _build_consumer(store)
        store.get_vectors = get_with_missing  # type: ignore[assignment]
        receipt = consumer.process(request)
        assert receipt.status == "failed"
        assert receipt.retryable is True
        assert store.upsert_calls == []
        assert store.disable_calls == []


class TestScopeQueryResultAlignment:
    def test_owner_listing_mismatch_returns_retryable_failed(self):
        contract = _profile_chunks(eid=700001)[0]
        store = FakeScopeVectorStore()
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])

        def wrong_owner(_owner_id: int) -> VectorOwnerList:
            return VectorOwnerList.model_construct(
                owner_id=700002,
                vector_doc_ids=[],
                count=0,
            )

        store.list_vectors_by_owner = wrong_owner  # type: ignore[assignment]
        consumer, store = _build_consumer(store)
        store.list_vectors_by_owner = wrong_owner  # type: ignore[assignment]

        receipt = consumer.process(request)

        assert receipt.status == "failed"
        assert receipt.retryable is True
        assert receipt.error_type == "scope_query_failed"
        assert store.upsert_calls == []
        assert store.disable_calls == []

    @pytest.mark.parametrize("mode", ["extra", "missing"])
    def test_query_id_set_mismatch_returns_retryable_failed(self, mode: str):
        contract = _profile_chunks(eid=700001)[0]
        store = FakeScopeVectorStore()
        store.seed(_existing_record(contract=contract))
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        request = _make_request(scope, [contract])

        def malformed_query(vector_doc_ids):
            if mode == "extra":
                extra = dict(store.existing[contract.vector_doc_id])
                extra["vector_doc_id"] = "expert:700001:profile_summary:extra:0"
                return VectorQueryResult.model_construct(
                    records=[store.existing[contract.vector_doc_id], extra],
                    missing=[],
                )
            return VectorQueryResult.model_construct(records=[], missing=[])

        store.get_vectors = malformed_query  # type: ignore[assignment]
        consumer, store = _build_consumer(store)
        store.get_vectors = malformed_query  # type: ignore[assignment]

        receipt = consumer.process(request)

        assert receipt.status == "failed"
        assert receipt.retryable is True
        assert receipt.error_type == "scope_query_failed"
        assert store.upsert_calls == []
        assert store.disable_calls == []


# ---------------------------------------------------------------------------
# 10. 结构断言：无 expert_index 依赖、无物理删除、无状态写入
# ---------------------------------------------------------------------------
class TestStructuralInvariants:
    def test_no_expert_index_import(self):
        path = Path(__file__).resolve().parents[1] / "src/rag_vector_store/incremental/scope_consumer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("expert_index"):
                raise AssertionError(f"scope_consumer.py 不得 import expert_index：{node.module}")
            if isinstance(node, ast.Import) and any(
                alias.name.startswith("expert_index") for alias in node.names
            ):
                raise AssertionError("scope_consumer.py 不得 import expert_index")

    def test_no_physical_delete_or_state_tokens(self):
        path = Path(__file__).resolve().parents[1] / "src/rag_vector_store/incremental/scope_consumer.py"
        text = path.read_text(encoding="utf-8")
        forbidden = [
            "delete_vectors",
            "delete_by_vector_doc_ids",
            "V15WorkItem",
            "v15_vectorization",
            "checkpoint",
            "StateStore",
            "Artifact",
            "pymysql",
            "MySQL",
        ]
        for token in forbidden:
            assert token not in text, f"scope_consumer.py 禁止出现 {token!r}"
