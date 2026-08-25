"""P5.1 正式内存交付链上游 Handoff/Receipt 契约与构建器测试。

复用真实 ``normalize_aggregate`` + ``ChunkFactory.build_all`` + ``contract_bridge``
生成三类真实 ChunkRecord，端到端验证 Handoff 构建与 Receipt 不变式。全部 Fake，不连接
MySQL/Milvus，不调用 Embedding/Consumer。
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import pytest

from expert_contracts import ExpertChunkEmbeddingInput
from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.contract_bridge import chunk_records_to_contracts
from expert_index.handoff import (
    HandoffReceiptStatus,
    ScopeHandoffBuilder,
    ScopeHandoffReceipt,
    ScopeHandoffRequest,
)
from expert_index.models import (
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    PreparedChunkSet,
    ProjectSource,
    SourceSyncTask,
    impact_scope_from_target,
)
from expert_index.normalize import normalize_aggregate

CREATED = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
TASK_UPDATED = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
LEASE_UNTIL = datetime(2026, 8, 21, 9, 30, tzinfo=timezone.utc)
NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
TOKEN = "tok-claim-001"


def _settings() -> Settings:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _factory() -> ChunkFactory:
    return ChunkFactory(_settings())


def _expert_source(eid: int = 700001, deleted_at=None) -> ExpertSource:
    return ExpertSource(
        id=eid,
        serial_number=f"TEST{eid}",
        company="虚构交付测试公司",
        job_status=1,
        job_name="项目经理",
        contact_status=3,
        doubtful_flag=1,
        abroad_flag=2,
        background="仅用于 Handoff 构建单元测试的虚构背景。",
        work_years=8.5,
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _employment(eid: int = 700001, emp_id: int = 21307, deleted_at=None) -> EmploymentSource:
    return EmploymentSource(
        id=emp_id,
        expert_id=eid,
        company_name="虚构交付测试公司",
        position_title="高级工程师",
        start_date=UPDATED,
        end_date=None,
        work_description="负责交付链路单元测试。",
        is_employed=1,
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _project(eid: int = 700001, proj_id: int = 161, deleted_at=None) -> ProjectSource:
    return ProjectSource(
        id=proj_id,
        expert_id=eid,
        project_title="交付测试项目",
        company_name="虚构交付测试公司",
        start_date=UPDATED,
        end_date=None,
        work_description="设计并实现 Handoff。",
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=deleted_at,
    )


def _education(eid: int = 700001, edu_id: int = 1) -> EducationSource:
    return EducationSource(
        id=edu_id,
        expert_id=eid,
        school_name="测试大学",
        major="计算机",
        updated_at=UPDATED,
        created_at=CREATED,
        deleted_at=None,
    )


def _aggregate(eid=700001, *, employments=(), projects=(), educations=(), deleted_at=None) -> ExpertAggregate:
    return ExpertAggregate(
        profile=_expert_source(eid, deleted_at=deleted_at),
        employments=tuple(employments),
        projects=tuple(projects),
        educations=tuple(educations),
    )


def _task(
    *,
    task_id=701,
    batch_id=8,
    expert_id=700001,
    source_table="experts",
    source_record_id=700001,
    operation_type="update",
    target_type="expert",
    policy_version="1.3",
    sync_status="processing",
    attempt_token=TOKEN,
    lease_until=LEASE_UNTIL,
) -> SourceSyncTask:
    import hashlib
    dedupe = hashlib.sha256(f"dk-{task_id}".encode()).hexdigest()
    return SourceSyncTask(
        task_id=task_id,
        dedupe_key=dedupe,
        batch_id=batch_id,
        event_source="poll",
        source_table=source_table,
        source_record_id=source_record_id,
        source_updated_at=UPDATED,
        expert_id=expert_id,
        operation_type=operation_type,
        target_type=target_type,
        processing_stage="validated",
        sync_status=sync_status,
        retry_count=0,
        policy_version=policy_version,
        created_at=CREATED,
        updated_at=TASK_UPDATED,
        attempt_token=attempt_token,
        lease_until=lease_until,
    )


def _prepared(scope, chunks, *, task_id=701, batch_id=8, data_version="8") -> PreparedChunkSet:
    return PreparedChunkSet(
        task_id=task_id,
        batch_id=batch_id,
        data_version=data_version,
        impact_scope=scope,
        desired_chunks=tuple(chunks),
    )


def _chunks_from_aggregate(agg: ExpertAggregate):
    profile, employments, projects = normalize_aggregate(
        agg, data_version="20260824_handoff", build_batch_id=701)
    return _factory().build_all(profile, employments, projects)


# ---------------------------------------------------------------------------
# 构建成功：三种 chunk_type 转换 + 三 Hash/chunk_type/source_field_id 保留
# ---------------------------------------------------------------------------


class TestThreeChunkTypeConversion:
    def test_profile_summary_handoff_preserves_hash_and_fields(self):
        agg = _aggregate(educations=(_education(),))
        chunks = _chunks_from_aggregate(agg)
        profile_chunk = [c for c in chunks if c.chunk_type == "profile_summary"]
        assert len(profile_chunk) == 1
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        prepared = _prepared(scope, profile_chunk)
        task = _task(
            target_type="profile", source_table="experts",
            source_record_id=700001, expert_id=700001,
        )
        req = ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)
        assert isinstance(req, ScopeHandoffRequest)
        assert len(req.contracts) == 1
        c = req.contracts[0]
        assert isinstance(c, ExpertChunkEmbeddingInput)
        assert c.chunk_type == "profile_summary"
        # 三 Hash 保留
        assert c.content_hash == profile_chunk[0].content_hash
        assert c.metadata_hash == profile_chunk[0].metadata_hash
        assert c.source.source_hash == profile_chunk[0].source_hash
        # source_field_id 保留
        assert c.source.source_field_id == profile_chunk[0].source_field_id == "700001"
        # 全部 Contract 共享同一 updated_at（取自 task.updated_at）
        assert c.updated_at.startswith("2026-08-21T09:00:00")
        assert req.payload_hash and len(req.payload_hash) == 64

    def test_employment_handoff_preserves_hash_and_source_field_id(self):
        agg = _aggregate(employments=(_employment(),))
        chunks = _chunks_from_aggregate(agg)
        emp_chunk = [c for c in chunks if c.chunk_type == "employment"]
        assert len(emp_chunk) == 1
        scope = impact_scope_from_target(
            target_type="employment", expert_id=700001, source_field_id="21307")
        prepared = _prepared(scope, emp_chunk)
        task = _task(
            target_type="employment", source_table="experts_work_experience",
            source_record_id=21307, expert_id=700001,
        )
        req = ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)
        c = req.contracts[0]
        assert c.chunk_type == "employment"
        assert c.content_hash == emp_chunk[0].content_hash
        assert c.source.source_field_id == "21307"
        assert c.source.source_table == "experts_work_experience"

    def test_project_handoff_preserves_hash_and_source_field_id(self):
        agg = _aggregate(projects=(_project(),))
        chunks = _chunks_from_aggregate(agg)
        proj_chunk = [c for c in chunks if c.chunk_type == "project"]
        assert len(proj_chunk) == 1
        scope = impact_scope_from_target(
            target_type="project", expert_id=700001, source_field_id="161")
        prepared = _prepared(scope, proj_chunk)
        task = _task(
            target_type="project", source_table="experts_project_experience",
            source_record_id=161, expert_id=700001,
        )
        req = ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)
        c = req.contracts[0]
        assert c.chunk_type == "project"
        assert c.source.source_field_id == "161"

    def test_expert_scope_carries_all_three_types(self):
        agg = _aggregate(
            employments=(_employment(),), projects=(_project(),), educations=(_education(),))
        chunks = _chunks_from_aggregate(agg)
        assert {c.chunk_type for c in chunks} == {"profile_summary", "employment", "project"}
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, chunks)
        task = _task(target_type="expert", source_table="experts", source_record_id=700001)
        req = ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)
        assert {c.chunk_type for c in req.contracts} == {
            "profile_summary", "employment", "project"}
        # vector_doc_id 唯一
        assert len({c.vector_doc_id for c in req.contracts}) == len(req.contracts)


# ---------------------------------------------------------------------------
# scope 与 Contract 归属校验 / 非法范围 fail-closed
# ---------------------------------------------------------------------------


class TestScopeOwnershipValidation:
    def test_cross_expert_contract_rejected_in_request(self):
        agg = _aggregate(eid=700001)
        chunks = _chunks_from_aggregate(agg)
        # PreparedChunkSet 自身会拒绝 cross-expert，但直接构造 Request 也要拒绝
        contracts = chunk_records_to_contracts(
            chunks, updated_at=TASK_UPDATED, timezone_name="Asia/Shanghai")
        bad_scope = impact_scope_from_target(target_type="expert", expert_id=700002)
        with pytest.raises(ValueError, match="不属于范围"):
            ScopeHandoffRequest(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                impact_scope=bad_scope, contracts=tuple(contracts),
                payload_hash="x" * 64,
            )

    def test_non_expert_scope_rejects_other_chunk_type(self):
        agg = _aggregate(educations=(_education(),), employments=(_employment(),))
        chunks = _chunks_from_aggregate(agg)
        profile_chunk = next(c for c in chunks if c.chunk_type == "profile_summary")
        # employment scope 携带 profile_summary Chunk
        scope = impact_scope_from_target(
            target_type="employment", expert_id=700001, source_field_id="21307")
        contracts = chunk_records_to_contracts(
            (profile_chunk,), updated_at=TASK_UPDATED, timezone_name="Asia/Shanghai")
        with pytest.raises(ValueError, match="employment 范围不允许 chunk_type"):
            ScopeHandoffRequest(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                impact_scope=scope, contracts=tuple(contracts), payload_hash="x" * 64,
            )

    def test_duplicate_vector_doc_id_rejected(self):
        agg = _aggregate(educations=(_education(),))
        chunks = _chunks_from_aggregate(agg)
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        contracts = chunk_records_to_contracts(
            chunks, updated_at=TASK_UPDATED, timezone_name="Asia/Shanghai")
        dup = tuple(contracts) + tuple(contracts)  # 重复同一 Contract
        with pytest.raises(ValueError, match="vector_doc_id=.* 重复"):
            ScopeHandoffRequest(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                impact_scope=scope, contracts=dup, payload_hash="x" * 64,
            )


# ---------------------------------------------------------------------------
# 空 desired 合法
# ---------------------------------------------------------------------------


class TestEmptyDesired:
    def test_empty_desired_produces_valid_handoff(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [])  # 空 desired
        task = _task(target_type="expert", source_table="experts", source_record_id=700001)
        req = ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)
        assert req.contracts == ()
        assert req.impact_scope.scope_type == "expert"
        # 空 desired 仍有合法 payload_hash
        assert len(req.payload_hash) == 64
        # 重新构造得到一致 hash
        req2 = ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)
        assert req.payload_hash == req2.payload_hash


# ---------------------------------------------------------------------------
# payload_hash 确定性及变化检测
# ---------------------------------------------------------------------------


class TestPayloadHash:
    def _build_req(self, task=None, prepared=None, token=TOKEN):
        task = task or _task()
        prepared = prepared or _prepared(
            impact_scope_from_target(target_type="expert", expert_id=700001), [])
        return ScopeHandoffBuilder("Asia/Shanghai").build(task, token, prepared)

    def test_same_input_deterministic_replay(self):
        r1 = self._build_req()
        r2 = self._build_req()
        assert r1.payload_hash == r2.payload_hash
        assert r1.request_id == r2.request_id

    def test_contract_order_change_alters_hash(self):
        agg = _aggregate(
            employments=(_employment(),), projects=(_project(),), educations=(_education(),))
        chunks = _chunks_from_aggregate(agg)
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared_a = _prepared(scope, chunks)
        # PreparedChunkSet 内部会重新 stable sort，所以两个 prepared 实际相同顺序
        # 为真实测试顺序敏感性，直接在 Request 层构造不同顺序的 contracts
        contracts_a = chunk_records_to_contracts(
            prepared_a.desired_chunks, updated_at=TASK_UPDATED, timezone_name="Asia/Shanghai")
        contracts_b = tuple(reversed(contracts_a))
        ra = ScopeHandoffRequest(
            request_id="handoff:a", task_id=701, batch_id=8, attempt_token=TOKEN,
            impact_scope=scope, contracts=tuple(contracts_a),
            payload_hash=_manual_hash("handoff:a", scope, contracts_a),
        )
        rb = ScopeHandoffRequest(
            request_id="handoff:a", task_id=701, batch_id=8, attempt_token=TOKEN,
            impact_scope=scope, contracts=contracts_b,
            payload_hash=_manual_hash("handoff:a", scope, contracts_b),
        )
        assert ra.payload_hash != rb.payload_hash

    def test_token_change_does_not_alter_payload_hash(self):
        # 同任务仅 attempt_token 变化：payload_hash 必须相同（attempt_token 不参与 hash）
        r1 = self._build_req()
        task2 = _task(attempt_token="tok-different")
        r2 = self._build_req(task=task2, token="tok-different")
        assert r1.payload_hash == r2.payload_hash
        # request_id 含 attempt_token，所以 request_id 仍变
        assert r1.request_id != r2.request_id
        # attempt_token 在 Request 中保留用于 fencing
        assert r2.attempt_token == "tok-different"

    def test_scope_change_alters_hash(self):
        r1 = self._build_req()
        scope2 = impact_scope_from_target(target_type="profile", expert_id=700001)
        prepared2 = _prepared(scope2, [])
        r2 = self._build_req(prepared=prepared2)
        assert r1.payload_hash != r2.payload_hash

    def test_contract_payload_change_alters_hash(self):
        chunks1 = _chunks_from_aggregate(_aggregate(eid=700001, educations=(_education(),)))
        chunks2 = _chunks_from_aggregate(_aggregate(eid=700002, educations=(_education(eid=700002),)))
        scope1 = impact_scope_from_target(target_type="expert", expert_id=700001)
        scope2 = impact_scope_from_target(target_type="expert", expert_id=700002)
        task1 = _task(expert_id=700001)
        task2 = _task(expert_id=700002)
        r1 = self._build_req(task1, _prepared(scope1, chunks1))
        r2 = self._build_req(task2, _prepared(scope2, chunks2))
        assert r1.payload_hash != r2.payload_hash

    def test_payload_hash_matches_manual_canonical(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [])
        req = self._build_req(prepared=prepared)
        manual = _manual_hash(req.request_id, scope, [])
        assert req.payload_hash == manual


def _manual_hash(request_id, scope, contracts):
    # attempt_token/request_id 均不参与 payload_hash（仅用于 fencing）
    payload = {
        "task_id": 701,
        "batch_id": 8,
        "impact_scope": {
            "scope_type": scope.scope_type,
            "expert_id": scope.expert_id,
            "chunk_type": scope.chunk_type,
            "source_table": scope.source_table,
            "source_field_id": scope.source_field_id,
        },
        "contracts": [c.model_dump(mode="json") for c in contracts],
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# token/task identity 保留
# ---------------------------------------------------------------------------


class TestIdentityPreservation:
    def test_request_carries_task_and_token_identity(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [], task_id=909, batch_id=77, data_version="77")
        task = _task(task_id=909, batch_id=77, attempt_token="tok-xyz")
        req = ScopeHandoffBuilder("Asia/Shanghai").build(task, "tok-xyz", prepared)
        assert req.task_id == 909
        assert req.batch_id == 77
        assert req.attempt_token == "tok-xyz"
        assert req.request_id.startswith("handoff:")
        # request_id 是任务/批次/token 身份的摘要
        expected = hashlib.sha256(b"909:77:tok-xyz").hexdigest()
        assert req.request_id == f"handoff:{expected}"

    def test_non_processing_task_rejected(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [])
        task = _task(sync_status="pending", attempt_token=None, lease_until=None)
        with pytest.raises(ValueError, match="只有 processing"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)

    def test_token_mismatch_rejected(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [])
        task = _task()
        with pytest.raises(ValueError, match="attempt_token 与任务认领 token 不一致"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, "wrong-token", prepared)

    def test_task_id_mismatch_rejected(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [], task_id=999)
        task = _task(task_id=701)
        with pytest.raises(ValueError, match="prepared.task_id"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)

    def test_batch_id_mismatch_rejected(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [], batch_id=5, data_version="5")
        task = _task(batch_id=8)
        with pytest.raises(ValueError, match="prepared.batch_id"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)

    def test_data_version_not_str_batch_id_rejected(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [], data_version="wrong")
        task = _task()
        with pytest.raises(ValueError, match="data_version"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)

    def test_expert_id_mismatch_with_task_rejected(self):
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [])
        task = _task(expert_id=700002)
        with pytest.raises(ValueError, match="task.expert_id"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)

    def test_task_expert_id_none_rejected(self):
        # P5.1 要求 task.expert_id 非空
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [])
        task = _task(expert_id=None)
        with pytest.raises(ValueError, match="task.expert_id 非空"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)

    def test_task_policy_version_none_rejected_even_empty_desired(self):
        # P5.1 要求 task.policy_version 为非空字符串，空 desired 也拒绝
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [])
        task = _task(policy_version=None)
        with pytest.raises(ValueError, match="task.policy_version"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)

    def test_contract_policy_mismatch_with_task_rejected(self):
        # Chunk policy_version 与 task.policy_version 不一致 fail-closed
        agg = _aggregate(eid=700001, educations=(_education(),))
        chunks = _chunks_from_aggregate(agg)
        profile_chunk = next(c for c in chunks if c.chunk_type == "profile_summary")
        scope = impact_scope_from_target(target_type="profile", expert_id=700001)
        prepared = _prepared(scope, [profile_chunk])
        task = _task(
            target_type="profile", source_table="experts",
            source_record_id=700001, expert_id=700001,
            policy_version="9.9",  # 与 Chunk 的 1.3 不一致
        )
        with pytest.raises(ValueError, match="policy_version 不一致"):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)


# ---------------------------------------------------------------------------
# 子 scope 父删除升级为 expert 空 desired
# ---------------------------------------------------------------------------


class TestChildScopeParentDeletion:
    def test_employment_task_with_deleted_parent_escalates_to_expert_empty(self):
        # 父专家删除：P4 Handler 升级为 expert 空 desired（删除专家无 Chunk）
        # 这里直接用 expert 空 desired，模拟 P4 输出
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        prepared = _prepared(scope, [])
        # 任务 target_type 仍是 employment（原始发现），但 prepared scope 是 expert
        task = _task(
            target_type="employment", source_table="experts_work_experience",
            source_record_id=21307, expert_id=700001)
        req = ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)
        # 不要求 task target scope == prepared scope；expert 空 desired 合法
        assert req.impact_scope.scope_type == "expert"
        assert req.contracts == ()
        assert len(req.payload_hash) == 64



# ---------------------------------------------------------------------------
# Bridge/Contract 校验传播
# ---------------------------------------------------------------------------


class TestBridgeValidationPropagation:
    def test_invalid_chunk_propagates_chunk_validation_error(self):
        from expert_index.validators import ChunkValidationError
        # 构造一个 content_hash 不匹配的非法 ChunkRecord
        agg = _aggregate(educations=(_education(),))
        chunks = _chunks_from_aggregate(agg)
        bad_chunk = chunks[0]
        # 用 dataclasses.replace 篡改 content_hash 制造非法 Chunk
        from dataclasses import replace
        bad = replace(bad_chunk, content_hash="0" * 64)
        scope = impact_scope_from_target(
            target_type=bundle_scope_type(bad_chunk.chunk_type), expert_id=700001,
            source_field_id=bad_chunk.source_field_id if bad_chunk.chunk_type != "profile_summary" else None)
        prepared = _prepared(scope, [bad])
        # profile 任务的 source_table 是物理表 experts（非聚合标识）
        _task_table = {"profile_summary": "experts", "employment": "experts_work_experience",
                       "project": "experts_project_experience"}[bad_chunk.chunk_type]
        task = _task(
            target_type=bundle_scope_type(bad_chunk.chunk_type),
            source_table=_task_table,
            source_record_id=int(bad_chunk.source_field_id) if bad_chunk.source_field_id.isdigit() else 700001,
            expert_id=700001,
        )
        with pytest.raises(ChunkValidationError):
            ScopeHandoffBuilder("Asia/Shanghai").build(task, TOKEN, prepared)


def bundle_scope_type(chunk_type):
    return {"profile_summary": "profile", "employment": "employment", "project": "project"}[chunk_type]


# ---------------------------------------------------------------------------
# Receipt 状态和计数不变式
# ---------------------------------------------------------------------------


class TestReceiptInvariants:
    def _scope(self):
        return impact_scope_from_target(target_type="expert", expert_id=700001)

    def test_committed_invariant_with_content_change(self):
        scope = self._scope()
        r = ScopeHandoffReceipt(
            request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
            payload_hash="x" * 64, impact_scope=scope, status="committed",
            result_code="converged", desired_count=3, embedding_count=2,
            metadata_only_count=1, unchanged_count=0, disable_count=1,
        )
        assert r.status == "committed"
        assert r.retryable is False

    def test_committed_empty_desired_with_disable(self):
        scope = self._scope()
        r = ScopeHandoffReceipt(
            request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
            payload_hash="x" * 64, impact_scope=scope, status="committed",
            result_code="empty_converged", desired_count=0, embedding_count=0,
            metadata_only_count=0, unchanged_count=0, disable_count=5,
        )
        # disable_count 独立，空 desired + disable 仍 committed
        assert r.disable_count == 5

    def test_committed_count_mismatch_rejected(self):
        scope = self._scope()
        # resolved(embedding+metadata_only+unchanged) > desired_count → 违反 committed 不变式
        with pytest.raises(ValueError, match="committed 回执收敛计数超过 desired_count"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="committed",
                result_code="converged", desired_count=2, embedding_count=3,
                metadata_only_count=0, unchanged_count=0,
            )

    def test_committed_unresolved_without_disable_rejected(self):
        scope = self._scope()
        # 存在未收敛且未下线的 desired（desired-resolved > disable_count）→ 违反不变式
        with pytest.raises(ValueError, match="committed 回执存在未收敛且未下线的 desired"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="committed",
                result_code="converged", desired_count=3, embedding_count=1,
                metadata_only_count=1, unchanged_count=0, disable_count=0,
            )

    def test_committed_resolved_with_disable_covers_unresolved_ok(self):
        scope = self._scope()
        # ineligible desired 经逻辑下线：disable_count 兑底未收敛 desired，committed 成立
        receipt = ScopeHandoffReceipt(
            request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
            payload_hash="x" * 64, impact_scope=scope, status="committed",
            result_code="converged", desired_count=3, embedding_count=1,
            metadata_only_count=1, unchanged_count=0, disable_count=1,
        )
        assert receipt.status == "committed"

    def test_committed_with_error_rejected(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="不得携带错误字段"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="committed",
                result_code="converged", desired_count=0, embedding_count=0,
                metadata_only_count=0, unchanged_count=0, error_type="boom",
                error_message="x",
            )

    def test_committed_retryable_must_be_false(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="retryable 必须为 False"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="committed",
                result_code="converged", desired_count=0, embedding_count=0,
                metadata_only_count=0, unchanged_count=0, retryable=True,
            )

    def test_rejected_requires_stable_error(self):
        scope = self._scope()
        r = ScopeHandoffReceipt(
            request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
            payload_hash="x" * 64, impact_scope=scope, status="rejected",
            result_code="contract_invalid", desired_count=1, embedding_count=0,
            error_type="ContractValidationError", error_message="chunk_text 疑似非法",
        )
        assert r.retryable is False

    def test_rejected_without_error_rejected(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="必须携带 error_type"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="rejected",
                result_code="contract_invalid",
            )

    def test_rejected_retryable_must_be_false(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="retryable 必须为 False"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="rejected",
                result_code="contract_invalid", retryable=True,
                error_type="X", error_message="y",
            )

    def test_failed_requires_error_retryable_optional(self):
        scope = self._scope()
        r_retry = ScopeHandoffReceipt(
            request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
            payload_hash="x" * 64, impact_scope=scope, status="failed",
            result_code="embedding_timeout", desired_count=2, embedding_count=1,
            error_type="VectorEmbeddingError", error_message="upstream timeout",
            retryable=True,
        )
        assert r_retry.retryable is True
        r_no_retry = ScopeHandoffReceipt(
            request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
            payload_hash="x" * 64, impact_scope=scope, status="failed",
            result_code="fatal", desired_count=2, error_type="X", error_message="y",
            retryable=False,
        )
        assert r_no_retry.retryable is False

    def test_failed_without_error_rejected(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="必须携带 error_type"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="failed",
                result_code="x",
            )

    def test_unknown_status_rejected(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="未知回执状态"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="skipped",  # type: ignore[arg-type]
                result_code="x",
            )

    def test_bool_count_rejected(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="desired_count 必须是非负整数"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="committed",
                result_code="converged", desired_count=True,  # type: ignore[arg-type]
                embedding_count=0, metadata_only_count=0, unchanged_count=0,
            )

    def test_negative_count_rejected(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="embedding_count 必须是非负整数"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="committed",
                result_code="converged", desired_count=0, embedding_count=-1,
                metadata_only_count=0, unchanged_count=0,
            )

    def test_multiline_error_rejected(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="单行"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="failed",
                result_code="x", error_type="X", error_message="line1\nline2",
            )

    def test_oversized_error_rejected(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="过长"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="failed",
                result_code="x", error_type="X", error_message="y" * 600,
            )

    def test_error_type_message_must_coexist(self):
        scope = self._scope()
        with pytest.raises(ValueError, match="同时提供"):
            ScopeHandoffReceipt(
                request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
                payload_hash="x" * 64, impact_scope=scope, status="failed",
                result_code="x", error_type="X", error_message=None,
            )

    def test_status_literal_only_three_values(self):
        assert set(get_args(HandoffReceiptStatus)) == {"committed", "rejected", "failed"}

    def test_receipt_has_no_transient_field(self):
        import dataclasses

        import expert_index.handoff as mod
        field_names = {f.name for f in dataclasses.fields(mod.ScopeHandoffReceipt)}
        assert "_transient" not in field_names
        # 文档化字段集
        assert field_names == {
            "request_id", "task_id", "batch_id", "attempt_token", "payload_hash",
            "impact_scope", "status", "result_code", "desired_count",
            "embedding_count", "metadata_only_count", "disable_count",
            "unchanged_count", "retryable", "error_type", "error_message",
        }


# ---------------------------------------------------------------------------
# 结构断言：不 import Work Item/Consumer/repository/MySQL/Embedding/Milvus
# ---------------------------------------------------------------------------


class TestModuleBoundary:
    def test_no_stateful_downstream_imports(self):
        import expert_index.handoff as mod

        tree = ast.parse(Path(mod.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                for n in node.names:
                    imported.add(n.name)
        forbidden = {
            "rag_vector_store",
            "pymilvus",
            "requests",
            "urllib",
            "pymysql",
            "expert_index.repositories.sync_task",
            "expert_index.checkpoint",
            "expert_index.scanner",
            "expert_index.vector_sync",
        }
        leaked = {m for f in forbidden for m in imported if f in m}
        assert not leaked, f"handoff 不得 import 状态/下游模块: {leaked}"
        # 不得 import Work Item / Consumer
        assert not any("incremental" in m and "contract_adapter" not in m for m in imported)

    def test_builder_is_pure_no_clock(self):
        import expert_index.handoff as mod

        src = Path(mod.__file__).read_text()
        # 不得调用时钟
        assert "datetime.now" not in src
        assert "datetime.utcnow" not in src
        assert "time.time" not in src

    def test_public_exports(self):
        import expert_index.handoff as mod

        assert set(mod.__all__) == {
            "HandoffReceiptStatus",
            "ScopeHandoffBuilder",
            "ScopeHandoffReceipt",
            "ScopeHandoffRequest",
        }

    def test_list_contracts_normalized_to_immutable_tuple(self):
        # Request 接受序列输入，内部归一为不可变 tuple
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        req = ScopeHandoffRequest(
            request_id="handoff:abc", task_id=701, batch_id=8, attempt_token=TOKEN,
            impact_scope=scope, contracts=[],  # list 输入
            payload_hash=_manual_hash("handoff:abc", scope, []),
        )
        assert isinstance(req.contracts, tuple)
        assert req.contracts == ()
        # 非空 list 也应归一为 tuple
        agg = _aggregate(educations=(_education(),))
        chunks = _chunks_from_aggregate(agg)
        contracts_list = chunk_records_to_contracts(
            chunks, updated_at=TASK_UPDATED, timezone_name="Asia/Shanghai")
        rid = "handoff:xyz"
        req2 = ScopeHandoffRequest(
            request_id=rid, task_id=701, batch_id=8, attempt_token=TOKEN,
            impact_scope=scope, contracts=list(contracts_list),
            payload_hash=_manual_hash(rid, scope, contracts_list),
        )
        assert isinstance(req2.contracts, tuple)
        assert len(req2.contracts) == 1

    def test_direct_request_mutation_recomputed_hash_still_scope_valid(self):
        # 直接构造 Request 并重算 hash：scope 归属规则全满足时才合法
        scope = impact_scope_from_target(target_type="expert", expert_id=700001)
        agg = _aggregate(eid=700001, educations=(_education(),))
        chunks = _chunks_from_aggregate(agg)
        contracts = chunk_records_to_contracts(
            chunks, updated_at=TASK_UPDATED, timezone_name="Asia/Shanghai")
        rid = "handoff:direct"
        # 合法 scope（expert 含三类）+ 重算 hash
        req = ScopeHandoffRequest(
            request_id=rid, task_id=701, batch_id=8, attempt_token=TOKEN,
            impact_scope=scope, contracts=tuple(contracts),
            payload_hash=_manual_hash(rid, scope, contracts),
        )
        assert req.contracts[0].expert_id == 700001  # scope 归属满足
