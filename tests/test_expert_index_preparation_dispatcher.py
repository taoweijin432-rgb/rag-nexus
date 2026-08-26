"""P4.8 单任务 Preparation Dispatcher 测试。

覆盖：四种 scope 路由、education→profile、task_id/batch_id/data_version 注入、
当前事实只读一次且仅调用目标 Handler、无效任务/token 异常向上传播、
父删除及空 desired 不被改写、策略版本校验、无任务状态/checkpoint 写入。

P4 验收修复后 Dispatcher 不再判断 ``lease_until``：真正的 lease 有效性由外层
:class:`PreparationCoordinator` 调用 Repository 续租以 Index MySQL
``CURRENT_TIMESTAMP(6)`` + ``attempt_token`` + rowcount fencing 权威判定。
故本文件不再测试 lease 过期 / naive now / 非 datetime now（这些属 Coordinator 职责，
见 ``tests/test_expert_index_preparation_coordinator.py``）。
全部使用 Fake source + 真实 ChunkFactory（策略版本 1.3），不连接 MySQL/Milvus。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.models import (
    CurrentFactReadRequest,
    EducationSource,
    EmploymentSource,
    ExpertSource,
    PreparedChunkSet,
    ProjectSource,
    ScopedCurrentFacts,
    SourceSyncTask,
    impact_scope_from_target,
)
from expert_index.preparation_dispatcher import (
    build_prepared_chunk_set,
    prepare_claimed_source_sync_task,
    read_scoped_facts,
)
from expert_index.source import ExpertNotFoundError
from expert_index.validators import ChunkValidationError

CREATED = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
UPDATED = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
# P4 验收修复后 Dispatcher 不再解读 lease_until；保留 LEASE_EXPIRY 仅供 _task 默认填充，
# 任意非空 datetime 均可通过（真正的 lease 判定在 Coordinator + Repository）。
LEASE_EXPIRY = datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc)
TOKEN = "tok-claim-001"
TASK_TS = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
# P4.3 把专家删除标记时间传到 Handler；语义上属来源事件时间，与 Dispatcher lease 无关。
NOW_FALLBACK = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
DEDUPE = "a" * 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _factory() -> ChunkFactory:
    return ChunkFactory(_settings())


def _expert(expert_id: int = 880001, *, deleted_at: datetime | None = None) -> ExpertSource:
    return ExpertSource(
        id=expert_id, serial_number="GOLDEN-专家-001", company="ＡＩ未来有限公司",
        job_status=1, job_name="首席执行官", contact_status=3, doubtful_flag=1,
        abroad_flag=2, background="专注知识图谱与检索增强，支持多语言语义分析。",
        work_years=8.0, updated_at=UPDATED, created_at=CREATED, deleted_at=deleted_at,
    )


def _education(edu_id: int, expert_id: int = 880001) -> EducationSource:
    return EducationSource(
        id=edu_id, expert_id=expert_id, school_name="清华大学", major="计算机",
        updated_at=UPDATED, created_at=CREATED, deleted_at=None,
    )


def _employment(emp_id: int = 9201, expert_id: int = 880001) -> EmploymentSource:
    return EmploymentSource(
        id=emp_id, expert_id=expert_id, company_name="北京京东世纪贸易有限公司",
        position_title="项目经理", start_date=datetime(2020, 2, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 12, 1, tzinfo=timezone.utc),
        work_description="工作职责：负责推荐系统交付与跨团队协作。",
        is_employed=1, updated_at=UPDATED, created_at=CREATED, deleted_at=None,
    )


def _project(proj_id: int = 9301, expert_id: int = 880001) -> ProjectSource:
    return ProjectSource(
        id=proj_id, expert_id=expert_id, project_title="推荐引擎重构", company_name="京东",
        start_date=datetime(2021, 3, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        work_description="项目目标：提升吞吐。\n关键动作：重写召回层。",
        updated_at=UPDATED, created_at=CREATED, deleted_at=None,
    )


def _task(
    *,
    target_type: str = "employment",
    source_table: str = "experts_work_experience",
    source_record_id: int = 9201,
    expert_id: int = 880001,
    sync_status: str = "processing",
    attempt_token: str | None = TOKEN,
    lease_until: datetime | None = LEASE_EXPIRY,
    policy_version: str | None = "1.3",
    **extra: object,
) -> SourceSyncTask:
    values: dict[str, object] = {
        "task_id": 701,
        "dedupe_key": DEDUPE,
        "batch_id": 8,
        "event_source": "poll",
        "source_table": source_table,
        "source_record_id": source_record_id,
        "source_updated_at": TASK_TS,
        "expert_id": expert_id,
        "operation_type": "update",
        "target_type": target_type,
        "processing_stage": "candidate",
        "sync_status": sync_status,
        "retry_count": 0,
        "policy_version": policy_version,
        "created_at": TASK_TS,
        "updated_at": TASK_TS,
    }
    if attempt_token is not None:
        values["attempt_token"] = attempt_token
        values["lease_until"] = lease_until
    values.update(extra)
    return SourceSyncTask(**values)  # type: ignore[arg-type]


class _FakeSource:
    """只暴露 read_current_facts 的 Fake，记录调用次数与入参。"""

    def __init__(self, facts: ScopedCurrentFacts) -> None:
        self._facts = facts
        self.read_calls: list[CurrentFactReadRequest] = []

    def read_current_facts(self, request: CurrentFactReadRequest) -> ScopedCurrentFacts:
        self.read_calls.append(request)
        return self._facts


class _ThrowingSource:
    """read_current_facts 抛异常的 Fake，验证异常向上传播。"""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.read_calls = 0

    def read_current_facts(self, request: CurrentFactReadRequest) -> ScopedCurrentFacts:
        self.read_calls += 1
        raise self._exc


# ---------------------------------------------------------------------------
# 成功路径：四种 scope 路由（真实 handler 端到端）
# ---------------------------------------------------------------------------


class TestFourScopeRouting:
    def test_expert_scope_routes_to_expert_handler(self):
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="expert", expert_id=880001),
            expert=_expert(), employments=(_employment(),),
            projects=(_project(),), educations=(_education(1),),
        )
        source = _FakeSource(facts)
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert isinstance(result, PreparedChunkSet)
        assert result.impact_scope.scope_type == "expert"
        # 当前事实只读一次
        assert len(source.read_calls) == 1
        # task_id/batch_id/data_version 注入正式值（在 PreparedChunkSet 顶层）
        assert result.task_id == 701
        assert result.batch_id == 8
        assert result.data_version == "8"  # str(batch_id)

    def test_profile_scope_routes_to_profile_handler(self):
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="profile", expert_id=880001),
            expert=_expert(), educations=(_education(1),),
        )
        source = _FakeSource(facts)
        task = _task(target_type="profile", source_table="experts",
                     source_record_id=880001)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert result.impact_scope.scope_type == "profile"
        assert all(c.chunk_type == "profile_summary" for c in result.desired_chunks)
        assert len(source.read_calls) == 1

    def test_employment_scope_routes_to_employment_handler(self):
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(
                target_type="employment", expert_id=880001, source_field_id="9201"),
            expert=_expert(), employments=(_employment(),),
        )
        source = _FakeSource(facts)
        task = _task(target_type="employment", source_table="experts_work_experience",
                     source_record_id=9201)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert result.impact_scope.scope_type == "employment"
        assert all(c.chunk_type == "employment" for c in result.desired_chunks)
        assert all(c.source_field_id == "9201" for c in result.desired_chunks)
        assert len(source.read_calls) == 1

    def test_project_scope_routes_to_project_handler(self):
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(
                target_type="project", expert_id=880001, source_field_id="9301"),
            expert=_expert(), projects=(_project(),),
        )
        source = _FakeSource(facts)
        task = _task(target_type="project", source_table="experts_project_experience",
                     source_record_id=9301)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert result.impact_scope.scope_type == "project"
        assert all(c.chunk_type == "project" for c in result.desired_chunks)
        assert all(c.source_field_id == "9301" for c in result.desired_chunks)
        assert len(source.read_calls) == 1

    def test_education_maps_to_profile_scope(self):
        # education 触发经 source_sync_task_to_read_request 映射为 profile scope
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="profile", expert_id=880001),
            expert=_expert(), educations=(_education(1),),
        )
        source = _FakeSource(facts)
        task = _task(target_type="education", source_table="experts_education_experience",
                     source_record_id=1)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert result.impact_scope.scope_type == "profile"
        assert all(c.chunk_type == "profile_summary" for c in result.desired_chunks)
        # 验证 mapping 已完成：request impact_scope 是 profile
        assert source.read_calls[0].impact_scope.scope_type == "profile"


# ---------------------------------------------------------------------------
# 恰好调用一个 Handler（monkeypatch 证明路由选择性）
# ---------------------------------------------------------------------------


class TestSingleHandlerDispatch:
    @pytest.mark.parametrize(
        "scope_type,target_type,source_table,record_id",
        [
            ("expert", "expert", "experts", 880001),
            ("profile", "profile", "experts", 880001),
            ("employment", "employment", "experts_work_experience", 9201),
            ("project", "project", "experts_project_experience", 9301),
        ],
    )
    def test_only_target_handler_called(self, monkeypatch, scope_type, target_type,
                                          source_table, record_id):
        import expert_index.scope_handlers as sh

        calls = {"expert": 0, "profile": 0, "employment": 0, "project": 0}

        def _spy(name):
            def _h(facts, factory, *, task_id, batch_id, data_version):
                calls[name] += 1
                return PreparedChunkSet(
                    impact_scope=facts.scope, desired_chunks=(),
                    task_id=task_id, batch_id=batch_id, data_version=data_version,
                )
            return _h

        monkeypatch.setattr(sh, "handle_expert_scope", _spy("expert"))
        monkeypatch.setattr(sh, "handle_profile_scope", _spy("profile"))
        monkeypatch.setattr(sh, "handle_employment_scope", _spy("employment"))
        monkeypatch.setattr(sh, "handle_project_scope", _spy("project"))

        from expert_index.models import ScopedCurrentFacts
        if scope_type == "expert":
            facts = ScopedCurrentFacts(
                scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                expert=_expert())
        elif scope_type == "profile":
            facts = ScopedCurrentFacts(
                scope=impact_scope_from_target(target_type="profile", expert_id=880001),
                expert=_expert(), educations=(_education(1),))
        elif scope_type == "employment":
            facts = ScopedCurrentFacts(
                scope=impact_scope_from_target(
                    target_type="employment", expert_id=880001, source_field_id=str(record_id)),
                expert=_expert(), employments=(_employment(record_id),))
        else:  # project
            facts = ScopedCurrentFacts(
                scope=impact_scope_from_target(
                    target_type="project", expert_id=880001, source_field_id=str(record_id)),
                expert=_expert(), projects=(_project(record_id),))
        source = _FakeSource(facts)
        task = _task(target_type=target_type, source_table=source_table,
                     source_record_id=record_id)
        prepare_claimed_source_sync_task(task, TOKEN, source, _factory())

        assert calls[scope_type] == 1, f"应只调用 {scope_type} handler 一次"
        others = {k: v for k, v in calls.items() if k != scope_type}
        assert all(v == 0 for v in others.values()), f"其他 handler 不应被调用: {others}"
        assert len(source.read_calls) == 1, "当前事实只读一次"


# ---------------------------------------------------------------------------
# 无效任务 / token：异常向上传播
#
# P4 验收修复后 Dispatcher 不再校验 lease_until（lease 有效性由 Coordinator + Repository
# 以 MySQL CURRENT_TIMESTAMP(6) 权威判定），故不再测试 lease 过期 / naive now。
# ---------------------------------------------------------------------------


class TestInvalidTaskToken:
    def test_non_source_sync_task_rejected(self):
        with pytest.raises(TypeError, match="必须是 SourceSyncTask"):
            prepare_claimed_source_sync_task(
                "not a task", TOKEN, _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=1),
                    expert=_expert())), _factory(),
            )

    def test_non_processing_status_rejected(self):
        # pending 任务 attempt_token/lease_until 必须为 None，由 model 校验
        task = _task(sync_status="pending", attempt_token=None, lease_until=None)
        with pytest.raises(ValueError, match="非 processing"):
            prepare_claimed_source_sync_task(
                task, TOKEN, _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                    expert=_expert())), _factory(),
            )

    def test_blank_token_rejected(self):
        task = _task(target_type="expert", source_table="experts", source_record_id=880001)
        with pytest.raises(ValueError, match="attempt_token"):
            prepare_claimed_source_sync_task(
                task, "   ", _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                    expert=_expert())), _factory(),
            )

    def test_token_too_long_rejected(self):
        task = _task(target_type="expert", source_table="experts", source_record_id=880001)
        with pytest.raises(ValueError, match="attempt_token"):
            prepare_claimed_source_sync_task(
                task, "x" * 65, _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                    expert=_expert())), _factory(),
            )

    def test_token_mismatch_rejected(self):
        task = _task(target_type="expert", source_table="experts", source_record_id=880001)
        with pytest.raises(ValueError, match="不一致"):
            prepare_claimed_source_sync_task(
                task, "wrong-token", _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                    expert=_expert())), _factory(),
            )


# ---------------------------------------------------------------------------
# 策略版本校验
# ---------------------------------------------------------------------------


class TestPolicyVersion:
    def test_missing_policy_version_rejected(self):
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001, policy_version=None)
        with pytest.raises(ValueError, match="缺少 policy_version"):
            prepare_claimed_source_sync_task(
                task, TOKEN, _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                    expert=_expert())), _factory(),
            )

    def test_mismatched_policy_version_rejected(self):
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001, policy_version="1.2")
        with pytest.raises(ValueError, match="不一致"):
            prepare_claimed_source_sync_task(
                task, TOKEN, _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                    expert=_expert())), _factory(),
            )

    def test_uses_injected_factory_policy_not_constant(self):
        # 注入一个 chunk_policy_version 不同的 Factory，证明读 Factory 配置而非硬编码常量
        import dataclasses
        custom_settings = dataclasses.replace(_settings(), chunk_policy_version="9.9")
        custom_factory = ChunkFactory(custom_settings)

        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001, policy_version="9.9")
        source = _FakeSource(ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="expert", expert_id=880001),
            expert=_expert(), employments=(), projects=(), educations=(),
        ))
        result = prepare_claimed_source_sync_task(task, TOKEN, source, custom_factory)
        assert result.impact_scope.scope_type == "expert"

        # 同一 task 用默认 Factory（policy 1.3）应被拒绝
        with pytest.raises(ValueError, match="不一致"):
            prepare_claimed_source_sync_task(task, TOKEN,
                _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                    expert=_expert())), _factory())


# ---------------------------------------------------------------------------
# 异常向上传播（不吞不包装）
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_mapping_error_propagates(self):
        # expert_id=None 任务，source_sync_task_to_read_request 会抛 ValueError
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001, expert_id=None)
        with pytest.raises(ValueError, match="expert_id"):
            prepare_claimed_source_sync_task(
                task, TOKEN, _FakeSource(ScopedCurrentFacts(
                    scope=impact_scope_from_target(target_type="expert", expert_id=880001),
                    expert=_expert())), _factory(),
            )

    def test_source_exception_propagates(self):
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        source = _ThrowingSource(RuntimeError("source boom"))
        with pytest.raises(RuntimeError, match="source boom"):
            prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert source.read_calls == 1

    def test_scope_mismatch_rejected(self):
        # source 返回 scope 与请求不一致
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        # 请求是 expert，但 source 返回 profile scope
        mismatch_facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="profile", expert_id=880001),
            expert=_expert(),
        )
        source = _FakeSource(mismatch_facts)
        with pytest.raises(ValueError, match="scope 不一致"):
            prepare_claimed_source_sync_task(task, TOKEN, source, _factory())

    def test_handler_exception_propagates_unwrapped(self, monkeypatch):
        import expert_index.scope_handlers as sh

        def boom(facts, factory, *, task_id, batch_id, data_version):
            raise ChunkValidationError("v=demo", "handler 故意失败")

        monkeypatch.setattr(sh, "handle_expert_scope", boom)
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        source = _FakeSource(ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="expert", expert_id=880001),
            expert=_expert(), employments=(), projects=(), educations=(),
        ))
        with pytest.raises(ChunkValidationError, match="handler 故意失败"):
            prepare_claimed_source_sync_task(task, TOKEN, source, _factory())

    def test_physical_missing_propagates_expert_not_found(self):
        # P4.3 物理行不存在独出 ExpertNotFoundError
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        source = _ThrowingSource(ExpertNotFoundError("专家 880001 物理不存在"))
        with pytest.raises(ExpertNotFoundError, match="物理不存在"):
            prepare_claimed_source_sync_task(task, TOKEN, source, _factory())

    def test_ownership_mismatch_propagates_value_error(self):
        # P4.3 子表-父专家归属不一致独出 ValueError
        task = _task(target_type="employment", source_table="experts_work_experience",
                     source_record_id=9201)
        source = _ThrowingSource(ValueError("employment 触发行 expert_id=999 与请求 880001 不一致"))
        with pytest.raises(ValueError, match="不一致"):
            prepare_claimed_source_sync_task(task, TOKEN, source, _factory())

    def test_timing_rollback_propagates_value_error(self):
        # P4.3 当前事实比候选旧独出 ValueError（事件比当前事实更新）
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        source = _ThrowingSource(ValueError("experts.updated_at 当前 X 早于候选 Y，事件比当前事实更新"))
        with pytest.raises(ValueError, match="事件比当前事实更新"):
            prepare_claimed_source_sync_task(task, TOKEN, source, _factory())

    def test_chunk_validation_error_propagates(self):
        # Handler 内 validate_chunks 独出 ChunkValidationError
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        source = _ThrowingSource(ChunkValidationError("v=demo", "validate 失败"))
        with pytest.raises(ChunkValidationError, match="validate 失败"):
            prepare_claimed_source_sync_task(task, TOKEN, source, _factory())


# ---------------------------------------------------------------------------
# 父删除及空 desired 不被改写
# ---------------------------------------------------------------------------


class TestPreservesHandlerResult:
    def test_expert_deleted_empty_result_preserved(self):
        # 父专家删除：Handler 返回 expert 空 desired，Dispatcher 原样返回
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="expert", expert_id=880001),
            expert=_expert(deleted_at=NOW_FALLBACK),  # P4.3 已删除专家子表为空
        )
        source = _FakeSource(facts)
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert result.impact_scope.scope_type == "expert"
        assert result.is_empty
        assert result.desired_chunks == ()

    def test_employment_row_deleted_preserves_scope(self):
        # employment 行逻辑删除：Handler 返回 employment 空 desired（保留原 scope）
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(
                target_type="employment", expert_id=880001, source_field_id="9201"),
            expert=_expert(),
            employments=(EmploymentSource(
                id=9201, expert_id=880001, company_name="X", position_title="Y",
                start_date=None, end_date=None, work_description=None, is_employed=None,
                updated_at=UPDATED, created_at=CREATED, deleted_at=NOW_FALLBACK,
            ),),
        )
        source = _FakeSource(facts)
        task = _task(target_type="employment", source_table="experts_work_experience",
                     source_record_id=9201)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert result.impact_scope.scope_type == "employment"
        assert result.desired_chunks == ()

    def test_child_scope_parent_deleted_escalates_to_expert_empty(self):
        # 子 scope 任务 + 父专家删除：Handler 升级为 expert 空 desired，Dispatcher 原样保留
        # employment scope 触发，但父专家 deleted_at 非空
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(
                target_type="employment", expert_id=880001, source_field_id="9201"),
            expert=_expert(deleted_at=NOW_FALLBACK),
            employments=(_employment(),),
        )
        source = _FakeSource(facts)
        task = _task(target_type="employment", source_table="experts_work_experience",
                     source_record_id=9201)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        # 升级为 expert scope + 空 desired，保留原范围类型语义
        assert result.impact_scope.scope_type == "expert"
        assert result.is_empty
        assert result.desired_chunks == ()


# ---------------------------------------------------------------------------
# 拆分函数：read_scoped_facts / build_prepared_chunk_set
# ---------------------------------------------------------------------------


class TestSplitHelpers:
    def test_read_scoped_facts_does_not_build(self):
        # read_scoped_facts 只读取事实，不调 Factory，不构造 Chunk
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="expert", expert_id=880001),
            expert=_expert(), employments=(), projects=(), educations=(),
        )
        source = _FakeSource(facts)
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        out = read_scoped_facts(task, source)
        assert out is facts
        assert len(source.read_calls) == 1

    def test_read_scoped_facts_rejects_scope_mismatch(self):
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        mismatch = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="profile", expert_id=880001),
            expert=_expert(),
        )
        source = _FakeSource(mismatch)
        with pytest.raises(ValueError, match="scope 不一致"):
            read_scoped_facts(task, source)

    def test_build_prepared_chunk_set_uses_facts_directly(self):
        # build_prepared_chunk_set 不再读取事实，直接用传入 facts 走 Handler
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="expert", expert_id=880001),
            expert=_expert(), employments=(), projects=(), educations=(),
        )
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        # source 未传给 build，证明不读取事实
        result = build_prepared_chunk_set(task, facts, _factory())
        assert result.impact_scope.scope_type == "expert"


# ---------------------------------------------------------------------------
# 无任务状态/checkpoint 写入（结构断言）
# ---------------------------------------------------------------------------


class TestNoStateWrites:
    def test_dispatcher_module_has_no_state_imports(self):
        import ast

        import expert_index.preparation_dispatcher as mod

        tree = ast.parse(Path(mod.__file__).read_text())
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                for n in node.names:
                    imported_modules.add(n.name)
        # 不得 import 任务状态/存储/Consumer/Embedding/Milvus/Contract 桁
        forbidden = {
            "repositories.sync_task", "expert_index.checkpoint",
            "expert_index.contract_bridge", "expert_index.vector_sync",
            "pymilvus", "requests", "urllib", "rag_vector_store",
        }
        leaked = {m for f in forbidden for m in imported_modules if f in m}
        assert not leaked, f"Dispatcher 不得 import 状态/下游模块: {leaked}"

    def test_dispatcher_exports_three_helpers(self):
        # P4 验收修复后 Dispatcher 拆出 read_scoped_facts / build_prepared_chunk_set
        # 供外层 Coordinator 在阶段推进间穿插使用。__all__ 按字母序。
        import expert_index.preparation_dispatcher as mod
        assert mod.__all__ == [
            "build_prepared_chunk_set",
            "prepare_claimed_source_sync_task",
            "read_scoped_facts",
        ]
        defined = [n for n, v in vars(mod).items()
                   if callable(v) and getattr(v, "__module__", "") == mod.__name__]
        assert set(defined) == {
            "_validate_token",
            "prepare_claimed_source_sync_task",
            "read_scoped_facts",
            "build_prepared_chunk_set",
        }

    def test_fake_source_records_single_read(self):
        # 证明 source 只被读一次，无重复读取
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="expert", expert_id=880001),
            expert=_expert(), employments=(), projects=(), educations=(),
        )
        source = _FakeSource(facts)
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001)
        prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert len(source.read_calls) == 1


# ---------------------------------------------------------------------------
# data_version 固定 str(batch_id)
# ---------------------------------------------------------------------------


class TestDataVersionInjection:
    def test_data_version_is_str_batch_id(self):
        facts = ScopedCurrentFacts(
            scope=impact_scope_from_target(target_type="expert", expert_id=880001),
            expert=_expert(), employments=(), projects=(), educations=(),
        )
        source = _FakeSource(facts)
        task = _task(target_type="expert", source_table="experts",
                     source_record_id=880001, batch_id=999)
        result = prepare_claimed_source_sync_task(task, TOKEN, source, _factory())
        assert result.data_version == "999"
        assert result.batch_id == 999
        assert result.task_id == 701
