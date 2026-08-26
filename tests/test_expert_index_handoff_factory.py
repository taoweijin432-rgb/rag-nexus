"""P5.4 生产依赖组装 Factory 测试。

全部使用 monkeypatch/spies：不连接真实 MySQL/Milvus/Embedding，不启动 Worker，
不调 initialize()。只验证组装正确性与 fail-closed 前置校验。
"""

from __future__ import annotations

import pytest

from expert_index.config import (
    DatabaseConfig,
    DatabaseRole,
    IncrementalScanConfig,
    Settings,
)
from expert_index.handoff_factory import (
    IncrementalHandoffRuntime,
    build_incremental_handoff_runtime,
)
from expert_index.scanner import IncrementalScanDisabledError


def _settings(*, enabled: bool = True, same_schema: bool = False) -> Settings:
    """构造测试 Settings；默认开启增量且 Index/Source 物理 Schema 隔离。"""

    source = DatabaseConfig(
        host="39.96.40.93", port=3306, user="reader", password="x",
        database="expert", role=DatabaseRole.SOURCE,
    )
    # same_schema=True 时 Index 与 Source 指向同一 host+port+database（触发边界拒绝）。
    if same_schema:
        index = DatabaseConfig(
            host="39.96.40.93", port=3306, user="idx", password="y",
            database="expert", role=DatabaseRole.INDEX,
        )
    else:
        index = DatabaseConfig(
            host="127.0.0.1", port=3306, user="idx", password="y",
            database="expert_dev", role=DatabaseRole.INDEX,
        )
    incremental = IncrementalScanConfig(
        enabled=enabled, batch_size=50, lease_seconds=300,
    )
    settings = Settings(source_db=source, index_db=index, incremental_scan=incremental)
    return settings


class _SpyVectorService:
    """仅用于注入的向量服务占位；不暴露任何业务方法。"""



class TestFactoryGates:
    def test_disabled_incremental_fails_before_anything(self, monkeypatch):
        # 即使下游 import 也不应发生：禁用时直接抛，不构造 vector_service。
        called = {"v": 0}

        def _fail(*a, **kw):
            called["v"] += 1
            raise AssertionError("不应构造 vector_service")

        monkeypatch.setattr(
            "rag_vector_store.create_vector_ingestion_service", _fail)
        settings = _settings(enabled=False)
        with pytest.raises(IncrementalScanDisabledError):
            build_incremental_handoff_runtime(
                settings, max_attempts=2, retry_delay_seconds=30)
        assert called["v"] == 0

    def test_v14_profile_fail_closed(self, monkeypatch):
        called = {"v": 0}

        def _fail(*a, **kw):
            called["v"] += 1
            raise AssertionError("不应构造 vector_service")

        monkeypatch.setattr(
            "rag_vector_store.create_vector_ingestion_service", _fail)
        settings = _settings()
        with pytest.raises(ValueError, match="schema_profile"):
            build_incremental_handoff_runtime(
                settings, schema_profile="v14",
                vector_service=_SpyVectorService(),
                max_attempts=2, retry_delay_seconds=30)
        assert called["v"] == 0

    def test_unknown_profile_fail_closed(self):
        settings = _settings()
        with pytest.raises(ValueError, match="schema_profile"):
            build_incremental_handoff_runtime(
                settings, schema_profile="weird",
                vector_service=_SpyVectorService(),
                max_attempts=2, retry_delay_seconds=30)

    def test_same_schema_boundary_fail_closed(self):
        from expert_index.config import DatabaseBoundaryError

        settings = _settings(same_schema=True)
        with pytest.raises(DatabaseBoundaryError):
            build_incremental_handoff_runtime(
                settings, vector_service=_SpyVectorService(),
                max_attempts=2, retry_delay_seconds=30)

    def test_bad_retry_args_fail_closed(self):
        settings = _settings()
        with pytest.raises(ValueError, match="max_attempts"):
            build_incremental_handoff_runtime(
                settings, vector_service=_SpyVectorService(),
                max_attempts=-1, retry_delay_seconds=30)
        with pytest.raises(ValueError, match="retry_delay_seconds"):
            build_incremental_handoff_runtime(
                settings, vector_service=_SpyVectorService(),
                max_attempts=2, retry_delay_seconds=0)


class TestAssemblyWiring:
    def test_assembles_all_components_without_connecting(self):
        from expert_index.cdc import CdcDeleteTaskProcessor
        from expert_index.preparation_coordinator import PreparationCoordinator

        settings = _settings()
        spy = _SpyVectorService()
        runtime = build_incremental_handoff_runtime(
            settings, vector_service=spy,
            max_attempts=3, retry_delay_seconds=45)

        assert isinstance(runtime, IncrementalHandoffRuntime)
        # 来源/构建器/持久化按配置构造
        assert runtime.source.config is settings.source_db
        assert runtime.chunk_factory.settings is settings
        assert runtime.task_repo.config is settings.index_db
        assert runtime.artifact_repo.config is settings.index_db
        assert runtime.builder.timezone_name == "Asia/Shanghai"
        # PreparationCoordinator 已组装，复用同一 task_repo/source/chunk_factory
        assert isinstance(runtime.preparation_coordinator, PreparationCoordinator)
        assert runtime.preparation_coordinator._repo is runtime.task_repo
        assert runtime.preparation_coordinator._source is runtime.source
        assert runtime.preparation_coordinator._factory is runtime.chunk_factory
        # lease_seconds 复用正式 incremental_scan 配置（与 P3 claim/renew 一致）
        assert (
            runtime.preparation_coordinator._lease_seconds
            == settings.incremental_scan.lease_seconds
        )
        # 同一 vector_service 注入 core 与 state_port
        assert runtime.vector_service is spy
        # orchestrator 持有同一批��件
        assert runtime.orchestrator._task_repo is runtime.task_repo
        assert runtime.orchestrator._artifact_repo is runtime.artifact_repo
        assert runtime.orchestrator._builder is runtime.builder
        assert runtime.orchestrator._consumer is runtime.scope_consumer
        assert runtime.orchestrator._max_attempts == 3
        assert runtime.orchestrator._retry_delay_seconds == 45
        # CDC DELETE 复用同一份 Task、Source、ChunkFactory 与 Handoff 编排依赖。
        assert isinstance(runtime.cdc_delete_processor, CdcDeleteTaskProcessor)
        assert runtime.cdc_delete_processor._task_repository is runtime.task_repo
        assert runtime.cdc_delete_processor._source is runtime.source
        assert runtime.cdc_delete_processor._factory is runtime.chunk_factory
        assert runtime.cdc_delete_processor._orchestrator is runtime.orchestrator
        assert (
            runtime.cdc_delete_processor._lease_seconds
            == settings.incremental_scan.lease_seconds
        )
        assert runtime.cdc_delete_processor._max_attempts == 3
        assert runtime.cdc_delete_processor._retry_delay_seconds == 45

    def test_eligibility_injected_non_none(self):
        # ContractConsumerCore 必须注入正式 Eligibility，process_contracts 不能因
        # None 而拒绝执行。
        from rag_vector_store.incremental.consumer_core import ContractConsumerCore

        settings = _settings()
        spy = _SpyVectorService()
        runtime = build_incremental_handoff_runtime(
            settings, vector_service=spy,
            max_attempts=3, retry_delay_seconds=45)
        core = runtime.scope_consumer._core  # type: ignore[attr-defined]
        assert isinstance(core, ContractConsumerCore)
        assert core._eligibility_skip_of is not None

    def test_eligibility_rejects_ineligible_contract(self):
        # is_active=false 的 Contract 经 Eligibility 应被标记 skip。
        settings = _settings()
        spy = _SpyVectorService()
        runtime = build_incremental_handoff_runtime(
            settings, vector_service=spy,
            max_attempts=3, retry_delay_seconds=45)
        skip_of = runtime.scope_consumer._core._eligibility_skip_of  # type: ignore[attr-defined]
        assert skip_of is not None and callable(skip_of)
        # 用真实 ChunkFactory 生成合法 employment Contract，再改 is_active=false 验证 skip。
        from datetime import datetime, timezone

        from expert_index.chunk_builder import ChunkFactory
        from expert_index.contract_bridge import chunk_records_to_contracts
        from expert_index.models import EducationSource, ExpertAggregate, ExpertSource
        from expert_index.normalize import normalize_aggregate

        created = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)
        agg = ExpertAggregate(
            profile=ExpertSource(
                id=700001, serial_number="T700001", company="虚构公司",
                job_status=1, job_name="项目经理", contact_status=3,
                doubtful_flag=1, abroad_flag=2, background="虚构背景",
                work_years=8.5, updated_at=created, created_at=created, deleted_at=None,
            ),
            employments=(),
            projects=(),
            educations=(EducationSource(
                id=1, expert_id=700001, school_name="测试大学",
                major="计算机", updated_at=created, created_at=created, deleted_at=None,
            ),),
        )
        profile_stage, emp_stages, proj_stages = normalize_aggregate(
            agg, data_version="8", build_batch_id=701)
        chunks = ChunkFactory(settings).build_all(profile_stage, emp_stages, proj_stages)
        contracts = chunk_records_to_contracts(
            chunks, updated_at=created, timezone_name=settings.timezone)
        inactive = contracts[0].model_copy(update={"is_active": False})
        reason = skip_of(inactive)
        assert reason is not None and "is_active" in reason
        # active 版应允许消费（None）
        assert skip_of(contracts[0]) is None

    def test_receipt_sink_adapter_constructs_receipt(self):
        from expert_index.handoff import ScopeHandoffReceipt
        from expert_index.handoff_factory import _ReceiptSinkAdapter
        from expert_index.models import IncrementalImpactScope

        scope = IncrementalImpactScope(
            scope_type="expert", expert_id=700001, chunk_type=None,
            source_table=None, source_field_id=None,
        )
        sink = _ReceiptSinkAdapter(ScopeHandoffReceipt)
        receipt = sink.make_receipt(
            request_id="r", task_id=1, batch_id=1, attempt_token="t",
            payload_hash="0" * 64, impact_scope=scope,
            status="committed", result_code="scope_committed",
            desired_count=1, embedding_count=1, metadata_only_count=0,
            disable_count=0, unchanged_count=0, retryable=False,
            error_type=None, error_message=None,
        )
        assert isinstance(receipt, ScopeHandoffReceipt)
        assert receipt.status == "committed"

    def test_receipt_sink_rejects_non_impact_scope(self):
        from expert_index.handoff import ScopeHandoffReceipt
        from expert_index.handoff_factory import _ReceiptSinkAdapter

        sink = _ReceiptSinkAdapter(ScopeHandoffReceipt)
        with pytest.raises(TypeError, match="impact_scope"):
            sink.make_receipt(
                request_id="r", task_id=1, batch_id=1, attempt_token="t",
                payload_hash="0" * 64, impact_scope="not-a-scope",
                status="committed", result_code="x",
                desired_count=0, embedding_count=0, metadata_only_count=0,
                disable_count=0, unchanged_count=0, retryable=False,
                error_type=None, error_message=None,
            )


    def test_runtime_exposes_coordinator_not_legacy_dispatcher(self):
        # 结构断言：正式调用说明不再绕过 Coordinator——runtime 必须暴露
        # preparation_coordinator，且不再引导调用方直接调旧的
        # prepare_claimed_source_sync_task（它不续租、不推进阶段）。
        from expert_index.handoff_factory import IncrementalHandoffRuntime
        from expert_index.preparation_coordinator import PreparationCoordinator

        settings = _settings()
        spy = _SpyVectorService()
        runtime = build_incremental_handoff_runtime(
            settings, vector_service=spy,
            max_attempts=2, retry_delay_seconds=30)
        # runtime 持有 Coordinator 实例
        assert isinstance(runtime.preparation_coordinator, PreparationCoordinator)
        # Runtime docstring 不再推荐直接调旧 dispatcher
        doc = IncrementalHandoffRuntime.__doc__ or ""
        assert "prepare_claimed_source_sync_task" not in doc
        assert "PreparationCoordinator.prepare" in doc


class TestNoExternalCalls:
    def test_lazy_vector_service_construction_only_when_absent(self, monkeypatch):
        # 注入 vector_service 时不调 create_vector_ingestion_service
        called = {"v": 0}

        def _fail(*a, **kw):
            called["v"] += 1
            raise AssertionError("不应惰性构造")

        monkeypatch.setattr(
            "rag_vector_store.create_vector_ingestion_service", _fail)
        settings = _settings()
        spy = _SpyVectorService()
        runtime = build_incremental_handoff_runtime(
            settings, vector_service=spy,
            max_attempts=2, retry_delay_seconds=30)
        assert runtime.vector_service is spy
        assert called["v"] == 0

    def test_lazy_construction_uses_v15_profile(self, monkeypatch):
        captured = {}

        def _fake(*a, **kw):
            captured["schema_profile"] = kw.get("schema_profile")
            return _SpyVectorService()

        monkeypatch.setattr(
            "rag_vector_store.create_vector_ingestion_service", _fake)
        settings = _settings()
        runtime = build_incremental_handoff_runtime(
            settings, max_attempts=2, retry_delay_seconds=30)
        assert captured["schema_profile"] == "v15"
        assert isinstance(runtime.vector_service, _SpyVectorService)


def _make_contract_dummy() -> None:
    """占位：真实 Contract 由 test_eligibility_rejects_ineligible_contract 中
    的 ChunkFactory 链路生成，避免手写字段与契约脱同步。"""

    return
