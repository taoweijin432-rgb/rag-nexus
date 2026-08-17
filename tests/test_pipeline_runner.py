"""Pipeline Runner（Task2 Step 2）单元测试。

覆盖：
1. 正常执行（created → running → completed，条目 success/skipped）
2. ingestion 失败（→ failed，条目全部标记 failed）
3. skipped 记录处理（条目 status=skipped, result=skip_reason）
4. Task State 状态变化时序
5. 前置条件（任务不存在 / 状态不是 created）
6. 读取失败（fail-closed）
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from expert_contracts.models import ExpertChunkEmbeddingInput
from rag_vector_store.chunking.contract_ingestion import (
    ContractBatchIngestionResult,
    ContractIngestionResult,
)
from rag_vector_store.mysql.expert_chunk_reader import (
    MySQLReadError,
    MySQLReadResult,
)
from rag_vector_store.runner import (
    PipelineRunnerError,
    VectorizationPipelineRunner,
)
from rag_vector_store.task_state import TaskState, TaskStateError

# ── Fake 组件 ──


class FakeTaskStateStore:
    """内存版 TaskStateStore，记录任务与条目状态变迁。"""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[str] = []

    def create_task(
        self,
        *,
        source: str,
        mode: str = "full",
        collection_name: str,
        total_count: int,
        request_id: str | None = None,
    ) -> TaskState:
        # 生成符合 CHAR(36) UUID 格式的 task_id
        import uuid

        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        self.tasks[task_id] = {
            "task_id": task_id,
            "source": source,
            "mode": mode,
            "status": "created",
            "total_count": total_count,
            "success_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "request_id": request_id,
            "error_message": None,
            "collection_name": collection_name,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
        }
        self.calls.append("create_task")
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> TaskState:
        self.calls.append(f"get_task:{task_id}")
        row = self.tasks.get(task_id)
        if row is None:
            raise TaskStateError(f"任务不存在: {task_id}")
        return TaskState.model_validate(row)

    def update_task_status(
        self,
        task_id: str,
        *,
        status: str,
        success_count: int | None = None,
        skipped_count: int | None = None,
        failed_count: int | None = None,
        error_message: str | None = None,
        started_at: bool = False,
        finished_at: bool = False,
    ) -> None:
        self.calls.append(f"update_task:{status}")
        row = self.tasks[task_id]
        row["status"] = status
        if success_count is not None:
            row["success_count"] = success_count
        if skipped_count is not None:
            row["skipped_count"] = skipped_count
        if failed_count is not None:
            row["failed_count"] = failed_count
        if error_message is not None:
            row["error_message"] = error_message
        now = datetime.now(timezone.utc)
        if started_at:
            row["started_at"] = now
        if finished_at:
            row["finished_at"] = now

    def create_items(self, items: list[Any]) -> None:
        self.calls.append("create_items")
        now = datetime.now(timezone.utc)
        for item in items:
            self.items[(item.task_id, item.vector_doc_id)] = {
                "task_id": item.task_id,
                "vector_doc_id": item.vector_doc_id,
                "status": "pending",
                "result": None,
                "error_message": None,
                "processed_at": None,
                "created_at": now,
            }

    def get_item(self, task_id: str, vector_doc_id: str) -> Any:
        return self.items.get((task_id, vector_doc_id))

    def update_item_status(
        self,
        task_id: str,
        vector_doc_id: str,
        *,
        status: str,
        result: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.calls.append(f"update_item:{status}")
        key = (task_id, vector_doc_id)
        if key not in self.items:
            self.items[key] = {
                "task_id": task_id,
                "vector_doc_id": vector_doc_id,
                "status": "pending",
                "result": None,
                "error_message": None,
                "processed_at": None,
                "created_at": datetime.now(timezone.utc),
            }
        row = self.items[key]
        row["status"] = status
        if result is not None:
            row["result"] = result
        if error_message is not None:
            row["error_message"] = error_message
        row["processed_at"] = datetime.now(timezone.utc)

    def list_items(self, task_id: str) -> list[Any]:
        return [
            v for k, v in self.items.items() if k[0] == task_id
        ]


class FakeReader:
    """内存版 MySqlExpertChunkReader。"""

    def __init__(
        self,
        records: list[ExpertChunkEmbeddingInput] | None = None,
        *,
        errors: list[MySQLReadError] | None = None,
        read_error: Exception | None = None,
    ) -> None:
        self.records = records or []
        self.errors = errors or []
        self.read_error = read_error
        self.read_calls = 0

    def read(self) -> MySQLReadResult:
        self.read_calls += 1
        if self.read_error is not None:
            raise self.read_error
        return MySQLReadResult(
            input_count=len(self.records),
            records=self.records,
            errors=self.errors,
        )


class FakeIngestionService:
    """内存版 ContractBatchIngestionService（不复制 Embedding/Milvus 逻辑）。"""

    def __init__(
        self,
        *,
        result: ContractBatchIngestionResult | None = None,
        ingest_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.ingest_error = ingest_error
        self.calls: list[list[ExpertChunkEmbeddingInput]] = []

    def ingest_batch(
        self,
        records: list[ExpertChunkEmbeddingInput],
    ) -> ContractBatchIngestionResult:
        self.calls.append(list(records))
        if self.ingest_error is not None:
            raise self.ingest_error
        if self.result is not None:
            return self.result
        # 默认全部 success
        return ContractBatchIngestionResult(
            results=[
                ContractIngestionResult(
                    status="embedded",
                    vector_doc_id=record.vector_doc_id,
                    affected_count=1,
                )
                for record in records
            ],
            embedded_count=len(records),
            skipped_count=0,
            affected_count=len(records),
        )


def _make_record(vector_doc_id: str) -> ExpertChunkEmbeddingInput:
    parts = vector_doc_id.split(":")
    expert_id = int(parts[1])
    chunk_index = int(parts[-1])
    source_field_id = parts[-2]
    text = "[专家能力概述]\n虚构测试专家具备知识检索系统建设经验。"
    return ExpertChunkEmbeddingInput(
        contract_version="4.0.0",
        vector_doc_id=vector_doc_id,
        expert_id=expert_id,
        serial_number=f"DEV{expert_id}",
        chunk_type="profile_summary",
        chunk_subtype=None,
        chunk_index=chunk_index,
        chunk_total=1,
        chunk_text=text,
        char_count=len(text),
        content_hash="a" * 64,
        metadata_hash="b" * 64,
        metadata={
            "company_name_raw": None,
            "company_name_norm": None,
            "role_name_raw": None,
            "role_code": None,
            "role_level": None,
            "job_status_norm": "active",
            "abroad_flag_norm": False,
            "work_years": None,
            "project_name": None,
            "start_month": None,
            "end_month": None,
            "is_current": None,
            "industry_codes": ["artificial_intelligence"],
            "function_codes": [],
            "topic_codes": ["knowledge_retrieval"],
            "school_names_raw": [],
            "major_names_raw": [],
            "eligible_for_recommendation": True,
            "origin_type": "compressed_from_source",
            "verification_status": "source_summary",
            "weight_tier": "B",
        },
        source={
            "source_table": "experts",
            "source_field": "company+job_name+background",
            "source_field_id": source_field_id,
            "source_version": "2026-08-12T10:00:00+08:00",
            "source_updated_at": "2026-08-12T10:00:00+08:00",
            "source_hash": "a" * 64,
        },
        versions={"chunk_policy_version": "1.3", "metadata_schema_version": "3.0"},
        is_active=True,
        updated_at="2026-08-12T10:05:00+08:00",
    )


def _make_runner(
    store: FakeTaskStateStore,
    reader: FakeReader,
    service: FakeIngestionService,
) -> VectorizationPipelineRunner:
    return VectorizationPipelineRunner(
        task_state_store=store,
        reader=reader,
        ingestion_service=service,
    )


# ── 1. 正常执行 ──


def test_run_success_transitions() -> None:
    store = FakeTaskStateStore()
    records = [_make_record("expert:1:profile_summary:1:0")]
    reader = FakeReader(records)
    service = FakeIngestionService()
    runner = _make_runner(store, reader, service)

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=1,
    )
    final = runner.run(task.task_id)

    assert final.status == "completed"
    assert final.success_count == 1
    assert final.skipped_count == 0
    assert final.failed_count == 0
    assert final.finished_at is not None

    # 状态变迁时序：created → running → completed
    transitions = [
        c for c in store.calls if c.startswith("update_task")
    ]
    assert transitions == [
        "update_task:running",
        "update_task:completed",
    ]

    # Reader 和 Ingestion 各调用一次
    assert reader.read_calls == 1
    assert len(service.calls) == 1

    # 条目成功
    item = store.get_item(task.task_id, records[0].vector_doc_id)
    assert item is not None
    assert item["status"] == "success"
    assert item["result"] == "embedded"
    assert item["processed_at"] is not None


def test_run_success_with_skipped_records() -> None:
    store = FakeTaskStateStore()
    records = [
        _make_record("expert:1:profile_summary:1:0"),
        _make_record("expert:1:profile_summary:2:0"),
    ]
    reader = FakeReader(records)
    service = FakeIngestionService(
        result=ContractBatchIngestionResult(
            results=[
                ContractIngestionResult(
                    status="embedded",
                    vector_doc_id=records[0].vector_doc_id,
                    affected_count=1,
                ),
                ContractIngestionResult(
                    status="skipped",
                    vector_doc_id=records[1].vector_doc_id,
                    skip_reason="verification_status=rejected",
                ),
            ],
            embedded_count=1,
            skipped_count=1,
            affected_count=1,
        )
    )
    runner = _make_runner(store, reader, service)

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=2,
    )
    final = runner.run(task.task_id)

    assert final.status == "completed"
    assert final.success_count == 1
    assert final.skipped_count == 1
    assert final.failed_count == 0

    # 条目：第一条 success，第二条 skipped（result 记录跳过原因）
    item0 = store.get_item(task.task_id, records[0].vector_doc_id)
    item1 = store.get_item(task.task_id, records[1].vector_doc_id)
    assert item0["status"] == "success"
    assert item0["result"] == "embedded"
    assert item1["status"] == "skipped"
    assert item1["result"] == "verification_status=rejected"


# ── 2. ingestion 失败 ──


def test_run_ingestion_failure_marks_task_failed() -> None:
    store = FakeTaskStateStore()
    records = [_make_record("expert:1:profile_summary:1:0")]
    reader = FakeReader(records)
    service = FakeIngestionService(
        ingest_error=RuntimeError("embedding api down"),
    )
    runner = _make_runner(store, reader, service)

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=1,
    )
    final = runner.run(task.task_id)

    assert final.status == "failed"
    assert final.failed_count == 1
    assert "embedding" in (final.error_message or "")
    assert final.finished_at is not None

    # 失败后不再走到 completed
    assert "update_task:completed" not in store.calls


# ── 3. 读取失败（fail-closed） ──


def test_run_read_error_marks_task_failed() -> None:
    store = FakeTaskStateStore()
    reader = FakeReader(read_error=RuntimeError("mysql connection refused"))
    service = FakeIngestionService()
    runner = _make_runner(store, reader, service)

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=1,
    )
    final = runner.run(task.task_id)

    assert final.status == "failed"
    assert "mysql" in (final.error_message or "").lower()
    # 未进入 ingestion
    assert service.calls == []


def test_run_row_mapping_error_fail_closed() -> None:
    store = FakeTaskStateStore()
    reader = FakeReader(
        records=[_make_record("expert:1:profile_summary:1:0")],
        errors=[
            MySQLReadError(
                row_id=10,
                vector_doc_id="expert:1:profile_summary:9:0",
                code="bad_json",
                message="json 解析失败",
            )
        ],
    )
    service = FakeIngestionService()
    runner = _make_runner(store, reader, service)

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=1,
    )
    final = runner.run(task.task_id)

    assert final.status == "failed"
    assert "行映射" in (final.error_message or "")
    assert service.calls == []


# ── 4. Task State 状态变化时序 ──


def test_run_state_transition_order() -> None:
    store = FakeTaskStateStore()
    records = [_make_record("expert:1:profile_summary:1:0")]
    reader = FakeReader(records)
    service = FakeIngestionService()
    runner = _make_runner(store, reader, service)

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=1,
    )
    runner.run(task.task_id)

    # 时序：created → running → create_items → ingest → update_item → completed
    assert store.calls[0] == "create_task"
    assert "update_task:running" in store.calls
    running_idx = store.calls.index("update_task:running")
    assert "create_items" in store.calls
    assert store.calls.index("create_items") > running_idx
    assert "update_item:success" in store.calls
    assert "update_task:completed" in store.calls
    completed_idx = store.calls.index("update_task:completed")
    assert completed_idx > store.calls.index("update_item:success")


# ── 5. 前置条件 ──


def test_run_task_not_found_raises() -> None:
    store = FakeTaskStateStore()
    runner = _make_runner(store, FakeReader(), FakeIngestionService())

    with pytest.raises(PipelineRunnerError, match="任务不存在"):
        runner.run("task-not-exist")


def test_run_non_created_task_rejected() -> None:
    store = FakeTaskStateStore()
    runner = _make_runner(store, FakeReader(), FakeIngestionService())

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=1,
    )
    # 直接模拟任务已 running
    store.update_task_status(task.task_id, status="running")

    with pytest.raises(PipelineRunnerError, match="仅 created 可运行"):
        runner.run(task.task_id)

    # 状态未被修改（仍为 running）
    assert store.get_task(task.task_id).status == "running"


# ── 6. 空输入 ──


def test_run_empty_records_completes() -> None:
    store = FakeTaskStateStore()
    reader = FakeReader([])
    service = FakeIngestionService()
    runner = _make_runner(store, reader, service)

    task = store.create_task(
        source="full_batch",
        collection_name="expert_candidates_v14_contract_dev",
        total_count=0,
    )
    final = runner.run(task.task_id)

    assert final.status == "completed"
    assert final.success_count == 0
    assert final.skipped_count == 0
    # ingestion 仍被调用（返回空结果语义）
    assert len(service.calls) == 1
    # 无条目创建
    assert store.list_items(task.task_id) == []