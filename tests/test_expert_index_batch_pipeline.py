"""全量 Pipeline 编排、批次记录和生命周期测试。

测试全部使用虚构 Source 与内存 Repository，不连接 MySQL。它验证 Pipeline 的调用
顺序、分页循环、数量汇总、专家终态和批次状态。Metadata Hash 的数据库行重算由
Repository 定向测试覆盖。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from expert_contracts import ExpertChunkEmbeddingInput
from expert_index.cli import build_parser
from expert_index.config import DatabaseConfig, Settings
from expert_index.models import ExpertAggregate, ExpertCandidate, ExpertSource
from expert_index.pipeline import ExpertIndexPipeline
from expert_index.repository import BatchItemSummary, FullBatchCheckpoint


def _settings() -> Settings:
    """构造不会实际建立连接的流水线配置。"""

    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _aggregate(expert_id: int, *, chunkable: bool = True) -> ExpertAggregate:
    """创建只含主档的虚构专家；chunkable=False 用于模拟无可生成内容。"""

    when = datetime(2026, 8, 14, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    profile = ExpertSource(
        id=expert_id,
        serial_number=f"TEST{expert_id:06d}",
        company="虚构批量测试公司" if chunkable else None,
        job_status=1,
        job_name="测试负责人" if chunkable else None,
        contact_status=3,
        doubtful_flag=1,
        abroad_flag=2,
        background="仅用于批量流程单元测试的虚构背景。" if chunkable else None,
        work_years=8,
        updated_at=when,
        created_at=when,
        deleted_at=None,
    )
    return ExpertAggregate(profile, (), (), ())


class FakeSource:
    """按内存候选列表模拟固定上界、Keyset 分页和单专家读取。"""

    def __init__(
        self,
        candidates: list[ExpertCandidate],
        *,
        failed_ids: set[int] | None = None,
        empty_ids: set[int] | None = None,
    ) -> None:
        self.candidates = candidates
        self.failed_ids = failed_ids or set()
        self.empty_ids = empty_ids or set()
        self.upper_bound_calls = 0
        self.upper_bound_limits: list[int | None] = []
        self.page_calls: list[tuple[int, int, int]] = []
        self.candidate_calls: list[int] = []
        self.loaded_ids: list[int] = []

    def get_full_scan_upper_bound(self, *, max_experts: int | None = None) -> int:
        self.upper_bound_calls += 1
        self.upper_bound_limits.append(max_experts)
        candidates = sorted(self.candidates, key=lambda candidate: candidate.id)
        scoped = candidates if max_experts is None else candidates[:max_experts]
        return max((candidate.id for candidate in scoped), default=0)

    def load_full_scan_candidates(
        self,
        *,
        after_id: int,
        upper_bound_id: int,
        limit: int,
    ) -> list[ExpertCandidate]:
        self.page_calls.append((after_id, upper_bound_id, limit))
        return [
            candidate
            for candidate in self.candidates
            if after_id < candidate.id <= upper_bound_id
        ][:limit]

    def load_expert(self, expert_id: int) -> ExpertAggregate:
        self.loaded_ids.append(expert_id)
        if expert_id in self.failed_ids:
            raise RuntimeError("虚构 Source 读取失败")
        return _aggregate(expert_id, chunkable=expert_id not in self.empty_ids)

    def load_expert_candidate(self, expert_id: int) -> ExpertCandidate:
        self.candidate_calls.append(expert_id)
        return next(candidate for candidate in self.candidates if candidate.id == expert_id)


class FakeRepository:
    """记录 Pipeline 的批次和专家事务调用，不执行任何 SQL。"""

    def __init__(self) -> None:
        self.lock_modes: list[str] = []
        self.start_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.finish_calls: list[tuple[int, dict[str, Any]]] = []
        self.events: list[tuple[str, Any]] = []
        self.progress_calls: list[dict[str, Any]] = []
        self.resume_calls: list[int] = []
        self.resume_checkpoint: FullBatchCheckpoint | None = None
        self.fail_finish_item_ids: set[int] = set()
        self.partial_retry_checkpoint: FullBatchCheckpoint | None = None
        self.retryable_ids: list[int] = []
        self.retry_item_calls: list[tuple[int, int, int]] = []
        self.retry_summary: BatchItemSummary | None = None

    @contextmanager
    def build_lock(self, run_mode: str) -> Iterator[None]:
        """模拟 Repository 全局锁，并记录 Pipeline 是否覆盖整个入口。"""

        self.lock_modes.append(run_mode)
        yield

    def start_batch(self, batch_no: str, run_mode: str, **kwargs: Any) -> int:
        self.start_calls.append((batch_no, run_mode, kwargs))
        return 701

    def resume_batch(self, batch_id: int) -> FullBatchCheckpoint:
        self.resume_calls.append(batch_id)
        if self.resume_checkpoint is None:
            raise AssertionError("测试未配置恢复检查点")
        return self.resume_checkpoint

    def load_partial_batch_for_retry(self, batch_id: int) -> FullBatchCheckpoint:
        if self.partial_retry_checkpoint is None:
            raise AssertionError("测试未配置 partial 重试检查点")
        assert batch_id == self.partial_retry_checkpoint.batch_id
        return self.partial_retry_checkpoint

    def load_retryable_batch_item_ids(
        self,
        batch_id: int,
        *,
        after_expert_id: int,
        limit: int,
    ) -> list[int]:
        self.retry_item_calls.append((batch_id, after_expert_id, limit))
        return [
            expert_id
            for expert_id in self.retryable_ids
            if expert_id > after_expert_id
        ][:limit]

    def summarize_batch_items(self, batch_id: int) -> BatchItemSummary:
        assert batch_id == 701
        if self.retry_summary is None:
            raise AssertionError("测试未配置重试后的明细汇总")
        return self.retry_summary

    def finish_batch(self, batch_id: int, **kwargs: Any) -> None:
        self.finish_calls.append((batch_id, kwargs))

    def prepare_batch_items(self, batch_id: int, expert_ids: list[int]) -> None:
        self.events.append(("prepare", (batch_id, expert_ids)))

    def mark_batch_item_running(self, batch_id: int, expert_id: int) -> None:
        self.events.append(("running", (batch_id, expert_id)))

    def finish_batch_item(self, batch_id: int, expert_id: int, **kwargs: Any) -> None:
        if expert_id in self.fail_finish_item_ids:
            raise RuntimeError("虚构批次明细写入失败")
        self.events.append(("finished", (batch_id, expert_id, kwargs)))

    def update_batch_progress(self, batch_id: int, **kwargs: Any) -> None:
        self.events.append(("progress", (batch_id, kwargs["cursor_id"])))
        self.progress_calls.append(kwargs)


class FakeVectorSync:
    """记录 Pipeline 交付的正式 Contract 快照。"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, list[ExpertChunkEmbeddingInput]]] = []

    def sync_expert(
        self,
        expert_id: int,
        records: list[ExpertChunkEmbeddingInput],
    ) -> object:
        self.calls.append((expert_id, list(records)))
        return object()


def _pipeline(
    source: FakeSource,
    repository: FakeRepository,
    vector_sync: FakeVectorSync | None = None,
) -> ExpertIndexPipeline:
    """通过依赖注入创建不连接数据库的真实 Pipeline。"""

    return ExpertIndexPipeline(
        _settings(),
        source=source,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        vector_sync=vector_sync or FakeVectorSync(),
    )


def _partial_retry_checkpoint() -> FullBatchCheckpoint:
    """构造已经完成固定范围扫描的 partial 批次检查点。"""

    return FullBatchCheckpoint(
        batch_id=701,
        batch_no="partial_retry_batch",
        data_version="20260818_06",
        cursor_id=30,
        upper_bound_id=30,
        candidate_count=3,
        success_count=1,
        failed_count=2,
        rejected_count=0,
        deactivated_count=0,
        extracted_count=1,
        stage_count=0,
        chunk_count=1,
    )


def test_build_single_expert_reuses_core_and_keeps_existing_result() -> None:
    """重构后单专家入口仍创建 single 批次并返回原有数量结构。"""

    source = FakeSource([ExpertCandidate(id=24, deleted_at=None)])
    repository = FakeRepository()

    result = _pipeline(source, repository).build_single_expert(
        24,
        data_version="20260814_01",
        batch_no="single_pipeline_refactor_test",
    )

    assert result.batch_id == 701
    assert result.expert_id == 24
    assert result.extracted_count == 1
    assert result.stage_count == 0
    assert result.chunk_count == 1
    assert result.partial_chunk_count == 0
    assert repository.lock_modes == ["single"]
    assert repository.start_calls == [
        ("single_pipeline_refactor_test", "single", {})
    ]
    assert repository.finish_calls[-1][1]["status"] == "success"


def test_build_single_expert_passes_strict_contract_snapshot_to_vector_sync() -> None:
    """内存标准化生成的 Chunk 应通过 Contract 直接交给下游。"""

    source = FakeSource([ExpertCandidate(id=25, deleted_at=None)])
    repository = FakeRepository()
    vector_sync = FakeVectorSync()

    result = _pipeline(source, repository, vector_sync).build_single_expert(
        25,
        data_version="20260818_01",
        batch_no="single_vector_sync_test",
    )

    assert result.chunk_count == 1
    assert len(vector_sync.calls) == 1
    expert_id, records = vector_sync.calls[0]
    assert expert_id == 25
    assert len(records) == 1
    assert isinstance(records[0], ExpertChunkEmbeddingInput)
    assert records[0].contract_version == "4.0.0"
    assert records[0].vector_doc_id.startswith("expert:25:profile_summary:")


def test_build_expert_batch_paginates_and_continues_after_one_expert_fails() -> None:
    """一位 active 专家失败后继续处理下一位，并把全量批次汇总为 partial。"""

    source = FakeSource(
        [
            ExpertCandidate(id=10, deleted_at=None),
            ExpertCandidate(id=20, deleted_at=None),
            ExpertCandidate(id=30, deleted_at=None),
        ],
        failed_ids={20},
    )
    repository = FakeRepository()

    result = _pipeline(source, repository).build_expert_batch(
        data_version="20260814_02",
        page_size=2,
        batch_no="full_pipeline_partial_test",
    )

    assert result.status == "partial"
    assert result.upper_bound_id == 30
    assert result.cursor_id == 30
    assert result.candidate_count == 3
    assert result.success_count == 2
    assert result.failed_count == 1
    assert result.extracted_count == 2
    assert result.stage_count == 0
    assert result.chunk_count == 2
    assert source.page_calls == [(0, 30, 2), (20, 30, 2)]
    assert source.loaded_ids == [10, 20, 30]
    assert repository.start_calls[0][2] == {
        "data_version": "20260814_02",
        "upper_bound_id": 30,
    }
    assert [call["cursor_id"] for call in repository.progress_calls] == [20, 30]
    finish = repository.finish_calls[-1][1]
    assert finish["status"] == "partial"
    assert finish["error_summary"]["failed_expert_count"] == 1
    assert finish["error_summary"]["sample_errors"] == [
        {"expert_id": 20, "error_type": "RuntimeError"}
    ]


def test_build_expert_batch_limits_scope_to_first_n_candidates() -> None:
    """max_experts 应在分页前收缩固定上界，并只处理前 N 位候选。"""

    source = FakeSource(
        [
            ExpertCandidate(id=10, deleted_at=None),
            ExpertCandidate(id=20, deleted_at=None),
            ExpertCandidate(id=30, deleted_at=None),
        ]
    )
    repository = FakeRepository()

    result = _pipeline(source, repository).build_expert_batch(
        data_version="20260818_02",
        page_size=1,
        max_experts=2,
        batch_no="limited_batch_test",
    )

    assert result.status == "success"
    assert result.upper_bound_id == 20
    assert result.cursor_id == 20
    assert result.candidate_count == 2
    assert source.upper_bound_limits == [2]
    assert source.page_calls == [(0, 20, 1), (10, 20, 1)]
    assert source.loaded_ids == [10, 20]
    assert result.stage_count == 0
    assert repository.start_calls[0][2]["upper_bound_id"] == 20


def test_resume_expert_batch_continues_after_last_complete_page_with_original_scope() -> None:
    """续跑应复用原批次范围和计数，只读取可靠游标之后的候选。"""

    source = FakeSource(
        [
            ExpertCandidate(id=10, deleted_at=None),
            ExpertCandidate(id=20, deleted_at=None),
            ExpertCandidate(id=30, deleted_at=None),
            ExpertCandidate(id=40, deleted_at=None),
        ]
    )
    repository = FakeRepository()
    repository.resume_checkpoint = FullBatchCheckpoint(
        batch_id=701,
        batch_no="original_interrupted_batch",
        data_version="20260818_04",
        cursor_id=20,
        upper_bound_id=40,
        candidate_count=2,
        success_count=2,
        failed_count=0,
        rejected_count=0,
        deactivated_count=0,
        extracted_count=2,
        stage_count=0,
        chunk_count=2,
    )

    result = _pipeline(source, repository).resume_expert_batch(701, page_size=1)

    assert result.batch_id == 701
    assert result.batch_no == "original_interrupted_batch"
    assert result.upper_bound_id == 40
    assert result.cursor_id == 40
    assert result.candidate_count == 4
    assert result.success_count == 4
    assert result.extracted_count == 4
    assert result.stage_count == 0
    assert result.chunk_count == 4
    assert repository.lock_modes == ["full"]
    assert repository.resume_calls == [701]
    assert repository.start_calls == []
    assert source.upper_bound_calls == 0
    assert source.page_calls == [(20, 40, 1), (30, 40, 1)]
    assert source.loaded_ids == [30, 40]


def test_retry_failed_batch_experts_only_reprocesses_failures_and_finishes_success() -> None:
    """失败专家全部修复后，应按明细重汇总并把原 partial 批次改为 success。"""

    source = FakeSource(
        [
            ExpertCandidate(id=10, deleted_at=None),
            ExpertCandidate(id=20, deleted_at=None),
            ExpertCandidate(id=30, deleted_at=None),
        ]
    )
    repository = FakeRepository()
    repository.partial_retry_checkpoint = _partial_retry_checkpoint()
    repository.retryable_ids = [20, 30]
    repository.retry_summary = BatchItemSummary(
        candidate_count=3,
        success_count=3,
        failed_count=0,
        rejected_count=0,
        deactivated_count=0,
        pending_count=0,
        running_count=0,
        extracted_count=3,
        stage_count=0,
        chunk_count=3,
    )

    result = _pipeline(source, repository).retry_failed_batch_experts(
        701,
        page_size=1,
    )

    assert result.batch_id == 701
    assert result.batch_no == "partial_retry_batch"
    assert result.status == "success"
    assert result.candidate_count == 3
    assert result.success_count == 3
    assert result.failed_count == 0
    assert source.upper_bound_calls == 0
    assert source.page_calls == []
    assert source.candidate_calls == [20, 30]
    assert source.loaded_ids == [20, 30]
    assert repository.retry_item_calls == [
        (701, 0, 1),
        (701, 20, 1),
        (701, 30, 1),
    ]
    assert repository.finish_calls[-1][1]["status"] == "success"
    assert repository.finish_calls[-1][1]["error_summary"] is None


def test_retry_failed_batch_experts_attempts_each_item_once_and_remains_partial() -> None:
    """专家再次失败时本次只尝试一次，并根据明细汇总保持 partial。"""

    source = FakeSource(
        [
            ExpertCandidate(id=10, deleted_at=None),
            ExpertCandidate(id=20, deleted_at=None),
        ],
        failed_ids={20},
    )
    repository = FakeRepository()
    repository.partial_retry_checkpoint = FullBatchCheckpoint(
        batch_id=701,
        batch_no="partial_retry_still_failed",
        data_version="20260818_06",
        cursor_id=20,
        upper_bound_id=20,
        candidate_count=2,
        success_count=1,
        failed_count=1,
        rejected_count=0,
        deactivated_count=0,
        extracted_count=1,
        stage_count=0,
        chunk_count=1,
    )
    repository.retryable_ids = [20]
    repository.retry_summary = BatchItemSummary(
        candidate_count=2,
        success_count=1,
        failed_count=1,
        rejected_count=0,
        deactivated_count=0,
        pending_count=0,
        running_count=0,
        extracted_count=1,
        stage_count=0,
        chunk_count=1,
    )

    result = _pipeline(source, repository).retry_failed_batch_experts(
        701,
        page_size=50,
    )

    assert result.status == "partial"
    assert result.failed_count == 1
    assert source.candidate_calls == [20]
    assert source.loaded_ids == [20]
    assert repository.retry_item_calls == [(701, 0, 50), (701, 20, 50)]
    failed_item = [event for event in repository.events if event[0] == "finished"][-1]
    assert failed_item[1][2]["status"] == "failed"
    finish = repository.finish_calls[-1][1]
    assert finish["status"] == "partial"
    assert finish["error_summary"]["sample_errors"] == [
        {"expert_id": 20, "error_type": "RuntimeError"}
    ]


def test_build_expert_batch_deactivates_deleted_candidate_without_loading_source() -> None:
    """deleted 专家不读取正文，完成历史停用后记为 deactivated。"""

    deleted_at = datetime(2026, 8, 14, 12, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    source = FakeSource([ExpertCandidate(id=40, deleted_at=deleted_at)])
    repository = FakeRepository()
    vector_sync = FakeVectorSync()

    result = _pipeline(source, repository, vector_sync).build_expert_batch(
        data_version="20260814_03",
        batch_no="full_deleted_pending_test",
    )

    assert result.status == "success"
    assert result.failed_count == 0
    assert result.deactivated_count == 1
    assert source.loaded_ids == []
    assert result.stage_count == 0
    assert vector_sync.calls == [(40, [])]
    finished = [event for event in repository.events if event[0] == "finished"]
    assert finished[0][1][2]["status"] == "deactivated"
    assert finished[0][1][2]["result_code"] == "expert_deleted"


def test_build_expert_batch_rejects_no_content_and_continues_next_expert() -> None:
    """无内容专家用空快照停用旧 Chunk 并记为 rejected，不算程序失败。"""

    source = FakeSource(
        [
            ExpertCandidate(id=50, deleted_at=None),
            ExpertCandidate(id=60, deleted_at=None),
        ],
        empty_ids={50},
    )
    repository = FakeRepository()
    vector_sync = FakeVectorSync()

    result = _pipeline(source, repository, vector_sync).build_expert_batch(
        data_version="20260814_04",
        page_size=1,
        batch_no="full_empty_pending_test",
    )

    assert result.status == "success"
    assert result.success_count == 1
    assert result.failed_count == 0
    assert result.rejected_count == 1
    assert result.extracted_count == 2
    assert result.stage_count == 0
    assert vector_sync.calls[0] == (50, [])
    assert len(vector_sync.calls[1][1]) == 1
    statuses = [
        event[1][2]["status"]
        for event in repository.events
        if event[0] == "finished"
    ]
    assert statuses == ["rejected", "success"]


def test_build_expert_batch_marks_batch_failed_when_candidate_page_is_invalid() -> None:
    """候选页倒序属于批次级契约错误，不能靠跳过单个专家继续运行。"""

    source = FakeSource(
        [
            ExpertCandidate(id=80, deleted_at=None),
            ExpertCandidate(id=70, deleted_at=None),
        ]
    )
    repository = FakeRepository()

    with pytest.raises(ValueError, match="严格升序"):
        _pipeline(source, repository).build_expert_batch(
            data_version="20260814_05",
            batch_no="full_invalid_page_test",
        )

    assert repository.finish_calls[-1][1]["status"] == "failed"


def test_failed_page_keeps_last_complete_page_counts_for_safe_resume() -> None:
    """页中途发生编排错误时，批次主表不能写入该页的部分内存计数。"""

    source = FakeSource(
        [
            ExpertCandidate(id=10, deleted_at=None),
            ExpertCandidate(id=20, deleted_at=None),
        ]
    )
    repository = FakeRepository()
    repository.fail_finish_item_ids = {20}

    with pytest.raises(RuntimeError, match="批次明细写入失败"):
        _pipeline(source, repository).build_expert_batch(
            data_version="20260818_05",
            page_size=2,
            batch_no="failed_mid_page_batch",
        )

    failed_finish = repository.finish_calls[-1][1]
    assert failed_finish["status"] == "failed"
    assert failed_finish["cursor_id"] == 0
    assert failed_finish["candidate_count"] == 0
    assert failed_finish["success_count"] == 0
    assert failed_finish["extracted_count"] == 0
    assert failed_finish["stage_count"] == 0
    assert failed_finish["chunk_count"] == 0


@pytest.mark.parametrize("page_size", [0, 1001])
def test_build_expert_batch_rejects_invalid_page_size_before_source_access(
    page_size: int,
) -> None:
    """无效页大小必须在固定上界查询和批次创建之前失败。"""

    source = FakeSource([ExpertCandidate(id=1, deleted_at=None)])
    repository = FakeRepository()

    with pytest.raises(ValueError, match="page_size"):
        _pipeline(source, repository).build_expert_batch(
            data_version="20260814_06",
            page_size=page_size,
        )

    assert source.upper_bound_calls == 0
    assert repository.start_calls == []


@pytest.mark.parametrize("max_experts", [0, -1, True, 1.5])
def test_build_expert_batch_rejects_invalid_max_experts_before_source_access(
    max_experts: object,
) -> None:
    """无效范围限制必须在查询上界和创建批次前失败。"""

    source = FakeSource([ExpertCandidate(id=1, deleted_at=None)])
    repository = FakeRepository()

    with pytest.raises(ValueError, match="max_experts"):
        _pipeline(source, repository).build_expert_batch(
            data_version="20260818_03",
            max_experts=max_experts,  # type: ignore[arg-type]
        )

    assert source.upper_bound_calls == 0
    assert repository.start_calls == []


def test_cli_exposes_full_batch_entry_with_explicit_page_size() -> None:
    """build-batch 命令应把全量版本、页大小和批次号交给 Pipeline。"""

    args = build_parser().parse_args(
        [
            "build-batch",
            "--data-version",
            "20260814_07",
            "--page-size",
            "300",
            "--max-experts",
            "500",
            "--batch-no",
            "full_cli_test",
        ]
    )

    assert args.command == "build-batch"
    assert args.data_version == "20260814_07"
    assert args.page_size == 300
    assert args.max_experts == 500
    assert args.batch_no == "full_cli_test"
    assert not hasattr(args, "vectorize")


def test_cli_exposes_resume_batch_without_scope_overrides() -> None:
    """续跑命令只接收原批次 ID 和本次分页大小。"""

    args = build_parser().parse_args(
        ["resume-batch", "--batch-id", "701", "--page-size", "50"]
    )

    assert args.command == "resume-batch"
    assert args.batch_id == 701
    assert args.page_size == 50
    assert not hasattr(args, "data_version")
    assert not hasattr(args, "max_experts")
    assert not hasattr(args, "batch_no")


def test_cli_exposes_partial_batch_failed_item_retry() -> None:
    """retry-batch 只接收原批次 ID 和失败明细分页大小。"""

    args = build_parser().parse_args(
        ["retry-batch", "--batch-id", "701", "--page-size", "25"]
    )

    assert args.command == "retry-batch"
    assert args.batch_id == 701
    assert args.page_size == 25
    assert not hasattr(args, "data_version")
    assert not hasattr(args, "max_experts")
    assert not hasattr(args, "batch_no")
