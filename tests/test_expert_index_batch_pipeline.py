"""第三、第四阶段 Pipeline 复用和全量失败隔离测试。

测试全部使用虚构 Source 与内存 Repository，不连接 MySQL。它验证 Pipeline 的调用
顺序、分页循环、数量汇总和批次状态，不测试下一阶段才会实现的批次明细、生命周期
停用、Metadata Hash 重算或并发锁。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from expert_index.config import DatabaseConfig, Settings
from expert_index.models import ExpertAggregate, ExpertCandidate, ExpertSource
from expert_index.pipeline import ExpertIndexPipeline


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
        self.page_calls: list[tuple[int, int, int]] = []
        self.loaded_ids: list[int] = []

    def get_full_scan_upper_bound(self) -> int:
        self.upper_bound_calls += 1
        return max((candidate.id for candidate in self.candidates), default=0)

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


class FakeRepository:
    """记录 Pipeline 的批次和专家事务调用，不执行任何 SQL。"""

    def __init__(self) -> None:
        self.start_calls: list[tuple[str, str]] = []
        self.finish_calls: list[tuple[int, dict[str, Any]]] = []
        self.persisted_expert_ids: list[int] = []

    def start_batch(self, batch_no: str, run_mode: str) -> int:
        self.start_calls.append((batch_no, run_mode))
        return 701

    def finish_batch(self, batch_id: int, **kwargs: Any) -> None:
        self.finish_calls.append((batch_id, kwargs))

    def persist_expert(
        self,
        profile: Any,
        employments: list[Any],
        projects: list[Any],
        chunks: list[Any],
    ) -> int:
        """模拟一位专家事务成功，并返回 Pipeline 约定的 Stage 行数。"""

        self.persisted_expert_ids.append(profile.expert_id)
        assert chunks
        return 1 + len(employments) + len(projects)


def _pipeline(
    source: FakeSource,
    repository: FakeRepository,
) -> ExpertIndexPipeline:
    """通过依赖注入创建不连接数据库的真实 Pipeline。"""

    return ExpertIndexPipeline(
        _settings(),
        source=source,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
    )


def test_build_one_reuses_expert_core_and_keeps_existing_result() -> None:
    """重构后单专家入口仍创建 single 批次并返回原有数量结构。"""

    source = FakeSource([ExpertCandidate(id=24, deleted_at=None)])
    repository = FakeRepository()

    result = _pipeline(source, repository).build_one(
        24,
        data_version="20260814_01",
        batch_no="single_pipeline_refactor_test",
    )

    assert result.batch_id == 701
    assert result.expert_id == 24
    assert result.extracted_count == 1
    assert result.stage_count == 1
    assert result.chunk_count == 1
    assert result.partial_chunk_count == 0
    assert repository.start_calls == [("single_pipeline_refactor_test", "single")]
    assert repository.persisted_expert_ids == [24]
    assert repository.finish_calls[-1][1]["status"] == "success"


def test_build_many_paginates_and_continues_after_one_expert_fails() -> None:
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

    result = _pipeline(source, repository).build_many(
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
    assert result.stage_count == 2
    assert result.chunk_count == 2
    assert source.page_calls == [(0, 30, 2), (20, 30, 2)]
    assert source.loaded_ids == [10, 20, 30]
    assert repository.persisted_expert_ids == [10, 30]
    finish = repository.finish_calls[-1][1]
    assert finish["status"] == "partial"
    assert finish["error_summary"]["failed_expert_count"] == 1
    assert finish["error_summary"]["sample_errors"] == [
        {"expert_id": 20, "error_type": "RuntimeError"}
    ]


def test_build_many_currently_isolates_deleted_candidate_for_next_stage() -> None:
    """本阶段不误读 deleted 专家；实际停用与 deactivated 计数留给下一阶段。"""

    deleted_at = datetime(2026, 8, 14, 12, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    source = FakeSource([ExpertCandidate(id=40, deleted_at=deleted_at)])
    repository = FakeRepository()

    result = _pipeline(source, repository).build_many(
        data_version="20260814_03",
        batch_no="full_deleted_pending_test",
    )

    assert result.status == "partial"
    assert result.failed_count == 1
    assert result.deactivated_count == 0
    assert source.loaded_ids == []
    assert repository.persisted_expert_ids == []
    error = repository.finish_calls[-1][1]["error_summary"]["sample_errors"][0]
    assert error["error_type"] == "DeletedExpertLifecyclePendingError"


def test_build_many_currently_isolates_no_content_for_next_stage() -> None:
    """无内容专家不会写空 Chunk；rejected 与历史停用计数留给下一阶段。"""

    source = FakeSource(
        [
            ExpertCandidate(id=50, deleted_at=None),
            ExpertCandidate(id=60, deleted_at=None),
        ],
        empty_ids={50},
    )
    repository = FakeRepository()

    result = _pipeline(source, repository).build_many(
        data_version="20260814_04",
        page_size=1,
        batch_no="full_empty_pending_test",
    )

    assert result.status == "partial"
    assert result.success_count == 1
    assert result.failed_count == 1
    assert result.rejected_count == 0
    assert result.extracted_count == 2
    assert repository.persisted_expert_ids == [60]
    errors = repository.finish_calls[-1][1]["error_summary"]["sample_errors"]
    assert errors == [{"expert_id": 50, "error_type": "NoChunkableContentError"}]


def test_build_many_marks_batch_failed_when_candidate_page_is_invalid() -> None:
    """候选页倒序属于批次级契约错误，不能靠跳过单个专家继续运行。"""

    source = FakeSource(
        [
            ExpertCandidate(id=80, deleted_at=None),
            ExpertCandidate(id=70, deleted_at=None),
        ]
    )
    repository = FakeRepository()

    with pytest.raises(ValueError, match="严格升序"):
        _pipeline(source, repository).build_many(
            data_version="20260814_05",
            batch_no="full_invalid_page_test",
        )

    assert repository.persisted_expert_ids == []
    assert repository.finish_calls[-1][1]["status"] == "failed"


@pytest.mark.parametrize("page_size", [0, 1001])
def test_build_many_rejects_invalid_page_size_before_source_access(page_size: int) -> None:
    """无效页大小必须在固定上界查询和批次创建之前失败。"""

    source = FakeSource([ExpertCandidate(id=1, deleted_at=None)])
    repository = FakeRepository()

    with pytest.raises(ValueError, match="page_size"):
        _pipeline(source, repository).build_many(
            data_version="20260814_06",
            page_size=page_size,
        )

    assert source.upper_bound_calls == 0
    assert repository.start_calls == []
