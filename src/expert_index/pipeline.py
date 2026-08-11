"""单专家端到端编排：Source → Stage → Chunk → Validator → MySQL。

本模块只做流程控制，不重复实现 SQL、清洗或 Chunk 规则。异常时批次会被标记为
failed，且错误摘要不包含原始业务正文；专家派生数据由 Repository 事务回滚。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .chunk_builder import ChunkFactory
from .config import Settings
from .normalize import normalize_aggregate
from .repository import MySQLIndexRepository
from .source import MySQLSourceAdapter
from .validators import validate_chunks


@dataclass(frozen=True)
class BuildResult:
    """可安全打印的运行结果，只含 ID 和数量，不含原文。"""

    batch_id: int
    batch_no: str
    expert_id: int
    extracted_count: int
    stage_count: int
    chunk_count: int
    partial_chunk_count: int


class ExpertIndexPipeline:
    """组装默认生产实现，也允许测试时注入替代 Adapter/Repository。"""

    def __init__(
        self,
        settings: Settings,
        *,
        source: MySQLSourceAdapter | None = None,
        repository: MySQLIndexRepository | None = None,
    ) -> None:
        self.settings = settings
        self.source = source or MySQLSourceAdapter(settings.source_db)
        self.repository = repository or MySQLIndexRepository(settings.index_db)
        self.factory = ChunkFactory(settings)

    def build_one(self, expert_id: int, *, data_version: str, batch_no: str | None = None) -> BuildResult:
        """完整重建一位专家；重复运行会按稳定 chunk_id 幂等 Upsert。"""

        if not data_version.strip():
            raise ValueError("data_version 不能为空")
        actual_batch_no = batch_no or _batch_no(expert_id)
        batch_id = self.repository.start_batch(actual_batch_no, "single")
        extracted_count = stage_count = chunk_count = 0
        try:
            aggregate = self.source.load_expert(expert_id)
            extracted_count = 1 + len(aggregate.employments) + len(aggregate.projects)
            profile, employments, projects = normalize_aggregate(
                aggregate,
                data_version=data_version,
                build_batch_id=batch_id,
            )
            chunks = self.factory.build_all(profile, employments, projects)
            if not chunks:
                raise ValueError(f"专家 {expert_id} 没有可生成的 Chunk")
            validate_chunks(chunks)
            stage_count = self.repository.persist_expert(profile, employments, projects, chunks)
            chunk_count = len(chunks)
            partial_count = sum(chunk.verification_status == "partial" for chunk in chunks)
            self.repository.finish_batch(
                batch_id,
                status="partial" if partial_count else "success",
                extracted_count=extracted_count,
                stage_count=stage_count,
                chunk_count=chunk_count,
                rejected_count=partial_count,
                error_summary={
                    "note": "partial Chunk 已落表，但按契约不会进入正式 Embedding 队列"
                } if partial_count else None,
            )
            return BuildResult(
                batch_id=batch_id,
                batch_no=actual_batch_no,
                expert_id=expert_id,
                extracted_count=extracted_count,
                stage_count=stage_count,
                chunk_count=chunk_count,
                partial_chunk_count=partial_count,
            )
        except Exception as exc:
            # 摘要只记录异常类型与 str(exc)。校验异常已被设计为不携带正文；底层 SQL
            # 异常也不会把绑定参数展开，因此不会把真实资料写入批次表。
            self.repository.finish_batch(
                batch_id,
                status="failed",
                extracted_count=extracted_count,
                stage_count=stage_count,
                chunk_count=chunk_count,
                rejected_count=1,
                error_summary={"error_type": type(exc).__name__, "message": str(exc)[:400]},
            )
            raise


def _batch_no(expert_id: int) -> str:
    """批次号包含专家 ID 和微秒时间，便于重放与排障。"""

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"single_{expert_id}_{timestamp}"

