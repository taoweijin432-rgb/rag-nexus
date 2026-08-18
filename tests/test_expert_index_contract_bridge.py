"""上游 ``ChunkRecord`` 到下游正式 Contract 的无外部依赖测试。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from expert_contracts import ExpertChunkEmbeddingInput
from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.contract_bridge import chunk_record_to_contract
from expert_index.models import ExpertAggregate, ExpertSource
from expert_index.normalize import normalize_aggregate


def _settings() -> Settings:
    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def test_chunk_record_maps_to_strict_contract_v400() -> None:
    """Bridge 应补齐嵌套结构、Decimal 和带时区时间，不改 ID/正文/hash。"""

    source_time = datetime(2026, 8, 18, 9, 30)  # noqa: DTZ001 - 模拟 MySQL naive DATETIME
    aggregate = ExpertAggregate(
        profile=ExpertSource(
            id=700001,
            serial_number="TEST700001",
            company="虚构桥接测试公司",
            job_status=1,
            job_name="项目经理",
            contact_status=3,
            doubtful_flag=1,
            abroad_flag=2,
            background="仅用于 Contract Bridge 单元测试的虚构背景。",
            work_years=8.5,
            updated_at=source_time,
            created_at=source_time,
            deleted_at=None,
        ),
        employments=(),
        projects=(),
        educations=(),
    )
    settings = _settings()
    profile, employments, projects = normalize_aggregate(
        aggregate,
        data_version="20260818_01",
        build_batch_id=701,
    )
    chunk = ChunkFactory(settings).build_all(profile, employments, projects)[0]

    contract = chunk_record_to_contract(
        chunk,
        updated_at=datetime(2026, 8, 18, 10, 0),  # noqa: DTZ001 - 验证 Bridge 补时区
        timezone_name=settings.timezone,
    )

    assert isinstance(contract, ExpertChunkEmbeddingInput)
    assert contract.contract_version == "4.0.0"
    assert contract.vector_doc_id == chunk.vector_doc_id
    assert contract.chunk_text == chunk.chunk_text
    assert contract.content_hash == chunk.content_hash
    assert contract.metadata_hash == chunk.metadata_hash
    assert contract.metadata.work_years == Decimal("8.5")
    assert not isinstance(contract.metadata.work_years, float)
    assert contract.source.source_updated_at == "2026-08-18T09:30:00.000000+08:00"
    assert contract.updated_at == "2026-08-18T10:00:00.000000+08:00"
    assert contract.metadata.industry_codes == list(chunk.industry_codes)
    assert "build_batch_id" not in contract.model_dump()
    assert "embedding_model_alias" not in contract.model_dump()
