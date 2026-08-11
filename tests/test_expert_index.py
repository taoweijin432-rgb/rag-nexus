"""专家索引核心规则单元测试。

测试全部使用虚构资料，不连接真实数据库。覆盖 PII、确定性哈希、三类 Chunk、
超长同源切分与契约导出，确保修改模板时能立即发现破坏性变化。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.exporter import export_jsonl
from expert_index.models import ExpertAggregate, ExpertSource, EmploymentSource, ProjectSource
from expert_index.normalize import clean_text, normalize_aggregate, pii_hits, sha256_value
from expert_index.source import (
    _employment_status,
    _parse_employment_description,
    _parse_project_description,
)
from expert_index.test_fixture import PROJECT_FIXTURES, TEST_EXPERT_SERIAL, WORK_FIXTURES
from expert_index.validators import validate_chunks


def _settings() -> Settings:
    """构造不会实际连接的占位配置，供纯领域测试使用。"""

    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _aggregate(long_employment: bool = False) -> ExpertAggregate:
    """返回含三类来源的虚构专家快照。"""

    when = datetime(2026, 8, 11, 10, 0, 0)
    profile = ExpertSource(
        expert_id=24,
        serial_number="EXP000024",
        company="京东零售集团有限公司",
        role="运营负责人",
        background="8年电商运营经验，邮箱 fake@example.com，手机号 13800138000。",
        education=None,
        work_years=8,
        updated_at=when,
        created_at=when,
    )
    responsibility = "负责品类规划、供应商管理，年度GMV达到5000万元。"
    if long_employment:
        responsibility = responsibility * 80
    employment = EmploymentSource(
        expert_id=24,
        source_table="experts_work_experience",
        source_record_id="1001",
        source_item_key="1001",
        company="京东零售集团有限公司",
        role="运营负责人",
        start_date=datetime(2018, 3, 1),
        end_date=datetime(2021, 6, 1),
        description=responsibility,
        responsibilities="负责酒水采销。",
        achievements="年度GMV达到5000万元。",
        is_current=False,
        updated_at=when,
        created_at=when,
    )
    project = ProjectSource(
        expert_id=24,
        source_table="experts_project_experience",
        source_record_id="2001",
        source_item_key="2001",
        project_name="组合投放优化",
        company="京东零售集团有限公司",
        project_role="项目负责人",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 1),
        description="提升广告投放效率。",
        responsibilities="按人群和素材分层测试，线索转化率提升18%。",
        results="线索转化率提升18%。",
        updated_at=when,
        created_at=when,
    )
    return ExpertAggregate(profile, (employment,), (project,))


class CleanerTest(unittest.TestCase):
    """验证禁止信息被删除，同时业务数字保持不变。"""

    def test_clean_text_removes_pii_and_preserves_metrics(self) -> None:
        cleaned = clean_text("联系 fake@example.com 或 13800138000，GMV 5000万元，提升18%。")
        self.assertEqual({}, pii_hits(cleaned))
        self.assertIn("5000万元", cleaned)
        self.assertIn("18%", cleaned)
        self.assertNotIn("fake@example.com", cleaned)


class SourceMappingTest(unittest.TestCase):
    """锁定正确新表的状态枚举和来源表名，防止以后回退到旧表。"""

    def test_employment_status_enum(self) -> None:
        self.assertIs(_employment_status(1), True)
        self.assertIs(_employment_status(2), False)
        self.assertIsNone(_employment_status(0))
        self.assertIsNone(_employment_status(None))

    def test_fixture_serial_fits_source_column(self) -> None:
        self.assertEqual(19, len(TEST_EXPERT_SERIAL))
        self.assertEqual(3, len(WORK_FIXTURES))
        self.assertEqual(3, len(PROJECT_FIXTURES))
        self.assertEqual(3, len({(item.company, item.position) for item in WORK_FIXTURES}))
        self.assertEqual(3, len({item.title for item in PROJECT_FIXTURES}))
        for item in WORK_FIXTURES:
            responsibilities, achievements = _parse_employment_description(item.description)
            self.assertTrue(responsibilities)
            self.assertTrue(achievements)
        for item in PROJECT_FIXTURES:
            self.assertTrue(all(_parse_project_description(item.description)))

    def test_fixture_uses_correct_source_tables(self) -> None:
        aggregate = _aggregate()
        self.assertEqual("experts_work_experience", aggregate.employments[0].source_table)
        self.assertEqual("experts_project_experience", aggregate.projects[0].source_table)

    def test_labeled_descriptions_are_source_parsed(self) -> None:
        responsibilities, achievements = _parse_employment_description(
            "工作职责：负责知识库建设。\n代表性成果：检索命中率提升18%。"
        )
        self.assertEqual("负责知识库建设。", responsibilities)
        self.assertEqual("检索命中率提升18%。", achievements)
        role, objective, actions, results = _parse_project_description(
            "项目角色：项目负责人\n项目目标：建设智能检索。\n"
            "关键动作：设计Chunk策略。\n项目结果：召回率提升17%。"
        )
        self.assertEqual("项目负责人", role)
        self.assertEqual("建设智能检索。", objective)
        self.assertEqual("设计Chunk策略。", actions)
        self.assertEqual("召回率提升17%。", results)


class ChunkFactoryTest(unittest.TestCase):
    """验证三类 Chunk、稳定 ID、Same Segment 与长度上限。"""

    def test_builds_three_chunk_types(self) -> None:
        settings = _settings()
        profile, employments, projects = normalize_aggregate(
            _aggregate(), data_version="20260811_01", build_batch_id=1
        )
        chunks = ChunkFactory(settings).build_all(profile, employments, projects)
        validate_chunks(chunks)
        self.assertEqual({"profile_summary", "employment", "project"}, {c.chunk_type for c in chunks})
        self.assertTrue(all(c.content_hash == sha256_value(c.chunk_text) for c in chunks))
        self.assertTrue(all(c.chunk_id.startswith("expert:24:") for c in chunks))
        self.assertNotIn("fake@example.com", "\n".join(c.chunk_text for c in chunks))

    def test_long_employment_splits_but_keeps_source_key(self) -> None:
        settings = _settings()
        profile, employments, projects = normalize_aggregate(
            _aggregate(long_employment=True), data_version="20260811_01", build_batch_id=1
        )
        chunks = ChunkFactory(settings).build_all(profile, employments, projects)
        employment = [chunk for chunk in chunks if chunk.chunk_type == "employment"]
        validate_chunks(chunks)
        self.assertGreater(len(employment), 1)
        self.assertEqual({"1001"}, {chunk.source_item_key for chunk in employment})
        self.assertTrue(all(chunk.char_count <= 900 for chunk in employment))
        self.assertEqual(list(range(len(employment))), [chunk.chunk_index for chunk in employment])

    def test_unknown_employment_status_becomes_partial(self) -> None:
        """is_employed=0 对应 None，不能作为当前/历史任职的硬证据。"""

        settings = _settings()
        aggregate = _aggregate()
        unknown = replace(aggregate.employments[0], is_current=None)
        aggregate = replace(aggregate, employments=(unknown,))
        profile, employments, projects = normalize_aggregate(
            aggregate, data_version="20260811_02", build_batch_id=1
        )
        chunks = ChunkFactory(settings).build_all(profile, employments, projects)
        employment = [chunk for chunk in chunks if chunk.chunk_type == "employment"]
        validate_chunks(chunks)
        self.assertTrue(employment)
        self.assertEqual({"partial"}, {chunk.verification_status for chunk in employment})


class ExporterTest(unittest.TestCase):
    """验证 MySQL 行到固定 JSONL 契约的映射。"""

    def test_export_jsonl(self) -> None:
        now = datetime(2026, 8, 11, 10, 0, 0)
        text = "[专家能力概述]\n核心背景：虚构测试资料"
        row = {
            "chunk_id": "expert:24:profile_summary:24:0",
            "expert_id": 24,
            "serial_number": "EXP000024",
            "chunk_type": "profile_summary",
            "chunk_subtype": None,
            "chunk_index": 0,
            "chunk_total": 1,
            "chunk_text": text,
            "char_count": len(text),
            "content_hash": sha256_value(text),
            "metadata_json": json.dumps({"industry_codes": [], "topic_codes": []}),
            "source_table": "experts",
            "source_record_id": "24",
            "source_item_key": "24",
            "source_field": "background+company+job_name",
            "source_field_id": "experts.background",
            "source_version": "2026-08-11T10:00:00+08:00",
            "source_updated_at": now,
            "source_hash": "a" * 64,
            "chunk_policy_version": "1.2",
            "metadata_schema_version": "1.0",
            "dictionary_version": "2026.08",
            "data_version": "20260811_01",
            "is_active": 1,
            "updated_at": now,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            self.assertEqual(1, export_jsonl([row], path, "Asia/Shanghai"))
            contract = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("1.0.0", contract["contract_version"])
        self.assertEqual(text, contract["chunk_text"])
        self.assertEqual("20260811_01", contract["versions"]["data_version"])


if __name__ == "__main__":
    unittest.main()
