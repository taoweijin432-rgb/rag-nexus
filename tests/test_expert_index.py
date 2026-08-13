"""专家索引核心规则单元测试。

测试全部使用虚构资料，不连接真实数据库。覆盖 PII、确定性哈希、三类 Chunk、
超长同源切分与契约导出，确保修改模板时能立即发现破坏性变化。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.exporter import export_jsonl
from expert_index.models import (
    ChunkRecord,
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
)
from expert_index.normalize import (
    clean_text,
    employment_status,
    normalize_abroad_flag,
    normalize_aggregate,
    normalize_job_status,
    parse_employment_description,
    parse_project_description,
    pii_hits,
    profile_verification_status,
    recommendation_eligibility,
    sha256_value,
)
from expert_index.repository import MySQLIndexRepository
from expert_index.source import MySQLSourceAdapter
from expert_index.test_fixture import (
    EDUCATION_FIXTURES,
    PROJECT_FIXTURES,
    TEST_EXPERT_SERIAL,
    WORK_FIXTURES,
)
from expert_index.validators import validate_chunks


def _settings() -> Settings:
    """构造不会实际连接的占位配置，供纯领域测试使用。"""

    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _aggregate(long_employment: bool = False) -> ExpertAggregate:
    """返回含三类来源的虚构专家快照。"""

    when = datetime(2026, 8, 11, 10, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    profile = ExpertSource(
        id=24,
        serial_number="EXP000024",
        company="京东零售集团有限公司",
        job_status=1,
        job_name="运营负责人",
        contact_status=4,
        doubtful_flag=1,
        abroad_flag=2,
        background="8年电商运营经验，邮箱 fake@example.com，手机号 13800138000。",
        work_years=8,
        updated_at=when,
        created_at=when,
        deleted_at=None,
    )
    responsibility = "负责品类规划、供应商管理，年度GMV达到5000万元。"
    if long_employment:
        responsibility = responsibility * 80
    employment = EmploymentSource(
        id=1001,
        expert_id=24,
        company_name="京东零售集团有限公司",
        position_title="运营负责人",
        start_date=datetime(2018, 3, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        end_date=datetime(2021, 6, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        work_description=(
            f"工作职责：{responsibility}\n"
            "代表性成果：年度GMV达到5000万元。"
        ),
        is_employed=2,
        updated_at=when,
        created_at=when,
        deleted_at=None,
    )
    project = ProjectSource(
        id=2001,
        expert_id=24,
        project_title="组合投放优化",
        company_name="京东零售集团有限公司",
        start_date=datetime(2024, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        end_date=datetime(2024, 6, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        work_description=(
            "项目角色：项目负责人\n"
            "项目目标：提升广告投放效率。\n"
            "关键动作：按人群和素材分层测试，线索转化率提升18%。\n"
            "项目结果：线索转化率提升18%。"
        ),
        updated_at=when,
        created_at=when,
        deleted_at=None,
    )
    education = EducationSource(
        id=3001,
        expert_id=24,
        school_name="华东流程测试大学",
        major="信息管理与信息系统",
        updated_at=when,
        created_at=when,
        deleted_at=None,
    )
    return ExpertAggregate(profile, (employment,), (project,), (education,))


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
        self.assertIs(employment_status(1), True)
        self.assertIs(employment_status(2), False)
        self.assertIsNone(employment_status(0))
        self.assertIsNone(employment_status(None))

    def test_profile_status_enums(self) -> None:
        """锁定四个 experts 标量字段的数据库枚举映射。"""

        self.assertEqual("active", normalize_job_status(1))
        self.assertEqual("former", normalize_job_status(2))
        self.assertIsNone(normalize_job_status(0))
        self.assertIs(recommendation_eligibility(1), False)
        self.assertIs(recommendation_eligibility(2), False)
        self.assertIs(recommendation_eligibility(3), True)
        self.assertIs(recommendation_eligibility(4), True)
        self.assertIsNone(recommendation_eligibility(0))
        self.assertEqual("source_summary", profile_verification_status(1))
        self.assertEqual("rejected", profile_verification_status(2))
        self.assertEqual("rejected", profile_verification_status(0))
        self.assertIs(normalize_abroad_flag(1), True)
        self.assertIs(normalize_abroad_flag(2), False)
        self.assertIsNone(normalize_abroad_flag(0))

    def test_source_model_names_match_mysql_columns(self) -> None:
        """Source 层不得再用 role/description 等派生别名隐藏源列。"""

        self.assertEqual(
            {
                "id", "serial_number", "company", "job_name", "background",
                "job_status", "contact_status", "doubtful_flag", "abroad_flag",
                "work_years", "updated_at", "created_at", "deleted_at",
            },
            {item.name for item in fields(ExpertSource)},
        )
        self.assertEqual(
            {
                "id", "expert_id", "company_name", "position_title", "start_date",
                "end_date", "work_description", "is_employed", "updated_at", "created_at",
                "deleted_at",
            },
            {item.name for item in fields(EmploymentSource)},
        )
        self.assertEqual(
            {
                "id", "expert_id", "project_title", "company_name", "start_date",
                "end_date", "work_description", "updated_at", "created_at",
                "deleted_at",
            },
            {item.name for item in fields(ProjectSource)},
        )
        self.assertEqual(
            {
                "id", "expert_id", "school_name", "major",
                "updated_at", "created_at", "deleted_at",
            },
            {item.name for item in fields(EducationSource)},
        )

    def test_fixture_serial_fits_source_column(self) -> None:
        self.assertEqual(19, len(TEST_EXPERT_SERIAL))
        self.assertEqual(3, len(WORK_FIXTURES))
        self.assertEqual(3, len(PROJECT_FIXTURES))
        self.assertEqual(2, len(EDUCATION_FIXTURES))
        self.assertEqual(3, len({(item.company, item.position) for item in WORK_FIXTURES}))
        self.assertEqual(3, len({item.title for item in PROJECT_FIXTURES}))
        for item in WORK_FIXTURES:
            responsibilities, achievements = parse_employment_description(item.description)
            self.assertTrue(responsibilities)
            self.assertTrue(achievements)
        for item in PROJECT_FIXTURES:
            self.assertTrue(all(parse_project_description(item.description)))

    def test_fixture_uses_correct_source_tables(self) -> None:
        profile, employments, projects = normalize_aggregate(
            _aggregate(), data_version="20260812_01", build_batch_id=1
        )
        self.assertEqual("experts_work_experience", employments[0].source_table)
        self.assertEqual("experts_project_experience", projects[0].source_table)
        self.assertEqual(
            "company_name+position_title+start_date+end_date+work_description+is_employed",
            employments[0].source_field,
        )
        self.assertEqual(
            "project_title+company_name+start_date+end_date+work_description",
            projects[0].source_field,
        )
        self.assertEqual("1001", employments[0].source_field_id)
        self.assertEqual("2001", projects[0].source_field_id)
        self.assertEqual(("华东流程测试大学",), profile.school_names_raw)
        self.assertEqual(("信息管理与信息系统",), profile.major_names_raw)
        self.assertEqual("active", profile.job_status_norm)
        self.assertIs(profile.abroad_flag_norm, False)
        self.assertIs(profile.eligible_for_recommendation, True)
        self.assertEqual("source_summary", profile.verification_status)

    def test_new_month_format_is_yyyymm(self) -> None:
        """新标准不再保留年月之间的连字符。"""

        _, employments, projects = normalize_aggregate(
            _aggregate(), data_version="20260813_01", build_batch_id=1
        )
        self.assertEqual("201803", employments[0].start_month)
        self.assertEqual("202106", employments[0].end_month)
        self.assertEqual("202401", projects[0].start_month)

    def test_deleted_fields_are_present_and_current_snapshot_is_clean(self) -> None:
        """三个经历 Source 都承接 deleted_at；有效查询结果应为 None。"""

        aggregate = _aggregate()
        self.assertIsNone(aggregate.employments[0].deleted_at)
        self.assertIsNone(aggregate.projects[0].deleted_at)
        self.assertIsNone(aggregate.educations[0].deleted_at)

    def test_labeled_descriptions_are_source_parsed(self) -> None:
        responsibilities, achievements = parse_employment_description(
            "工作职责：负责知识库建设。\n代表性成果：检索命中率提升18%。"
        )
        self.assertEqual("负责知识库建设。", responsibilities)
        self.assertEqual("检索命中率提升18%。", achievements)
        role, objective, actions, results = parse_project_description(
            "项目角色：项目负责人\n项目目标：建设智能检索。\n"
            "关键动作：设计Chunk策略。\n项目结果：召回率提升17%。"
        )
        self.assertEqual("项目负责人", role)
        self.assertEqual("建设智能检索。", objective)
        self.assertEqual("设计Chunk策略。", actions)
        self.assertEqual("召回率提升17%。", results)

    def test_all_three_experience_queries_filter_deleted_at(self) -> None:
        """工作、项目、教育三张经历表都只能返回未逻辑删除的当前快照。"""

        class EmptyCursor:
            def __init__(self) -> None:
                self.sql: list[str] = []

            def execute(self, sql: str, params: tuple[object, ...]) -> None:
                self.sql.append(" ".join(sql.split()))

            @staticmethod
            def fetchall() -> tuple[()]:
                return ()

        cursor = EmptyCursor()
        MySQLSourceAdapter._query_employment_rows(cursor, 24)
        MySQLSourceAdapter._load_projects(cursor, 24)
        MySQLSourceAdapter._load_educations(cursor, 24)
        self.assertEqual(3, len(cursor.sql))
        self.assertTrue(all("deleted_at IS NULL" in sql for sql in cursor.sql))
        self.assertTrue(any("experts_education_experience" in sql for sql in cursor.sql))


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
        self.assertTrue(all(c.vector_doc_id.startswith("expert:24:") for c in chunks))
        self.assertTrue(
            all(
                c.vector_doc_id
                == f"expert:{c.expert_id}:{c.chunk_type}:{c.source_field_id}:{c.chunk_index}"
                for c in chunks
            )
        )
        self.assertEqual(
            len(chunks),
            len({c.vector_doc_id for c in chunks}),
        )
        self.assertNotIn("fake@example.com", "\n".join(c.chunk_text for c in chunks))

    def test_final_chunk_model_omits_removed_database_fields(self) -> None:
        """产品删除的五个字段不得继续出现在最终 ChunkRecord 中。"""

        chunk_fields = {item.name for item in fields(ChunkRecord)}
        self.assertTrue(
            {
                "chunk_id",
                "source_item_key",
                "dictionary_version",
                "data_version",
            }.isdisjoint(chunk_fields)
        )

    def test_long_employment_splits_but_keeps_source_key(self) -> None:
        settings = _settings()
        profile, employments, projects = normalize_aggregate(
            _aggregate(long_employment=True), data_version="20260811_01", build_batch_id=1
        )
        chunks = ChunkFactory(settings).build_all(profile, employments, projects)
        employment = [chunk for chunk in chunks if chunk.chunk_type == "employment"]
        validate_chunks(chunks)
        self.assertGreater(len(employment), 1)
        self.assertEqual({"1001"}, {chunk.source_field_id for chunk in employment})
        self.assertTrue(all(chunk.char_count <= 900 for chunk in employment))
        self.assertEqual(list(range(len(employment))), [chunk.chunk_index for chunk in employment])
        self.assertEqual(
            len(employment),
            len({chunk.vector_doc_id for chunk in employment}),
        )

    def test_unknown_employment_status_remains_null(self) -> None:
        """is_employed=0 对应 None；V1.3 不再用 partial 表示解析质量。"""

        settings = _settings()
        aggregate = _aggregate()
        unknown = replace(aggregate.employments[0], is_employed=0)
        aggregate = replace(aggregate, employments=(unknown,))
        profile, employments, projects = normalize_aggregate(
            aggregate, data_version="20260811_02", build_batch_id=1
        )
        chunks = ChunkFactory(settings).build_all(profile, employments, projects)
        employment = [chunk for chunk in chunks if chunk.chunk_type == "employment"]
        validate_chunks(chunks)
        self.assertTrue(employment)
        self.assertEqual({None}, {chunk.is_current for chunk in employment})
        self.assertEqual({"verified"}, {chunk.verification_status for chunk in employment})

    def test_metadata_hash_is_independent_from_content_hash(self) -> None:
        """正文不变、业务 Metadata 变化时，只应改变 metadata_hash。"""

        profile, _, _ = normalize_aggregate(
            _aggregate(), data_version="20260813_01", build_batch_id=1
        )
        original = ChunkFactory(_settings()).build_profile(profile)[0]
        changed = ChunkFactory(_settings()).build_profile(
            replace(profile, school_names_raw=("另一所测试大学",))
        )[0]
        self.assertEqual(original.content_hash, changed.content_hash)
        self.assertNotEqual(original.metadata_hash, changed.metadata_hash)


class RepositoryMappingTest(unittest.TestCase):
    """不连接 MySQL，仅锁定 Repository SQL 占位符与模型参数数量。"""

    def test_insert_parameter_counts_match_sql(self) -> None:
        class CountingCursor:
            def execute(self, sql: str, params: tuple[object, ...]) -> None:
                self.assert_equal(sql.count("%s"), len(params))

            @staticmethod
            def assert_equal(expected: int, actual: int) -> None:
                if expected != actual:
                    raise AssertionError(
                        f"SQL 占位符数={expected}，参数数={actual}"
                    )

        profile, employments, projects = normalize_aggregate(
            _aggregate(), data_version="20260812_01", build_batch_id=1
        )
        chunk = ChunkFactory(_settings()).build_all(profile, employments, projects)[0]
        cursor = CountingCursor()
        MySQLIndexRepository._upsert_profile(cursor, profile)
        MySQLIndexRepository._insert_employment(cursor, employments[0])
        MySQLIndexRepository._insert_project(cursor, projects[0])
        MySQLIndexRepository._upsert_chunk(cursor, chunk)

    def test_final_chunk_insert_omits_removed_columns(self) -> None:
        """最终表 INSERT 不得重新携带已删除的五个物理列。"""

        captured: list[str] = []

        class CapturingCursor:
            @staticmethod
            def execute(sql: str, params: tuple[object, ...]) -> None:
                captured.append(sql.split("ON DUPLICATE KEY UPDATE", 1)[0])

        profile, employments, projects = normalize_aggregate(
            _aggregate(), data_version="20260813_03", build_batch_id=1
        )
        chunk = ChunkFactory(_settings()).build_all(profile, employments, projects)[0]
        MySQLIndexRepository._upsert_chunk(CapturingCursor(), chunk)
        insert_sql = captured[0]
        for removed in (
            "chunk_id",
            "token_count_estimate",
            "source_item_key",
            "dictionary_version",
            "data_version",
        ):
            self.assertNotIn(removed, insert_sql)

    def test_stale_chunks_are_deactivated_after_snapshot_rebuild(self) -> None:
        """经历被逻辑删除后，其旧 vector_doc_id 不在新快照中，Repository 应停用。"""

        calls: list[tuple[str, tuple[object, ...]]] = []

        class RecordingCursor:
            @staticmethod
            def execute(sql: str, params: tuple[object, ...]) -> None:
                calls.append((" ".join(sql.split()), params))

        MySQLIndexRepository._deactivate_stale_chunks(
            RecordingCursor(), 24, ["expert:24:employment:1001:0"]
        )
        self.assertEqual(1, len(calls))
        self.assertIn("SET is_active=0", calls[0][0])
        self.assertIn("vector_doc_id NOT IN", calls[0][0])
        self.assertEqual((24, "expert:24:employment:1001:0"), calls[0][1])


class ExporterTest(unittest.TestCase):
    """验证 MySQL 行到固定 JSONL 契约的映射。"""

    def test_export_jsonl(self) -> None:
        now = datetime(2026, 8, 11, 10, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        text = "[专家能力概述]\n核心背景：虚构测试资料"
        row = {
            "expert_id": 24,
            "serial_number": "EXP000024",
            "chunk_type": "profile_summary",
            "chunk_subtype": None,
            "chunk_index": 0,
            "chunk_total": 1,
            "chunk_text": text,
            "char_count": len(text),
            "content_hash": sha256_value(text),
            "metadata_hash": "b" * 64,
            "vector_doc_id": "expert:24:profile_summary:24:0",
            "source_table": "experts+experts_education_experience",
            "source_field": "company+job_name+background+school_name+major",
            "source_field_id": "24",
            "source_version": "2026-08-11T10:00:00+08:00",
            "source_updated_at": now,
            "source_hash": "a" * 64,
            "company_name_raw": "虚构公司",
            "company_name_norm": "虚构公司",
            "role_name_raw": "产品经理",
            "role_code": None,
            "role_level": None,
            "job_status_norm": None,
            "abroad_flag_norm": 1,
            "work_years": 8,
            "project_name": None,
            "start_month": None,
            "end_month": None,
            "is_current": None,
            "industry_codes": "[]",
            "function_codes": "[]",
            "topic_codes": "[]",
            "school_names_raw": '["华东流程测试大学"]',
            "major_names_raw": '["信息管理与信息系统"]',
            "eligible_for_recommendation": None,
            "origin_type": "compressed_from_source",
            "verification_status": "source_summary",
            "weight_tier": "B",
            "chunk_policy_version": "1.3",
            "metadata_schema_version": "3.0",
            "is_active": 1,
            "updated_at": now,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            self.assertEqual(1, export_jsonl([row], path, "Asia/Shanghai"))
            contract = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("4.0.0", contract["contract_version"])
        self.assertEqual(text, contract["chunk_text"])
        self.assertNotIn("source_record_id", contract["source"])
        self.assertEqual("company+job_name+background+school_name+major", contract["source"]["source_field"])
        self.assertEqual("24", contract["source"]["source_field_id"])
        self.assertEqual(
            "expert:24:profile_summary:24:0",
            contract["metadata"]["vector_doc_id"],
        )
        self.assertNotIn("data_version", contract["versions"])
        self.assertNotIn("dictionary_version", contract["versions"])


if __name__ == "__main__":
    unittest.main()
