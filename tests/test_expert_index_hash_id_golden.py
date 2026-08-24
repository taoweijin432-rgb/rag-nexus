"""锁定上游稳定 ID、正文和三 Hash 的精确 golden 输出。"""

from __future__ import annotations

from datetime import datetime

from expert_index.chunk_builder import ChunkFactory
from expert_index.config import DatabaseConfig, Settings
from expert_index.contract_bridge import chunk_records_to_contracts
from expert_index.models import (
    ChunkRecord,
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
)
from expert_index.normalize import canonical_json, normalize_aggregate
from expert_index.validators import validate_chunks


def _settings() -> Settings:
    """构造不连接外部服务的固定业务配置。"""

    database = DatabaseConfig("127.0.0.1", 3306, "test", "test", "test")
    return Settings(source_db=database, index_db=database)


def _normal_aggregate() -> ExpertAggregate:
    """返回覆盖三类 Chunk、Unicode、空值和教育去重的固定快照。"""

    profile_time = datetime(2026, 1, 2, 3, 4, 5, 123456)  # noqa: DTZ001
    education_time = datetime(2026, 1, 4, 5, 6, 7, 654321)  # noqa: DTZ001
    return ExpertAggregate(
        profile=ExpertSource(
            id=880001,
            serial_number="GOLDEN-专家-001",
            company="ＡＩ未来有限公司",
            job_status=1,
            job_name="首席执行官",
            contact_status=3,
            doubtful_flag=1,
            abroad_flag=2,
            background="专注知识图谱与检索增强，支持多语言语义分析🚀。",
            work_years=8.0,
            updated_at=profile_time,
            created_at=profile_time,
            deleted_at=None,
        ),
        employments=(
            EmploymentSource(
                id=9201,
                expert_id=880001,
                company_name="北京京东世纪贸易有限公司",
                position_title="项目经理",
                start_date=datetime(2020, 2, 1),  # noqa: DTZ001
                end_date=datetime(2025, 12, 1),  # noqa: DTZ001
                work_description=(
                    "工作职责：负责推荐系统交付与跨团队协作。\n"
                    "代表性成果：检索准确率提升18%。"
                ),
                is_employed=1,
                updated_at=datetime(2026, 1, 3, 4, 5, 6, 111222),  # noqa: DTZ001
                created_at=datetime(2020, 2, 1),  # noqa: DTZ001
                deleted_at=None,
            ),
        ),
        projects=(
            ProjectSource(
                id=9301,
                expert_id=880001,
                project_title="跨语言检索 PoC",
                company_name="ibm中国",
                start_date=None,
                end_date=datetime(2025, 11, 1),  # noqa: DTZ001
                work_description=(
                    "项目角色：技术负责人\n"
                    "项目目标：验证中英混合检索。\n"
                    "关键动作：构建 Unicode 规范化与召回评估。\n"
                    "项目结果：Top-K 命中率提升12%。"
                ),
                updated_at=datetime(2026, 1, 3, 8, 9, 10, 333444),  # noqa: DTZ001
                created_at=datetime(2025, 6, 1),  # noqa: DTZ001
                deleted_at=None,
            ),
        ),
        educations=(
            EducationSource(
                id=9101,
                expert_id=880001,
                school_name=" 北方测试大学 ",
                major="人工智能",
                updated_at=education_time,
                created_at=education_time,
                deleted_at=None,
            ),
            EducationSource(
                id=9102,
                expert_id=880001,
                school_name="北方测试大学",
                major=None,
                updated_at=None,
                created_at=None,
                deleted_at=None,
            ),
        ),
    )


def _segmented_aggregate() -> ExpertAggregate:
    """返回各产生多个子 Chunk 的固定 Employment 与 Project 快照。"""

    source_time = datetime(2026, 2, 3, 4, 5, 6, 789012)  # noqa: DTZ001
    return ExpertAggregate(
        profile=ExpertSource(
            id=880002,
            serial_number=None,
            company=None,
            job_status=None,
            job_name=None,
            contact_status=None,
            doubtful_flag=1,
            abroad_flag=None,
            background=None,
            work_years=None,
            updated_at=source_time,
            created_at=source_time,
            deleted_at=None,
        ),
        employments=(
            EmploymentSource(
                id=9202,
                expert_id=880002,
                company_name="虚构韧性实验室",
                position_title="研发工程师",
                start_date=datetime(2022, 1, 1),  # noqa: DTZ001
                end_date=datetime(2025, 1, 1),  # noqa: DTZ001
                work_description=(
                    f"工作职责：{'负责稳定性验证。' * 70}\n"
                    f"代表性成果：{'故障率下降20%。' * 55}"
                ),
                is_employed=2,
                updated_at=source_time,
                created_at=source_time,
                deleted_at=None,
            ),
        ),
        projects=(
            ProjectSource(
                id=9302,
                expert_id=880002,
                project_title="Unicode 数据迁移🚀",
                company_name=None,
                start_date=None,
                end_date=None,
                work_description=(
                    f"项目目标：{'验证字符一致性。' * 45}\n"
                    f"关键动作：{'执行可重复迁移。' * 45}\n"
                    f"项目结果：{'校验结果全部通过。' * 45}"
                ),
                updated_at=source_time,
                created_at=source_time,
                deleted_at=None,
            ),
        ),
        educations=(),
    )


def _build(aggregate: ExpertAggregate) -> list[ChunkRecord]:
    """通过正式上游主链路构建并校验固定快照。"""

    profile, employments, projects = normalize_aggregate(
        aggregate,
        data_version="golden-data-version-excluded",
        build_batch_id=987654,
    )
    chunks = ChunkFactory(_settings()).build_all(profile, employments, projects)
    validate_chunks(chunks)
    return chunks


def _golden_rows(aggregate: ExpertAggregate) -> list[dict[str, object]]:
    """提取需要跨运行保持精确一致的 golden 字段。"""

    return [
        {
            "vector_doc_id": chunk.vector_doc_id,
            "chunk_type": chunk.chunk_type,
            "chunk_subtype": chunk.chunk_subtype,
            "chunk_index": chunk.chunk_index,
            "chunk_total": chunk.chunk_total,
            "char_count": chunk.char_count,
            "source_hash": chunk.source_hash,
            "content_hash": chunk.content_hash,
            "metadata_hash": chunk.metadata_hash,
        }
        for chunk in _build(aggregate)
    ]


def test_canonical_json_ignores_dictionary_insertion_order() -> None:
    """相同字典值必须得到完全相同的 canonical JSON。"""

    first = {"unicode": "专家🚀", "nullable": None, "items": ["b", "a"]}
    second = {"items": ["b", "a"], "nullable": None, "unicode": "专家🚀"}

    expected = '{"items":["b","a"],"nullable":null,"unicode":"专家🚀"}'
    assert canonical_json(first) == canonical_json(second) == expected


def test_normal_chunks_match_exact_golden_output() -> None:
    """锁定 Profile、Employment 和 Project 的精确 ID、正文与三 Hash。"""

    expected: list[dict[str, object]] = [
        {
            "vector_doc_id": "expert:880001:profile_summary:880001:0",
            "chunk_type": "profile_summary",
            "chunk_subtype": None,
            "chunk_index": 0,
            "chunk_total": 1,
            "char_count": 62,
            "source_hash": "7f434e87947533fa87d45b31f8335032fea9fb9de9d4b7ea9960c3cea513fd50",
            "content_hash": "68e44df323ec2adea6ba8f0e00e880e3465171fb5ab5a1398941d5e16485a132",
            "metadata_hash": "90f4e8e524ffc79a0f9906a7bb70566014ffc251fe0c8cd43b116b5ce5162b67",
        },
        {
            "vector_doc_id": "expert:880001:employment:9201:0",
            "chunk_type": "employment",
            "chunk_subtype": None,
            "chunk_index": 0,
            "chunk_total": 1,
            "char_count": 84,
            "source_hash": "8fffaf1fd4f1a305b5ce2e892edd7902afdae258ad083a44c8c4782f3d7d3e23",
            "content_hash": "7a1befec9a6db38c6ceddf9ce9ffa2bdaac368bdcfb0b1b651ea1116739a7ee4",
            "metadata_hash": "6208eb9c178b5cf7679f83e53e52fb6c4105463db26968c19ead58044c5cff8f",
        },
        {
            "vector_doc_id": "expert:880001:project:9301:0",
            "chunk_type": "project",
            "chunk_subtype": None,
            "chunk_index": 0,
            "chunk_total": 1,
            "char_count": 112,
            "source_hash": "debd9f7bfc4d5e24fc93c1e961158df09f60fe2cc10291bf58e9c6d30400e6e2",
            "content_hash": "1c3af4402323911bae01c401f1d6239027d8e0f551ba8021a948c15949e224cf",
            "metadata_hash": "a0d432ed857b0e8e7b7e17a01e95a40b109d7e20d79ae0b1c4eab2f029fd7e79",
        },
    ]
    chunks = _build(_normal_aggregate())

    assert [chunk.chunk_text for chunk in chunks] == [
        (
            "[专家能力概述]\n"
            "当前/最近任职：AI未来有限公司 · 首席执行官\n"
            "核心背景：专注知识图谱与检索增强,支持多语言语义分析🚀。"
        ),
        (
            "[任职经历]\n"
            "公司：京东零售\n"
            "岗位：项目经理\n"
            "时间：202002—至今\n"
            "任职状态：当前任职\n"
            "工作职责：负责推荐系统交付与跨团队协作。\n"
            "代表性成果：检索准确率提升18%。"
        ),
        (
            "[项目经历]\n"
            "项目名称：跨语言检索 PoC\n"
            "所属公司/客户：IBM中国\n"
            "时间：未知—202511\n"
            "项目描述：验证中英混合检索。\n"
            "项目工作内容：构建 Unicode 规范化与召回评估。\n"
            "项目结果：Top-K 命中率提升12%。"
        ),
    ]
    assert _golden_rows(_normal_aggregate()) == expected
    assert _golden_rows(_normal_aggregate()) == expected


def test_segmented_chunks_match_exact_golden_output() -> None:
    """锁定超长 Employment 与 Project 的分片身份和三 Hash。"""

    expected: list[dict[str, object]] = [
        {
            "vector_doc_id": "expert:880002:employment:9202:0",
            "chunk_type": "employment",
            "chunk_subtype": "responsibility",
            "chunk_index": 0,
            "chunk_total": 2,
            "char_count": 619,
            "source_hash": "52d66edf75ae144e3e637860077729d631afc8414c65925e26665fd210b4fb3f",
            "content_hash": "caa73f89aea03ae74b6a7dc1cb89ece98076d370b0752fae685b959178337e00",
            "metadata_hash": "6f3c2e753a002ee6d4def7fdbea98827f37185eab682950b9f55088620d05c49",
        },
        {
            "vector_doc_id": "expert:880002:employment:9202:1",
            "chunk_type": "employment",
            "chunk_subtype": "achievement",
            "chunk_index": 1,
            "chunk_total": 2,
            "char_count": 555,
            "source_hash": "52d66edf75ae144e3e637860077729d631afc8414c65925e26665fd210b4fb3f",
            "content_hash": "33597dfb72fc7ba65ce33b10c9ea49f41fb0410675a77c5ff8fa7ef85dff226c",
            "metadata_hash": "c65d80ba80f5cd4d2f45dd409669d63a6c422647d4cfa2da88a42538758cc373",
        },
        {
            "vector_doc_id": "expert:880002:project:9302:0",
            "chunk_type": "project",
            "chunk_subtype": "context",
            "chunk_index": 0,
            "chunk_total": 3,
            "char_count": 391,
            "source_hash": "462bba53f77eec8b62f2ff73c942e2d840255821f07a067503e11034a7b54453",
            "content_hash": "507f3169f048b6fba8e4dc4d277f60594ed661021d889de6317f156f44da2455",
            "metadata_hash": "e47502ee757c06ced1ccdddb3461fd21e04d376bcfb69f53b6106ad57316b78d",
        },
        {
            "vector_doc_id": "expert:880002:project:9302:1",
            "chunk_type": "project",
            "chunk_subtype": "action",
            "chunk_index": 1,
            "chunk_total": 3,
            "char_count": 393,
            "source_hash": "462bba53f77eec8b62f2ff73c942e2d840255821f07a067503e11034a7b54453",
            "content_hash": "a75fdaa58005d91db881ef0279c0853459d51d6c2bd6bbc2657a89f662f02f7f",
            "metadata_hash": "325c1922a8eb1ff6720a6fc52e13deb832caad1346f73b96379672935332b76d",
        },
        {
            "vector_doc_id": "expert:880002:project:9302:2",
            "chunk_type": "project",
            "chunk_subtype": "result",
            "chunk_index": 2,
            "chunk_total": 3,
            "char_count": 436,
            "source_hash": "462bba53f77eec8b62f2ff73c942e2d840255821f07a067503e11034a7b54453",
            "content_hash": "edb21868880814118a2fbf610dffaa224dfb6bce4e0180551554205a00a3c8d0",
            "metadata_hash": "5a0a66c86b3a0ecb3d8f3c21f022e33f0ea75c7f1c0189461706fc857c24e908",
        },
    ]
    assert _golden_rows(_segmented_aggregate()) == expected
    assert _golden_rows(_segmented_aggregate()) == expected


def test_golden_chunks_pass_contract_formula_without_hash_rewrite() -> None:
    """正式 Contract 必须接受所有 golden ID 并原样保留三个 Hash。"""

    chunks = _build(_normal_aggregate()) + _build(_segmented_aggregate())
    contracts = chunk_records_to_contracts(
        chunks,
        updated_at=datetime(2026, 3, 4, 5, 6, 7, 890123),  # noqa: DTZ001
        timezone_name=_settings().timezone,
    )

    assert [item.vector_doc_id for item in contracts] == [item.vector_doc_id for item in chunks]
    assert [item.source.source_hash for item in contracts] == [item.source_hash for item in chunks]
    assert [item.content_hash for item in contracts] == [item.content_hash for item in chunks]
    assert [item.metadata_hash for item in contracts] == [item.metadata_hash for item in chunks]

    profile_metadata = contracts[0].metadata.model_dump()
    assert profile_metadata["school_names_raw"] == ["北方测试大学"]
    assert profile_metadata["major_names_raw"] == ["人工智能"]
    assert profile_metadata["industry_codes"] == []
    assert profile_metadata["function_codes"] == []
    assert profile_metadata["topic_codes"] == []

    segmented_project_metadata = contracts[-1].metadata.model_dump()
    assert segmented_project_metadata["company_name_raw"] is None
    assert segmented_project_metadata["company_name_norm"] is None
