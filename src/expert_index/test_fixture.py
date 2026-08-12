"""测试环境完整专家 Fixture 的幂等写入器。

本模块会写业务源表，因此不会被普通 ``build`` 隐式调用；只有操作者显式执行
``seed-test-expert`` 才会运行。三张表在一个 MySQL 事务中创建或更新，任何失败都
整体回滚。所有内容都带“流程测试/虚构”标记，不使用真实姓名或联系方式。
"""

from __future__ import annotations

from dataclasses import dataclass

import pymysql

from .config import DatabaseConfig

# 固定唯一编号使重复执行具有幂等性，也让清理测试数据时可以精确定位。
TEST_EXPERT_SERIAL = "TST2026081100000001"
TEST_COMPANY = "星瀚智检（流程测试）科技有限公司"
TEST_POSITION = "AI检索产品负责人"
TEST_PROJECT_TITLE = "企业知识库智能检索升级（流程测试项目）"

# 工作和项目描述故意使用明确标签。Source Adapter 只解析这些原文标签，不通过
# LLM 推断角色、目标、成果，因此仍满足 Same Segment 与来源可追溯要求。
TEST_WORK_DESCRIPTION = (
    "工作职责：负责企业知识库、数据治理和AI检索产品的规划与交付，"
    "组织产品、算法和工程团队制定Chunk策略、混合检索方案及验收标准。\n"
    "代表性成果：测试周期内检索命中率由68%提升至86%，平均响应时间下降35%，"
    "覆盖1200名虚构内部用户。"
)

TEST_PROJECT_DESCRIPTION = (
    "项目角色：项目负责人\n"
    "项目目标：建设面向虚构企业内部员工的知识库智能检索系统，提升资料查找效率。\n"
    "关键动作：完成数据清洗规范、Chunk切分策略、关键词与向量混合检索方案设计，"
    "并组织离线评测和灰度验收。\n"
    "项目结果：测试数据Top10召回率由72%提升至89%，平均资料检索时间下降40%。"
)


@dataclass(frozen=True)
class WorkFixture:
    """一段虚构任职的全部源字段；每个对象会对应一条独立子表记录。"""

    company: str
    position: str
    start_date: str
    end_date: str | None
    is_employed: int
    description: str


@dataclass(frozen=True)
class ProjectFixture:
    """一个虚构项目的全部源字段；title 在同专家内作为幂等定位键。"""

    company: str
    title: str
    start_date: str
    end_date: str
    description: str


# 三段任职覆盖“当前 + 两段历史”，用于验证多个 source_field_id 仍属于同一专家。
WORK_FIXTURES = (
    WorkFixture(
        company=TEST_COMPANY,
        position=TEST_POSITION,
        start_date="2022-03-01 00:00:00",
        end_date=None,
        is_employed=1,
        description=TEST_WORK_DESCRIPTION,
    ),
    WorkFixture(
        company="云岚数据（流程测试）有限公司",
        position="高级数据产品经理",
        start_date="2018-07-01 00:00:00",
        end_date="2022-02-01 00:00:00",
        is_employed=2,
        description=(
            "工作职责：负责虚构数据中台、指标平台和企业搜索产品建设，推进数据标准与"
            "权限体系落地。\n"
            "代表性成果：测试数据口径一致率由75%提升至93%，报表交付周期缩短50%。"
        ),
    ),
    WorkFixture(
        company="启明软件（流程测试）有限公司",
        position="搜索产品经理",
        start_date="2014-05-01 00:00:00",
        end_date="2018-06-01 00:00:00",
        is_employed=2,
        description=(
            "工作职责：负责虚构站内搜索需求分析、相关性评测和搜索运营工具设计。\n"
            "代表性成果：测试查询无结果率由12%下降至4%，月均服务300万次虚构查询。"
        ),
    ),
)


# 三个项目属于不同 source_field_id；正文均显式给出角色、目标、动作和结果。
PROJECT_FIXTURES = (
    ProjectFixture(
        company=TEST_COMPANY,
        title=TEST_PROJECT_TITLE,
        start_date="2024-01-01 00:00:00",
        end_date="2024-08-01 00:00:00",
        description=TEST_PROJECT_DESCRIPTION,
    ),
    ProjectFixture(
        company=TEST_COMPANY,
        title="智能客服知识库建设（流程测试项目）",
        start_date="2023-03-01 00:00:00",
        end_date="2023-12-01 00:00:00",
        description=(
            "项目角色：产品负责人\n"
            "项目目标：建设虚构智能客服知识库，降低重复咨询处理成本。\n"
            "关键动作：设计问答数据清洗、知识审核和召回评测流程，组织产品与算法联调。\n"
            "项目结果：测试首问解决率由61%提升至82%，人工转接率下降28%。"
        ),
    ),
    ProjectFixture(
        company="云岚数据（流程测试）有限公司",
        title="研发文档统一搜索（流程测试项目）",
        start_date="2020-04-01 00:00:00",
        end_date="2021-01-01 00:00:00",
        description=(
            "项目角色：核心产品经理\n"
            "项目目标：统一检索虚构需求、接口和测试文档。\n"
            "关键动作：梳理文档权限、同义词和排序规则，建立200条虚构评测集。\n"
            "项目结果：测试Top5命中率由70%提升至88%，资料查找时间下降45%。"
        ),
    ),
)


@dataclass(frozen=True)
class TestFixtureResult:
    """可安全打印的 Fixture 主键结果；数组顺序与 Fixture 定义顺序一致。"""

    expert_id: int
    work_experience_ids: tuple[int, ...]
    project_experience_ids: tuple[int, ...]
    serial_number: str = TEST_EXPERT_SERIAL


def ensure_complete_test_expert(config: DatabaseConfig) -> TestFixtureResult:
    """在测试库幂等创建一位主档、任职、项目均完整关联的虚构专家。"""

    connection = pymysql.connect(**config.as_kwargs())
    try:
        with connection.cursor() as cursor:
            expert_id = _upsert_expert(cursor)
            work_ids = tuple(_upsert_work(cursor, expert_id, item) for item in WORK_FIXTURES)
            project_ids = tuple(
                _upsert_project(cursor, expert_id, item) for item in PROJECT_FIXTURES
            )
        connection.commit()
        return TestFixtureResult(expert_id, work_ids, project_ids)
    except Exception:
        # 三张源表必须保持全有或全无；失败时不允许留下孤立经历记录。
        connection.rollback()
        raise
    finally:
        connection.close()


def _upsert_expert(cursor: object) -> int:
    """按固定 serial_number 锁定并创建/更新虚构专家主档。"""

    cursor.execute(
        "SELECT id FROM experts WHERE serial_number=%s FOR UPDATE",
        (TEST_EXPERT_SERIAL,),
    )
    row = cursor.fetchone()
    background = (
        "【流程测试虚构数据】12年企业软件、数据治理和人工智能检索产品经验，"
        "熟悉知识库建设、Chunk策略、混合检索和检索效果评测。"
    )
    if row:
        expert_id = int(row["id"])
        cursor.execute(
            """
            UPDATE experts
            SET name='流程测试专家A001', interview_nickname='流程测试专家',
                company=%s, job_status=1, job_name=%s, background=%s,
                source_type=99, work_years=12, education='本科', gender=0,
                email='', phone='', other_contacts='', deleted_at=NULL,
                updated_at=CURRENT_TIMESTAMP(3)
            WHERE id=%s
            """,
            (TEST_COMPANY, TEST_POSITION, background, expert_id),
        )
        return expert_id

    cursor.execute(
        """
        INSERT INTO experts(
            serial_number, name, interview_nickname, company, job_status,
            job_name, background, source_type, work_years, education, gender,
            email, phone, other_contacts, created_at, updated_at
        ) VALUES (
            %s, '流程测试专家A001', '流程测试专家', %s, 1,
            %s, %s, 99, 12, '本科', 0,
            '', '', '', CURRENT_TIMESTAMP(3), CURRENT_TIMESTAMP(3)
        )
        """,
        (TEST_EXPERT_SERIAL, TEST_COMPANY, TEST_POSITION, background),
    )
    return int(cursor.lastrowid)


def _upsert_work(cursor: object, expert_id: int, item: WorkFixture) -> int:
    """按公司和岗位幂等创建/更新一段字段完整的虚构任职。"""

    cursor.execute(
        """
        SELECT id FROM experts_work_experience
        WHERE expert_id=%s AND company_name=%s AND position_title=%s
        ORDER BY id LIMIT 1 FOR UPDATE
        """,
        (expert_id, item.company, item.position),
    )
    row = cursor.fetchone()
    if row:
        record_id = int(row["id"])
        cursor.execute(
            """
            UPDATE experts_work_experience
            SET start_date=%s, end_date=%s,
                work_description=%s, is_employed=%s, deleted_at=NULL,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (item.start_date, item.end_date, item.description, item.is_employed, record_id),
        )
        return record_id
    cursor.execute(
        """
        INSERT INTO experts_work_experience(
            company_name, start_date, end_date, position_title,
            work_description, expert_id, is_employed, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s,
                  CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            item.company, item.start_date, item.end_date, item.position,
            item.description, expert_id, item.is_employed,
        ),
    )
    return int(cursor.lastrowid)


def _upsert_project(cursor: object, expert_id: int, item: ProjectFixture) -> int:
    """按项目名幂等创建/更新一个含完整显式标签的虚构项目。"""

    cursor.execute(
        """
        SELECT id FROM experts_project_experience
        WHERE expert_id=%s AND project_title=%s
        ORDER BY id LIMIT 1 FOR UPDATE
        """,
        (expert_id, item.title),
    )
    row = cursor.fetchone()
    if row:
        record_id = int(row["id"])
        cursor.execute(
            """
            UPDATE experts_project_experience
            SET company_name=%s, start_date=%s,
                end_date=%s, work_description=%s,
                deleted_at=NULL, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (item.company, item.start_date, item.end_date, item.description, record_id),
        )
        return record_id
    cursor.execute(
        """
        INSERT INTO experts_project_experience(
            company_name, start_date, end_date, project_title,
            work_description, expert_id, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            item.company, item.start_date, item.end_date, item.title,
            item.description, expert_id,
        ),
    )
    return int(cursor.lastrowid)
