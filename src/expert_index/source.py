"""MySQL Source Adapter：只读抽取三张正确的专家业务表。

所有 SQL 均使用字段白名单，禁止 ``SELECT *``。工作和项目只读取能够通过
``expert_id`` 正确关联主档的 ``experts_work_experience`` 与
``experts_project_experience``；历史错误表和 experts 中的旧经历文本不再参与链路。
"""

from __future__ import annotations

from typing import Any

import pymysql

from .config import DatabaseConfig
from .models import EmploymentSource, ExpertAggregate, ExpertSource, ProjectSource


class ExpertNotFoundError(LookupError):
    """请求的专家不存在或已逻辑删除。"""


class MySQLSourceAdapter:
    """按专家 ID 读取完整快照；该类没有任何写方法。"""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config

    def load_expert(self, expert_id: int) -> ExpertAggregate:
        """一次连接中读取主档、任职和项目，减少跨时刻数据漂移。"""

        if expert_id <= 0:
            raise ValueError("expert_id 必须大于 0")
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                profile = self._load_profile(cursor, expert_id)
                employments = self._load_employments(cursor, profile.id)
                projects = self._load_projects(cursor, expert_id)
        finally:
            connection.close()
        return ExpertAggregate(profile, tuple(employments), tuple(projects))

    @staticmethod
    def _load_profile(cursor: Any, expert_id: int) -> ExpertSource:
        # 姓名、电话、邮箱、旧经历长文本、personal_intro、comment、remark、
        # source_link 等字段都不进入查询结果。
        cursor.execute(
            """
            SELECT id, serial_number, company, job_name, background,
                   edu_experience, work_years, updated_at, created_at
            FROM experts
            WHERE id = %s AND deleted_at IS NULL
            """,
            (expert_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ExpertNotFoundError(f"专家 {expert_id} 不存在或已删除")
        return ExpertSource(
            id=int(row["id"]),
            serial_number=row.get("serial_number"),
            company=row.get("company"),
            job_name=row.get("job_name"),
            background=row.get("background"),
            edu_experience=row.get("edu_experience"),
            work_years=float(row["work_years"]) if row.get("work_years") is not None else None,
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )

    def _load_employments(self, cursor: Any, expert_id: int) -> list[EmploymentSource]:
        # 新表明确提供 expert_id，不再经过 relate_id 或 resume_id 猜测关联关系。
        rows = self._query_employment_rows(cursor, expert_id)
        result: list[EmploymentSource] = []
        for row in rows:
            result.append(EmploymentSource(
                # Source 模型只承接 SELECT 返回的真实列；正文解析在
                # normalize.py 进入 Stage 时完成，不在 Source 层增加派生字段。
                id=int(row["id"]),
                expert_id=int(row["expert_id"]),
                company_name=row.get("company_name"),
                position_title=row.get("position_title"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                work_description=row.get("work_description"),
                is_employed=row.get("is_employed"),
                updated_at=row.get("updated_at"),
                created_at=row.get("created_at"),
            ))
        return result

    @staticmethod
    def _query_employment_rows(cursor: Any, expert_id: int) -> list[dict[str, Any]]:
        cursor.execute(
            """
            SELECT id, expert_id, company_name, position_title, start_date, end_date,
                   work_description, is_employed, updated_at, created_at
            FROM experts_work_experience
            WHERE expert_id = %s AND deleted_at IS NULL
            ORDER BY COALESCE(end_date, '9999-12-31') DESC, start_date DESC, id ASC
            """,
            (expert_id,),
        )
        return list(cursor.fetchall())

    @staticmethod
    def _load_projects(cursor: Any, expert_id: int) -> list[ProjectSource]:
        # 新表 id 是自增主键，进入 Stage 后会同时作为
        # source_field_id 和稳定 source_item_key。
        cursor.execute(
            """
            SELECT id, expert_id, project_title, company_name, start_date, end_date,
                   work_description, updated_at, created_at
            FROM experts_project_experience
            WHERE expert_id = %s AND deleted_at IS NULL
            ORDER BY COALESCE(end_date, '9999-12-31') DESC, start_date DESC, id ASC
            """,
            (expert_id,),
        )
        result: list[ProjectSource] = []
        for row in cursor.fetchall():
            result.append(ProjectSource(
                id=int(row["id"]),
                expert_id=int(row["expert_id"]),
                project_title=row.get("project_title"),
                company_name=row.get("company_name"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                work_description=row.get("work_description"),
                updated_at=row.get("updated_at"),
                created_at=row.get("created_at"),
            ))
        return result
