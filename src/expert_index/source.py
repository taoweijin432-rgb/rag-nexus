"""MySQL Source Adapter：只读抽取三张正确的专家业务表。

所有 SQL 均使用字段白名单，禁止 ``SELECT *``。工作和项目只读取能够通过
``expert_id`` 正确关联主档的 ``experts_work_experience`` 与
``experts_project_experience``；历史错误表和 experts 中的旧经历文本不再参与链路。
"""

from __future__ import annotations

import re
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
                employments = self._load_employments(cursor, profile)
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
            expert_id=int(row["id"]),
            serial_number=row.get("serial_number"),
            company=row.get("company"),
            role=row.get("job_name"),
            background=row.get("background"),
            education=row.get("edu_experience"),
            work_years=float(row["work_years"]) if row.get("work_years") is not None else None,
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
        )

    def _load_employments(self, cursor: Any, profile: ExpertSource) -> list[EmploymentSource]:
        # 新表明确提供 expert_id，不再经过 relate_id 或 resume_id 猜测关联关系。
        rows = self._query_employment_rows(cursor, profile.expert_id)
        result: list[EmploymentSource] = []
        for row in rows:
            responsibilities, achievements = _parse_employment_description(
                row.get("work_description")
            )
            result.append(EmploymentSource(
                expert_id=profile.expert_id,
                source_table="experts_work_experience",
                source_record_id=str(row["id"]),
                source_item_key=str(row["id"]),
                company=row.get("company_name"),
                role=row.get("position_title"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                # 若工作内容包含显式标签，则分别承接职责和成果；普通自由文本仍完整
                # 进入职责字段，不会因为解析失败而丢失。
                description=None,
                responsibilities=responsibilities,
                achievements=achievements,
                is_current=_employment_status(row.get("is_employed")),
                updated_at=row.get("updated_at"),
                created_at=row.get("created_at"),
            ))
        return result

    @staticmethod
    def _query_employment_rows(cursor: Any, expert_id: int) -> list[dict[str, Any]]:
        cursor.execute(
            """
            SELECT id, company_name, position_title, start_date, end_date,
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
        # 新表 id 是自增主键，可直接作为稳定 source_item_key。
        cursor.execute(
            """
            SELECT id, project_title, company_name, start_date, end_date,
                   work_description, updated_at, created_at
            FROM experts_project_experience
            WHERE expert_id = %s AND deleted_at IS NULL
            ORDER BY COALESCE(end_date, '9999-12-31') DESC, start_date DESC, id ASC
            """,
            (expert_id,),
        )
        result: list[ProjectSource] = []
        for row in cursor.fetchall():
            role, objective, actions, results = _parse_project_description(
                row.get("work_description")
            )
            result.append(ProjectSource(
                expert_id=expert_id,
                source_table="experts_project_experience",
                source_record_id=str(row["id"]),
                source_item_key=str(row["id"]),
                project_name=row.get("project_title"),
                company=row.get("company_name"),
                # 表没有独立项目角色列，仅接受 work_description 中明确写出的
                # “项目角色：...”来源证据；没有标签时仍保持 None。
                project_role=role,
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                description=objective,
                responsibilities=actions,
                results=results,
                updated_at=row.get("updated_at"),
                created_at=row.get("created_at"),
            ))
        return result


def _employment_status(value: object) -> bool | None:
    """按数据库字段注释映射：1=在职，2=不在职，其他值=未知。"""

    if value == 1:
        return True
    if value == 2:
        return False
    return None


def _parse_employment_description(value: object) -> tuple[str | None, str | None]:
    """解析显式的工作职责/代表性成果标签；普通文本整体作为职责。"""

    text = str(value).strip() if value is not None else ""
    if not text:
        return None, None
    sections = _labeled_sections(text, ("工作职责", "代表性成果"))
    if not sections:
        return text, None
    return sections.get("工作职责"), sections.get("代表性成果")


def _parse_project_description(
    value: object,
) -> tuple[str | None, str | None, str | None, str | None]:
    """从同一项目正文提取角色、目标、动作、结果四类显式来源事实。

    只有带 ``标签：值`` 的文本才进入结构化字段。既有无标签项目仍把完整正文
    当作项目工作内容，但项目角色保持空，因此会安全标记为 partial。
    """

    text = str(value).strip() if value is not None else ""
    if not text:
        return None, None, None, None
    labels = ("项目角色", "项目目标", "关键动作", "项目结果")
    sections = _labeled_sections(text, labels)
    if not sections:
        return None, None, text, None
    return (
        sections.get("项目角色"),
        sections.get("项目目标"),
        sections.get("关键动作"),
        sections.get("项目结果"),
    )


def _labeled_sections(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """确定性解析中文标签区段，支持区段值跨行且不改写任何事实。"""

    alternation = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"(?:^|\n)\s*({alternation})\s*[：:]\s*(.*?)"
        rf"(?=(?:\n\s*(?:{alternation})\s*[：:])|\Z)",
        re.DOTALL,
    )
    result: dict[str, str] = {}
    for match in pattern.finditer(text.replace("\r\n", "\n").replace("\r", "\n")):
        cleaned = match.group(2).strip()
        if cleaned:
            result[match.group(1)] = cleaned
    return result
