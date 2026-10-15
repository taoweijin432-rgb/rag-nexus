"""Resume MySQL Source Adapter。

本模块只读 ``resume`` 主表和三张 ``resume_*`` 派生表，并把允许字段映射到现有公共
Source 模型。
``resume_id`` 在映射后承载内部兼容字段 ``expert_id`` 的语义；人物主表物理 ``id``
只用于全量分页游标和来源追踪，绝不替代人物身份。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pymysql

from expert_index.config import DatabaseBoundaryError, DatabaseConfig, DatabaseRole
from expert_index.dataset import DatasetSpec
from expert_index.models import (
    CurrentFactReadRequest,
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertSource,
    ProjectSource,
    ScopedCurrentFacts,
    SourceChangeCandidate,
    SourceScanCursor,
    impact_scope_from_target,
)

from .dataset import RESUME_SPEC

# 与 Expert Source 的访问上限保持同一数量级，但由 Resume 入口独立校验和使用。
MAX_RESUME_FULL_SCAN_PAGE_SIZE = 1000
MAX_RESUME_SOURCE_SCAN_PAGE_SIZE = 1000


class ResumeNotFoundError(LookupError):
    """请求的 Resume 人物不存在或不满足当前读取条件。"""


@dataclass(frozen=True)
class ResumeCandidate:
    """全量扫描中的最小 Resume 人物候选。

    Parameters
    ----------
    source_record_id:
        ``resume.id`` 物理行 ID，只用于 Keyset Pagination 和来源追踪。
    resume_id:
        Resume 全链路唯一人物/任务身份，后续映射到公共模型的 ``expert_id``。
    deleted_at:
        ``resume.resume_deleted_at`` 的别名；非空时应停用该人物历史 Chunk。
    """

    source_record_id: int
    resume_id: int
    deleted_at: datetime | None

    def __post_init__(self) -> None:
        """确保物理游标和人物身份都是有效正整数。"""

        _require_positive_integer(self.source_record_id, "source_record_id")
        _require_positive_integer(self.resume_id, "resume_id")
        if self.deleted_at is not None and not isinstance(self.deleted_at, datetime):
            raise TypeError("deleted_at 必须是 datetime 或 None")

    @property
    def is_deleted(self) -> bool:
        """返回人物是否已按 Resume 生命周期契约逻辑删除。"""

        return self.deleted_at is not None


class MySQLResumeSourceAdapter:
    """按 ``resume_id`` 读取完整快照，并按物理主键分页扫描人物候选。

    Adapter 不提供任何写方法。完整快照的四次查询共享一个 REPEATABLE READ 只读事务，
    避免主档与经历来自不同数据库时刻。传入规格必须等于冻结的 :data:`RESUME_SPEC`，
    防止 Resume 入口被误配置为 Expert 或任意其他来源表。
    """

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        dataset_spec: DatasetSpec = RESUME_SPEC,
    ) -> None:
        if not isinstance(config, DatabaseConfig):
            raise TypeError("config 必须是 DatabaseConfig")
        if config.role is not DatabaseRole.SOURCE:
            raise DatabaseBoundaryError(
                "Resume Source Adapter 只能使用 RESUME_SOURCE_DB_* 构造的 SOURCE 连接"
            )
        if dataset_spec != RESUME_SPEC:
            raise ValueError("Resume Source Adapter 只能使用冻结的 RESUME_SPEC")
        self.config = config
        self.dataset_spec = dataset_spec

    def get_full_scan_upper_bound(self, *, max_resumes: int | None = None) -> int:
        """固定本批次允许扫描到的最大 ``resume.id``。

        ``max_resumes=None`` 时读取全表最大物理 ID；传入正整数 N 时只把按物理 ID
        升序排列的前 N 位人物纳入批次。查询不排除逻辑删除人物，因为删除记录同样需要
        进入后续停用分支。本方法不读取任何简历正文。
        """

        if max_resumes is not None:
            _require_positive_integer(max_resumes, "max_resumes")

        profile_table = self.dataset_spec.profile_table
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                if max_resumes is None:
                    cursor.execute(
                        f"SELECT COALESCE(MAX(id), 0) AS upper_bound_id FROM {profile_table}"
                    )
                else:
                    cursor.execute(
                        f"""
                        SELECT COALESCE(MAX(id), 0) AS upper_bound_id
                        FROM (
                            SELECT id FROM {profile_table} ORDER BY id ASC LIMIT %s
                        ) AS scoped_resumes
                        """,
                        (max_resumes,),
                    )
                row = cursor.fetchone()
            return int(row["upper_bound_id"]) if row else 0
        finally:
            connection.close()

    def load_full_scan_candidates(
        self,
        *,
        after_source_record_id: int,
        upper_bound_source_record_id: int,
        limit: int,
    ) -> list[ResumeCandidate]:
        """按 ``resume.id`` Keyset 游标读取一页人物候选。

        结果同时返回物理 ``source_record_id`` 和业务 ``resume_id``，从类型层面阻止调用方
        混用两个身份。查询只读取调度字段和生命周期标记，不读取背景或经历正文。
        """

        _require_non_negative_integer(after_source_record_id, "after_source_record_id")
        _require_non_negative_integer(
            upper_bound_source_record_id,
            "upper_bound_source_record_id",
        )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_RESUME_FULL_SCAN_PAGE_SIZE
        ):
            raise ValueError(f"limit 必须位于 1 到 {MAX_RESUME_FULL_SCAN_PAGE_SIZE} 之间")
        if (
            upper_bound_source_record_id == 0
            or after_source_record_id >= upper_bound_source_record_id
        ):
            return []

        spec = self.dataset_spec
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id AS source_record_id,
                           {spec.profile_identity_column} AS resume_id,
                           {spec.profile_deleted_at_column} AS deleted_at
                    FROM {spec.profile_table}
                    WHERE id > %s AND id <= %s
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (after_source_record_id, upper_bound_source_record_id, limit),
                )
                rows = cursor.fetchall()
            return [
                ResumeCandidate(
                    source_record_id=int(row["source_record_id"]),
                    resume_id=int(row["resume_id"]),
                    deleted_at=row.get("deleted_at"),
                )
                for row in rows
            ]
        finally:
            connection.close()

    def load_resume_candidate(self, resume_id: int) -> ResumeCandidate:
        """按业务 ``resume_id`` 读取人物存在状态，不读取业务正文。"""

        _require_positive_integer(resume_id, "resume_id")
        spec = self.dataset_spec
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id AS source_record_id,
                           {spec.profile_identity_column} AS resume_id,
                           {spec.profile_deleted_at_column} AS deleted_at
                    FROM {spec.profile_table}
                    WHERE {spec.profile_identity_column} = %s
                    """,
                    (resume_id,),
                )
                row = cursor.fetchone()
            if row is None:
                raise ResumeNotFoundError(f"Resume {resume_id} 不存在")
            return ResumeCandidate(
                source_record_id=int(row["source_record_id"]),
                resume_id=int(row["resume_id"]),
                deleted_at=row.get("deleted_at"),
            )
        finally:
            connection.close()

    def get_source_scan_upper_bound(self, source_table: str) -> SourceScanCursor | None:
        """读取一张 Resume 来源表当前最大的 ``(updated_at, id)`` 固定上界。

        表名只能来自冻结的 ``RESUME_SPEC``；查询不读取任何业务正文，也不排除逻辑
        删除行。空表返回 ``None``，Checkpoint 的读取和推进由独立 Index Repository
        负责。
        """

        self._require_source_table(source_table)
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT updated_at, id FROM {source_table} "
                    "ORDER BY updated_at DESC, id DESC LIMIT 1"
                )
                row = cursor.fetchone()
            if row is None:
                return None
            return SourceScanCursor(
                updated_at=row["updated_at"],
                record_id=int(row["id"]),
            )
        finally:
            connection.close()

    def load_source_change_candidates(
        self,
        source_table: str,
        *,
        after_cursor: SourceScanCursor,
        upper_bound: SourceScanCursor,
        limit: int,
    ) -> list[SourceChangeCandidate]:
        """在固定范围内按 ``(updated_at, id)`` 读取一页 Resume 变化候选。

        主表候选同时返回物理 ``id`` 和业务 ``resume_id``；子表以自身 ``id`` 作为
        来源记录身份，并通过 ``resume_id`` 关联人物。候选只含调度字段，不含简历正文。
        """

        self._require_source_table(source_table)
        if not isinstance(after_cursor, SourceScanCursor):
            raise TypeError("after_cursor 必须是 SourceScanCursor")
        if not isinstance(upper_bound, SourceScanCursor):
            raise TypeError("upper_bound 必须是 SourceScanCursor")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_RESUME_SOURCE_SCAN_PAGE_SIZE
        ):
            raise ValueError(f"limit 必须位于 1 到 {MAX_RESUME_SOURCE_SCAN_PAGE_SIZE} 之间")
        try:
            if after_cursor >= upper_bound:
                return []
        except TypeError as exc:
            raise ValueError("扫描起点和上界必须使用一致的时间类型") from exc

        spec = self.dataset_spec
        person_expression = (
            f"{spec.profile_identity_column} AS internal_person_id"
            if source_table == spec.profile_table
            else f"{spec.child_identity_column} AS internal_person_id"
        )
        deleted_expression = (
            f"{spec.lifecycle_deleted_at_column_for_source_table(source_table)}"
            " AS lifecycle_deleted_at"
        )
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id, {person_expression}, updated_at, {deleted_expression}
                    FROM {source_table}
                    WHERE (updated_at > %s OR (updated_at = %s AND id > %s))
                      AND (updated_at < %s OR (updated_at = %s AND id <= %s))
                    ORDER BY updated_at ASC, id ASC
                    LIMIT %s
                    """,
                    (
                        after_cursor.updated_at,
                        after_cursor.updated_at,
                        after_cursor.record_id,
                        upper_bound.updated_at,
                        upper_bound.updated_at,
                        upper_bound.record_id,
                        limit,
                    ),
                )
                rows = cursor.fetchall()
            return [
                SourceChangeCandidate(
                    source_table=source_table,
                    record_id=int(row["id"]),
                    expert_id=(
                        int(row["internal_person_id"])
                        if row.get("internal_person_id") is not None
                        else None
                    ),
                    updated_at=row["updated_at"],
                    deleted_at=row.get("lifecycle_deleted_at"),
                )
                for row in rows
            ]
        finally:
            connection.close()

    def load_resume(self, resume_id: int) -> ExpertAggregate:
        """在同一只读一致性快照中读取一位有效 Resume 人物的四表事实。

        返回公共 :class:`~expert_index.models.ExpertAggregate` 是刻意的兼容边界：
        ``resume_id`` 映射到 ``ExpertSource.id`` 和三类经历的 ``expert_id``，经历物理
        ``id`` 原样保留。这样公共 Normalize/Chunk/Contract 不需要新增 Resume 分支。
        """

        _require_positive_integer(resume_id, "resume_id")
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=False))
        try:
            with connection.cursor() as cursor:
                self._begin_consistent_read(cursor)
                profile = self._load_profile(cursor, resume_id)
                employments = self._load_employments(cursor, resume_id)
                projects = self._load_projects(cursor, resume_id)
                educations = self._load_educations(cursor, resume_id)
            # 只读事务用 rollback 明确结束；未来即使误加入写 SQL，也不会被提交。
            connection.rollback()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return ExpertAggregate(
            profile=profile,
            employments=tuple(employments),
            projects=tuple(projects),
            educations=tuple(educations),
        )

    def read_current_facts(self, request: CurrentFactReadRequest) -> ScopedCurrentFacts:
        """在同一只读快照中重读一个 Resume 增量任务的最小当前事实。

        触发行必须属于 ``RESUME_SPEC``，主档物理 ``id`` 会与任务中的
        ``source_record_id`` 精确核对，而返回的公共 ``ExpertSource.id`` 始终承载
        ``resume_id``。逻辑删除行正常返回，由共享 Scope Handler 生成空 desired。
        """

        if not isinstance(request, CurrentFactReadRequest):
            raise TypeError("request 必须是 CurrentFactReadRequest")
        self._require_source_table(request.trigger_source_table)

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=False))
        try:
            with connection.cursor() as cursor:
                self._begin_consistent_read(cursor)
                facts = self._read_scoped_facts(cursor, request)
            connection.rollback()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return facts

    def read_cdc_education_delete_facts(
        self,
        resume_id: int,
    ) -> ScopedCurrentFacts | None:
        """重读教育经历物理删除后的 Resume Profile 当前事实。

        物理删除事件中已无法再次读取触发行，因此这里只在同一个只读一致性快照中
        读取人物主档和剩余有效教育经历。主档也已物理删除时返回 ``None``，由公共
        CDC Delete Processor 将影响范围提升为整个人物；逻辑删除主档则保留生命周期
        事实并返回空教育集合。本方法不读取工作、项目，也不执行任何写入。

        :param resume_id: Resume 全链路唯一人物身份。
        :return: Profile 范围当前事实；人物主档物理不存在时返回 ``None``。
        """

        _require_positive_integer(resume_id, "resume_id")
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=False))
        try:
            with connection.cursor() as cursor:
                self._begin_consistent_read(cursor)
                scope = impact_scope_from_target(
                    target_type="profile",
                    expert_id=resume_id,
                    dataset_spec=self.dataset_spec,
                )
                try:
                    profile = self._load_profile_row(cursor, resume_id)
                except ResumeNotFoundError:
                    facts = None
                else:
                    educations = (
                        ()
                        if profile.deleted_at is not None
                        else tuple(self._load_educations(cursor, resume_id))
                    )
                    facts = ScopedCurrentFacts(
                        scope=scope,
                        expert=profile,
                        educations=educations,
                    )
            # 只读事务仍以 rollback 明确收口，避免连接复用时残留事务状态。
            connection.rollback()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return facts

    def _read_scoped_facts(
        self,
        cursor: Any,
        request: CurrentFactReadRequest,
    ) -> ScopedCurrentFacts:
        """按影响范围读取主档、触发行和必要的当前有效子记录。"""

        scope = request.impact_scope
        spec = self.dataset_spec
        if scope.scope_type == "expert":
            profile = self._load_profile_trigger_row(
                cursor,
                int(request.trigger_source_field_id),
                expected_resume_id=scope.expert_id,
            )
            self._assert_current_not_older_than_candidate(
                profile.scan_updated_at,
                request.trigger_source_updated_at,
                f"{spec.profile_table}.updated_at",
            )
            if profile.deleted_at is not None:
                return ScopedCurrentFacts(scope=scope, expert=profile)
            return ScopedCurrentFacts(
                scope=scope,
                expert=profile,
                employments=tuple(self._load_employments(cursor, scope.expert_id)),
                projects=tuple(self._load_projects(cursor, scope.expert_id)),
                educations=tuple(self._load_educations(cursor, scope.expert_id)),
            )

        if scope.scope_type == "profile":
            profile = self._load_profile_row(cursor, scope.expert_id)
            if request.trigger_source_table == spec.education_table:
                education, education_scan_updated_at = self._load_education_row_by_id(
                    cursor,
                    int(request.trigger_source_field_id),
                )
                self._assert_trigger_owns_resume(
                    education,
                    scope.expert_id,
                    "education",
                )
                self._assert_current_not_older_than_candidate(
                    education_scan_updated_at,
                    request.trigger_source_updated_at,
                    f"{spec.education_table}.updated_at",
                )
            elif request.trigger_source_table == spec.profile_table:
                profile = self._load_profile_trigger_row(
                    cursor,
                    int(request.trigger_source_field_id),
                    expected_resume_id=scope.expert_id,
                )
                self._assert_current_not_older_than_candidate(
                    profile.scan_updated_at,
                    request.trigger_source_updated_at,
                    f"{spec.profile_table}.updated_at",
                )
            else:  # pragma: no cover - 请求模型和表白名单已提前拒绝。
                raise ValueError("profile 范围收到不支持的 Resume 触发表")
            educations = (
                ()
                if profile.deleted_at is not None
                else tuple(self._load_educations(cursor, scope.expert_id))
            )
            return ScopedCurrentFacts(
                scope=scope,
                expert=profile,
                educations=educations,
            )

        if scope.scope_type == "employment":
            employment, employment_scan_updated_at = self._load_employment_row_by_id(
                cursor,
                int(request.trigger_source_field_id),
            )
            self._assert_trigger_owns_resume(employment, scope.expert_id, "employment")
            self._assert_current_not_older_than_candidate(
                employment_scan_updated_at,
                request.trigger_source_updated_at,
                f"{spec.employment_table}.updated_at",
            )
            return ScopedCurrentFacts(
                scope=scope,
                expert=self._load_profile_row(cursor, scope.expert_id),
                employments=(employment,),
            )

        if scope.scope_type == "project":
            project, project_scan_updated_at = self._load_project_row_by_id(
                cursor,
                int(request.trigger_source_field_id),
            )
            self._assert_trigger_owns_resume(project, scope.expert_id, "project")
            self._assert_current_not_older_than_candidate(
                project_scan_updated_at,
                request.trigger_source_updated_at,
                f"{spec.project_table}.updated_at",
            )
            return ScopedCurrentFacts(
                scope=scope,
                expert=self._load_profile_row(cursor, scope.expert_id),
                projects=(project,),
            )

        raise ValueError(f"不支持的影响范围类型: {scope.scope_type}")

    @staticmethod
    def _begin_consistent_read(cursor: Any) -> None:
        """建立四表共用的 REPEATABLE READ 只读一致性快照。"""

        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")

    def _load_profile(self, cursor: Any, resume_id: int) -> ExpertSource:
        """读取有效人物主档，并把 ``resume_id`` 映射为内部人物身份。"""

        return self._load_profile_row(cursor, resume_id, active_only=True)

    def _load_profile_row(
        self,
        cursor: Any,
        resume_id: int,
        *,
        active_only: bool = False,
    ) -> ExpertSource:
        """按 ``resume_id`` 读取人物生命周期，不过滤逻辑删除。"""

        spec = self.dataset_spec
        active_predicate = f"AND {spec.profile_deleted_at_column} IS NULL" if active_only else ""
        cursor.execute(
            f"""
            SELECT {spec.profile_identity_column} AS internal_person_id,
                   serial_number, company, job_status, job_name,
                   contact_status, doubtful_flag, abroad_flag, background,
                   work_years,
                   {spec.profile_source_updated_at_column} AS business_updated_at,
                   updated_at AS scan_updated_at,
                   created_at,
                   {spec.profile_deleted_at_column} AS lifecycle_deleted_at
            FROM {spec.profile_table}
            WHERE {spec.profile_identity_column} = %s
              {active_predicate}
            """,
            (resume_id,),
        )
        row = cursor.fetchone()
        if row is None:
            suffix = "不存在或已删除" if active_only else "物理不存在"
            raise ResumeNotFoundError(f"Resume {resume_id} {suffix}")
        return self._profile_source_from_row(row)

    def _load_profile_trigger_row(
        self,
        cursor: Any,
        source_record_id: int,
        *,
        expected_resume_id: int,
    ) -> ExpertSource:
        """按 ``resume.id`` 读取触发行并核对其业务人物身份。"""

        spec = self.dataset_spec
        cursor.execute(
            f"""
            SELECT id AS physical_source_id,
                   {spec.profile_identity_column} AS internal_person_id,
                   serial_number, company, job_status, job_name,
                   contact_status, doubtful_flag, abroad_flag, background,
                   work_years,
                   {spec.profile_source_updated_at_column} AS business_updated_at,
                   updated_at AS scan_updated_at,
                   created_at,
                   {spec.profile_deleted_at_column} AS lifecycle_deleted_at
            FROM {spec.profile_table}
            WHERE id = %s
            """,
            (source_record_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ResumeNotFoundError(f"Resume 主档记录 {source_record_id} 物理不存在")
        actual_resume_id = int(row["internal_person_id"])
        if actual_resume_id != expected_resume_id:
            raise ValueError(
                f"Resume 主档记录属于 resume_id={actual_resume_id}，"
                f"与任务 resume_id={expected_resume_id} 不一致"
            )
        return self._profile_source_from_row(row)

    @staticmethod
    def _profile_source_from_row(row: dict[str, Any]) -> ExpertSource:
        """把已别名化的 Resume 主档行转换为公共人物事实模型。"""

        return ExpertSource(
            id=int(row["internal_person_id"]),
            serial_number=row.get("serial_number"),
            company=row.get("company"),
            job_status=row.get("job_status"),
            job_name=row.get("job_name"),
            contact_status=row.get("contact_status"),
            doubtful_flag=row.get("doubtful_flag"),
            abroad_flag=row.get("abroad_flag"),
            background=row.get("background"),
            work_years=(float(row["work_years"]) if row.get("work_years") is not None else None),
            updated_at=row.get("business_updated_at"),
            created_at=row.get("created_at"),
            deleted_at=row.get("lifecycle_deleted_at"),
            scan_updated_at=row.get("scan_updated_at"),
        )

    def _load_employment_row_by_id(
        self,
        cursor: Any,
        employment_id: int,
    ) -> tuple[EmploymentSource, datetime | None]:
        """按工作物理主键读取触发行，不过滤其权威逻辑删除列。"""

        spec = self.dataset_spec
        deleted_at_column = spec.lifecycle_deleted_at_column_for_source_table(spec.employment_table)
        cursor.execute(
            f"""
            SELECT id, {spec.child_identity_column} AS internal_person_id,
                   company_name, position_title, start_date, end_date,
                   work_description, is_employed,
                   {spec.employment_source_updated_at_column} AS business_updated_at,
                   updated_at AS scan_updated_at, created_at,
                   {deleted_at_column} AS deleted_at
            FROM {spec.employment_table}
            WHERE id = %s
            """,
            (employment_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ResumeNotFoundError(f"Resume 工作记录 {employment_id} 物理不存在")
        return (
            EmploymentSource(
                id=int(row["id"]),
                expert_id=int(row["internal_person_id"]),
                company_name=row.get("company_name"),
                position_title=row.get("position_title"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                work_description=row.get("work_description"),
                is_employed=row.get("is_employed"),
                updated_at=row.get("business_updated_at"),
                created_at=row.get("created_at"),
                deleted_at=row.get("deleted_at"),
            ),
            row.get("scan_updated_at"),
        )

    def _load_project_row_by_id(
        self,
        cursor: Any,
        project_id: int,
    ) -> tuple[ProjectSource, datetime | None]:
        """按项目物理主键读取触发行，不过滤其权威逻辑删除列。"""

        spec = self.dataset_spec
        deleted_at_column = spec.lifecycle_deleted_at_column_for_source_table(spec.project_table)
        cursor.execute(
            f"""
            SELECT id, {spec.child_identity_column} AS internal_person_id,
                   project_title, company_name, start_date, end_date,
                   work_description,
                   {spec.project_source_updated_at_column} AS business_updated_at,
                   updated_at AS scan_updated_at, created_at,
                   {deleted_at_column} AS deleted_at
            FROM {spec.project_table}
            WHERE id = %s
            """,
            (project_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ResumeNotFoundError(f"Resume 项目记录 {project_id} 物理不存在")
        return (
            ProjectSource(
                id=int(row["id"]),
                expert_id=int(row["internal_person_id"]),
                project_title=row.get("project_title"),
                company_name=row.get("company_name"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                work_description=row.get("work_description"),
                updated_at=row.get("business_updated_at"),
                created_at=row.get("created_at"),
                deleted_at=row.get("deleted_at"),
            ),
            row.get("scan_updated_at"),
        )

    def _load_education_row_by_id(
        self,
        cursor: Any,
        education_id: int,
    ) -> tuple[EducationSource, datetime | None]:
        """按教育物理主键读取触发行，不过滤其权威逻辑删除列。"""

        spec = self.dataset_spec
        deleted_at_column = spec.lifecycle_deleted_at_column_for_source_table(spec.education_table)
        cursor.execute(
            f"""
            SELECT id, {spec.child_identity_column} AS internal_person_id,
                   school_name, major,
                   {spec.education_source_updated_at_column} AS business_updated_at,
                   updated_at AS scan_updated_at, created_at,
                   {deleted_at_column} AS deleted_at
            FROM {spec.education_table}
            WHERE id = %s
            """,
            (education_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ResumeNotFoundError(f"Resume 教育记录 {education_id} 物理不存在")
        return (
            EducationSource(
                id=int(row["id"]),
                expert_id=int(row["internal_person_id"]),
                school_name=row.get("school_name"),
                major=row.get("major"),
                updated_at=row.get("business_updated_at"),
                created_at=row.get("created_at"),
                deleted_at=row.get("deleted_at"),
            ),
            row.get("scan_updated_at"),
        )

    def _load_employments(
        self,
        cursor: Any,
        resume_id: int,
    ) -> list[EmploymentSource]:
        """读取全部有效工作经历；每条物理 ``id`` 保持为稳定 Chunk 来源 ID。"""

        spec = self.dataset_spec
        deleted_at_column = spec.lifecycle_deleted_at_column_for_source_table(spec.employment_table)
        cursor.execute(
            f"""
            SELECT id, {spec.child_identity_column} AS internal_person_id,
                   company_name, position_title, start_date, end_date,
                   work_description, is_employed,
                   {spec.employment_source_updated_at_column} AS business_updated_at,
                   created_at, {deleted_at_column} AS deleted_at
            FROM {spec.employment_table}
            WHERE {spec.child_identity_column} = %s
              AND {deleted_at_column} IS NULL
            ORDER BY COALESCE(end_date, '9999-12-31') DESC,
                     start_date DESC, id ASC
            """,
            (resume_id,),
        )
        return [
            EmploymentSource(
                id=int(row["id"]),
                expert_id=int(row["internal_person_id"]),
                company_name=row.get("company_name"),
                position_title=row.get("position_title"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                work_description=row.get("work_description"),
                is_employed=row.get("is_employed"),
                updated_at=row.get("business_updated_at"),
                created_at=row.get("created_at"),
                deleted_at=row.get("deleted_at"),
            )
            for row in cursor.fetchall()
        ]

    def _load_projects(self, cursor: Any, resume_id: int) -> list[ProjectSource]:
        """读取全部有效项目经历；每条物理 ``id`` 保持为稳定 Chunk 来源 ID。"""

        spec = self.dataset_spec
        deleted_at_column = spec.lifecycle_deleted_at_column_for_source_table(spec.project_table)
        cursor.execute(
            f"""
            SELECT id, {spec.child_identity_column} AS internal_person_id,
                   project_title, company_name, start_date, end_date,
                   work_description,
                   {spec.project_source_updated_at_column} AS business_updated_at,
                   created_at, {deleted_at_column} AS deleted_at
            FROM {spec.project_table}
            WHERE {spec.child_identity_column} = %s
              AND {deleted_at_column} IS NULL
            ORDER BY COALESCE(end_date, '9999-12-31') DESC,
                     start_date DESC, id ASC
            """,
            (resume_id,),
        )
        return [
            ProjectSource(
                id=int(row["id"]),
                expert_id=int(row["internal_person_id"]),
                project_title=row.get("project_title"),
                company_name=row.get("company_name"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                work_description=row.get("work_description"),
                updated_at=row.get("business_updated_at"),
                created_at=row.get("created_at"),
                deleted_at=row.get("deleted_at"),
            )
            for row in cursor.fetchall()
        ]

    def _load_educations(self, cursor: Any, resume_id: int) -> list[EducationSource]:
        """读取有效教育经历；教育仅参与 Profile 聚合，不单独生成 Chunk。"""

        spec = self.dataset_spec
        deleted_at_column = spec.lifecycle_deleted_at_column_for_source_table(spec.education_table)
        cursor.execute(
            f"""
            SELECT id, {spec.child_identity_column} AS internal_person_id,
                   school_name, major,
                   {spec.education_source_updated_at_column} AS business_updated_at,
                   created_at,
                   {deleted_at_column} AS deleted_at
            FROM {spec.education_table}
            WHERE {spec.child_identity_column} = %s
              AND {deleted_at_column} IS NULL
            ORDER BY id ASC
            """,
            (resume_id,),
        )
        return [
            EducationSource(
                id=int(row["id"]),
                expert_id=int(row["internal_person_id"]),
                school_name=row.get("school_name"),
                major=row.get("major"),
                updated_at=row.get("business_updated_at"),
                created_at=row.get("created_at"),
                deleted_at=row.get("deleted_at"),
            )
            for row in cursor.fetchall()
        ]

    def _require_source_table(self, source_table: str) -> None:
        """把动态 SQL 表名收窄到冻结的四张 Resume 来源表。"""

        if (
            not isinstance(source_table, str)
            or source_table not in self.dataset_spec.checkpoint_source_tables
        ):
            raise ValueError(f"不支持的 Resume 来源表: {source_table}")

    @staticmethod
    def _assert_trigger_owns_resume(
        row: EmploymentSource | ProjectSource | EducationSource,
        resume_id: int,
        label: str,
    ) -> None:
        """确认子表触发行属于任务声明的 ``resume_id``。"""

        if row.expert_id != resume_id:
            raise ValueError(
                f"{label} 触发行 resume_id={row.expert_id} 与请求 resume_id={resume_id} 不一致"
            )

    @staticmethod
    def _assert_current_not_older_than_candidate(
        current: datetime | None,
        candidate: datetime | None,
        field_label: str,
    ) -> None:
        """拒绝当前事实扫描时间早于任务候选时间的异常回退。"""

        if candidate is None:
            return
        if current is None:
            raise ValueError(f"{field_label} 当前为空，无法校验候选时间")
        if current < candidate:
            raise ValueError(f"{field_label} 当前时间早于候选时间，拒绝处理")


def _require_positive_integer(value: object, label: str) -> int:
    """校验并返回正整数，显式排除 ``bool``。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} 必须是正整数")
    return value


def _require_non_negative_integer(value: object, label: str) -> int:
    """校验并返回非负整数，供物理分页游标使用。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} 必须是非负整数")
    return value


__all__ = [
    "MAX_RESUME_FULL_SCAN_PAGE_SIZE",
    "MAX_RESUME_SOURCE_SCAN_PAGE_SIZE",
    "MySQLResumeSourceAdapter",
    "ResumeCandidate",
    "ResumeNotFoundError",
]
