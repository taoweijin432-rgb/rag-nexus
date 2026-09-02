"""MySQL Source Adapter：只读抽取四张正确的专家业务表。

所有 SQL 均使用字段白名单，禁止 ``SELECT *``。工作和项目只读取能够通过
``expert_id`` 正确关联主档的工作、项目和教育经历表；历史错误表以及 experts
主表中的旧经历长文本不再参与链路。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pymysql

from .config import DatabaseConfig
from .dataset import EXPERT_SPEC
from .models import (
    CurrentFactReadRequest,
    EducationSource,
    EmploymentSource,
    ExpertAggregate,
    ExpertCandidate,
    ExpertSource,
    ProjectSource,
    ScopedCurrentFacts,
    SourceChangeCandidate,
    SourceScanCursor,
    impact_scope_from_target,
)

# 第一版把分页大小限制在 1000 以内，防止调用方误传过大 limit，一次读取过多候选。
# 默认页大小将在后续 CLI/Pipeline 阶段决定；Source 这里只约束数据库访问安全上限。
MAX_FULL_SCAN_PAGE_SIZE = 1000
# 增量扫描同样限制单页规模，避免一次把大量来源变化加载进内存。
MAX_SOURCE_SCAN_PAGE_SIZE = 1000

# MySQL 不能用参数占位符绑定表名，因此这里只允许从固定白名单选择完整 SQL。
# 上界始终使用中间表 ``updated_at`` 和物理 ``id``，逻辑删除不会改变扫描字段。
_SOURCE_SCAN_UPPER_BOUND_SQL = {
    source_table: (
        f"SELECT updated_at, id FROM {source_table} "
        "ORDER BY updated_at DESC, id DESC LIMIT 1"
    )
    for source_table in EXPERT_SPEC.checkpoint_source_tables
}

# experts 使用自身主键作为 expert_id，并把正式来源删除时间映射为统一的 deleted_at；
# 三张经历表读取实际关联字段和各自的 deleted_at。所有表仍用中间表 updated_at 扫描，
# 查询只返回调度所需的 ID、扫描时间和删除标记。
_SOURCE_CHANGE_PAGE_SQL = {
    source_table: f"""
        SELECT id, {"id AS expert_id" if source_table == "experts" else "expert_id"},
               updated_at,
               {"experts_deleted_at AS deleted_at" if source_table == "experts" else "deleted_at"}
        FROM {source_table}
        WHERE (updated_at > %s OR (updated_at = %s AND id > %s))
          AND (updated_at < %s OR (updated_at = %s AND id <= %s))
        ORDER BY updated_at ASC, id ASC
        LIMIT %s
    """
    for source_table in EXPERT_SPEC.checkpoint_source_tables
}


class ExpertNotFoundError(LookupError):
    """请求的专家不存在或已逻辑删除。"""


class MySQLSourceAdapter:
    """按专家 ID 读取完整快照；该类没有任何写方法。"""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config

    def get_source_scan_upper_bound(
        self,
        source_table: str,
    ) -> SourceScanCursor | None:
        """读取一张来源表启动增量批次时的固定扫描上界。

        返回最大 ``(updated_at, id)``；空表返回 ``None``。该方法只固定本批次将来允许
        扫描到的位置，不读取业务正文、不更新 Checkpoint，也不实现增量分页。
        """

        if (
            not isinstance(source_table, str)
            or source_table not in EXPERT_SPEC.checkpoint_source_tables
        ):
            raise ValueError(f"不支持的 Checkpoint 来源表: {source_table}")

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(_SOURCE_SCAN_UPPER_BOUND_SQL[source_table])
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
        """在固定复合游标范围内读取一页最小来源变化。

        查询范围为 ``after_cursor < (updated_at, id) <= upper_bound``，结果按相同字段
        升序排列。逻辑删除记录不会被过滤；无法关联专家的经历记录保留 ``expert_id=None``，
        留给后续任务层隔离处理。本方法不读取正文，也不推进任何 Checkpoint。
        """

        if (
            not isinstance(source_table, str)
            or source_table not in EXPERT_SPEC.checkpoint_source_tables
        ):
            raise ValueError(f"不支持的 Checkpoint 来源表: {source_table}")
        if not isinstance(after_cursor, SourceScanCursor):
            raise TypeError("after_cursor 必须是 SourceScanCursor")
        if not isinstance(upper_bound, SourceScanCursor):
            raise TypeError("upper_bound 必须是 SourceScanCursor")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_SOURCE_SCAN_PAGE_SIZE
        ):
            raise ValueError(
                f"limit 必须位于 1 到 {MAX_SOURCE_SCAN_PAGE_SIZE} 之间"
            )
        try:
            range_is_empty = after_cursor >= upper_bound
        except TypeError as exc:
            raise ValueError("扫描起点和上界必须使用一致的时间类型") from exc
        if range_is_empty:
            return []

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    _SOURCE_CHANGE_PAGE_SQL[source_table],
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
                        int(row["expert_id"])
                        if row.get("expert_id") is not None
                        else None
                    ),
                    updated_at=row["updated_at"],
                    deleted_at=row.get("deleted_at"),
                )
                for row in rows
            ]
        finally:
            connection.close()

    def get_full_scan_upper_bound(
        self,
        *,
        max_experts: int | None = None,
    ) -> int:
        """读取批量任务固定的 ``experts.id`` 上界。

        构成与用途
        ----------
        ``max_experts=None`` 时读取全表 ``MAX(id)``；传入正整数 N 时，查询按 ID 升序
        的前 N 位候选，并以其中最后一个 ID 作为上界。后续每页都限制
        ``id <= upper_bound_id``，因此批次不会超过指定专家数，运行期间新增专家也不会
        扩大当前范围。

        查询故意不排除正式来源删除时间 ``experts_deleted_at``，因为已删除专家仍需
        进入批次，触发历史 Chunk 停用。该方法只读取边界，不读取任何专家正文或经历数据。
        """

        if max_experts is not None and (
            isinstance(max_experts, bool)
            or not isinstance(max_experts, int)
            or max_experts < 1
        ):
            raise ValueError("max_experts 必须是正整数或 None")

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                if max_experts is None:
                    cursor.execute(
                        "SELECT COALESCE(MAX(id), 0) AS upper_bound_id FROM experts"
                    )
                else:
                    cursor.execute(
                        """
                        SELECT COALESCE(MAX(id), 0) AS upper_bound_id
                        FROM (
                            SELECT id FROM experts ORDER BY id ASC LIMIT %s
                        ) AS scoped_experts
                        """,
                        (max_experts,),
                    )
                row = cursor.fetchone()
                return int(row["upper_bound_id"]) if row else 0
        finally:
            connection.close()

    def load_full_scan_candidates(
        self,
        *,
        after_id: int,
        upper_bound_id: int,
        limit: int,
    ) -> list[ExpertCandidate]:
        """按主键游标读取一页全量专家候选。

        Parameters
        ----------
        after_id:
            上一页最后完成调度的 ``experts.id``；第一页面传 0。查询使用严格大于，
            避免相邻页重复读取同一专家。
        upper_bound_id:
            :meth:`get_full_scan_upper_bound` 在批次启动时固定的上界。
        limit:
            本页最多返回的专家数，必须位于 1 到 ``MAX_FULL_SCAN_PAGE_SIZE`` 之间。

        Returns
        -------
        list[ExpertCandidate]
            按 ``id`` 升序排列的候选。结果包含逻辑删除专家，但不包含正文；后续
            Pipeline 会逐个选择“完整四表读取”或“停用历史 Chunk”分支。

        Notes
        -----
        这里使用 Keyset Pagination，而不是 OFFSET。前者可以利用主键范围扫描，且不会
        因前面数据新增或删除而让页边界漂移。当前方法只有一条 SELECT，不需要持有
        多语句一致性事务。
        """

        if after_id < 0:
            raise ValueError("after_id 不能小于 0")
        if upper_bound_id < 0:
            raise ValueError("upper_bound_id 不能小于 0")
        if not 1 <= limit <= MAX_FULL_SCAN_PAGE_SIZE:
            raise ValueError(
                f"limit 必须位于 1 到 {MAX_FULL_SCAN_PAGE_SIZE} 之间"
            )
        # 空源表或游标已经到达固定上界时，无需建立数据库连接。
        if upper_bound_id == 0 or after_id >= upper_bound_id:
            return []

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, experts_deleted_at AS deleted_at
                    FROM experts
                    WHERE id > %s AND id <= %s
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (after_id, upper_bound_id, limit),
                )
                return [
                    ExpertCandidate(
                        id=int(row["id"]),
                        deleted_at=row.get("deleted_at"),
                    )
                    for row in cursor.fetchall()
                ]
        finally:
            connection.close()

    def load_expert_candidate(self, expert_id: int) -> ExpertCandidate:
        """读取一位专家的存在状态和逻辑删除标记，不读取任何正文。

        partial 批次重试失败专家时使用该方法重新判断当前生命周期状态：active 专家
        继续读取四表快照，已逻辑删除专家进入停用分支。物理记录不存在时抛出
        :class:`ExpertNotFoundError`，由 Pipeline 把本次尝试继续记录为 failed。
        """

        if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id < 1:
            raise ValueError("expert_id 必须是正整数")
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, experts_deleted_at AS deleted_at FROM experts WHERE id=%s",
                    (expert_id,),
                )
                row = cursor.fetchone()
            if not row:
                raise ExpertNotFoundError(f"专家 {expert_id} 不存在")
            return ExpertCandidate(
                id=int(row["id"]),
                deleted_at=row.get("deleted_at"),
            )
        finally:
            connection.close()

    def load_expert(self, expert_id: int) -> ExpertAggregate:
        """在同一只读一致性快照中读取一位专家的四表当前数据。

        本方法被现有单专家 Pipeline 调用，后续也会作为第一版全量任务的专家级读取
        单元。它依次读取专家主档、工作、项目和教育经历，并返回一个
        :class:`ExpertAggregate`。三张经历表没有关联行时分别返回空元组，属于正常结果。

        四次 SELECT 必须看到同一个数据库时刻，因此连接关闭 autocommit，并由
        :meth:`_begin_consistent_read` 启动 REPEATABLE READ 的只读一致性快照。方法只读
        不写，成功和失败都用 rollback 结束事务，最后关闭短连接。
        """

        if expert_id <= 0:
            raise ValueError("expert_id 必须大于 0")
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=False))
        try:
            with connection.cursor() as cursor:
                self._begin_consistent_read(cursor)
                profile = self._load_profile(cursor, expert_id)
                employments = self._load_employments(cursor, profile.id)
                projects = self._load_projects(cursor, profile.id)
                educations = self._load_educations(cursor, profile.id)
            # 只读事务无需 commit；rollback 明确结束一致性快照，也避免未来误加入写 SQL
            # 后被意外提交。异常分支执行相同收口后继续抛出原错误。
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

    @staticmethod
    def _begin_consistent_read(cursor: Any) -> None:
        """为 :meth:`load_expert` 启动四表一致的只读数据库快照。

        第一条语句把当前事务隔离级别固定为 REPEATABLE READ，第二条语句立即创建一致性
        快照并声明只读。后续主档、工作、项目、教育四次 SELECT 都使用该快照，
        避免读取过程中业务库恰好更新某张经历表而形成“半新半旧”的专家聚合。

        本函数只负责事务开端，不负责 commit/rollback；事务结束和连接关闭统一由
        :meth:`load_expert` 管理。
        """

        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")

    @staticmethod
    def _load_profile(cursor: Any, expert_id: int) -> ExpertSource:
        # 姓名、电话、邮箱、旧经历长文本、personal_intro、comment、remark、
        # source_link 等字段都不进入查询结果。
        # Profile 将正式来源时间映射到领域模型字段：experts_updated_at 决定后续
        # source_updated_at/source_version，experts_deleted_at 决定专家是否仍然有效。
        cursor.execute(
            """
            SELECT id, serial_number, company, job_status, job_name,
                   contact_status, doubtful_flag, abroad_flag, background,
                   work_years, experts_updated_at AS updated_at, created_at,
                   experts_deleted_at AS deleted_at
            FROM experts
            WHERE id = %s AND experts_deleted_at IS NULL
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
            job_status=row.get("job_status"),
            job_name=row.get("job_name"),
            contact_status=row.get("contact_status"),
            doubtful_flag=row.get("doubtful_flag"),
            abroad_flag=row.get("abroad_flag"),
            background=row.get("background"),
            work_years=float(row["work_years"]) if row.get("work_years") is not None else None,
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
            deleted_at=row.get("deleted_at"),
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
                deleted_at=row.get("deleted_at"),
            ))
        return result

    @staticmethod
    def _query_employment_rows(cursor: Any, expert_id: int) -> list[dict[str, Any]]:
        cursor.execute(
            """
            SELECT id, expert_id, company_name, position_title, start_date, end_date,
                   work_description, is_employed, updated_at, created_at, deleted_at
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
                   work_description, updated_at, created_at, deleted_at
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
                deleted_at=row.get("deleted_at"),
            ))
        return result

    @staticmethod
    def _load_educations(cursor: Any, expert_id: int) -> list[EducationSource]:
        """读取未逻辑删除的教育经历。

        教育经历只为 profile_summary 提供学校和专业 Metadata，不产生独立 Chunk。
        查询仍带上 ``deleted_at`` 字段，使 Source 模型能和真实源表逐列审查；由于
        WHERE 条件已经过滤，正常返回的该字段必然为空。

        排序只使用两套环境都存在的 ``id``。本机源表虽然有 ``start_date/end_date``，
        公司中间表已在脱敏加工时移除这两列；而当前 EducationSource 和最终 Metadata
        都不使用教育起止日期，因此按主键稳定升序既不损失业务字段，又能兼容两套表。
        """

        cursor.execute(
            """
            SELECT id, expert_id, school_name, major,
                   updated_at, created_at, deleted_at
            FROM experts_education_experience
            WHERE expert_id = %s AND deleted_at IS NULL
            ORDER BY id ASC
            """,
            (expert_id,),
        )
        result: list[EducationSource] = []
        for row in cursor.fetchall():
            result.append(EducationSource(
                id=int(row["id"]),
                expert_id=int(row["expert_id"]),
                school_name=row.get("school_name"),
                major=row.get("major"),
                updated_at=row.get("updated_at"),
                created_at=row.get("created_at"),
                deleted_at=row.get("deleted_at"),
            ))
        return result

    # ------------------------------------------------------------------
    # P4.3 定向只读事实读取
    # ------------------------------------------------------------------

    def read_current_facts(self, request: CurrentFactReadRequest) -> ScopedCurrentFacts:
        """在同一只读一致性快照中按 :class:`CurrentFactReadRequest` 读取当前事实。

        四类影响范围共用一个短连接与一致性快照（REPEATABLE READ, READ ONLY）。
        物理行不存在 fail-closed；逻辑删除行（deleted_at 非空）正常返回，
        由后续 Handler 决定生成空 desired 或升级范围。触发行的 ``updated_at``
        早于候选 ``trigger_source_updated_at`` 时 fail-closed（事件比当前事实更新）。

        本方法不调用公开 :meth:`load_expert`（会另开连接破坏一致性快照），而是复用
        内部按主键查询的 helper，在同一个 cursor 上执行所有 SELECT。
        """

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
        expert_id: int,
    ) -> ScopedCurrentFacts | None:
        """重读教育物理删除后的专家与剩余教育事实。

        CDC DELETE 的触发行已经物理消失，因此本入口不查询或重建该教育行，只在同一个
        只读一致性快照中读取父专家和当前有效教育。父专家也已物理删除时返回 ``None``，
        由 P7.5 提升为 expert scope 空 desired；逻辑删除父专家仍返回生命周期事实。
        """

        if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id < 1:
            raise ValueError("expert_id 必须是正整数")
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=False))
        try:
            with connection.cursor() as cursor:
                self._begin_consistent_read(cursor)
                scope = impact_scope_from_target(
                    target_type="profile",
                    expert_id=expert_id,
                )
                try:
                    expert = self._load_expert_row_by_id(cursor, expert_id)
                except ExpertNotFoundError:
                    facts = None
                else:
                    educations = (
                        ()
                        if expert.deleted_at is not None
                        else tuple(self._load_educations(cursor, expert_id))
                    )
                    facts = ScopedCurrentFacts(
                        scope=scope,
                        expert=expert,
                        educations=educations,
                    )
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
        scope = request.impact_scope
        scope_type = scope.scope_type

        if scope_type == "expert":
            expert = self._load_expert_row_by_id(cursor, scope.expert_id)
            # 专家任务时序只与中间表扫描时间 experts.updated_at 比较，
            # 不与正式来源时间 experts_updated_at 比较。
            self._assert_current_not_older_than_candidate(
                expert.scan_updated_at, request.trigger_source_updated_at,
                "experts.updated_at",
            )
            if expert.deleted_at is not None:
                # 已删除专家：扫描时间不回退即返回 expert scope 空 desired，
                # 不与 source_updated_at 比较以避免误报事实版本回退。
                return ScopedCurrentFacts(scope=scope, expert=expert)
            employments = self._load_employments(cursor, scope.expert_id)
            projects = self._load_projects(cursor, scope.expert_id)
            educations = self._load_educations(cursor, scope.expert_id)
            return ScopedCurrentFacts(
                scope=scope, expert=expert,
                employments=tuple(employments), projects=tuple(projects),
                educations=tuple(educations),
            )

        if scope_type == "profile":
            expert = self._load_expert_row_by_id(cursor, scope.expert_id)
            if request.trigger_source_table == "experts_education_experience":
                # education 触发：先读教育行校验归属与时序，再决定是否读父专家有效教育。
                edu = self._load_education_row_by_id(
                    cursor, int(request.trigger_source_field_id),
                )
                self._assert_trigger_owns_expert(edu, scope.expert_id, "education")
                self._assert_current_not_older_than_candidate(
                    edu.updated_at, request.trigger_source_updated_at,
                    "experts_education_experience.updated_at",
                )
            else:
                # profile 自身触发：校验专家主档中间表扫描时间，
                # 不使用正式来源时间 experts_updated_at。
                self._assert_current_not_older_than_candidate(
                    expert.scan_updated_at, request.trigger_source_updated_at,
                    "experts.updated_at",
                )
            if expert.deleted_at is not None:
                return ScopedCurrentFacts(scope=scope, expert=expert)
            educations = self._load_educations(cursor, scope.expert_id)
            return ScopedCurrentFacts(
                scope=scope, expert=expert, educations=tuple(educations),
            )

        if scope_type == "employment":
            trigger = self._load_employment_row_by_id(
                cursor, int(request.trigger_source_field_id),
            )
            self._assert_trigger_owns_expert(trigger, scope.expert_id, "employment")
            self._assert_current_not_older_than_candidate(
                trigger.updated_at, request.trigger_source_updated_at,
                "experts_work_experience.updated_at",
            )
            expert = self._load_expert_row_by_id(cursor, scope.expert_id)
            return ScopedCurrentFacts(
                scope=scope, expert=expert, employments=(trigger,),
            )

        if scope_type == "project":
            trigger = self._load_project_row_by_id(
                cursor, int(request.trigger_source_field_id),
            )
            self._assert_trigger_owns_expert(trigger, scope.expert_id, "project")
            self._assert_current_not_older_than_candidate(
                trigger.updated_at, request.trigger_source_updated_at,
                "experts_project_experience.updated_at",
            )
            expert = self._load_expert_row_by_id(cursor, scope.expert_id)
            return ScopedCurrentFacts(
                scope=scope, expert=expert, projects=(trigger,),
            )

        raise ValueError(f"不支持的影响范围类型: {scope_type}")

    @staticmethod
    def _assert_trigger_owns_expert(
        row: EmploymentSource | ProjectSource, expert_id: int, label: str,
    ) -> None:
        if row.expert_id != expert_id:
            raise ValueError(
                f"{label} 触发行 expert_id={row.expert_id} 与请求 expert_id={expert_id} 不一致"
            )

    @staticmethod
    def _assert_current_not_older_than_candidate(
        current: datetime | None,
        candidate: datetime | None,
        field_label: str,
    ) -> None:
        """当前事实更新时间晚于候选才合法；早于候选 fail-closed（事件比当前新）。

        候选时间为空时不做时序校验（上游尚未提供）。当前时间为空而候选非空也
        fail-closed：来源行存在却无 updated_at 属于异常。
        """

        if candidate is None:
            return
        if current is None:
            raise ValueError(f"{field_label} 当前为空，无法校验时序，候选时间非空")
        if current < candidate:
            raise ValueError(
                f"{field_label} 当前 {current} 早于候选 {candidate}，事件比当前事实更新"
            )

    @staticmethod
    def _load_expert_row_by_id(cursor: Any, expert_id: int) -> ExpertSource:
        """按主键读取专家主档，不过滤 ``experts_deleted_at``。

        与 :meth:`_load_profile` 不同：Full Build 路径要求专家有效（deleted_at 为空）
        且只关心正式来源时间；P4.3 路径需要同时保留中间表扫描时间 ``experts.updated_at``
        （事件时序校验）、正式来源时间 ``experts_updated_at``（Profile source_version）
        与生命周期标记 ``experts_deleted_at`` 三种语义，故读取全部行并由调用方判断。
        """

        cursor.execute(
            """
            SELECT id, serial_number, company, job_status, job_name,
                   contact_status, doubtful_flag, abroad_flag, background,
                   work_years, updated_at, experts_updated_at, created_at,
                   experts_deleted_at
            FROM experts
            WHERE id = %s
            """,
            (expert_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ExpertNotFoundError(f"专家 {expert_id} 物理不存在")
        return ExpertSource(
            id=int(row["id"]),
            serial_number=row.get("serial_number"),
            company=row.get("company"),
            job_status=row.get("job_status"),
            job_name=row.get("job_name"),
            contact_status=row.get("contact_status"),
            doubtful_flag=row.get("doubtful_flag"),
            abroad_flag=row.get("abroad_flag"),
            background=row.get("background"),
            work_years=float(row["work_years"]) if row.get("work_years") is not None else None,
            updated_at=row.get("experts_updated_at"),
            created_at=row.get("created_at"),
            deleted_at=row.get("experts_deleted_at"),
            scan_updated_at=row.get("updated_at"),
        )

    @staticmethod
    def _load_employment_row_by_id(
        cursor: Any, employment_id: int,
    ) -> EmploymentSource:
        """按工作主键读取单条工作行，不过滤 ``deleted_at``。"""

        cursor.execute(
            """
            SELECT id, expert_id, company_name, position_title, start_date, end_date,
                   work_description, is_employed, updated_at, created_at, deleted_at
            FROM experts_work_experience
            WHERE id = %s
            """,
            (employment_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ExpertNotFoundError(f"工作记录 {employment_id} 物理不存在")
        return EmploymentSource(
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
            deleted_at=row.get("deleted_at"),
        )

    @staticmethod
    def _load_project_row_by_id(cursor: Any, project_id: int) -> ProjectSource:
        """按项目主键读取单条项目行，不过滤 ``deleted_at``。"""

        cursor.execute(
            """
            SELECT id, expert_id, project_title, company_name, start_date, end_date,
                   work_description, updated_at, created_at, deleted_at
            FROM experts_project_experience
            WHERE id = %s
            """,
            (project_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ExpertNotFoundError(f"项目记录 {project_id} 物理不存在")
        return ProjectSource(
            id=int(row["id"]),
            expert_id=int(row["expert_id"]),
            project_title=row.get("project_title"),
            company_name=row.get("company_name"),
            start_date=row.get("start_date"),
            end_date=row.get("end_date"),
            work_description=row.get("work_description"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
            deleted_at=row.get("deleted_at"),
        )

    @staticmethod
    def _load_education_row_by_id(cursor: Any, education_id: int) -> EducationSource:
        """按教育主键读取单条教育行，不过滤 ``deleted_at``。"""

        cursor.execute(
            """
            SELECT id, expert_id, school_name, major,
                   updated_at, created_at, deleted_at
            FROM experts_education_experience
            WHERE id = %s
            """,
            (education_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ExpertNotFoundError(f"教育记录 {education_id} 物理不存在")
        return EducationSource(
            id=int(row["id"]),
            expert_id=int(row["expert_id"]),
            school_name=row.get("school_name"),
            major=row.get("major"),
            updated_at=row.get("updated_at"),
            created_at=row.get("created_at"),
            deleted_at=row.get("deleted_at"),
        )
