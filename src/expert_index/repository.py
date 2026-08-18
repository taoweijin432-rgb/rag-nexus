"""专家索引派生库的 MySQL 数据访问层。

本文件负责索引加工过程中的数据库协调和持久化：

1. 执行受版本控制的建表或迁移 SQL；
2. 用 MySQL 命名锁协调 single/full 任务，并管理批次主表；
3. 记录每位候选专家的 pending、running 和最终状态，以及页游标；
4. 保存专家、工作和项目 Stage 当前快照；
5. 对无内容或已删除专家维护 Stage 的拒绝状态。

本层不读取业务源表，不清洗文本，不生成 Chunk，也不调用向量服务。一位专家的
Profile、工作和项目 Stage 始终在同一事务中整体替换；Chunk 仅通过内存 Contract
传给下游，不在本 Repository 落库。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymysql

from .config import DatabaseConfig
from .models import EmploymentStage, ProfileStage, ProjectStage

# single/full 共用一个由 MySQL 连接持有的全局锁，防止构建任务并发改写派生数据。
BUILD_LOCK_PREFIX = "expert_index_build_v1"

_BATCH_ITEM_FINAL_STATUSES = {"success", "rejected", "deactivated", "failed"}


class ConcurrentBuildError(RuntimeError):
    """表示另一个专家索引加工任务正在持有全局互斥锁。"""


class BatchNotResumableError(RuntimeError):
    """指定批次不存在、已经完成，或不具备可靠的全量续跑状态。"""


@dataclass(frozen=True)
class FullBatchCheckpoint:
    """保存全量批次最后完整提交页的游标、范围和累计数量。"""

    batch_id: int
    batch_no: str
    data_version: str
    cursor_id: int
    upper_bound_id: int
    candidate_count: int
    success_count: int
    failed_count: int
    rejected_count: int
    deactivated_count: int
    extracted_count: int
    stage_count: int
    chunk_count: int


class MySQLIndexRepository:
    """封装批次跟踪和 Stage 当前快照的全部 MySQL 操作。

    每个公开方法自行管理短连接；需要原子更新多张 Stage 表的方法会在同一连接事务中
    提交或回滚。构造对象只保存连接配置，不立即访问数据库。
    """

    def __init__(self, config: DatabaseConfig) -> None:
        # 仅保存索引库连接配置，实际连接由各公开操作按需创建并关闭。
        self.config = config

    @contextmanager
    def build_lock(self, run_mode: str) -> Iterator[None]:
        """在一次 single/full 加工的完整生命周期内持有 MySQL 命名锁。

        ``GET_LOCK(..., 0)`` 不等待；锁已被占用时立即抛出
        :class:`ConcurrentBuildError`。锁连接在上下文退出前保持打开，保证批次创建、处理
        和收口期间没有其他 single/full 任务并发执行。该锁只负责任务互斥，不作为业务
        SQL 的事务连接。
        """

        if run_mode not in {"single", "full"}:
            raise ValueError("build_lock 第一版只支持 single/full")
        # 将数据库名纳入锁名，防止同一 MySQL 实例上的不同索引 Schema 相互阻塞。
        lock_name = f"{BUILD_LOCK_PREFIX}:{self.config.database}"[:64]
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        acquired = False
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT GET_LOCK(%s, 0) AS acquired", (lock_name,))
                row = cursor.fetchone()
                acquired = bool(row and row.get("acquired") == 1)
            if not acquired:
                raise ConcurrentBuildError(
                    f"已有专家索引加工任务运行，当前 {run_mode} 任务未启动"
                )
            yield
        finally:
            if acquired:
                # 释放失败不覆盖原业务异常；关闭连接仍会释放该会话持有的命名锁。
                with suppress(Exception), connection.cursor() as cursor:
                    cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
            connection.close()

    def initialize_schema(self, sql_path: Path) -> None:
        """在一个连接中依次执行 SQL 文件中的全部非空语句。

        Parameters
        ----------
        sql_path:
            要执行的建表或迁移 SQL 文件路径。

        Effects
        -------
        - 按分号拆分并依次执行文件内 SQL；
        - 全部语句成功后提交；
        - 任意语句失败时回滚当前连接并继续抛出原异常；
        - 只执行文件内已有语句，不生成额外 DDL。

        Notes
        -----
        MySQL DDL 可能隐式提交，因此 rollback 只能撤销数据库仍允许回滚的部分。
        """

        # 固定使用 UTF-8，确保 SQL 中的中文注释和字段说明按预期读取。
        script = sql_path.read_text(encoding="utf-8")
        # SQL 文件不包含存储过程，可按分号拆分并忽略空语句。
        statements = [statement.strip() for statement in script.split(";") if statement.strip()]
        # 关闭自动提交，在语句全部执行后统一提交连接。
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            # 全部语句成功后统一提交。
            connection.commit()
        except Exception:
            # 回滚仍处于事务中的语句，再把异常交给调用方。
            connection.rollback()
            raise
        finally:
            # 无论成功或失败都关闭本次连接。
            connection.close()

    def start_batch(
        self,
        batch_no: str,
        run_mode: str,
        *,
        data_version: str | None = None,
        upper_bound_id: int | None = None,
    ) -> int:
        """创建一条 ``running`` 状态的构建批次，并返回批次主键。

        ``single`` 只写入批次号、运行模式和状态。``full`` 还要求数据版本和固定扫描
        上界，并把初始游标设为 0。返回的 ``batch_id`` 用于关联 Stage 和专家级批次
        明细；批量任务随后通过 :meth:`update_batch_progress` 按完整页推进游标。
        """

        if run_mode == "full":
            if not data_version or upper_bound_id is None:
                raise ValueError("full 批次必须提供 data_version 和 upper_bound_id")
            if upper_bound_id < 0:
                raise ValueError("upper_bound_id 不能小于 0")

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                if run_mode == "full":
                    cursor.execute(
                        """
                        INSERT INTO expert_index_batch(
                            batch_no, run_mode, data_version, status,
                            cursor_id, upper_bound_id
                        ) VALUES (%s, 'full', %s, 'running', 0, %s)
                        """,
                        (batch_no, data_version, upper_bound_id),
                    )
                else:
                    # single 批次不需要全量扫描版本、游标和固定上界。
                    cursor.execute(
                        """
                        INSERT INTO expert_index_batch(batch_no, run_mode, status)
                        VALUES (%s, %s, 'running')
                        """,
                        (batch_no, run_mode),
                    )
                # 返回刚创建批次的自增主键，供后续数据关联使用。
                batch_id = int(cursor.lastrowid)
            # 先提交 running 批次，后续任何阶段失败时才能回写其最终状态。
            connection.commit()
            return batch_id
        finally:
            connection.close()

    def finish_batch(
        self,
        batch_id: int,
        *,
        status: str,
        extracted_count: int = 0,
        stage_count: int = 0,
        chunk_count: int = 0,
        rejected_count: int = 0,
        cursor_id: int | None = None,
        candidate_count: int | None = None,
        success_count: int | None = None,
        failed_count: int | None = None,
        deactivated_count: int | None = None,
        error_summary: dict[str, Any] | None = None,
    ) -> None:
        """写回批次最终状态、各阶段计数和结束时间。

        本方法只更新批次主表，不写 Stage 或 Chunk。``single`` 写入处理数量和错误摘要；
        ``full`` 还必须完整提供最终游标及专家分类计数。两种模式都会设置结束时间。

        ``error_summary`` 会序列化为 JSON，只应包含异常类型、机器码或简短安全说明，
        不应包含专家背景、经历正文、联系方式或 Chunk Text。
        """

        full_counts = (
            cursor_id,
            candidate_count,
            success_count,
            failed_count,
            deactivated_count,
        )
        is_full = any(value is not None for value in full_counts)
        if is_full and any(value is None for value in full_counts):
            raise ValueError("full 批次结束时必须提供完整游标和专家计数")

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                error_json = _json(error_summary) if error_summary else None
                if is_full:
                    cursor.execute(
                        """
                        UPDATE expert_index_batch
                        SET status=%s, cursor_id=%s, candidate_count=%s,
                            success_count=%s, failed_count=%s, deactivated_count=%s,
                            extracted_count=%s, stage_count=%s, chunk_count=%s,
                            rejected_count=%s, error_summary=%s,
                            finished_at=CURRENT_TIMESTAMP(6)
                        WHERE id=%s
                        """,
                        (
                            status, cursor_id, candidate_count, success_count,
                            failed_count, deactivated_count, extracted_count,
                            stage_count, chunk_count, rejected_count, error_json,
                            batch_id,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE expert_index_batch
                        SET status=%s, extracted_count=%s, stage_count=%s,
                            chunk_count=%s, rejected_count=%s, error_summary=%s,
                            finished_at=CURRENT_TIMESTAMP(6)
                        WHERE id=%s
                        """,
                        (
                            status, extracted_count, stage_count, chunk_count,
                            rejected_count, error_json, batch_id,
                        ),
                    )
            connection.commit()
        finally:
            connection.close()

    def resume_batch(self, batch_id: int) -> FullBatchCheckpoint:
        """锁定并重新激活中断的 full 批次，返回可靠页级检查点。

        只允许恢复 ``running`` 或 ``failed`` 的 full 批次。``success`` 已完成，
        ``partial`` 也已扫描至固定上界，因此两者都不能续跑。方法在一个事务中校验批次
        类型、状态、游标和累计计数，将状态改回 running，并清除结束时间和错误摘要。
        返回值供 Pipeline 从原版本、固定上界和最后完整页之后继续处理。
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, batch_no, run_mode, data_version, status,
                           cursor_id, upper_bound_id, candidate_count,
                           success_count, failed_count, rejected_count,
                           deactivated_count, extracted_count, stage_count,
                           chunk_count
                    FROM expert_index_batch
                    WHERE id=%s
                    FOR UPDATE
                    """,
                    (batch_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise BatchNotResumableError(f"批次 {batch_id} 不存在")
                if row.get("run_mode") != "full":
                    raise BatchNotResumableError(
                        f"批次 {batch_id} 不是 full 批次，不能断点续跑"
                    )
                status = row.get("status")
                if status not in {"running", "failed"}:
                    raise BatchNotResumableError(
                        f"批次 {batch_id} 状态为 {status}，不能断点续跑"
                    )

                data_version = row.get("data_version")
                cursor_id = row.get("cursor_id")
                upper_bound_id = row.get("upper_bound_id")
                if not data_version or cursor_id is None or upper_bound_id is None:
                    raise BatchNotResumableError(
                        f"批次 {batch_id} 缺少 data_version 或扫描游标"
                    )
                cursor_id = int(cursor_id)
                upper_bound_id = int(upper_bound_id)
                if cursor_id < 0 or upper_bound_id < 0 or cursor_id > upper_bound_id:
                    raise BatchNotResumableError(f"批次 {batch_id} 的扫描游标无效")

                count_names = (
                    "candidate_count",
                    "success_count",
                    "failed_count",
                    "rejected_count",
                    "deactivated_count",
                    "extracted_count",
                    "stage_count",
                    "chunk_count",
                )
                counts = {name: int(row.get(name) or 0) for name in count_names}
                if any(value < 0 for value in counts.values()):
                    raise BatchNotResumableError(f"批次 {batch_id} 的累计计数无效")

                checkpoint = FullBatchCheckpoint(
                    batch_id=int(row["id"]),
                    batch_no=str(row["batch_no"]),
                    data_version=str(data_version),
                    cursor_id=cursor_id,
                    upper_bound_id=upper_bound_id,
                    **counts,
                )
                cursor.execute(
                    """
                    UPDATE expert_index_batch
                    SET status='running', error_summary=NULL, finished_at=NULL
                    WHERE id=%s
                    """,
                    (batch_id,),
                )
            connection.commit()
            return checkpoint
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def prepare_batch_items(self, batch_id: int, expert_ids: list[int]) -> None:
        """在处理一页候选前，幂等登记该页专家的 ``pending`` 明细。

        唯一键 ``(batch_id, expert_id)`` 保证重复登记不产生新行，冲突时也不覆盖已有
        状态。该操作只声明本页候选已进入批次，不表示专家已经开始处理。
        """

        if not expert_ids:
            return
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                for expert_id in expert_ids:
                    cursor.execute(
                        """
                        INSERT INTO expert_index_batch_item(batch_id, expert_id, status)
                        VALUES (%s, %s, 'pending')
                        ON DUPLICATE KEY UPDATE expert_id=VALUES(expert_id)
                        """,
                        (batch_id, expert_id),
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_batch_item_running(self, batch_id: int, expert_id: int) -> None:
        """把已登记专家标记为 running，并重置本次尝试的结果字段。

        每次调用都会增加 ``attempt_count``，清空上次计数、结果码、错误和结束时间，
        并记录新的开始时间。后续处理结果由 :meth:`finish_batch_item` 写入。
        """

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE expert_index_batch_item
                    SET status='running', attempt_count=attempt_count+1,
                        extracted_count=0, stage_count=0, chunk_count=0,
                        result_code=NULL, error_type=NULL, error_message=NULL,
                        started_at=CURRENT_TIMESTAMP(6), finished_at=NULL
                    WHERE batch_id=%s AND expert_id=%s
                    """,
                    (batch_id, expert_id),
                )
            connection.commit()
        finally:
            connection.close()

    def finish_batch_item(
        self,
        batch_id: int,
        expert_id: int,
        *,
        status: str,
        extracted_count: int = 0,
        stage_count: int = 0,
        chunk_count: int = 0,
        result_code: str | None = None,
        error_type: str | None = None,
    ) -> None:
        """写入一位专家的终态、数量和安全机器结果。

        只接受 success/rejected/deactivated/failed 四个终态。``result_code`` 用于区分
        ``expert_deleted``、``no_chunkable_content`` 等可预期业务结果；``error_type``
        只保存异常类名。方法同时写入本次实际处理数量和结束时间，不接收异常消息、
        源字段或 Chunk Text。
        """

        if status not in _BATCH_ITEM_FINAL_STATUSES:
            raise ValueError(f"不支持的批次明细终态: {status}")
        safe_result_code = _safe_identifier(result_code, max_length=64)
        safe_error_type = _safe_identifier(error_type, max_length=128)
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE expert_index_batch_item
                    SET status=%s, extracted_count=%s, stage_count=%s,
                        chunk_count=%s, result_code=%s, error_type=%s,
                        error_message=NULL, finished_at=CURRENT_TIMESTAMP(6)
                    WHERE batch_id=%s AND expert_id=%s
                    """,
                    (
                        status, extracted_count, stage_count, chunk_count,
                        safe_result_code, safe_error_type, batch_id, expert_id,
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def update_batch_progress(
        self,
        batch_id: int,
        *,
        cursor_id: int,
        candidate_count: int,
        success_count: int,
        failed_count: int,
        rejected_count: int,
        deactivated_count: int,
        extracted_count: int,
        stage_count: int,
        chunk_count: int,
    ) -> None:
        """在一整页专家到达终态后，持久化全量游标与累计计数。

        该方法把游标和所有累计数量作为同一个页级检查点写入 running 批次。Pipeline
        只在整页明细全部到达终态后调用，因此中断时游标不会越过未可靠记录的专家；
        续跑从该游标之后开始，未完成页会整体重跑。
        """

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE expert_index_batch
                    SET cursor_id=%s, candidate_count=%s, success_count=%s,
                        failed_count=%s, rejected_count=%s, deactivated_count=%s,
                        extracted_count=%s, stage_count=%s, chunk_count=%s
                    WHERE id=%s AND status='running'
                    """,
                    (
                        cursor_id, candidate_count, success_count, failed_count,
                        rejected_count, deactivated_count, extracted_count,
                        stage_count, chunk_count, batch_id,
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def persist_stage_snapshot(
        self,
        profile: ProfileStage,
        employments: list[EmploymentStage],
        projects: list[ProjectStage],
    ) -> int:
        """在一个事务中保存一位专家的完整 Stage 快照。

        Parameters
        ----------
        profile:
            专家级 Stage，一位专家只有一行，按 ``expert_id`` Upsert。
        employments/projects:
            本轮从有效源记录生成的工作、项目 Stage 完整列表。

        Effects
        -------
        1. Upsert 专家 Stage；
        2. 删除该专家旧工作/项目 Stage，再插入本轮完整 Stage 快照；
        3. 全部成功才 commit，任意失败则 rollback。

        Returns
        -------
        int
            本轮 Stage 行数，即 ``1 + 工作经历数 + 项目经历数``。教育经历不会单独
            落一张 Stage 表，而是已聚合进 Profile Stage 的学校和专业数组。
        """

        # 一位专家的全部 Stage 写入共用同一连接和事务。
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                self._replace_stage_snapshot(cursor, profile, employments, projects)
            # 全部表更新成功后，新快照才整体对其他连接可见。
            connection.commit()
            return 1 + len(employments) + len(projects)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def persist_rejected_stage_snapshot(
        self,
        profile: ProfileStage,
        employments: list[EmploymentStage],
        projects: list[ProjectStage],
        *,
        reason: str,
    ) -> int:
        """保存“存在但无可生成内容”的当前 Stage，并标记拒绝原因。

        方法先整体替换 Profile、工作和项目 Stage，再在同一事务中把 Profile 标记为
        rejected。``reason`` 只接受安全机器码。返回本轮 Stage 行数；向量删除不属于
        本事务，由 Pipeline 另行发送空快照。
        """

        safe_reason = _safe_identifier(reason, max_length=64)
        if not safe_reason:
            raise ValueError("rejected reason 不能为空")
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                self._replace_stage_snapshot(cursor, profile, employments, projects)
                cursor.execute(
                    """
                    UPDATE expert_index_stage
                    SET stage_status='rejected', reject_reason=%s,
                        build_batch_id=%s
                    WHERE expert_id=%s
                    """,
                    (safe_reason, profile.build_batch_id, profile.expert_id),
                )
            connection.commit()
            return 1 + len(employments) + len(projects)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_deleted_stage(self, expert_id: int, *, build_batch_id: int) -> None:
        """清理已逻辑删除专家的经历 Stage，并标记 Profile Stage。

        在同一事务中删除工作和项目 Stage，并把已存在的 Profile Stage 标记为
        ``rejected/expert_deleted``。没有历史 Stage 时不会新增占位数据，操作仍成功。
        本方法不读取源正文，也不删除向量；向量由 Pipeline 通过空快照清理。
        """

        if expert_id <= 0 or build_batch_id <= 0:
            raise ValueError("expert_id 和 build_batch_id 必须大于 0")
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM expert_index_employment_stage WHERE expert_id=%s",
                    (expert_id,),
                )
                cursor.execute(
                    "DELETE FROM expert_index_project_stage WHERE expert_id=%s",
                    (expert_id,),
                )
                cursor.execute(
                    """
                    UPDATE expert_index_stage
                    SET stage_status='rejected', reject_reason='expert_deleted',
                        build_batch_id=%s
                    WHERE expert_id=%s
                    """,
                    (build_batch_id, expert_id),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _replace_stage_snapshot(
        self,
        cursor: Any,
        profile: ProfileStage,
        employments: list[EmploymentStage],
        projects: list[ProjectStage],
    ) -> None:
        """在调用方事务内替换一位专家的 Stage 当前快照。

        先 Upsert Profile，再删除该专家旧工作和项目 Stage，最后插入本轮完整经历列表。
        本方法不提交或回滚，事务由调用它的公开方法统一管理。
        """

        self._upsert_profile(cursor, profile)
        cursor.execute(
            "DELETE FROM expert_index_employment_stage WHERE expert_id=%s",
            (profile.expert_id,),
        )
        cursor.execute(
            "DELETE FROM expert_index_project_stage WHERE expert_id=%s",
            (profile.expert_id,),
        )
        for item in employments:
            self._insert_employment(cursor, item)
        for item in projects:
            self._insert_project(cursor, item)

    @staticmethod
    def _upsert_profile(cursor: Any, item: ProfileStage) -> None:
        """在调用方事务中插入或更新一行 Profile Stage。

        以 ``expert_id`` 为主键执行 Upsert，完整覆盖标准化字段、来源哈希、数据版本和
        构建批次。更新已有行时恢复 ``stage_status=ready`` 并清除旧拒绝原因；学校和
        专业集合序列化为 JSON 数组。本方法不提交事务。
        """

        cursor.execute(
            """
            INSERT INTO expert_index_stage(
                expert_id, serial_number, company_name_raw, company_name_norm,
                company_normalization_status, role_name_raw, role_code, role_level,
                job_status_norm, abroad_flag_norm, background_clean,
                school_names_json, major_names_json, work_years,
                eligible_for_recommendation, verification_status, source_hash,
                source_updated_at, data_version, stage_status, build_batch_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'ready',%s)
            ON DUPLICATE KEY UPDATE
                serial_number=VALUES(serial_number), company_name_raw=VALUES(company_name_raw),
                company_name_norm=VALUES(company_name_norm),
                company_normalization_status=VALUES(company_normalization_status),
                role_name_raw=VALUES(role_name_raw), role_code=VALUES(role_code),
                role_level=VALUES(role_level), job_status_norm=VALUES(job_status_norm),
                abroad_flag_norm=VALUES(abroad_flag_norm),
                background_clean=VALUES(background_clean),
                school_names_json=VALUES(school_names_json),
                major_names_json=VALUES(major_names_json), work_years=VALUES(work_years),
                eligible_for_recommendation=VALUES(eligible_for_recommendation),
                verification_status=VALUES(verification_status),
                source_hash=VALUES(source_hash), source_updated_at=VALUES(source_updated_at),
                data_version=VALUES(data_version), stage_status='ready', reject_reason=NULL,
                build_batch_id=VALUES(build_batch_id)
            """,
            (
                item.expert_id, item.serial_number, item.company_name_raw, item.company_name_norm,
                item.company_normalization_status, item.role_name_raw, item.role_code,
                item.role_level, item.job_status_norm, item.abroad_flag_norm,
                item.background_clean, _json(list(item.school_names_raw)),
                _json(list(item.major_names_raw)), item.work_years,
                item.eligible_for_recommendation, item.verification_status, item.source_hash,
                item.source_updated_at, item.data_version,
                item.build_batch_id,
            ),
        )

    @staticmethod
    def _insert_employment(cursor: Any, item: EmploymentStage) -> None:
        """在调用方事务中插入一条标准化后的工作经历 Stage。

        旧工作快照已由调用方删除，因此这里只执行 INSERT。来源定位字段随 Stage 保存，
        行业和主题集合序列化为 JSON 数组。本方法不提交事务。
        """

        cursor.execute(
            """
            INSERT INTO expert_index_employment_stage(
                expert_id, source_table, source_field, source_field_id, source_item_key,
                company_name_raw, company_name_norm, company_normalization_status,
                role_name_raw, role_code, role_level, start_month, end_month, is_current,
                responsibilities_clean, achievements_clean, industry_codes_json,
                topic_codes_json, source_hash, source_updated_at,
                data_version, build_batch_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                item.expert_id, item.source_table, item.source_field, item.source_field_id,
                item.source_item_key,
                item.company_name_raw, item.company_name_norm,
                item.company_normalization_status, item.role_name_raw, item.role_code,
                item.role_level, item.start_month, item.end_month, item.is_current,
                item.responsibilities_clean, item.achievements_clean,
                _json(list(item.industry_codes)), _json(list(item.topic_codes)),
                item.source_hash, item.source_updated_at, item.data_version,
                item.build_batch_id,
            ),
        )

    @staticmethod
    def _insert_project(cursor: Any, item: ProjectStage) -> None:
        """在调用方事务中插入一条标准化后的项目经历 Stage。

        旧项目快照已由调用方删除，因此这里只执行 INSERT。项目内容、来源定位、数据
        版本和构建批次一并写入，行业和主题集合序列化为 JSON 数组。本方法不提交事务。
        """

        cursor.execute(
            """
            INSERT INTO expert_index_project_stage(
                expert_id, source_table, source_field, source_field_id, source_item_key,
                project_name,
                company_or_client_raw, company_or_client_norm,
                company_normalization_status, start_month, end_month,
                objective_clean, actions_clean, results_clean, industry_codes_json,
                topic_codes_json, source_hash, source_updated_at,
                data_version, build_batch_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                item.expert_id, item.source_table, item.source_field, item.source_field_id,
                item.source_item_key, item.project_name, item.company_or_client_raw,
                item.company_or_client_norm,
                item.company_normalization_status, item.start_month,
                item.end_month, item.objective_clean, item.actions_clean, item.results_clean,
                _json(list(item.industry_codes)), _json(list(item.topic_codes)),
                item.source_hash, item.source_updated_at, item.data_version,
                item.build_batch_id,
            ),
        )

def _json(value: Any) -> str:
    """把数组或字典序列化为稳定且符合 JSON 标准的紧凑字符串。

    字典键排序保证相同输入得到相同字符串；中文不转义；NaN 和 Infinity 会触发异常，
    从而由外层事务回滚，而不是写入非标准 JSON。
    """

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_identifier(value: str | None, *, max_length: int) -> str | None:
    """截断并校验写入批次明细的机器标识。

    空值返回 ``None``；非空值只允许字母、数字、下划线、点和短横线，防止把异常消息
    或专家正文误写入 ``result_code``、``error_type`` 等审计字段。
    """

    if value is None:
        return None
    candidate = value.strip()[:max_length]
    if not candidate:
        return None
    if not all(character.isalnum() or character in "_.-" for character in candidate):
        raise ValueError("批次结果标识只能包含字母、数字、下划线、点和短横线")
    return candidate
