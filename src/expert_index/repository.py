"""专家索引派生库的 MySQL Repository（数据访问层）。

这个文件集中管理“加工结果如何写入/读出 MySQL”，自身不负责读取业务源表，
也不负责清洗文本或生成 Chunk。完整职责包括：

1. 执行受版本控制的建表或迁移 SQL；
2. 用 MySQL 命名锁协调 single/full 任务，并管理批次主表；
3. 记录每位候选专家的 pending、running 和最终状态，以及页游标；
4. 保存专家、工作和项目 Stage 当前快照；
5. 按 ``vector_doc_id`` 幂等写入最终 ``expert_chunk``；
6. 停用删除、拒绝或不再属于当前快照的旧 Chunk，并重算 ``metadata_hash``；
7. 为导出和人工验收提供只读查询。

最重要的事务边界在 :meth:`persist_expert`、:meth:`persist_rejected_expert` 和
:meth:`deactivate_deleted_expert`：一位专家的 Stage、Chunk 与停用状态必须作为整体
提交。任意 SQL 失败都会回滚这一整位专家，避免出现“Stage 已更新、旧 Chunk 仍有效”
等半更新状态。

本文件不是独立运行脚本，直接运行不会自动写库；它由 ``cli.py`` 和 ``pipeline.py``
创建并调用。实际效果取决于调用的方法，例如 ``init-db`` 执行建表，``build`` 与
``build-batch`` 调用批次和专家持久化方法，``export`` 查询 active Chunk。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import pymysql

from .chunk_builder import calculate_metadata_hash
from .config import DatabaseConfig
from .models import ChunkRecord, EmploymentStage, ProfileStage, ProjectStage

# MySQL 命名锁由连接持有。第一版让 single/full 共用一个全局加工锁，以最简单、可靠的
# 方式避免两个全量任务并发，也避免全量期间单独重建专家。后续若有并发吞吐需求，可以
# 在保持本接口不变的前提下升级为“全量锁 + 专家级锁”。
BUILD_LOCK_PREFIX = "expert_index_build_v1"

_BATCH_ITEM_FINAL_STATUSES = {"success", "rejected", "deactivated", "failed"}


class ConcurrentBuildError(RuntimeError):
    """另一个专家索引加工任务已持有第一版全局互斥锁。"""


class MySQLIndexRepository:
    """封装索引派生表的全部 MySQL 操作。

    ``MySQLSourceAdapter`` 只从业务源表读取，Repository 只管理派生表，因此源库读取
    与索引库写入不会相互耦合。测试可以用假 Repository 替换本类而无需连接 MySQL。
    """

    def __init__(self, config: DatabaseConfig) -> None:
        # 保存索引库连接配置。这里只保存参数，不会在构造对象时立即建立连接；
        # 每个公开操作都会创建自己的短连接，并在 finally 中可靠关闭。
        self.config = config

    @contextmanager
    def build_lock(self, run_mode: str) -> Iterator[None]:
        """在一次 single/full 加工期间持有 MySQL 命名锁。

        功能与用途
        ----------
        Pipeline 在创建批次之前进入本上下文，并在整次任务完成或抛出异常后退出。
        ``GET_LOCK(..., 0)`` 不等待：若已有任务运行，立即抛出
        :class:`ConcurrentBuildError`，因此不会出现两个全量任务同时写批次游标，也不会
        出现单专家任务与全量任务同时重建同一专家。

        锁只用于任务编排，不包裹业务 SQL 事务。连接必须在整个 ``yield`` 期间保持打开，
        因为关闭连接会让 MySQL 自动释放命名锁。``run_mode`` 只用于校验和错误说明；
        第一版 single/full 刻意共用同一个锁名，以正确性优先。
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
                # RELEASE_LOCK 失败不应掩盖 Pipeline 原本的业务异常；连接关闭后 MySQL
                # 仍会兜底释放该会话持有的命名锁。
                with suppress(Exception), connection.cursor() as cursor:
                    cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
            connection.close()

    def initialize_schema(self, sql_path: Path) -> None:
        """执行指定的建表或迁移 SQL 文件。

        Parameters
        ----------
        sql_path:
            SQL 文件路径。新库通常传入 ``001_create_expert_index_tables.sql``；已有库
            做结构升级时传入对应迁移脚本，例如 ``004``。

        Effects
        -------
        - 按分号拆分并依次执行文件内 SQL；
        - 全部语句成功后提交；
        - 任意语句失败时回滚当前连接并把原异常继续抛给 CLI；
        - 本方法只执行文件已有内容，不会自行删除或创建额外业务表。

        Notes
        -----
        DDL 在不同 MySQL 版本中可能隐式提交，因此迁移脚本仍应设计为可审查、可备份，
        不能仅依赖 Python 的 rollback 保证所有结构变化都可撤销。
        """

        # 统一用 UTF-8 读取版本库中的 SQL，避免中文 COMMENT 受系统编码影响。
        script = sql_path.read_text(encoding="utf-8")
        # 当前项目 SQL 不包含存储过程，因此可以安全地按分号切成单条语句；空语句丢弃。
        statements = [statement.strip() for statement in script.split(";") if statement.strip()]
        # DDL 需要显式 commit，因此此处不开启 autocommit。
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            # 只有所有语句都没有异常时才提交连接级事务。
            connection.commit()
        except Exception:
            # 尽可能撤销尚未被 MySQL 隐式提交的语句，再把异常交给上层处理。
            connection.rollback()
            raise
        finally:
            # 无论成功还是失败都关闭连接，避免 CLI 多次运行后耗尽连接数。
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

        ``single`` 保留原来的最小 INSERT，兼容尚未执行 005 迁移的单专家环境。
        ``full`` 必须同时提供 ``data_version`` 和固定的 ``upper_bound_id``，并把初始
        ``cursor_id`` 写为 0；Pipeline 随后通过 :meth:`update_batch_progress` 按页推进。
        返回的自增 ``batch_id`` 会进入 Stage、Chunk 和专家级批次明细。
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
                    # single 不依赖 005 新列，保留现有数据库的向后兼容能力。
                    cursor.execute(
                        """
                        INSERT INTO expert_index_batch(batch_no, run_mode, status)
                        VALUES (%s, %s, 'running')
                        """,
                        (batch_no, run_mode),
                    )
                # lastrowid 是刚插入 expert_index_batch 的自增主键。
                batch_id = int(cursor.lastrowid)
            # 批次必须先落库，后续 Source/Normalize 失败时才能回写 failed 状态。
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

        该方法通常由 Pipeline 的成功或异常分支调用。它只更新批次统计，不写 Stage
        或 Chunk。``error_summary`` 会序列化为 JSON；调用方只能传异常类型、简短说明
        等安全信息，不应放入专家背景、工作描述、项目正文或联系方式。

        Effects
        -------
        full 调用额外传入最终游标和五类专家计数；single 不传这些参数，继续使用旧列。
        两种模式都会写入结束时间。该分支设计让单专家命令不强依赖 005 迁移，同时让
        全量批次主表能完整反映最终处理结果。
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

    def prepare_batch_items(self, batch_id: int, expert_ids: list[int]) -> None:
        """在处理一页候选前，幂等登记该页专家的 ``pending`` 明细。

        Pipeline 每取得并校验一页候选后调用一次。唯一键 ``(batch_id, expert_id)``
        保证重复登记不会产生重复行；冲突时不覆盖已有状态，便于保留已完成结果。
        本方法只准备明细，不表示专家已经开始处理。
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
        """把一位已登记专家标记为 running，并增加本批次尝试次数。

        该方法在 Pipeline 真正读取或停用专家之前调用。重新尝试时会清理上次终态计数
        和结果码；``started_at`` 记录本次开始时间。若明细不存在，受影响行数为 0，
        Pipeline 的测试会通过调用顺序防止这种用法。
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
        只保存异常类名。本方法不接收异常消息、源字段或 Chunk Text，避免批次明细泄露
        真实数据。它由 Pipeline 在单个专家处理完成后调用。
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

        Pipeline 只有在该页所有批次明细都成功写成终态后才调用本方法。因此数据库中的
        ``cursor_id`` 永远不会越过尚未可靠记录的专家，为以后断点续跑保留可信基础。
        当前第一版仍从 0 开始新任务，不自动恢复旧批次。
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

    def persist_expert(
        self,
        profile: ProfileStage,
        employments: list[EmploymentStage],
        projects: list[ProjectStage],
        chunks: list[ChunkRecord],
    ) -> int:
        """在一个事务中保存一位专家的完整派生快照。

        Parameters
        ----------
        profile:
            专家级 Stage，一位专家只有一行，按 ``expert_id`` Upsert。
        employments/projects:
            本轮从有效源记录生成的工作、项目 Stage 完整列表。
        chunks:
            已通过 Validator 的最终 Chunk 完整列表。

        Effects
        -------
        1. Upsert 专家 Stage；
        2. 删除该专家旧工作/项目 Stage，再插入本轮完整 Stage 快照；
        3. 按 ``vector_doc_id`` Upsert 本轮 Chunk；
        4. 将没有出现在本轮 ``vector_doc_id`` 集合中的旧 active Chunk 停用；
        5. 全部成功才 commit，任意失败则 rollback。

        Returns
        -------
        int
            本轮 Stage 行数，即 ``1 + 工作经历数 + 项目经历数``。教育经历不会单独
            落一张 Stage 表，而是已聚合进 Profile Stage 的学校和专业数组。
        """

        # 一位专家的全部写操作共用此连接，才能形成同一个数据库事务。
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                # 正常产出与 rejected 路径复用同一套 Stage 当前快照替换规则。
                self._replace_stage_snapshot(cursor, profile, employments, projects)
                # Chunk 不先整批删除，以 vector_doc_id Upsert 可以保留数据库行主键和创建时间。
                for chunk in chunks:
                    self._upsert_chunk(cursor, chunk)
                # 最后才停用陈旧 Chunk。若之前任一步失败，本步骤不会执行。
                self._deactivate_stale_chunks(
                    cursor,
                    profile.expert_id,
                    [chunk.vector_doc_id for chunk in chunks],
                    build_batch_id=profile.build_batch_id,
                )
            # commit 是这位专家的新快照对其他连接正式可见的时刻。
            connection.commit()
            return 1 + len(employments) + len(projects)
        except Exception:
            # 回滚包括 Stage 删除、Stage 插入、Chunk Upsert 和停用，旧快照仍保持完整。
            connection.rollback()
            raise
        finally:
            connection.close()

    def persist_rejected_expert(
        self,
        profile: ProfileStage,
        employments: list[EmploymentStage],
        projects: list[ProjectStage],
        *,
        reason: str,
    ) -> int:
        """保存“存在但无可生成内容”的当前 Stage，并停用全部旧 Chunk。

        用途
        ----
        ``Pipeline._process_expert`` 已完成四表读取和标准化，但 ChunkFactory 返回空列表
        时调用本方法。该情况是可预期的 rejected 业务结果，不应留下上一次构建的 active
        Chunk。Profile、经历 Stage 刷新、Profile 拒绝标记、旧 Chunk 停用及
        ``metadata_hash`` 重算都在同一个专家事务中完成。

        返回值仍是本轮 Stage 行数，供批次统计使用。``reason`` 只允许安全机器码，不能
        传入源数据或异常正文。
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
                self._deactivate_stale_chunks(
                    cursor,
                    profile.expert_id,
                    [],
                    build_batch_id=profile.build_batch_id,
                )
            connection.commit()
            return 1 + len(employments) + len(projects)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def deactivate_deleted_expert(self, expert_id: int, *, build_batch_id: int) -> int:
        """处理源主档已逻辑删除的专家，并返回实际停用的 Chunk 数。

        全量候选扫描会保留 ``experts.deleted_at`` 非空的行，但不会再读取其四表正文。
        Pipeline 对这类候选调用本方法：清空工作/项目当前 Stage，将已有 Profile Stage
        标记为 rejected，并把全部 active Chunk 置为 inactive。每条被停用 Chunk 都按
        ``is_active=False`` 同步重算 ``metadata_hash``，让下游能够识别 Metadata 变化。

        所有修改属于一个事务；不存在历史 Stage/Chunk 时也是成功的幂等操作。
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
                deactivated_count = self._deactivate_stale_chunks(
                    cursor,
                    expert_id,
                    [],
                    build_batch_id=build_batch_id,
                )
            connection.commit()
            return deactivated_count
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

        ``persist_expert`` 和 ``persist_rejected_expert`` 共用这个内部步骤，避免两条路径
        的 Profile/经历保存规则产生漂移。本方法不提交也不回滚，事务由公开方法管理。
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
        """插入或更新一行专家级 Stage。

        ``expert_index_stage.expert_id`` 是主键：首次构建执行 INSERT，重复构建触发
        ``ON DUPLICATE KEY UPDATE``。更新时会清除旧拒绝原因并恢复 ``stage_status``
        为 ready。学校、专业元组在写入 JSON 列前转换为列表并严格序列化。

        该私有方法不 commit，事务由 :meth:`persist_expert` 统一控制。
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
        """插入一条工作经历 Stage 当前快照。

        调用前，:meth:`persist_expert` 已删除该专家的全部旧工作 Stage，因此这里只需
        INSERT，不需要 Upsert。``source_field_id`` 可回查源表行；``source_item_key``
        仍保留在中间层用于 Same Segment 和快照唯一约束，但不再进入最终 expert_chunk。
        行业、主题元组以 JSON 数组写入。该方法不提交事务。
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
        """把一条已经清洗、标准化完成的项目经历 Stage，插入MySQL的expert_index_project_stage 中间表。
        它接收两个参数：
            cursor：当前数据库事务的游标，用来执行 SQL。
            item：一条 ProjectStage 对象，包含项目名称、公司、日期、描述、来源追踪和版本信息。
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

    @staticmethod
    def _upsert_chunk(cursor: Any, item: ChunkRecord) -> None:
        """把一条最终 ChunkRecord 保存到 MySQL 的 expert_chunk 表中。

        “Upsert”表示：
            数据不存在时：执行 INSERT，新增一条 Chunk。
            vector_doc_id 已存在时：执行 UPDATE，更新原来的 Chunk。
            不会因为重复构建同一专家而新增重复记录。
        """

        cursor.execute(
            """
            INSERT INTO expert_chunk(
                vector_doc_id, expert_id, serial_number, chunk_type, chunk_subtype,
                chunk_index, chunk_total, chunk_text, char_count,
                source_table, source_field, source_field_id,
                source_version, source_updated_at, source_hash, content_hash, metadata_hash,
                company_name_raw, company_name_norm, role_name_raw, role_code, role_level,
                job_status_norm, abroad_flag_norm, work_years, project_name,
                start_month, end_month, is_current,
                industry_codes, function_codes, topic_codes,
                school_names_raw, major_names_raw, eligible_for_recommendation,
                origin_type, verification_status, weight_tier, chunk_policy_version,
                metadata_schema_version,
                embedding_model_alias, embedding_model_version, indexed_at,
                is_active, build_batch_id
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s
            )
            ON DUPLICATE KEY UPDATE
                -- vector_doc_id 是命中重复行的唯一键，本身无须更新；其余上游权威列覆盖。
                serial_number=VALUES(serial_number), chunk_type=VALUES(chunk_type),
                chunk_subtype=VALUES(chunk_subtype), chunk_index=VALUES(chunk_index),
                chunk_total=VALUES(chunk_total), chunk_text=VALUES(chunk_text),
                char_count=VALUES(char_count),
                source_table=VALUES(source_table), source_field=VALUES(source_field),
                source_field_id=VALUES(source_field_id),
                source_version=VALUES(source_version), source_updated_at=VALUES(source_updated_at),
                source_hash=VALUES(source_hash), content_hash=VALUES(content_hash),
                metadata_hash=VALUES(metadata_hash),
                company_name_raw=VALUES(company_name_raw),
                company_name_norm=VALUES(company_name_norm),
                role_name_raw=VALUES(role_name_raw), role_code=VALUES(role_code),
                role_level=VALUES(role_level), job_status_norm=VALUES(job_status_norm),
                abroad_flag_norm=VALUES(abroad_flag_norm), work_years=VALUES(work_years),
                project_name=VALUES(project_name),
                start_month=VALUES(start_month), end_month=VALUES(end_month),
                is_current=VALUES(is_current), industry_codes=VALUES(industry_codes),
                function_codes=VALUES(function_codes), topic_codes=VALUES(topic_codes),
                school_names_raw=VALUES(school_names_raw),
                major_names_raw=VALUES(major_names_raw),
                eligible_for_recommendation=VALUES(eligible_for_recommendation),
                origin_type=VALUES(origin_type),
                verification_status=VALUES(verification_status), weight_tier=VALUES(weight_tier),
                chunk_policy_version=VALUES(chunk_policy_version),
                metadata_schema_version=VALUES(metadata_schema_version),
                is_active=VALUES(is_active), build_batch_id=VALUES(build_batch_id)
            """,
            (
                # 参数顺序必须与 INSERT 列顺序完全一致。测试会比较 %s 数量与参数数量，
                # 防止以后增删字段时出现“SQL 可执行但值写错列”的问题。
                item.vector_doc_id, item.expert_id, item.serial_number, item.chunk_type,
                item.chunk_subtype, item.chunk_index, item.chunk_total, item.chunk_text,
                item.char_count, item.source_table, item.source_field, item.source_field_id,
                item.source_version,
                item.source_updated_at, item.source_hash, item.content_hash,
                item.metadata_hash, item.company_name_raw, item.company_name_norm,
                item.role_name_raw, item.role_code, item.role_level, item.job_status_norm,
                item.abroad_flag_norm, item.work_years, item.project_name,
                item.start_month, item.end_month, item.is_current,
                _json(list(item.industry_codes)), _json(list(item.function_codes)),
                _json(list(item.topic_codes)), _json(list(item.school_names_raw)),
                _json(list(item.major_names_raw)), item.eligible_for_recommendation,
                item.origin_type, item.verification_status, item.weight_tier,
                item.chunk_policy_version, item.metadata_schema_version,
                item.embedding_model_alias, item.embedding_model_version,
                item.indexed_at, item.is_active,
                item.build_batch_id,
            ),
        )

    @staticmethod
    def _deactivate_stale_chunks(
        cursor: Any,
        expert_id: int,
        current_ids: list[str],
        *,
        build_batch_id: int,
    ) -> int:
        """锁定并停用本轮未生成的旧 Chunk，同时重算 Metadata Hash。

        ``current_ids`` 是本轮完整 Chunk 集合的 ``vector_doc_id``。源经历被逻辑删除、
        不再满足生成条件或分片数量减少后，对应旧 ID 不会出现在该集合中，本方法将其
        ``is_active`` 更新为 0。由于 ``is_active`` 属于 Metadata Hash 载荷，不能只改
        状态列；本方法先用 ``FOR UPDATE`` 读取待停用行，复用 Chunk Builder 的统一算法
        计算 inactive 状态哈希，再逐行更新 ``metadata_hash`` 和追溯批次。

        - 列表非空：只停用不在当前集合中的 active 行；
        - 列表为空：停用该专家全部 active Chunk；
        - 本方法位于 Upsert 之后且属于同一事务，失败时全部回滚。

        返回实际停用行数。它是 Repository 内部事务步骤，不自行 commit。
        """

        select_columns = """
            vector_doc_id, expert_id, serial_number, chunk_type, chunk_subtype,
            chunk_index, chunk_total, chunk_text, char_count,
            source_table, source_field, source_field_id, source_version,
            source_updated_at, source_hash, content_hash, metadata_hash,
            company_name_raw, company_name_norm, role_name_raw, role_code, role_level,
            job_status_norm, abroad_flag_norm, work_years, project_name,
            start_month, end_month, is_current,
            industry_codes, function_codes, topic_codes,
            school_names_raw, major_names_raw, eligible_for_recommendation,
            origin_type, verification_status, weight_tier, chunk_policy_version,
            metadata_schema_version, embedding_model_alias, embedding_model_version,
            indexed_at, is_active, build_batch_id
        """
        if current_ids:
            placeholders = ",".join(["%s"] * len(current_ids))
            cursor.execute(
                f"""
                SELECT {select_columns}
                FROM expert_chunk
                WHERE expert_id=%s AND is_active=1
                  AND vector_doc_id NOT IN ({placeholders})
                FOR UPDATE
                """,
                (expert_id, *current_ids),
            )
        else:
            cursor.execute(
                f"""
                SELECT {select_columns}
                FROM expert_chunk
                WHERE expert_id=%s AND is_active=1
                FOR UPDATE
                """,
                (expert_id,),
            )

        rows = list(cursor.fetchall())
        for row in rows:
            active_record = _chunk_record_from_row(row)
            inactive_record = replace(active_record, is_active=False, build_batch_id=build_batch_id)
            inactive_hash = calculate_metadata_hash(inactive_record)
            cursor.execute(
                """
                UPDATE expert_chunk
                SET is_active=0, metadata_hash=%s, build_batch_id=%s
                WHERE vector_doc_id=%s AND is_active=1
                """,
                (inactive_hash, build_batch_id, inactive_record.vector_doc_id),
            )
        return len(rows)

    def fetch_active_chunks(self, expert_id: int, *, eligible_only: bool = False) -> list[dict[str, Any]]:
        """从 MySQL 的 expert_chunk 表中，读取某一位专家当前有效的 Chunk。

        为了让后续导出、Embedding 或检查流程只读取当前有效、符合条件的 Chunk，避免使用已停用或被拒绝的数据。
        不会在生成 Chunk 或写入数据库时调用。
        它主要在 Chunk 已经生成并落库之后使用，例如执行 export 命令时，从 expert_chunk 查询有效数据并交给后续导出或处理流程。

        """

        # 资格条件只影响可选导出，不影响 expert_chunk 中保留的完整 active 数据。
        eligibility = " AND verification_status IN ('verified','source_summary') AND weight_tier IN ('A','B')" if eligible_only else ""
        sql = f"""
            SELECT vector_doc_id, expert_id, serial_number, chunk_type, chunk_subtype,
                   chunk_index, chunk_total, chunk_text, char_count, content_hash,
                   metadata_hash, source_table, source_field, source_field_id,
                   source_version, source_updated_at,
                   source_hash, company_name_raw, company_name_norm,
                   role_name_raw, role_code, role_level, job_status_norm,
                   abroad_flag_norm, work_years, project_name, start_month, end_month,
                   is_current, industry_codes, function_codes, topic_codes,
                   school_names_raw, major_names_raw, eligible_for_recommendation,
                   origin_type, verification_status, weight_tier,
                   chunk_policy_version, metadata_schema_version, embedding_model_alias,
                   embedding_model_version, indexed_at, is_active, updated_at
            FROM expert_chunk
            WHERE expert_id=%s AND is_active=1 {eligibility}
            ORDER BY FIELD(chunk_type,'profile_summary','employment','project'),
                     source_field_id, chunk_index
        """
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, (expert_id,))
                return list(cursor.fetchall())
        finally:
            connection.close()

    def summary(self, expert_id: int) -> dict[str, Any]:
        """summary() 用于统计某位专家的 Chunk 情况，包括：
            每种 chunk_type 的数量
            verification_status
            最小和最大字符数
            有效 Chunk 数量
            它主要由 cli.py 的 inspect 命令调用
            用于开发验收、数据对账和排查问题，不参与 Chunk 生成或落库。
            因此可在包含真实专家数据的环境中安全查看结构性结果。

        本方法只读且不修改数据库，返回格式为：
        ``{"expert_id": ..., "chunks": [分组统计行, ...]}``。
        """

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_type, verification_status, COUNT(*) count,
                           MIN(char_count) min_chars, MAX(char_count) max_chars,
                           SUM(is_active=1) active_count
                    FROM expert_chunk WHERE expert_id=%s
                    GROUP BY chunk_type, verification_status
                    ORDER BY chunk_type, verification_status
                    """,
                    (expert_id,),
                )
                return {"expert_id": expert_id, "chunks": list(cursor.fetchall())}
        finally:
            connection.close()


def _json(value: Any) -> str:
    """将数组或字典转换为可写入 MySQL JSON 列的确定性字符串。

    ``sort_keys=True`` 使字典键顺序稳定；紧凑分隔符减少无意义空格；中文保持可读；
    ``allow_nan=False`` 禁止写入 JSON 标准不支持的 NaN/Infinity。非法值会立即抛出
    异常，并由外层专家事务回滚，而不是把坏数据静默写入派生表。
    """

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_identifier(value: str | None, *, max_length: int) -> str | None:
    """清理写入批次明细的机器标识，拒绝正文式内容。

    result_code/error_type 只允许字母、数字、下划线、点和短横线。这样 Repository 的
    接口层就能阻止调用方误把异常消息或专家正文写入批次审计表。
    """

    if value is None:
        return None
    candidate = value.strip()[:max_length]
    if not candidate:
        return None
    if not all(character.isalnum() or character in "_.-" for character in candidate):
        raise ValueError("批次结果标识只能包含字母、数字、下划线、点和短横线")
    return candidate


def _json_tuple(value: Any) -> tuple[str, ...]:
    """把 PyMySQL 返回的 JSON 字符串或数组统一还原为不可变字符串元组。"""

    if value is None:
        return ()
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)):
        raise TypeError("Chunk JSON Metadata 必须是数组")
    return tuple(str(item) for item in parsed)


def _optional_bool(value: Any) -> bool | None:
    """把 MySQL TINYINT 可空值转换为 ChunkRecord 使用的可空布尔值。"""

    return None if value is None else bool(value)


def _chunk_record_from_row(row: dict[str, Any]) -> ChunkRecord:
    """把已锁定的 expert_chunk 数据库行还原为哈希计算模型。

    本函数只被 :meth:`MySQLIndexRepository._deactivate_stale_chunks` 使用。完整还原而不在
    Repository 复制一份哈希字段清单，可以保证停用时和首次生成时始终调用同一个
    ``calculate_metadata_hash`` 实现。JSON、DECIMAL 和 TINYINT 在这里统一转换为领域
    模型类型，避免数据库驱动返回类型影响确定性哈希。
    """

    work_years = row.get("work_years")
    return ChunkRecord(
        expert_id=int(row["expert_id"]),
        serial_number=row.get("serial_number"),
        chunk_type=str(row["chunk_type"]),
        chunk_subtype=row.get("chunk_subtype"),
        chunk_index=int(row["chunk_index"]),
        chunk_total=int(row["chunk_total"]),
        chunk_text=str(row["chunk_text"]),
        char_count=int(row["char_count"]),
        vector_doc_id=str(row["vector_doc_id"]),
        source_table=str(row["source_table"]),
        source_field=str(row["source_field"]),
        source_field_id=str(row["source_field_id"]),
        source_version=str(row["source_version"]),
        source_updated_at=row.get("source_updated_at"),
        source_hash=str(row["source_hash"]),
        content_hash=str(row["content_hash"]),
        metadata_hash=str(row["metadata_hash"]),
        company_name_raw=row.get("company_name_raw"),
        company_name_norm=row.get("company_name_norm"),
        role_name_raw=row.get("role_name_raw"),
        role_code=row.get("role_code"),
        role_level=row.get("role_level"),
        job_status_norm=row.get("job_status_norm"),
        abroad_flag_norm=_optional_bool(row.get("abroad_flag_norm")),
        work_years=float(work_years) if work_years is not None else None,
        project_name=row.get("project_name"),
        start_month=row.get("start_month"),
        end_month=row.get("end_month"),
        is_current=_optional_bool(row.get("is_current")),
        industry_codes=_json_tuple(row.get("industry_codes")),
        function_codes=_json_tuple(row.get("function_codes")),
        topic_codes=_json_tuple(row.get("topic_codes")),
        school_names_raw=_json_tuple(row.get("school_names_raw")),
        major_names_raw=_json_tuple(row.get("major_names_raw")),
        eligible_for_recommendation=_optional_bool(
            row.get("eligible_for_recommendation")
        ),
        origin_type=str(row["origin_type"]),
        verification_status=str(row["verification_status"]),
        weight_tier=str(row["weight_tier"]),
        chunk_policy_version=str(row["chunk_policy_version"]),
        metadata_schema_version=str(row["metadata_schema_version"]),
        embedding_model_alias=row.get("embedding_model_alias"),
        embedding_model_version=row.get("embedding_model_version"),
        indexed_at=row.get("indexed_at"),
        is_active=bool(row["is_active"]),
        build_batch_id=int(row["build_batch_id"]),
    )
