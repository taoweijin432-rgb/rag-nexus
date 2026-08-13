"""专家索引派生库的 MySQL Repository（数据访问层）。

这个文件集中管理“加工结果如何写入/读出 MySQL”，自身不负责读取业务源表，
也不负责清洗文本或生成 Chunk。完整职责包括：

1. 执行受版本控制的建表或迁移 SQL；
2. 创建、结束 ``expert_index_batch`` 构建批次；
3. 保存专家、工作和项目 Stage 当前快照；
4. 按 ``vector_doc_id`` 幂等写入最终 ``expert_chunk``；
5. 停用同一专家已经不属于当前快照的旧 Chunk；
6. 为导出和人工验收提供只读查询。

最重要的事务边界在 :meth:`persist_expert`：一位专家的 Profile Stage、工作 Stage、
项目 Stage、全部 Chunk 以及旧 Chunk 停用必须作为一个整体提交。任意 SQL 失败都会
回滚这一整位专家，避免出现“主档是新版、经历还是旧版”的半更新状态。

本文件不是独立运行脚本，直接运行不会自动写库；它由 ``cli.py`` 和 ``pipeline.py``
创建并调用。实际效果取决于调用的方法，例如 ``init-db`` 调用建表方法，``build``
调用批次和专家持久化方法，``export`` 调用 active Chunk 查询方法。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymysql

from .config import DatabaseConfig
from .models import ChunkRecord, EmploymentStage, ProfileStage, ProjectStage


class MySQLIndexRepository:
    """封装索引派生表的全部 MySQL 操作。

    ``MySQLSourceAdapter`` 只从业务源表读取，Repository 只管理派生表，因此源库读取
    与索引库写入不会相互耦合。测试可以用假 Repository 替换本类而无需连接 MySQL。
    """

    def __init__(self, config: DatabaseConfig) -> None:
        # 保存索引库连接配置。这里只保存参数，不会在构造对象时立即建立连接；
        # 每个公开操作都会创建自己的短连接，并在 finally 中可靠关闭。
        self.config = config

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

    def start_batch(self, batch_no: str, run_mode: str) -> int:
        """创建一条 ``running`` 状态的构建批次，并返回批次主键。

        ``batch_no`` 是调用方可读的业务批次号，``run_mode`` 表示 single/full 等模式。
        返回的自增 ``batch_id`` 会写入 Stage 和 Chunk 的 ``build_batch_id``，使最终行
        可以追溯到具体运行。
        """

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                # 新批次刚创建时还没有计数结果，因此只写编号、模式和 running 状态。
                cursor.execute(
                    "INSERT INTO expert_index_batch(batch_no, run_mode, status) VALUES (%s, %s, 'running')",
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
        error_summary: dict[str, Any] | None = None,
    ) -> None:
        """写回批次最终状态、各阶段计数和结束时间。

        该方法通常由 Pipeline 的成功或异常分支调用。它只更新批次统计，不写 Stage
        或 Chunk。``error_summary`` 会序列化为 JSON；调用方只能传异常类型、简短说明
        等安全信息，不应放入专家背景、工作描述、项目正文或联系方式。

        Effects
        -------
        对指定 ``batch_id`` 更新状态、抽取数、Stage 数、Chunk 数、拒绝数、错误摘要，
        并由 MySQL 写入微秒级 ``finished_at``。方法返回 ``None``。
        """

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                # 使用参数化 SQL，避免状态或错误信息被解释为 SQL 代码。
                cursor.execute(
                    """
                    UPDATE expert_index_batch
                    SET status=%s, extracted_count=%s, stage_count=%s,
                        chunk_count=%s, rejected_count=%s, error_summary=%s,
                        finished_at=CURRENT_TIMESTAMP(6)
                    WHERE id=%s
                    """,
                    (
                        status,
                        extracted_count,
                        stage_count,
                        chunk_count,
                        rejected_count,
                        # 没有错误时保存 SQL NULL，而不是无意义的空 JSON 对象。
                        _json(error_summary) if error_summary else None,
                        batch_id,
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
                # Profile 使用 Upsert，因为专家级 Stage 以 expert_id 为唯一主键。
                self._upsert_profile(cursor, profile)

                # Stage 是当前规范化快照，因此先删除该专家旧经历，再写入完整新集合。
                # 删除仅作用于派生表且被当前事务保护，异常时可完整恢复。
                cursor.execute("DELETE FROM expert_index_employment_stage WHERE expert_id=%s", (profile.expert_id,))
                cursor.execute("DELETE FROM expert_index_project_stage WHERE expert_id=%s", (profile.expert_id,))
                # 删除后写入本轮完整列表，使 Stage 表严格表示“当前有效快照”，而不是历史流水。
                for item in employments:
                    self._insert_employment(cursor, item)
                for item in projects:
                    self._insert_project(cursor, item)
                # Chunk 不先整批删除，以 vector_doc_id Upsert 可以保留数据库行主键和创建时间。
                for chunk in chunks:
                    self._upsert_chunk(cursor, chunk)
                # 最后才停用陈旧 Chunk。若之前任一步失败，本步骤不会执行。
                self._deactivate_stale_chunks(
                    cursor,
                    profile.expert_id,
                    [chunk.vector_doc_id for chunk in chunks],
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
    def _deactivate_stale_chunks(cursor: Any, expert_id: int, current_ids: list[str]) -> None:
        """停用该专家本轮未生成的旧 Chunk。

        ``current_ids`` 是本轮完整 Chunk 集合的 ``vector_doc_id``。源经历被逻辑删除、
        不再满足生成条件或分片数量减少后，对应旧 ID 不会出现在该集合中，本方法将其
        ``is_active`` 更新为 0。数据库行仍被保留，便于审计和下游执行停用操作。

        - 列表非空：只停用不在当前集合中的 active 行；
        - 列表为空：停用该专家全部 active Chunk；
        - 本方法位于 Upsert 之后且属于同一事务，失败时全部回滚。
        """

        if current_ids:
            # 占位符数量随当前 ID 数量生成，所有值仍通过参数绑定传给 PyMySQL。
            placeholders = ",".join(["%s"] * len(current_ids))
            cursor.execute(
                f"UPDATE expert_chunk SET is_active=0 WHERE expert_id=%s AND is_active=1 AND vector_doc_id NOT IN ({placeholders})",
                (expert_id, *current_ids),
            )
        else:
            # 当前专家没有任何可生成 Chunk 时，调用方可用此分支停用其全部历史记录。
            cursor.execute("UPDATE expert_chunk SET is_active=0 WHERE expert_id=%s AND is_active=1", (expert_id,))

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
