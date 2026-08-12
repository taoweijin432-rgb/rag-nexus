"""派生索引库 Repository：Schema 初始化、批次、Stage 与 Chunk 持久化。

一位专家的三张 Stage 和完整 Chunk 集合在同一个事务中写入。只有全部成功后才
停用旧 Chunk；任一步骤异常都会 rollback，满足“不能出现半更新专家”的要求。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymysql

from .config import DatabaseConfig
from .models import ChunkRecord, EmploymentStage, ProfileStage, ProjectStage


class MySQLIndexRepository:
    """封装所有派生表写入；Source Adapter 不依赖本类，读写边界清晰。"""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config

    def initialize_schema(self, sql_path: Path) -> None:
        """执行受版本控制的 DDL；脚本应只包含派生表定义。"""

        script = sql_path.read_text(encoding="utf-8")
        statements = [statement.strip() for statement in script.split(";") if statement.strip()]
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def start_batch(self, batch_no: str, run_mode: str) -> int:
        """创建 running 批次并返回数据库主键。"""

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO expert_index_batch(batch_no, run_mode, status) VALUES (%s, %s, 'running')",
                    (batch_no, run_mode),
                )
                batch_id = int(cursor.lastrowid)
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
        """结束批次；错误摘要只保存异常类型和说明，不保存任何源正文。"""

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
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
        """原子写入一位专家，返回写入的 Stage 行数。"""

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                self._upsert_profile(cursor, profile)

                # Stage 是当前规范化快照，因此先删除该专家旧经历，再写入完整新集合。
                # 删除仅作用于派生表且被当前事务保护，异常时可完整恢复。
                cursor.execute("DELETE FROM expert_index_employment_stage WHERE expert_id=%s", (profile.expert_id,))
                cursor.execute("DELETE FROM expert_index_project_stage WHERE expert_id=%s", (profile.expert_id,))
                for item in employments:
                    self._insert_employment(cursor, item)
                for item in projects:
                    self._insert_project(cursor, item)
                for chunk in chunks:
                    self._upsert_chunk(cursor, chunk)
                self._deactivate_stale_chunks(cursor, profile.expert_id, [chunk.chunk_id for chunk in chunks])
            connection.commit()
            return 1 + len(employments) + len(projects)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _upsert_profile(cursor: Any, item: ProfileStage) -> None:
        cursor.execute(
            """
            INSERT INTO expert_index_stage(
                expert_id, serial_number, company_name_raw, company_name_norm,
                company_normalization_status, role_name_raw, role_code, role_level,
                background_clean, school_names_json, work_years, source_hash,
                source_updated_at, data_version, stage_status, build_batch_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'ready',%s)
            ON DUPLICATE KEY UPDATE
                serial_number=VALUES(serial_number), company_name_raw=VALUES(company_name_raw),
                company_name_norm=VALUES(company_name_norm),
                company_normalization_status=VALUES(company_normalization_status),
                role_name_raw=VALUES(role_name_raw), role_code=VALUES(role_code),
                role_level=VALUES(role_level), background_clean=VALUES(background_clean),
                school_names_json=VALUES(school_names_json), work_years=VALUES(work_years),
                source_hash=VALUES(source_hash), source_updated_at=VALUES(source_updated_at),
                data_version=VALUES(data_version), stage_status='ready', reject_reason=NULL,
                build_batch_id=VALUES(build_batch_id)
            """,
            (
                item.expert_id, item.serial_number, item.company_name_raw, item.company_name_norm,
                item.company_normalization_status, item.role_name_raw, item.role_code,
                item.role_level, item.background_clean, _json(list(item.school_names)),
                item.work_years, item.source_hash, item.source_updated_at, item.data_version,
                item.build_batch_id,
            ),
        )

    @staticmethod
    def _insert_employment(cursor: Any, item: EmploymentStage) -> None:
        cursor.execute(
            """
            INSERT INTO expert_index_employment_stage(
                expert_id, source_table, source_field, source_field_id, source_item_key,
                company_name_raw, company_name_norm, company_normalization_status,
                role_name_raw, role_code, role_level, start_month, end_month, is_current,
                responsibilities_clean, achievements_clean, industry_codes_json,
                topic_codes_json, parse_status, source_hash, source_updated_at,
                data_version, build_batch_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                item.expert_id, item.source_table, item.source_field, item.source_field_id,
                item.source_item_key,
                item.company_name_raw, item.company_name_norm,
                item.company_normalization_status, item.role_name_raw, item.role_code,
                item.role_level, item.start_month, item.end_month, item.is_current,
                item.responsibilities_clean, item.achievements_clean,
                _json(list(item.industry_codes)), _json(list(item.topic_codes)),
                item.parse_status, item.source_hash, item.source_updated_at,
                item.data_version, item.build_batch_id,
            ),
        )

    @staticmethod
    def _insert_project(cursor: Any, item: ProjectStage) -> None:
        cursor.execute(
            """
            INSERT INTO expert_index_project_stage(
                expert_id, source_table, source_field, source_field_id, source_item_key,
                project_name,
                company_or_client_raw, company_or_client_norm,
                company_normalization_status, project_role, start_month, end_month,
                objective_clean, actions_clean, results_clean, industry_codes_json,
                topic_codes_json, parse_status, source_hash, source_updated_at,
                data_version, build_batch_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                item.expert_id, item.source_table, item.source_field, item.source_field_id,
                item.source_item_key, item.project_name, item.company_or_client_raw,
                item.company_or_client_norm,
                item.company_normalization_status, item.project_role, item.start_month,
                item.end_month, item.objective_clean, item.actions_clean, item.results_clean,
                _json(list(item.industry_codes)), _json(list(item.topic_codes)),
                item.parse_status, item.source_hash, item.source_updated_at,
                item.data_version, item.build_batch_id,
            ),
        )

    @staticmethod
    def _upsert_chunk(cursor: Any, item: ChunkRecord) -> None:
        cursor.execute(
            """
            INSERT INTO expert_chunk(
                chunk_id, expert_id, serial_number, chunk_type, chunk_subtype,
                chunk_index, chunk_total, chunk_text, char_count, token_count_estimate,
                metadata_json, source_table, source_field, source_field_id, source_item_key,
                source_version, source_updated_at,
                source_hash, content_hash, company_name_norm, role_code, start_month,
                end_month, is_current, industry_codes_json, topic_codes_json,
                origin_type, verification_status, weight_tier, chunk_policy_version,
                metadata_schema_version, dictionary_version, data_version,
                is_active, build_batch_id
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,1,%s
            )
            ON DUPLICATE KEY UPDATE
                serial_number=VALUES(serial_number), chunk_type=VALUES(chunk_type),
                chunk_subtype=VALUES(chunk_subtype), chunk_index=VALUES(chunk_index),
                chunk_total=VALUES(chunk_total), chunk_text=VALUES(chunk_text),
                char_count=VALUES(char_count), token_count_estimate=VALUES(token_count_estimate),
                metadata_json=VALUES(metadata_json), source_table=VALUES(source_table),
                source_field=VALUES(source_field), source_field_id=VALUES(source_field_id),
                source_item_key=VALUES(source_item_key),
                source_version=VALUES(source_version), source_updated_at=VALUES(source_updated_at),
                source_hash=VALUES(source_hash), content_hash=VALUES(content_hash),
                company_name_norm=VALUES(company_name_norm), role_code=VALUES(role_code),
                start_month=VALUES(start_month), end_month=VALUES(end_month),
                is_current=VALUES(is_current), industry_codes_json=VALUES(industry_codes_json),
                topic_codes_json=VALUES(topic_codes_json), origin_type=VALUES(origin_type),
                verification_status=VALUES(verification_status), weight_tier=VALUES(weight_tier),
                chunk_policy_version=VALUES(chunk_policy_version),
                metadata_schema_version=VALUES(metadata_schema_version),
                dictionary_version=VALUES(dictionary_version), data_version=VALUES(data_version),
                is_active=1, build_batch_id=VALUES(build_batch_id)
            """,
            (
                item.chunk_id, item.expert_id, item.serial_number, item.chunk_type,
                item.chunk_subtype, item.chunk_index, item.chunk_total, item.chunk_text,
                item.char_count, max(1, item.char_count // 2), _json(item.metadata),
                item.source_table, item.source_field, item.source_field_id,
                item.source_item_key, item.source_version,
                item.source_updated_at, item.source_hash, item.content_hash,
                item.company_name_norm, item.role_code, item.start_month, item.end_month,
                item.is_current, _json(list(item.industry_codes)), _json(list(item.topic_codes)),
                item.origin_type, item.verification_status, item.weight_tier,
                item.chunk_policy_version, item.metadata_schema_version,
                item.dictionary_version, item.data_version, item.build_batch_id,
            ),
        )

    @staticmethod
    def _deactivate_stale_chunks(cursor: Any, expert_id: int, current_ids: list[str]) -> None:
        """停用本轮未生成的旧 Chunk；仅在前面的全部 Upsert 成功后执行。"""

        if current_ids:
            placeholders = ",".join(["%s"] * len(current_ids))
            cursor.execute(
                f"UPDATE expert_chunk SET is_active=0 WHERE expert_id=%s AND is_active=1 AND chunk_id NOT IN ({placeholders})",
                (expert_id, *current_ids),
            )
        else:
            cursor.execute("UPDATE expert_chunk SET is_active=0 WHERE expert_id=%s AND is_active=1", (expert_id,))

    def fetch_active_chunks(self, expert_id: int, *, eligible_only: bool = False) -> list[dict[str, Any]]:
        """读取导出所需固定列；可选择只取正式向量化允许的记录。"""

        eligibility = " AND verification_status IN ('source_evidence','validated') AND weight_tier IN ('A','B')" if eligible_only else ""
        sql = f"""
            SELECT chunk_id, expert_id, serial_number, chunk_type, chunk_subtype,
                   chunk_index, chunk_total, chunk_text, char_count, content_hash,
                   metadata_json, source_table, source_field, source_field_id,
                   source_item_key, source_version, source_updated_at,
                   source_hash, chunk_policy_version, metadata_schema_version,
                   dictionary_version, data_version, is_active, updated_at,
                   verification_status, weight_tier
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
        """返回不含正文的验收统计，适合日志和 CLI 展示。"""

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
    """严格 JSON 序列化：禁止 NaN/Infinity，中文不转义。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
