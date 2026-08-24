"""构建批次主表的 MySQL 数据访问层。

本模块负责创建和收口构建批次、保存完整页进度，以及读取断点续跑和失败重试所需的
批次状态。每个公开方法按需建立连接，写操作在完成后提交并关闭连接。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pymysql

from ..config import DatabaseConfig, require_index_database


class BatchNotResumableError(RuntimeError):
    """表示指定批次不满足断点续跑条件。"""


class BatchNotRetryableError(RuntimeError):
    """表示指定批次不满足失败专家重试条件。"""


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
    vector_sync_expert_count: int = 0  # 已可靠完成向量同步的累计专家数。
    vector_current_count: int = 0  # 累计提交给下游的当前 Chunk 数。
    vector_eligible_count: int = 0  # 累计应保留向量的 Chunk 数。
    vector_embedded_count: int = 0  # 累计实际重新向量化的 Chunk 数。
    vector_metadata_updated_count: int = 0  # 累计只更新 Metadata 的 Chunk 数。
    vector_unchanged_count: int = 0  # 累计无需写入的 Chunk 数。
    vector_deleted_count: int = 0  # 累计成功删除的旧向量数。
    vector_delete_not_found_count: int = 0  # 累计删除时未找到的向量数。


class MySQLBatchRepository:
    """持久化构建批次的生命周期、扫描位置和累计计数。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存索引库连接配置，数据库连接由各方法按需创建。"""

        require_index_database(config)
        self.config = config

    def start_batch(
        self,
        batch_no: str,
        run_mode: str,
        *,
        data_version: str | None = None,
        upper_bound_id: int | None = None,
    ) -> int:
        """创建 running 状态的构建批次并返回批次主键。

        ``single`` 只写入批次号、运行模式和状态。``full`` 必须提供数据版本和固定扫描
        上界，初始游标写为 0。返回值用于关联批次专家明细和后续状态更新。
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
                    cursor.execute(
                        """
                        INSERT INTO expert_index_batch(batch_no, run_mode, status)
                        VALUES (%s, %s, 'running')
                        """,
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
        cursor_id: int | None = None,
        candidate_count: int | None = None,
        success_count: int | None = None,
        failed_count: int | None = None,
        deactivated_count: int | None = None,
        vector_sync_expert_count: int = 0,
        vector_current_count: int = 0,
        vector_eligible_count: int = 0,
        vector_embedded_count: int = 0,
        vector_metadata_updated_count: int = 0,
        vector_unchanged_count: int = 0,
        vector_deleted_count: int = 0,
        vector_delete_not_found_count: int = 0,
        error_summary: dict[str, Any] | None = None,
    ) -> None:
        """写入批次终态、处理计数、同步统计和结束时间。

        full 批次必须同时提供最终游标及全部专家分类计数。``error_summary`` 会被转换为
        稳定 JSON，调用方只应传入异常类型、机器码或简短安全说明。
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
                            rejected_count=%s, vector_sync_expert_count=%s,
                            vector_current_count=%s, vector_eligible_count=%s,
                            vector_embedded_count=%s,
                            vector_metadata_updated_count=%s,
                            vector_unchanged_count=%s, vector_deleted_count=%s,
                            vector_delete_not_found_count=%s, error_summary=%s,
                            finished_at=CURRENT_TIMESTAMP(6)
                        WHERE id=%s
                        """,
                        (
                            status,
                            cursor_id,
                            candidate_count,
                            success_count,
                            failed_count,
                            deactivated_count,
                            extracted_count,
                            stage_count,
                            chunk_count,
                            rejected_count,
                            vector_sync_expert_count,
                            vector_current_count,
                            vector_eligible_count,
                            vector_embedded_count,
                            vector_metadata_updated_count,
                            vector_unchanged_count,
                            vector_deleted_count,
                            vector_delete_not_found_count,
                            error_json,
                            batch_id,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE expert_index_batch
                        SET status=%s, extracted_count=%s, stage_count=%s,
                            chunk_count=%s, rejected_count=%s,
                            vector_sync_expert_count=%s, vector_current_count=%s,
                            vector_eligible_count=%s, vector_embedded_count=%s,
                            vector_metadata_updated_count=%s,
                            vector_unchanged_count=%s, vector_deleted_count=%s,
                            vector_delete_not_found_count=%s, error_summary=%s,
                            finished_at=CURRENT_TIMESTAMP(6)
                        WHERE id=%s
                        """,
                        (
                            status,
                            extracted_count,
                            stage_count,
                            chunk_count,
                            rejected_count,
                            vector_sync_expert_count,
                            vector_current_count,
                            vector_eligible_count,
                            vector_embedded_count,
                            vector_metadata_updated_count,
                            vector_unchanged_count,
                            vector_deleted_count,
                            vector_delete_not_found_count,
                            error_json,
                            batch_id,
                        ),
                    )
            connection.commit()
        finally:
            connection.close()

    def resume_batch(self, batch_id: int) -> FullBatchCheckpoint:
        """重新激活可续跑的 full 批次并返回可靠页级检查点。

        只接受 running 或 failed 状态。方法在同一事务中锁定并校验批次，将状态写为
        running，清除结束时间和错误摘要，然后返回原数据版本、固定上界、可靠游标及
        累计计数。
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
                           chunk_count, vector_sync_expert_count,
                           vector_current_count, vector_eligible_count,
                           vector_embedded_count, vector_metadata_updated_count,
                           vector_unchanged_count, vector_deleted_count,
                           vector_delete_not_found_count
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
                    raise BatchNotResumableError(f"批次 {batch_id} 不是 full 批次，不能断点续跑")
                status = row.get("status")
                if status not in {"running", "failed"}:
                    raise BatchNotResumableError(f"批次 {batch_id} 状态为 {status}，不能断点续跑")

                data_version = row.get("data_version")
                cursor_id = row.get("cursor_id")
                upper_bound_id = row.get("upper_bound_id")
                if not data_version or cursor_id is None or upper_bound_id is None:
                    raise BatchNotResumableError(f"批次 {batch_id} 缺少 data_version 或扫描游标")
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
                    "vector_sync_expert_count",
                    "vector_current_count",
                    "vector_eligible_count",
                    "vector_embedded_count",
                    "vector_metadata_updated_count",
                    "vector_unchanged_count",
                    "vector_deleted_count",
                    "vector_delete_not_found_count",
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

    def load_partial_batch_for_retry(self, batch_id: int) -> FullBatchCheckpoint:
        """读取并校验可执行失败专家重试的 partial full 批次。

        批次必须已经扫描到固定上界。方法只读取数据版本、范围和累计计数，不锁定记录
        或修改批次状态。调用方应在外层持有全量构建锁。
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, batch_no, run_mode, data_version, status,
                           cursor_id, upper_bound_id, candidate_count,
                           success_count, failed_count, rejected_count,
                           deactivated_count, extracted_count, stage_count,
                           chunk_count, vector_sync_expert_count,
                           vector_current_count, vector_eligible_count,
                           vector_embedded_count, vector_metadata_updated_count,
                           vector_unchanged_count, vector_deleted_count,
                           vector_delete_not_found_count
                    FROM expert_index_batch
                    WHERE id=%s
                    """,
                    (batch_id,),
                )
                row = cursor.fetchone()
            if not row:
                raise BatchNotRetryableError(f"批次 {batch_id} 不存在")
            if row.get("run_mode") != "full" or row.get("status") != "partial":
                raise BatchNotRetryableError(
                    f"批次 {batch_id} 不是 partial full 批次，不能重试失败专家"
                )

            data_version = row.get("data_version")
            cursor_id = row.get("cursor_id")
            upper_bound_id = row.get("upper_bound_id")
            if not data_version or cursor_id is None or upper_bound_id is None:
                raise BatchNotRetryableError(f"批次 {batch_id} 缺少 data_version 或扫描游标")
            cursor_id = int(cursor_id)
            upper_bound_id = int(upper_bound_id)
            if cursor_id < 0 or cursor_id != upper_bound_id:
                raise BatchNotRetryableError(
                    f"批次 {batch_id} 尚未完成固定范围扫描，不能重试失败专家"
                )

            count_names = (
                "candidate_count",
                "success_count",
                "failed_count",
                "rejected_count",
                "deactivated_count",
                "extracted_count",
                "stage_count",
                "chunk_count",
                "vector_sync_expert_count",
                "vector_current_count",
                "vector_eligible_count",
                "vector_embedded_count",
                "vector_metadata_updated_count",
                "vector_unchanged_count",
                "vector_deleted_count",
                "vector_delete_not_found_count",
            )
            counts = {name: int(row.get(name) or 0) for name in count_names}
            if any(value < 0 for value in counts.values()):
                raise BatchNotRetryableError(f"批次 {batch_id} 的累计计数无效")
            return FullBatchCheckpoint(
                batch_id=int(row["id"]),
                batch_no=str(row["batch_no"]),
                data_version=str(data_version),
                cursor_id=cursor_id,
                upper_bound_id=upper_bound_id,
                **counts,
            )
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
        vector_sync_expert_count: int = 0,
        vector_current_count: int = 0,
        vector_eligible_count: int = 0,
        vector_embedded_count: int = 0,
        vector_metadata_updated_count: int = 0,
        vector_unchanged_count: int = 0,
        vector_deleted_count: int = 0,
        vector_delete_not_found_count: int = 0,
    ) -> None:
        """保存 full 批次最后完整处理页的游标和累计计数。

        所有值在一次 UPDATE 中写入 running 批次。调用成功后，续跑会从 ``cursor_id``
        之后继续；未包含在该游标内的记录不会被视为安全完成。
        """

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE expert_index_batch
                    SET cursor_id=%s, candidate_count=%s, success_count=%s,
                        failed_count=%s, rejected_count=%s, deactivated_count=%s,
                        extracted_count=%s, stage_count=%s, chunk_count=%s,
                        vector_sync_expert_count=%s, vector_current_count=%s,
                        vector_eligible_count=%s, vector_embedded_count=%s,
                        vector_metadata_updated_count=%s,
                        vector_unchanged_count=%s, vector_deleted_count=%s,
                        vector_delete_not_found_count=%s
                    WHERE id=%s AND status='running'
                    """,
                    (
                        cursor_id,
                        candidate_count,
                        success_count,
                        failed_count,
                        rejected_count,
                        deactivated_count,
                        extracted_count,
                        stage_count,
                        chunk_count,
                        vector_sync_expert_count,
                        vector_current_count,
                        vector_eligible_count,
                        vector_embedded_count,
                        vector_metadata_updated_count,
                        vector_unchanged_count,
                        vector_deleted_count,
                        vector_delete_not_found_count,
                        batch_id,
                    ),
                )
            connection.commit()
        finally:
            connection.close()


def _json(value: Any) -> str:
    """将批次错误摘要序列化为稳定且符合 JSON 标准的紧凑字符串。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
