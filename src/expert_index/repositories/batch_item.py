"""批次专家明细的 MySQL 数据访问层。

本模块负责登记批次候选专家、记录单次处理状态和结果，以及分页读取重试对象和
汇总批次明细。每个公开方法按需建立连接；读取使用自动提交连接，写入成功后提交。
"""

from __future__ import annotations

from dataclasses import dataclass

import pymysql

from ..config import DatabaseConfig

MAX_BATCH_ITEM_PAGE_SIZE = 1000
_BATCH_ITEM_FINAL_STATUSES = {"success", "rejected", "deactivated", "failed"}


@dataclass(frozen=True)
class BatchItemSummary:
    """保存一个批次全部专家明细的状态和处理数量汇总。"""

    candidate_count: int
    success_count: int
    failed_count: int
    rejected_count: int
    deactivated_count: int
    pending_count: int
    running_count: int
    extracted_count: int
    stage_count: int
    chunk_count: int


class MySQLBatchItemRepository:
    """持久化批次内每位专家的处理状态、结果和阶段计数。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存索引库连接配置，数据库连接由各方法按需创建。"""

        self.config = config

    def load_retryable_batch_item_ids(
        self,
        batch_id: int,
        *,
        after_expert_id: int,
        limit: int,
    ) -> list[int]:
        """按专家 ID 分页读取 failed 和 running 状态的明细。

        ``after_expert_id`` 指定当前页的排他起点，``limit`` 限制返回数量。running
        明细会被返回，使已标记开始但未写入终态的专家能够再次处理。
        """

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        if (
            isinstance(after_expert_id, bool)
            or not isinstance(after_expert_id, int)
            or after_expert_id < 0
        ):
            raise ValueError("after_expert_id 不能小于 0")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_BATCH_ITEM_PAGE_SIZE
        ):
            raise ValueError(f"limit 必须位于 1 到 {MAX_BATCH_ITEM_PAGE_SIZE} 之间")

        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT expert_id
                    FROM expert_index_batch_item
                    WHERE batch_id=%s AND expert_id>%s
                      AND status IN ('failed', 'running')
                    ORDER BY expert_id ASC
                    LIMIT %s
                    """,
                    (batch_id, after_expert_id, limit),
                )
                return [int(row["expert_id"]) for row in cursor.fetchall()]
        finally:
            connection.close()

    def summarize_batch_items(self, batch_id: int) -> BatchItemSummary:
        """汇总指定批次全部专家明细的状态数量和阶段计数。"""

        if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id < 1:
            raise ValueError("batch_id 必须是正整数")
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*) AS candidate_count,
                           COALESCE(SUM(status='success'), 0) AS success_count,
                           COALESCE(SUM(status='failed'), 0) AS failed_count,
                           COALESCE(SUM(status='rejected'), 0) AS rejected_count,
                           COALESCE(SUM(status='deactivated'), 0) AS deactivated_count,
                           COALESCE(SUM(status='pending'), 0) AS pending_count,
                           COALESCE(SUM(status='running'), 0) AS running_count,
                           COALESCE(SUM(extracted_count), 0) AS extracted_count,
                           COALESCE(SUM(stage_count), 0) AS stage_count,
                           COALESCE(SUM(chunk_count), 0) AS chunk_count
                    FROM expert_index_batch_item
                    WHERE batch_id=%s
                    """,
                    (batch_id,),
                )
                row = cursor.fetchone() or {}
            return BatchItemSummary(
                candidate_count=int(row.get("candidate_count") or 0),
                success_count=int(row.get("success_count") or 0),
                failed_count=int(row.get("failed_count") or 0),
                rejected_count=int(row.get("rejected_count") or 0),
                deactivated_count=int(row.get("deactivated_count") or 0),
                pending_count=int(row.get("pending_count") or 0),
                running_count=int(row.get("running_count") or 0),
                extracted_count=int(row.get("extracted_count") or 0),
                stage_count=int(row.get("stage_count") or 0),
                chunk_count=int(row.get("chunk_count") or 0),
            )
        finally:
            connection.close()

    def prepare_batch_items(self, batch_id: int, expert_ids: list[int]) -> None:
        """幂等登记一页候选专家的 pending 明细。

        空列表不访问数据库。同一 ``batch_id`` 和 ``expert_id`` 已有记录时保留当前
        状态，只确保该专家已经登记到批次中。
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
        """把已登记专家标记为 running 并初始化本次尝试。

        方法增加 ``attempt_count``，清空上次处理计数和结果字段，并记录新的开始时间。
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
        """写入一位专家的终态、处理计数和机器结果标识。

        ``status`` 只接受 success、rejected、deactivated 或 failed。结果码和错误类型
        只允许安全标识字符，方法不接收异常消息、来源字段或 Chunk Text。
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
                        status,
                        extracted_count,
                        stage_count,
                        chunk_count,
                        safe_result_code,
                        safe_error_type,
                        batch_id,
                        expert_id,
                    ),
                )
            connection.commit()
        finally:
            connection.close()


def _safe_identifier(value: str | None, *, max_length: int) -> str | None:
    """返回适合写入机器标识字段的截断值。

    空值返回 ``None``；非空值只允许字母、数字、下划线、点和短横线。
    """

    if value is None:
        return None
    candidate = value.strip()[:max_length]
    if not candidate:
        return None
    if not all(character.isalnum() or character in "_.-" for character in candidate):
        raise ValueError("批次结果标识只能包含字母、数字、下划线、点和短横线")
    return candidate
