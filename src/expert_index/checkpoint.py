"""增量来源批次的安全 Checkpoint 提交器。

本模块只在本地 Index MySQL 内校验并收口已经完成的来源批次，不读取来源业务事实，
也不调用下游处理接口。success 与具备同 retry_count DLQ 证据的 dead 都是可收口终态；
所有状态更新位于同一事务，任一条件失败都会整体回滚。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pymysql

from .config import DatabaseConfig, require_index_database
from .models import (
    NO_WRITE_RESULT_CODES,
    PERSISTED_SOURCE_TABLES,
    TASK_RESULT_CODES,
    SourceCheckpoint,
    SourceScanCursor,
)


class SourceBatchCommitConflictError(RuntimeError):
    """表示 lease、Checkpoint 版本或批次状态已变化，不能安全提交。"""


class SourceBatchNotReadyError(RuntimeError):
    """表示扫描范围或任务成功证据尚未满足 Checkpoint 提交条件。"""


@dataclass(frozen=True)
class SourceBatchCommitResult:
    """返回一次来源批次安全提交后的持久化结果。"""

    checkpoint: SourceCheckpoint
    batch_id: int
    candidate_count: int
    skipped_count: int
    already_committed: bool


class CheckpointManager:
    """在单个 Index MySQL 事务内提交来源批次安全水位。"""

    def __init__(self, config: DatabaseConfig) -> None:
        """保存只允许写入 Index MySQL 的连接配置。"""

        require_index_database(config)
        self.config = config

    def try_commit_source_batch(
        self,
        source_table: str,
        batch_id: int,
        lease_owner: str,
        *,
        checkpoint_lock_version: int,
    ) -> SourceBatchCommitResult:
        """校验终态证据并原子提交一张来源表的固定范围。

        首次提交要求调用方仍持有有效来源 lease。若相同批次已经完整提交，方法在核对
        Checkpoint、Batch Source、Batch 和任务证据后幂等返回，不再次写入任何状态。
        ``dead`` 任务只有在对应 retry_count 的 DLQ 证据存在时才算终态；它不计入
        success_count，但允许安全水位越过并把批次标记为 partial。
        """

        _validate_source_table(source_table)
        _validate_positive_integer("batch_id", batch_id)
        _validate_lease_owner(lease_owner)
        _validate_non_negative_integer(
            "checkpoint_lock_version",
            checkpoint_lock_version,
        )

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                checkpoint_row = _lock_checkpoint(cursor, source_table)
                batch_source_row = _lock_batch_source(
                    cursor,
                    batch_id=batch_id,
                    source_table=source_table,
                )
                batch_row = _lock_incremental_batch(cursor, batch_id)
                task_rows = _lock_source_tasks(
                    cursor,
                    batch_id=batch_id,
                    source_table=source_table,
                )

                start_cursor = SourceScanCursor(
                    updated_at=batch_source_row["start_updated_at"],
                    record_id=int(batch_source_row["start_id"]),
                )
                current_cursor = SourceScanCursor(
                    updated_at=batch_source_row["cursor_updated_at"],
                    record_id=int(batch_source_row["cursor_id"]),
                )
                upper_bound = SourceScanCursor(
                    updated_at=batch_source_row["upper_bound_updated_at"],
                    record_id=int(batch_source_row["upper_bound_id"]),
                )
                checkpoint_cursor = SourceScanCursor(
                    updated_at=checkpoint_row["last_updated_at"],
                    record_id=int(checkpoint_row["last_id"]),
                )
                candidate_count = int(batch_source_row["candidate_count"])
                completed_success_count, dead_count = _validate_task_terminal_evidence(
                    task_rows,
                    candidate_count=candidate_count,
                )
                # P5.6 §4：skipped_count 不在每个 Task 成功时独立 +1，避免事务提交后
                # 重试造成重复计数。由 CheckpointManager 在锁定全部成功任务时按冻结的
                # NO_WRITE_RESULT_CODES 重新统计，与 success_count/candidate_count
                # 一起原子写 Batch Source。Batch Source 行里的旧 skipped_count 不参与
                # （frozen #5），只校验重算值不超出候选范围。recovered_unchanged 属
                # 幂等重放已收敛，不计 skipped（frozen #4）。expert_deleted/
                # no_chunkable_content 是 P4 阶段零写入的业务终态，不在
                # NO_WRITE_RESULT_CODES 内（frozen #6）。
                skipped_count = sum(
                    1
                    for row in task_rows
                    if str(row["sync_status"]) == "success"
                    and str(row.get("result_code") or "") in NO_WRITE_RESULT_CODES
                )
                if not 0 <= skipped_count <= candidate_count:
                    raise SourceBatchCommitConflictError(
                        "Batch Source skipped_count 超出候选范围"
                    )
                if _is_already_committed(
                    checkpoint_row,
                    batch_source_row,
                    batch_row,
                    batch_id=batch_id,
                    checkpoint_lock_version=checkpoint_lock_version,
                    upper_bound=upper_bound,
                    current_cursor=current_cursor,
                    checkpoint_cursor=checkpoint_cursor,
                    candidate_count=candidate_count,
                    completed_success_count=completed_success_count,
                    dead_count=dead_count,
                ):
                    checkpoint = _source_checkpoint_from_row(
                        checkpoint_row,
                        expected_source_table=source_table,
                    )
                    connection.commit()
                    return SourceBatchCommitResult(
                        checkpoint=checkpoint,
                        batch_id=batch_id,
                        candidate_count=candidate_count,
                        skipped_count=skipped_count,
                        already_committed=True,
                    )

                _validate_new_commit_state(
                    checkpoint_row,
                    batch_source_row,
                    batch_row,
                    source_table=source_table,
                    batch_id=batch_id,
                    lease_owner=lease_owner,
                    checkpoint_lock_version=checkpoint_lock_version,
                    checkpoint_cursor=checkpoint_cursor,
                    start_cursor=start_cursor,
                    current_cursor=current_cursor,
                    upper_bound=upper_bound,
                    candidate_count=candidate_count,
                )

                cursor.execute(
                    """
                    UPDATE expert_index_batch_source
                    SET status='success', success_count=%s,
                        skipped_count=%s, failed_count=%s,
                        finished_at=CURRENT_TIMESTAMP(6)
                    WHERE id=%s AND batch_id=%s AND source_table=%s
                      AND status='running'
                      AND cursor_updated_at=upper_bound_updated_at
                      AND cursor_id=upper_bound_id
                      AND checkpoint_lock_version=%s
                    """,
                    (
                        completed_success_count,
                        skipped_count,
                        dead_count,
                        int(batch_source_row["id"]),
                        batch_id,
                        source_table,
                        checkpoint_lock_version,
                    ),
                )
                _require_single_update(
                    cursor,
                    "Batch Source 收口 CAS 失败",
                )

                cursor.execute(
                    """
                    UPDATE expert_index_batch
                    SET status=CASE WHEN %s > 0 THEN 'partial' ELSE 'success' END,
                        cursor_updated_at=%s, cursor_id=%s,
                        candidate_count=%s, success_count=%s,
                        failed_count=%s,
                        error_summary=CASE
                            WHEN %s > 0 THEN JSON_OBJECT(
                                'dead_task_count', %s,
                                'reason', 'terminal_dead_task'
                            )
                            ELSE NULL
                        END,
                        finished_at=CURRENT_TIMESTAMP(6)
                    WHERE id=%s AND run_mode='incremental'
                      AND status='running'
                    """,
                    (
                        dead_count,
                        upper_bound.updated_at,
                        upper_bound.record_id,
                        candidate_count,
                        completed_success_count,
                        dead_count,
                        dead_count,
                        dead_count,
                        batch_id,
                    ),
                )
                _require_single_update(cursor, "incremental Batch 收口 CAS 失败")

                cursor.execute(
                    """
                    UPDATE expert_index_source_checkpoint
                    SET last_updated_at=%s, last_id=%s,
                        last_batch_id=%s,
                        last_success_time=CURRENT_TIMESTAMP(6),
                        lock_version=lock_version+1,
                        lease_owner=NULL, lease_until=NULL
                    WHERE source_table=%s
                      AND last_updated_at=%s AND last_id=%s
                      AND lock_version=%s
                      AND lease_owner=%s
                      AND lease_until > CURRENT_TIMESTAMP(6)
                    """,
                    (
                        upper_bound.updated_at,
                        upper_bound.record_id,
                        batch_id,
                        source_table,
                        start_cursor.updated_at,
                        start_cursor.record_id,
                        checkpoint_lock_version,
                        lease_owner,
                    ),
                )
                _require_single_update(cursor, "Checkpoint 乐观锁提交失败")
                checkpoint = _read_checkpoint(cursor, source_table)
            connection.commit()
            return SourceBatchCommitResult(
                checkpoint=checkpoint,
                batch_id=batch_id,
                candidate_count=candidate_count,
                skipped_count=skipped_count,
                already_committed=False,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _validate_new_commit_state(
    checkpoint_row: dict[str, Any],
    batch_source_row: dict[str, Any],
    batch_row: dict[str, Any],
    *,
    source_table: str,
    batch_id: int,
    lease_owner: str,
    checkpoint_lock_version: int,
    checkpoint_cursor: SourceScanCursor,
    start_cursor: SourceScanCursor,
    current_cursor: SourceScanCursor,
    upper_bound: SourceScanCursor,
    candidate_count: int,
) -> None:
    """校验首次提交所需的 lease、游标、版本、状态和批次计数。"""

    if (
        checkpoint_row.get("lease_owner") != lease_owner
        or not bool(checkpoint_row["lease_is_active"])
    ):
        raise SourceBatchCommitConflictError(
            f"调用方不再持有来源表 {source_table} 的有效扫描 lease"
        )
    if (
        int(checkpoint_row["lock_version"]) != checkpoint_lock_version
        or checkpoint_cursor != start_cursor
        or int(batch_source_row["checkpoint_lock_version"])
        != checkpoint_lock_version
    ):
        raise SourceBatchCommitConflictError(
            "Checkpoint 版本、当前位置或 Batch Source 起点已变化"
        )
    if batch_source_row.get("status") != "running":
        raise SourceBatchCommitConflictError("Batch Source 不是 running 状态")
    if current_cursor != upper_bound:
        raise SourceBatchNotReadyError("Batch Source 临时游标尚未到达固定上界")
    if batch_row.get("run_mode") != "incremental" or batch_row.get("status") != "running":
        raise SourceBatchCommitConflictError("批次不是可提交的 running incremental Batch")
    if int(batch_row["id"]) != batch_id:
        raise RuntimeError("锁定的批次主记录与请求 batch_id 不一致")
    batch_candidate_count = int(batch_row.get("candidate_count") or 0)
    if batch_candidate_count not in {0, candidate_count}:
        raise SourceBatchCommitConflictError("Batch 与 Batch Source 的候选计数不一致")
    for field_name in ("success_count", "failed_count"):
        value = int(batch_row.get(field_name) or 0)
        if not 0 <= value <= candidate_count:
            raise SourceBatchCommitConflictError(
                f"Batch {field_name} 超出候选范围"
            )


def _validate_task_terminal_evidence(
    task_rows: list[dict[str, Any]],
    *,
    candidate_count: int,
) -> tuple[int, int]:
    """校验 success/dead 两类终态证据并返回 ``(success, dead)`` 计数。

    ``dead`` 不是成功，但它是经过 DLQ 记录确认的不可自动恢复终态。只有这种
    终态才能被 Checkpoint 闸门接受；仍处于 pending/processing 或缺少 DLQ 证据的
    任务继续阻断水位，避免以“跳过校验”的方式丢失可重试数据。
    """

    if candidate_count < 0:
        raise SourceBatchCommitConflictError("Batch Source candidate_count 不能为负数")
    if len(task_rows) != candidate_count:
        raise SourceBatchNotReadyError("任务总数与 Batch Source candidate_count 不一致")
    incomplete_statuses: dict[str, int] = {}
    success_count = 0
    dead_count = 0
    for row in task_rows:
        status = str(row["sync_status"])
        if status == "dead":
            if (
                row.get("completed_at") is None
                or row.get("attempt_token") is not None
                or row.get("lease_until") is not None
                or not bool(row.get("dead_letter_exists"))
            ):
                raise SourceBatchNotReadyError(
                    "dead 任务缺少完成时间、无租约或 DLQ 证据"
                )
            dead_count += 1
            continue
        if status != "success":
            incomplete_statuses[status] = incomplete_statuses.get(status, 0) + 1
            continue
        if row.get("processing_stage") != "completed" or row.get("completed_at") is None:
            raise SourceBatchNotReadyError(
                "success 任务缺少 completed 阶段或完成时间证据"
            )
        result_code = row.get("result_code")
        if (
            not isinstance(result_code, str)
            or result_code not in TASK_RESULT_CODES
        ):
            raise SourceBatchNotReadyError(
                "success 任务缺少可靠 result_code 证据"
            )
        if row.get("attempt_token") is not None or row.get("lease_until") is not None:
            raise SourceBatchNotReadyError("success 任务仍残留 attempt lease")
        success_count += 1
    if incomplete_statuses:
        status_summary = ",".join(
            f"{status}:{count}"
            for status, count in sorted(incomplete_statuses.items())
        )
        raise SourceBatchNotReadyError(
            f"来源批次仍有未完成任务: {status_summary}"
        )
    return success_count, dead_count


def _is_already_committed(
    checkpoint_row: dict[str, Any],
    batch_source_row: dict[str, Any],
    batch_row: dict[str, Any],
    *,
    batch_id: int,
    checkpoint_lock_version: int,
    upper_bound: SourceScanCursor,
    current_cursor: SourceScanCursor,
    checkpoint_cursor: SourceScanCursor,
    candidate_count: int,
    completed_success_count: int,
    dead_count: int,
) -> bool:
    """识别相同批次已经完整提交的幂等重放，并校验终态计数。"""

    committed_identity = (
        checkpoint_row.get("last_batch_id") is not None
        and int(checkpoint_row["last_batch_id"]) == batch_id
        and checkpoint_cursor == upper_bound
        and int(checkpoint_row["lock_version"]) == checkpoint_lock_version + 1
    )
    if not committed_identity:
        return False
    if checkpoint_row.get("lease_owner") is not None or checkpoint_row.get("lease_until") is not None:
        raise SourceBatchCommitConflictError("已提交 Checkpoint 仍残留来源 lease")
    if (
        batch_source_row.get("status") != "success"
        or current_cursor != upper_bound
        or int(batch_source_row["checkpoint_lock_version"])
        != checkpoint_lock_version
        or int(batch_source_row["success_count"]) != completed_success_count
        or int(batch_source_row["failed_count"]) != dead_count
    ):
        raise SourceBatchCommitConflictError("已提交 Batch Source 的终态计数不一致")
    if (
        batch_row.get("run_mode") != "incremental"
        or batch_row.get("status") not in {"success", "partial"}
        or int(batch_row.get("candidate_count") or 0) != candidate_count
        or int(batch_row.get("success_count") or 0) != completed_success_count
        or int(batch_row.get("failed_count") or 0) != dead_count
        or batch_row.get("cursor_updated_at") != upper_bound.updated_at
        or int(batch_row.get("cursor_id") or -1) != upper_bound.record_id
    ):
        raise SourceBatchCommitConflictError("已提交 Batch 的终态计数不一致")
    return True


def _lock_checkpoint(cursor: Any, source_table: str) -> dict[str, Any]:
    """锁定来源 Checkpoint，并由数据库时钟计算 lease 是否有效。"""

    cursor.execute(
        """
        SELECT source_table, last_updated_at, last_id,
               last_batch_id, last_success_time, lock_version,
               lease_owner, lease_until,
               (
                   lease_owner IS NOT NULL
                   AND lease_until > CURRENT_TIMESTAMP(6)
               ) AS lease_is_active
        FROM expert_index_source_checkpoint
        WHERE source_table=%s
        FOR UPDATE
        """,
        (source_table,),
    )
    row = cursor.fetchone()
    if row is None:
        raise SourceBatchCommitConflictError(
            f"来源表 {source_table} 尚未初始化 Checkpoint"
        )
    if row.get("source_table") != source_table:
        raise RuntimeError("锁定的 Checkpoint 与请求来源表不一致")
    return row


def _lock_batch_source(
    cursor: Any,
    *,
    batch_id: int,
    source_table: str,
) -> dict[str, Any]:
    """锁定提交目标对应的唯一 Batch Source。"""

    cursor.execute(
        """
        SELECT id, batch_id, source_table,
               start_updated_at, start_id,
               cursor_updated_at, cursor_id,
               upper_bound_updated_at, upper_bound_id,
               checkpoint_lock_version, status,
               candidate_count, success_count,
               failed_count, skipped_count,
               started_at, finished_at
        FROM expert_index_batch_source
        WHERE batch_id=%s AND source_table=%s
        FOR UPDATE
        """,
        (batch_id, source_table),
    )
    row = cursor.fetchone()
    if row is None:
        raise SourceBatchCommitConflictError("Batch Source 不存在")
    if int(row["batch_id"]) != batch_id or row.get("source_table") != source_table:
        raise RuntimeError("锁定的 Batch Source 与请求范围不一致")
    return row


def _lock_incremental_batch(cursor: Any, batch_id: int) -> dict[str, Any]:
    """锁定 Batch Source 所属的批次主记录。"""

    cursor.execute(
        """
        SELECT id, run_mode, status,
               cursor_updated_at, cursor_id,
               candidate_count, success_count, failed_count,
               error_summary, finished_at
        FROM expert_index_batch
        WHERE id=%s
        FOR UPDATE
        """,
        (batch_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise SourceBatchCommitConflictError(f"批次 {batch_id} 不存在")
    return row


def _lock_source_tasks(
    cursor: Any,
    *,
    batch_id: int,
    source_table: str,
) -> list[dict[str, Any]]:
    """按任务 ID 锁定当前来源批次的终态证据行并关联 DLQ 存在性。"""

    cursor.execute(
        """
        SELECT id, processing_stage, sync_status, result_code,
               attempt_token, lease_until, completed_at,
               EXISTS (
                   SELECT 1
                   FROM expert_index_sync_dead_letter AS dl
                   WHERE dl.task_id=expert_index_sync_task.id
                     AND dl.retry_count=expert_index_sync_task.retry_count
               ) AS dead_letter_exists
        FROM expert_index_sync_task
        WHERE batch_id=%s AND source_table=%s
        ORDER BY id ASC
        FOR UPDATE
        """,
        (batch_id, source_table),
    )
    return list(cursor.fetchall())


def _read_checkpoint(cursor: Any, source_table: str) -> SourceCheckpoint:
    """在提交事务内读取更新后的无 lease Checkpoint。"""

    cursor.execute(
        """
        SELECT source_table, last_updated_at, last_id,
               last_batch_id, last_success_time, lock_version,
               lease_owner, lease_until
        FROM expert_index_source_checkpoint
        WHERE source_table=%s
        """,
        (source_table,),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Checkpoint 更新后无法读取")
    return _source_checkpoint_from_row(row, expected_source_table=source_table)


def _source_checkpoint_from_row(
    row: dict[str, Any],
    *,
    expected_source_table: str,
) -> SourceCheckpoint:
    """把已校验表名的数据库行转换为安全 Checkpoint 模型。"""

    if row.get("source_table") != expected_source_table:
        raise RuntimeError("Checkpoint 查询结果与请求来源表不一致")
    return SourceCheckpoint(
        source_table=expected_source_table,
        last_updated_at=row["last_updated_at"],
        last_id=int(row["last_id"]),
        last_batch_id=(
            int(row["last_batch_id"])
            if row.get("last_batch_id") is not None
            else None
        ),
        last_success_time=row.get("last_success_time"),
        lock_version=int(row["lock_version"]),
        lease_owner=row.get("lease_owner"),
        lease_until=row.get("lease_until"),
    )


def _require_single_update(cursor: Any, message: str) -> None:
    """把更新 0 行或多行统一转换为安全提交并发异常。"""

    if cursor.rowcount != 1:
        raise SourceBatchCommitConflictError(message)


def _validate_source_table(source_table: str) -> None:
    """校验来源表属于四表 Checkpoint 白名单。"""

    if source_table not in PERSISTED_SOURCE_TABLES:
        raise ValueError(f"不支持的 Checkpoint 来源表: {source_table}")


def _validate_positive_integer(field_name: str, value: int) -> None:
    """校验数据库主键参数为正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} 必须是正整数")


def _validate_non_negative_integer(field_name: str, value: int) -> None:
    """校验乐观锁版本为非负整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} 必须是非负整数")


def _validate_lease_owner(lease_owner: str) -> None:
    """校验来源 lease owner 可精确映射到数据库字段。"""

    if (
        not isinstance(lease_owner, str)
        or not lease_owner.strip()
        or lease_owner != lease_owner.strip()
        or len(lease_owner) > 128
    ):
        raise ValueError("lease_owner 必须是长度不超过 128 的非空字符串")
