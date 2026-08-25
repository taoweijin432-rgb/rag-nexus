"""来源同步任务的 MySQL 数据访问层。

本模块负责幂等创建任务、并发领取、任务租约和受保护的阶段/终态写回。
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from typing import Any

import pymysql

from ..config import DatabaseConfig, require_index_database
from ..models import (
    CHECKPOINT_SOURCE_TABLES,
    SOURCE_SYNC_ERROR_STAGES,
    SOURCE_SYNC_PROCESSING_STAGE_ORDER,
    NewSourceSyncTask,
    SourceSyncTask,
)

MAX_SOURCE_SYNC_TASK_CLAIM_SIZE = 500
_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_TASK_SELECT_COLUMNS = """
    id, dedupe_key, batch_id, event_source,
    source_table, source_record_id, source_updated_at,
    expert_id, operation_type, target_type,
    processing_stage, sync_status, retry_count,
    attempt_token, lease_until, next_retry_at,
    policy_version, error_stage, error_code,
    error_type, error_message, result_code,
    started_at, completed_at, created_at, updated_at
"""


class SourceSyncTaskLeaseConflictError(RuntimeError):
    """表示任务不存在或调用方不再持有有效 attempt token 租约。"""


class SourceSyncTaskStageConflictError(RuntimeError):
    """表示任务当前阶段不允许执行请求的状态转换。"""


class MySQLSourceSyncTaskRepository:
    """持久化来源记录的增量同步任务。"""

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        attempt_token_factory: Callable[[], str] | None = None,
    ) -> None:
        """保存 Index 配置和 attempt token 生成器，连接由各方法按需创建。"""

        require_index_database(config)
        self.config = config
        self.attempt_token_factory = attempt_token_factory or _new_attempt_token

    def create_source_sync_task(
        self,
        task: NewSourceSyncTask,
    ) -> int:
        """按跨扫描 ``dedupe_key`` 幂等创建 pending 任务并返回 ID。

        新任务固定从 ``candidate/pending`` 开始。唯一键冲突后会再按
        ``dedupe_key`` 及其公式字段查询；只有确认是同一业务身份时才返回
        原任务 ID。旧批次内唯一键如果误命中不同任务，方法会回滚并报错。
        """

        if not isinstance(task, NewSourceSyncTask):
            raise TypeError("task 必须是 NewSourceSyncTask")

        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO expert_index_sync_task(
                        dedupe_key, batch_id, event_source,
                        source_table, source_record_id, source_updated_at,
                        expert_id, operation_type, target_type,
                        processing_stage, sync_status, retry_count,
                        policy_version
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'candidate', 'pending', 0, %s
                    )
                    ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)
                    """,
                    (
                        task.dedupe_key,
                        task.batch_id,
                        task.event_source,
                        task.source_table,
                        task.source_record_id,
                        task.source_updated_at,
                        task.expert_id,
                        task.operation_type,
                        task.target_type,
                        task.policy_version,
                    ),
                )
                # 不依赖命中的具体唯一索引；只认可完整幂等身份对应的任务。
                cursor.execute(
                    """
                    SELECT id
                    FROM expert_index_sync_task
                    WHERE dedupe_key=%s
                      AND event_source=%s
                      AND source_table=%s
                      AND source_record_id=%s
                      AND source_updated_at <=> %s
                      AND operation_type=%s
                      AND target_type=%s
                      AND policy_version=%s
                    """,
                    (
                        task.dedupe_key,
                        task.event_source,
                        task.source_table,
                        task.source_record_id,
                        task.source_updated_at,
                        task.operation_type,
                        task.target_type,
                        task.policy_version,
                    ),
                )
                identity_row = cursor.fetchone()
                if identity_row is None:
                    raise RuntimeError("同步任务唯一键冲突，但未找到匹配的 dedupe_key 身份")
                task_id = int(identity_row["id"])
            connection.commit()
            return task_id
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_source_sync_task(self, task_id: int) -> SourceSyncTask | None:
        """按任务 ID 读取同步任务。

        返回 ``None`` 表示任务不存在。方法只读取已提交状态，不锁定或修改任务记录。
        """

        _validate_positive_integer("task_id", task_id)
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_TASK_SELECT_COLUMNS} "
                    "FROM expert_index_sync_task WHERE id=%s",
                    (task_id,),
                )
                row = cursor.fetchone()
            if row is None:
                return None
            if int(row["id"]) != task_id:
                raise RuntimeError("同步任务查询结果与请求 ID 不一致")
            return _source_sync_task_from_row(row)
        finally:
            connection.close()

    def claim_source_sync_tasks(
        self,
        batch_id: int,
        source_table: str,
        *,
        limit: int,
        lease_seconds: int,
    ) -> tuple[SourceSyncTask, ...]:
        """并发安全领取 pending 或 lease 已过期的 processing 任务。

        任务按 ID 升序通过 ``FOR UPDATE SKIP LOCKED`` 领取。每条任务获得新的
        ``attempt_token``；回收过期 processing 任务时增加 ``retry_count``，但不改变
        已完成的处理阶段。方法只访问本地 Index MySQL。
        """

        _validate_positive_integer("batch_id", batch_id)
        _validate_source_table(source_table)
        _validate_claim_limit(limit)
        _validate_lease_seconds(lease_seconds)
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, sync_status
                    FROM expert_index_sync_task
                    WHERE batch_id=%s AND source_table=%s
                      AND (
                          (
                              sync_status='pending'
                              AND attempt_token IS NULL
                              AND lease_until IS NULL
                          )
                          OR (
                              sync_status='processing'
                              AND attempt_token IS NOT NULL
                              AND lease_until <= CURRENT_TIMESTAMP(6)
                          )
                      )
                    ORDER BY id ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                    """,
                    (batch_id, source_table, limit),
                )
                candidates = cursor.fetchall()
                if not candidates:
                    connection.commit()
                    return ()

                claimed_tokens = [self.attempt_token_factory() for _ in candidates]
                for token in claimed_tokens:
                    _validate_attempt_token(token)
                if len(set(claimed_tokens)) != len(claimed_tokens):
                    raise RuntimeError("本次领取生成了重复的 attempt_token")

                task_ids: list[int] = []
                for candidate, attempt_token in zip(candidates, claimed_tokens, strict=True):
                    task_id = int(candidate["id"])
                    previous_status = str(candidate["sync_status"])
                    if previous_status not in {"pending", "processing"}:
                        raise RuntimeError("领取查询返回了不可认领任务状态")
                    cursor.execute(
                        """
                        UPDATE expert_index_sync_task
                        SET sync_status='processing', attempt_token=%s,
                            lease_until=TIMESTAMPADD(
                                SECOND, %s, CURRENT_TIMESTAMP(6)
                            ),
                            retry_count=retry_count+%s,
                            started_at=CURRENT_TIMESTAMP(6), completed_at=NULL
                        WHERE id=%s AND sync_status=%s
                        """,
                        (
                            attempt_token,
                            lease_seconds,
                            1 if previous_status == "processing" else 0,
                            task_id,
                            previous_status,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise SourceSyncTaskLeaseConflictError(
                            f"任务 {task_id} 在领取期间状态已变化"
                        )
                    task_ids.append(task_id)

                placeholders = ", ".join(["%s"] * len(task_ids))
                cursor.execute(
                    f"SELECT {_TASK_SELECT_COLUMNS} "
                    f"FROM expert_index_sync_task WHERE id IN ({placeholders}) "
                    "ORDER BY id ASC",
                    tuple(task_ids),
                )
                rows = cursor.fetchall()
                tasks = tuple(_source_sync_task_from_row(row) for row in rows)
                _validate_claimed_tasks(tasks, task_ids, claimed_tokens)
            connection.commit()
            return tasks
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def renew_source_sync_task_lease(
        self,
        task_id: int,
        attempt_token: str,
        *,
        lease_seconds: int,
    ) -> SourceSyncTask:
        """为当前有效 attempt 延长任务 lease，并返回数据库中的最新任务。"""

        _validate_positive_integer("task_id", task_id)
        _validate_attempt_token(attempt_token)
        _validate_lease_seconds(lease_seconds)
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                _require_active_task_attempt(
                    _lock_source_sync_task(cursor, task_id),
                    task_id=task_id,
                    attempt_token=attempt_token,
                )
                cursor.execute(
                    """
                    UPDATE expert_index_sync_task
                    SET lease_until=TIMESTAMPADD(
                        SECOND, %s, CURRENT_TIMESTAMP(6)
                    )
                    WHERE id=%s AND sync_status='processing'
                      AND attempt_token=%s
                      AND lease_until > CURRENT_TIMESTAMP(6)
                    """,
                    (lease_seconds, task_id, attempt_token),
                )
                if cursor.rowcount != 1:
                    raise SourceSyncTaskLeaseConflictError(
                        f"任务 {task_id} 的 attempt lease 已变化"
                    )
                task = _select_source_sync_task(cursor, task_id)
            connection.commit()
            return task
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def advance_source_sync_task_stage(
        self,
        task_id: int,
        attempt_token: str,
        *,
        next_stage: str,
    ) -> SourceSyncTask:
        """把有效 processing 任务幂等推进到紧邻的下一个标准阶段。

        重复写入当前阶段直接返回现状。``completed`` 必须由可靠下游回执触发的成功方法
        写入，不能通过普通阶段推进绕过。
        """

        _validate_positive_integer("task_id", task_id)
        _validate_attempt_token(attempt_token)
        _validate_next_processing_stage(next_stage)
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                locked_row = _lock_source_sync_task(cursor, task_id)
                _require_active_task_attempt(
                    locked_row,
                    task_id=task_id,
                    attempt_token=attempt_token,
                )
                current_stage = str(locked_row["processing_stage"])
                if current_stage == next_stage:
                    task = _source_sync_task_from_row(locked_row)
                else:
                    expected_stage = _previous_processing_stage(next_stage)
                    if current_stage != expected_stage:
                        raise SourceSyncTaskStageConflictError(
                            f"任务 {task_id} 不能从 {current_stage} 推进到 {next_stage}"
                        )
                    cursor.execute(
                        """
                        UPDATE expert_index_sync_task
                        SET processing_stage=%s
                        WHERE id=%s AND sync_status='processing'
                          AND processing_stage=%s AND attempt_token=%s
                          AND lease_until > CURRENT_TIMESTAMP(6)
                        """,
                        (next_stage, task_id, current_stage, attempt_token),
                    )
                    if cursor.rowcount != 1:
                        raise SourceSyncTaskLeaseConflictError(
                            f"任务 {task_id} 的阶段或 attempt lease 已变化"
                        )
                    task = _select_source_sync_task(cursor, task_id)
            connection.commit()
            return task
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_source_sync_task_failed(
        self,
        task_id: int,
        attempt_token: str,
        *,
        error_stage: str,
        error_code: str,
        error_type: str | None = None,
        safe_error_message: str | None = None,
    ) -> SourceSyncTask:
        """把当前有效 attempt 标记为 failed，并写入受限的安全错误摘要。"""

        _validate_positive_integer("task_id", task_id)
        _validate_attempt_token(attempt_token)
        _validate_error_stage(error_stage)
        _validate_safe_identifier("error_code", error_code, max_length=64)
        _validate_optional_safe_identifier("error_type", error_type, max_length=128)
        _validate_safe_error_message(safe_error_message)
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                _require_active_task_attempt(
                    _lock_source_sync_task(cursor, task_id),
                    task_id=task_id,
                    attempt_token=attempt_token,
                )
                cursor.execute(
                    """
                    UPDATE expert_index_sync_task
                    SET sync_status='failed', attempt_token=NULL,
                        lease_until=NULL, next_retry_at=NULL,
                        error_stage=%s, error_code=%s, error_type=%s,
                        error_message=%s, result_code=NULL,
                        completed_at=NULL
                    WHERE id=%s AND sync_status='processing'
                      AND attempt_token=%s
                      AND lease_until > CURRENT_TIMESTAMP(6)
                    """,
                    (
                        error_stage,
                        error_code,
                        error_type,
                        safe_error_message,
                        task_id,
                        attempt_token,
                    ),
                )
                if cursor.rowcount != 1:
                    raise SourceSyncTaskLeaseConflictError(
                        f"任务 {task_id} 的 attempt lease 已变化"
                    )
                task = _select_source_sync_task(cursor, task_id)
            connection.commit()
            return task
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_source_sync_task_success(
        self,
        task_id: int,
        attempt_token: str,
        *,
        result_code: str,
    ) -> SourceSyncTask:
        """在 P5 可靠回执后把 handed_off 任务写成 completed/success。

        P3 不调用此方法；该入口只固化未来回执写回所需的 token、lease 和阶段守卫。
        """

        _validate_positive_integer("task_id", task_id)
        _validate_attempt_token(attempt_token)
        _validate_safe_identifier("result_code", result_code, max_length=64)
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                locked_row = _lock_source_sync_task(cursor, task_id)
                _require_active_task_attempt(
                    locked_row,
                    task_id=task_id,
                    attempt_token=attempt_token,
                )
                if locked_row["processing_stage"] != "handed_off":
                    raise SourceSyncTaskStageConflictError(
                        f"任务 {task_id} 只有 handed_off 阶段可以写入 success"
                    )
                cursor.execute(
                    """
                    UPDATE expert_index_sync_task
                    SET processing_stage='completed', sync_status='success',
                        attempt_token=NULL, lease_until=NULL,
                        next_retry_at=NULL, error_stage=NULL,
                        error_code=NULL, error_type=NULL, error_message=NULL,
                        result_code=%s, completed_at=CURRENT_TIMESTAMP(6)
                    WHERE id=%s AND sync_status='processing'
                      AND processing_stage='handed_off'
                      AND attempt_token=%s
                      AND lease_until > CURRENT_TIMESTAMP(6)
                    """,
                    (result_code, task_id, attempt_token),
                )
                if cursor.rowcount != 1:
                    raise SourceSyncTaskLeaseConflictError(
                        f"任务 {task_id} 的阶段或 attempt lease 已变化"
                    )
                task = _select_source_sync_task(cursor, task_id)
            connection.commit()
            return task
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _validate_positive_integer(field_name: str, value: int) -> None:
    """校验数据库标识字段为正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} 必须是正整数")


def _new_attempt_token() -> str:
    """生成不包含业务信息的随机 attempt token。"""

    return uuid.uuid4().hex


def _validate_source_table(source_table: str) -> None:
    """校验领取范围只使用四张冻结的来源表名。"""

    if source_table not in CHECKPOINT_SOURCE_TABLES:
        raise ValueError(f"不支持的同步任务来源表: {source_table}")


def _validate_claim_limit(limit: int) -> None:
    """校验单次任务领取数量，限制锁定范围。"""

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_SOURCE_SYNC_TASK_CLAIM_SIZE
    ):
        raise ValueError(
            f"limit 必须位于 1 到 {MAX_SOURCE_SYNC_TASK_CLAIM_SIZE} 之间"
        )


def _validate_lease_seconds(lease_seconds: int) -> None:
    """校验任务租约时长为正整数秒。"""

    if (
        isinstance(lease_seconds, bool)
        or not isinstance(lease_seconds, int)
        or lease_seconds < 1
    ):
        raise ValueError("lease_seconds 必须是正整数")


def _validate_attempt_token(attempt_token: str) -> None:
    """校验 attempt token 可精确写入 VARCHAR(64)。"""

    if (
        not isinstance(attempt_token, str)
        or not attempt_token.strip()
        or attempt_token != attempt_token.strip()
        or len(attempt_token) > 64
    ):
        raise ValueError("attempt_token 必须是长度不超过 64 的非空字符串")


def _validate_next_processing_stage(next_stage: str) -> None:
    """校验普通阶段推进的目标，completed 由成功写回专用入口处理。"""

    if next_stage not in SOURCE_SYNC_PROCESSING_STAGE_ORDER:
        raise ValueError(f"不支持的同步任务处理阶段: {next_stage}")
    if next_stage in {"candidate", "completed"}:
        raise ValueError("普通阶段推进不能写入 candidate 或 completed")


def _previous_processing_stage(stage: str) -> str:
    """返回标准状态机中指定阶段的直接前驱。"""

    index = SOURCE_SYNC_PROCESSING_STAGE_ORDER.index(stage)
    return SOURCE_SYNC_PROCESSING_STAGE_ORDER[index - 1]


def _validate_error_stage(error_stage: str) -> None:
    """校验失败位置使用冻结的安全阶段术语。"""

    if error_stage not in SOURCE_SYNC_ERROR_STAGES:
        raise ValueError(f"不支持的同步任务失败阶段: {error_stage}")


def _validate_safe_identifier(field_name: str, value: str, *, max_length: int) -> None:
    """校验机器标识仅包含稳定的非业务字符。"""

    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= max_length
        or _SAFE_IDENTIFIER_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            f"{field_name} 必须是长度不超过 {max_length} 的安全机器标识"
        )


def _validate_optional_safe_identifier(
    field_name: str,
    value: str | None,
    *,
    max_length: int,
) -> None:
    """校验可选机器标识；None 表示调用方没有安全类型信息。"""

    if value is not None:
        _validate_safe_identifier(field_name, value, max_length=max_length)


def _validate_safe_error_message(value: str | None) -> None:
    """校验调用方已脱敏的单行错误摘要，不负责保存异常正文。"""

    if value is None:
        return
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 512
        or any(character in value for character in "\r\n\x00")
    ):
        raise ValueError("safe_error_message 必须是长度不超过 512 的脱敏单行摘要或 None")


def _lock_source_sync_task(cursor: Any, task_id: int) -> dict[str, Any]:
    """锁定任务，并由 Index MySQL 判断当前任务 lease 是否仍有效。"""

    cursor.execute(
        f"SELECT {_TASK_SELECT_COLUMNS}, "
        "(sync_status='processing' "
        "AND attempt_token IS NOT NULL "
        "AND lease_until > CURRENT_TIMESTAMP(6)) AS lease_is_active "
        "FROM expert_index_sync_task WHERE id=%s FOR UPDATE",
        (task_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise SourceSyncTaskLeaseConflictError(f"任务 {task_id} 不存在")
    if int(row["id"]) != task_id:
        raise RuntimeError("锁定的同步任务与请求 ID 不一致")
    return row


def _require_active_task_attempt(
    row: dict[str, Any],
    *,
    task_id: int,
    attempt_token: str,
) -> None:
    """要求任务仍由调用方 token 持有有效 processing lease。"""

    if row.get("attempt_token") != attempt_token or not bool(row["lease_is_active"]):
        raise SourceSyncTaskLeaseConflictError(
            f"调用方不再持有任务 {task_id} 的有效 attempt lease"
        )


def _select_source_sync_task(cursor: Any, task_id: int) -> SourceSyncTask:
    """在当前事务内读取一条必然存在的同步任务。"""

    cursor.execute(
        f"SELECT {_TASK_SELECT_COLUMNS} "
        "FROM expert_index_sync_task WHERE id=%s",
        (task_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"任务 {task_id} 更新后无法读取")
    if int(row["id"]) != task_id:
        raise RuntimeError("同步任务更新结果与请求 ID 不一致")
    return _source_sync_task_from_row(row)


def _source_sync_task_from_row(row: dict[str, Any]) -> SourceSyncTask:
    """把任务表查询行完整映射为领域模型。"""

    return SourceSyncTask(
        task_id=int(row["id"]),
        dedupe_key=str(row["dedupe_key"]),
        batch_id=int(row["batch_id"]),
        event_source=str(row["event_source"]),
        source_table=str(row["source_table"]),
        source_record_id=int(row["source_record_id"]),
        source_updated_at=row.get("source_updated_at"),
        expert_id=(int(row["expert_id"]) if row.get("expert_id") is not None else None),
        operation_type=str(row["operation_type"]),
        target_type=str(row["target_type"]),
        processing_stage=str(row["processing_stage"]),
        sync_status=str(row["sync_status"]),
        retry_count=int(row["retry_count"]),
        attempt_token=row.get("attempt_token"),
        lease_until=row.get("lease_until"),
        next_retry_at=row.get("next_retry_at"),
        policy_version=row.get("policy_version"),
        error_stage=row.get("error_stage"),
        error_code=row.get("error_code"),
        error_type=row.get("error_type"),
        error_message=row.get("error_message"),
        result_code=row.get("result_code"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _validate_claimed_tasks(
    tasks: tuple[SourceSyncTask, ...],
    task_ids: list[int],
    attempt_tokens: list[str],
) -> None:
    """核对领取后回读结果与当前事务生成的任务身份完全一致。"""

    if [task.task_id for task in tasks] != task_ids:
        raise RuntimeError("任务领取后的回读结果不完整或顺序错误")
    if [task.attempt_token for task in tasks] != attempt_tokens:
        raise RuntimeError("任务领取后的 attempt_token 与生成值不一致")
    if any(task.sync_status != "processing" for task in tasks):
        raise RuntimeError("任务领取后的状态不是 processing")
