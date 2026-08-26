"""Index MySQL 专家级执行栅栏，串行化会修改同一专家向量状态的请求。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pymysql

from ..config import DatabaseConfig, require_index_database
from ..models import IncrementalImpactScope, SourceSyncTask
from .sync_task import SourceSyncTaskLeaseConflictError

_LOCK_PREFIX = "expert-index:scope:"


class ScopeFenceUnavailableError(RuntimeError):
    """专家级执行栅栏未能在限定时间内取得。"""


@dataclass(frozen=True)
class ScopeFenceDecision:
    """持锁后得到的顺序判定；superseded 表示已有更晚的重叠任务。"""

    superseded: bool


class MySQLScopeExecutionFence:
    """通过 MySQL 命名锁串行同一专家，并阻止旧重叠任务晚到覆盖。"""

    def __init__(self, config: DatabaseConfig) -> None:
        require_index_database(config)
        self.config = config

    @contextmanager
    def hold(
        self,
        task: SourceSyncTask,
        attempt_token: str,
        scope: IncrementalImpactScope,
        *,
        timeout_seconds: int,
    ) -> Iterator[ScopeFenceDecision]:
        """在 Consumer 执行期持有专家级锁，并返回旧请求判定。"""

        if scope.expert_id != task.expert_id:
            raise ValueError("scope.expert_id 与 task.expert_id 不一致")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or timeout_seconds < 1:
            raise ValueError("timeout_seconds 必须是正整数")
        lock_name = f"{_LOCK_PREFIX}{scope.expert_id}"
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        acquired = False
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT GET_LOCK(%s, %s) AS acquired", (lock_name, timeout_seconds))
                row = cursor.fetchone()
                acquired = row is not None and int(row.get("acquired") or 0) == 1
                if not acquired:
                    raise ScopeFenceUnavailableError("专家级执行栅栏获取超时")
                cursor.execute(
                    """
                    SELECT id
                    FROM expert_index_sync_task
                    WHERE id=%s AND sync_status='processing' AND attempt_token=%s
                      AND lease_until > CURRENT_TIMESTAMP(6)
                    """,
                    (task.task_id, attempt_token),
                )
                if cursor.fetchone() is None:
                    raise SourceSyncTaskLeaseConflictError(
                        f"任务 {task.task_id} 在取得 scope fence 后已失去有效 lease"
                    )
                cursor.execute(
                    _newer_overlap_sql(scope.scope_type),
                    _newer_overlap_params(task, scope),
                )
                newer = cursor.fetchone()
                yield ScopeFenceDecision(
                    superseded=newer is not None and bool(newer.get("has_newer")),
                )
        finally:
            if acquired:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                finally:
                    connection.close()
            else:
                connection.close()


def _newer_overlap_sql(scope_type: str) -> str:
    """返回当前 scope 对应的后继重叠任务查询。"""

    base = """
        SELECT EXISTS(
            SELECT 1 FROM expert_index_sync_task newer
            WHERE newer.id > %s AND newer.expert_id=%s
              AND newer.sync_status <> 'dead'
              AND (
                  newer.sync_status='success'
                  OR EXISTS (
                      SELECT 1 FROM expert_index_sync_artifact artifact
                      WHERE artifact.task_id=newer.id
                        AND artifact.artifact_type='contract_payload'
                        AND artifact.artifact_version='v1'
                  )
              )
              AND {overlap}
        ) AS has_newer
    """
    if scope_type == "expert":
        overlap = "1=1"
    elif scope_type == "profile":
        overlap = "newer.target_type IN ('expert', 'profile', 'education')"
    elif scope_type == "employment":
        overlap = "(newer.target_type='expert' OR (newer.target_type='employment' AND newer.source_record_id=%s))"
    elif scope_type == "project":
        overlap = "(newer.target_type='expert' OR (newer.target_type='project' AND newer.source_record_id=%s))"
    else:
        raise ValueError(f"不支持的 scope_type: {scope_type}")
    return base.format(overlap=overlap)


def _newer_overlap_params(
    task: SourceSyncTask,
    scope: IncrementalImpactScope,
) -> tuple[object, ...]:
    """生成后继重叠任务查询参数。"""

    params: list[object] = [task.task_id, scope.expert_id]
    if scope.scope_type in {"employment", "project"}:
        params.append(task.source_record_id)
    return tuple(params)


__all__ = [
    "MySQLScopeExecutionFence",
    "ScopeFenceDecision",
    "ScopeFenceUnavailableError",
]
