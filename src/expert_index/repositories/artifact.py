"""受控 Contract Artifact 的 MySQL 数据访问层。

P5.3A 正式交付链要求把 P5.1 ``ScopeHandoffRequest`` 持久化为可恢复的受控
Contract Artifact，供崩溃恢复和跨阶段重试幂等复用。本模块只复用 ``sql/012``
的 ``expert_index_sync_artifact`` 表，不新建表、不修改 DDL。

能力边界（严格）：

- 只使用 ``(task_id, artifact_type, artifact_version)`` 查询 Artifact；表本身没有
  ``request_id``/``attempt_token`` 列，fencing 由同一事务内对
  ``expert_index_sync_task`` 行的 ``FOR UPDATE`` 锁定完成（复用 ``sync_task``
  模块的 ``_lock_source_sync_task``/``_require_active_task_attempt``）。
- ``put_or_verify``：不存在则保存；同 identity 同 ``payload_hash`` 幂等复用；
  同 identity 不同 ``payload_hash`` 抛 :class:`ArtifactPayloadConflictError`
  并回滚。
- 写入前在同一事务校验 ``task_id`` 存在、``sync_status='processing'``、
  ``attempt_token`` 匹配且 ``lease_until`` 未过期。
- ``payload_json`` 白名单校验：只允许 request identity、exact scope、有序 Contract
  payload、``payload_hash``；**禁止** embedding vector、previous state、action
  预判、``stale_vector_doc_ids``、原始 MySQL 行、PII 或其他敏感字段。
- 不保存 ``attempt_token`` 到 Artifact JSON（fencing 已由 task 行锁完成，Artifact
  行本身只存稳定可恢复 Payload）。
"""

from __future__ import annotations

import json
from typing import Any

import pymysql

from ..config import DatabaseConfig, require_index_database
from .sync_task import (
    _lock_source_sync_task,
    _require_active_task_attempt,
    _validate_attempt_token,
    _validate_positive_integer,
    _validate_safe_identifier,
)

__all__ = [
    "ArtifactPayloadConflictError",
    "ArtifactPayloadValidationError",
    "ContractArtifact",
    "MySQLArtifactRepository",
]

_ARTIFACT_SELECT_COLUMNS = (
    "id, task_id, artifact_type, artifact_version, "
    "payload_json, payload_reference, payload_hash, expires_at, "
    "created_at, updated_at"
)

# payload_json 白名单允许的顶层键。Contract Payload 只携带稳定 request identity、
# exact scope 和有序 Contract 列表，不携带向量、previous state、动作预判或原始行。
_ALLOWED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "request_id",
        "task_id",
        "batch_id",
        "attempt_token_hash",
        "impact_scope",
        "contracts",
        "payload_hash",
    }
)

# impact_scope 子对象允许的字段（与 P5.1 IncrementalImpactScope 结构对齐）。
_ALLOWED_SCOPE_KEYS: frozenset[str] = frozenset(
    {
        "scope_type",
        "expert_id",
        "chunk_type",
        "source_table",
        "source_field_id",
    }
)

# Contract 内被显式禁止的顶层键，任何命中即 fail-closed。
_FORBIDDEN_CONTRACT_KEYS: frozenset[str] = frozenset(
    {
        "embedding",
        "embedding_dim",
        "embedding_model_alias",
        "embedding_model_version",
        "embedded_at",
        "previous_state",
        "action",
        "stale_vector_doc_ids",
        "raw_source_row",
    }
)

_SHA256_HEX_PATTERN_SET = frozenset("0123456789abcdef")


class ArtifactPayloadConflictError(RuntimeError):
    """同 identity Artifact 已存在但 payload_hash 不一致（fail-closed）。"""


class ArtifactPayloadValidationError(RuntimeError):
    """Artifact payload 违反白名单或禁止字段约束。"""


class ContractArtifact:
    """一条受控 Contract Artifact 的领域视图。

    只暴露稳定 identity、版本、payload 引用和完整性哈希；``payload_json`` 以原始
    JSON 字符串形式保存，调用方需在白名单校验通过后自行解释。
    """

    __slots__ = (
        "artifact_id",
        "artifact_type",
        "artifact_version",
        "created_at",
        "expires_at",
        "payload_hash",
        "payload_json",
        "payload_reference",
        "task_id",
        "updated_at",
    )

    def __init__(
        self,
        *,
        artifact_id: int,
        task_id: int,
        artifact_type: str,
        artifact_version: str,
        payload_json: str | None,
        payload_reference: str | None,
        payload_hash: str,
        expires_at: Any,
        created_at: Any,
        updated_at: Any,
    ) -> None:
        self.artifact_id = artifact_id
        self.task_id = task_id
        self.artifact_type = artifact_type
        self.artifact_version = artifact_version
        self.payload_json = payload_json
        self.payload_reference = payload_reference
        self.payload_hash = payload_hash
        self.expires_at = expires_at
        self.created_at = created_at
        self.updated_at = updated_at


class MySQLArtifactRepository:
    """持久化受控 Contract Artifact。

    所有写操作都先在同一事务内锁定 ``expert_index_sync_task`` 行并校验有效 attempt
    lease，再操作 Artifact 表。只使用 012 已有表，不新增表、不修改 DDL。
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """保存 Index 配置，连接由各方法按需创建。"""

        require_index_database(config)
        self.config = config

    def get_contract_artifact(
        self,
        task_id: int,
        *,
        artifact_type: str,
        artifact_version: str,
    ) -> ContractArtifact | None:
        """按 ``(task_id, artifact_type, artifact_version)`` 查询 Artifact。

        返回 ``None`` 表示不存在。只读，不锁定、不校验 token。
        """

        _validate_positive_integer("task_id", task_id)
        _validate_artifact_type(artifact_type)
        _validate_artifact_version(artifact_version)
        connection = pymysql.connect(**self.config.as_kwargs(autocommit=True))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_ARTIFACT_SELECT_COLUMNS} "
                    "FROM expert_index_sync_artifact "
                    "WHERE task_id=%s AND artifact_type=%s AND artifact_version=%s",
                    (task_id, artifact_type, artifact_version),
                )
                row = cursor.fetchone()
            if row is None:
                return None
            return _artifact_from_row(row)
        finally:
            connection.close()

    def put_or_verify_contract_artifact(
        self,
        task_id: int,
        attempt_token: str,
        *,
        artifact_type: str,
        artifact_version: str,
        payload_json: str,
        payload_hash: str,
        expires_at: Any = None,
    ) -> ContractArtifact:
        """幂等保存或校验受控 Contract Artifact。

        在同一事务内：锁定 task 行 → 校验 processing + token + 未过期 lease →
        查询 Artifact → 不存在则白名单校验 payload 并 INSERT；存在且 hash 相同
        则复用；存在且 hash 不同则抛 :class:`ArtifactPayloadConflictError`。
        """

        _validate_positive_integer("task_id", task_id)
        _validate_attempt_token(attempt_token)
        _validate_artifact_type(artifact_type)
        _validate_artifact_version(artifact_version)
        _validate_safe_identifier("payload_hash", payload_hash, max_length=64)
        _validate_payload_hash_hex(payload_hash)
        _validate_payload_json(payload_json)
        connection = pymysql.connect(**self.config.as_kwargs())
        try:
            with connection.cursor() as cursor:
                # fencing：先锁定 task 行，校验调用方仍持有有效 processing lease。
                _require_active_task_attempt(
                    _lock_source_sync_task(cursor, task_id),
                    task_id=task_id,
                    attempt_token=attempt_token,
                )
                cursor.execute(
                    f"SELECT {_ARTIFACT_SELECT_COLUMNS} "
                    "FROM expert_index_sync_artifact "
                    "WHERE task_id=%s AND artifact_type=%s AND artifact_version=%s "
                    "FOR UPDATE",
                    (task_id, artifact_type, artifact_version),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    existing_artifact = _artifact_from_row(existing)
                    if existing_artifact.payload_hash != payload_hash:
                        raise ArtifactPayloadConflictError(
                            f"任务 {task_id} 的 Artifact ({artifact_type}/"
                            f"{artifact_version}) payload_hash 冲突，拒绝覆盖"
                        )
                    artifact = existing_artifact
                else:
                    _validate_payload_whitelist(payload_json)
                    cursor.execute(
                        """
                        INSERT INTO expert_index_sync_artifact(
                            task_id, artifact_type, artifact_version,
                            payload_json, payload_reference, payload_hash, expires_at
                        ) VALUES (%s, %s, %s, %s, NULL, %s, %s)
                        """,
                        (
                            task_id,
                            artifact_type,
                            artifact_version,
                            payload_json,
                            payload_hash,
                            expires_at,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("Artifact 写入失败")
                    cursor.execute(
                        f"SELECT {_ARTIFACT_SELECT_COLUMNS} "
                        "FROM expert_index_sync_artifact WHERE id=LAST_INSERT_ID()",
                        (),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise RuntimeError("Artifact 写入后无法读取")
                    artifact = _artifact_from_row(row)
            connection.commit()
            return artifact
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _artifact_from_row(row: dict[str, Any]) -> ContractArtifact:
    """把 Artifact 表查询行映射为 :class:`ContractArtifact`。"""

    return ContractArtifact(
        artifact_id=int(row["id"]),
        task_id=int(row["task_id"]),
        artifact_type=str(row["artifact_type"]),
        artifact_version=str(row["artifact_version"]),
        payload_json=row.get("payload_json"),
        payload_reference=row.get("payload_reference"),
        payload_hash=str(row["payload_hash"]),
        expires_at=row.get("expires_at"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _validate_artifact_type(artifact_type: str) -> None:
    """校验 artifact_type 在 012 CHECK 允许范围内。"""

    if artifact_type not in ("contract_payload", "contract_reference"):
        raise ValueError(f"不支持的 artifact_type: {artifact_type}")


def _validate_artifact_version(artifact_version: str) -> None:
    """校验 artifact_version 是非空安全标识，长度不超过 VARCHAR(32)。"""

    _validate_safe_identifier("artifact_version", artifact_version, max_length=32)


def _validate_payload_hash_hex(payload_hash: str) -> None:
    """校验 payload_hash 是 64 位小写 SHA-256 十六进制。"""

    if len(payload_hash) != 64 or any(
        character not in _SHA256_HEX_PATTERN_SET for character in payload_hash
    ):
        raise ValueError("payload_hash 必须是 64 位小写 SHA-256 十六进制")


def _validate_payload_json(payload_json: str) -> None:
    """校验 payload_json 是合法 JSON 字符串。"""

    if not isinstance(payload_json, str) or not payload_json.strip():
        raise ValueError("payload_json 必须是非空 JSON 字符串")
    try:
        json.loads(payload_json)
    except json.JSONDecodeError as error:
        raise ValueError(f"payload_json 不是合法 JSON: {error}") from error


def _validate_payload_whitelist(payload_json: str) -> None:
    """校验 payload_json 顶层键和 Contract payload 内容符合白名单。

    只允许 request identity、exact scope、有序 Contract 列表和 payload_hash。
    禁止 embedding vector、previous state、action、stale_vector_doc_ids、原始行
    等字段出现在 Contract 内或 payload 顶层。
    """

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as error:
        raise ArtifactPayloadValidationError(
            f"payload_json 不是合法 JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ArtifactPayloadValidationError("payload_json 顶层必须是 JSON 对象")
    unknown_keys = set(payload) - _ALLOWED_PAYLOAD_KEYS
    if unknown_keys:
        raise ArtifactPayloadValidationError(
            f"payload_json 包含未授权顶层键: {sorted(unknown_keys)}"
        )
    scope = payload.get("impact_scope")
    if scope is not None:
        if not isinstance(scope, dict):
            raise ArtifactPayloadValidationError("impact_scope 必须是 JSON 对象")
        scope_unknown = set(scope) - _ALLOWED_SCOPE_KEYS
        if scope_unknown:
            raise ArtifactPayloadValidationError(
                f"impact_scope 包含未授权键: {sorted(scope_unknown)}"
            )
    contracts = payload.get("contracts")
    if contracts is not None:
        if not isinstance(contracts, list):
            raise ArtifactPayloadValidationError("contracts 必须是 JSON 数组")
        for index, contract in enumerate(contracts):
            if not isinstance(contract, dict):
                raise ArtifactPayloadValidationError(
                    f"contracts[{index}] 必须是 JSON 对象"
                )
            forbidden = set(contract) & _FORBIDDEN_CONTRACT_KEYS
            if forbidden:
                raise ArtifactPayloadValidationError(
                    f"contracts[{index}] 包含禁止字段: {sorted(forbidden)}"
                )
    declared_hash = payload.get("payload_hash")
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise ArtifactPayloadValidationError("payload_hash 必须是 64 位字符串")


# 显式导出供测试断言使用的校验函数（内部使用，不进 __all__）。
__all__ += [
    "_ALLOWED_PAYLOAD_KEYS",
    "_ALLOWED_SCOPE_KEYS",
    "_FORBIDDEN_CONTRACT_KEYS",
    "_validate_payload_hash_hex",
    "_validate_payload_json",
    "_validate_payload_whitelist",
]
