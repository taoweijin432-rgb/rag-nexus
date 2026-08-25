"""受控 Contract Artifact Repository 的幂等保存、复用和冲突校验测试。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Self

import pytest

from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.repositories import (
    ArtifactPayloadConflictError,
    ArtifactPayloadValidationError,
    ContractArtifact,
    MySQLArtifactRepository,
)


class RecordingCursor:
    """记录 SQL 并返回测试指定的查询结果。"""

    def __init__(
        self,
        *,
        one: dict[str, Any] | None = None,
        ones: list[dict[str, Any] | None] | None = None,
        rowcounts: list[int] | None = None,
        lastrowid: int = 0,
    ) -> None:
        self.one = one
        self.ones = list(ones or [])
        self.rowcounts = list(rowcounts or [])
        self.lastrowid = lastrowid
        self.rowcount = 0
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append((" ".join(sql.split()), params))
        self.rowcount = self.rowcounts.pop(0) if self.rowcounts else 0

    def fetchone(self) -> dict[str, Any] | None:
        if self.ones:
            return self.ones.pop(0)
        return self.one

    def fetchall(self) -> list[dict[str, Any]]:
        return []


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self.cursor_instance = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> RecordingCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


_NOW = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
_LEASE = datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc)
_PAYLOAD_HASH = "a" * 64
_PAYLOAD_HASH_OTHER = "b" * 64


def _repository() -> MySQLArtifactRepository:
    config = DatabaseConfig(
        "127.0.0.1",
        3306,
        "test",
        "test",
        "index_test",
        role=DatabaseRole.INDEX,
    )
    return MySQLArtifactRepository(config)


def _patch_connection(
    monkeypatch: pytest.MonkeyPatch,
    connection: RecordingConnection,
) -> None:
    monkeypatch.setattr(
        "expert_index.repositories.artifact.pymysql.connect",
        lambda **_kwargs: connection,
    )


def _locked_task_row(**changes: object) -> dict[str, Any]:
    """构造 task 表 processing + 有效 lease 的锁定行。"""

    base: dict[str, Any] = {
        "id": 801,
        "dedupe_key": "b" * 64,
        "batch_id": 701,
        "event_source": "poll",
        "source_table": "experts_work_experience",
        "source_record_id": 901,
        "source_updated_at": _NOW,
        "expert_id": 101,
        "operation_type": "update",
        "target_type": "employment",
        "processing_stage": "contract_ready",
        "sync_status": "processing",
        "retry_count": 0,
        "attempt_token": "attempt-801",
        "lease_until": _LEASE,
        "next_retry_at": None,
        "policy_version": "1.3",
        "error_stage": None,
        "error_code": None,
        "error_type": None,
        "error_message": None,
        "result_code": None,
        "started_at": _NOW,
        "completed_at": None,
        "created_at": _NOW,
        "updated_at": _NOW,
        "lease_is_active": 1,
    }
    base.update(changes)
    return base


def _artifact_row(**changes: object) -> dict[str, Any]:
    """构造 artifact 表查询行。"""

    base: dict[str, Any] = {
        "id": 5001,
        "task_id": 801,
        "artifact_type": "contract_payload",
        "artifact_version": "4.1.0",
        "payload_json": _canonical_payload(),
        "payload_reference": None,
        "payload_hash": _PAYLOAD_HASH,
        "expires_at": None,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    base.update(changes)
    return base


def _canonical_payload() -> str:
    """构造一个符合白名单的合法 Contract Payload JSON 字符串。"""

    return json.dumps(
        {
            "request_id": "handoff:" + "c" * 64,
            "task_id": 801,
            "batch_id": 701,
            "attempt_token_hash": "d" * 64,
            "impact_scope": {
                "scope_type": "employment",
                "expert_id": 101,
                "chunk_type": "employment",
                "source_table": "experts_work_experience",
                "source_field_id": "901",
            },
            "contracts": [
                {
                    "vector_doc_id": "expert:101:employment:901:0",
                    "chunk_text": "示例文本",
                    "content_hash": "e" * 64,
                    "metadata_hash": "f" * 64,
                }
            ],
            "payload_hash": _PAYLOAD_HASH,
        },
        ensure_ascii=False,
    )


def test_get_contract_artifact_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不存在时只读查询返回 None，不锁定 task。"""

    cursor = RecordingCursor(one=None)
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    assert _repository().get_contract_artifact(
        801,
        artifact_type="contract_payload",
        artifact_version="4.1.0",
    ) is None
    assert connection.commit_count == 0
    assert connection.closed is True


def test_get_contract_artifact_maps_complete_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查询命中时返回完整 ContractArtifact。"""

    cursor = RecordingCursor(one=_artifact_row())
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    artifact = _repository().get_contract_artifact(
        801,
        artifact_type="contract_payload",
        artifact_version="4.1.0",
    )
    assert isinstance(artifact, ContractArtifact)
    assert artifact.artifact_id == 5001
    assert artifact.task_id == 801
    assert artifact.payload_hash == _PAYLOAD_HASH


def test_put_or_verify_inserts_artifact_on_first_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次写入应锁定 task、校验 token、白名单校验 payload 并 INSERT。"""

    inserted = _artifact_row(id=9001)
    cursor = RecordingCursor(
        ones=[_locked_task_row(), None, inserted],
        rowcounts=[0, 0, 1, 0],
        lastrowid=9001,
    )
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    artifact = _repository().put_or_verify_contract_artifact(
        801,
        "attempt-801",
        artifact_type="contract_payload",
        artifact_version="4.1.0",
        payload_json=_canonical_payload(),
        payload_hash=_PAYLOAD_HASH,
    )

    assert "FOR UPDATE" in cursor.executed[0][0]
    assert "expert_index_sync_task" in cursor.executed[0][0]
    assert "INSERT INTO expert_index_sync_artifact" in cursor.executed[2][0]
    insert_params = cursor.executed[2][1]
    assert insert_params == (
        801,
        "contract_payload",
        "4.1.0",
        _canonical_payload(),
        _PAYLOAD_HASH,
        None,
    )
    assert artifact.artifact_id == 9001
    assert connection.commit_count == 1
    assert connection.rollback_count == 0


def test_put_or_verify_reuses_existing_artifact_on_same_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同 identity 同 hash 幂等复用，不 INSERT，返回已有行。"""

    existing = _artifact_row()
    cursor = RecordingCursor(ones=[_locked_task_row(), existing], rowcounts=[0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    artifact = _repository().put_or_verify_contract_artifact(
        801,
        "attempt-801",
        artifact_type="contract_payload",
        artifact_version="4.1.0",
        payload_json=_canonical_payload(),
        payload_hash=_PAYLOAD_HASH,
    )

    assert artifact.artifact_id == 5001
    assert all("INSERT" not in sql for sql, _ in cursor.executed)
    assert connection.commit_count == 1


def test_put_or_verify_conflict_on_different_hash_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同 identity 不同 hash 必须 fail-closed 并回滚。"""

    existing = _artifact_row(payload_hash=_PAYLOAD_HASH_OTHER)
    cursor = RecordingCursor(ones=[_locked_task_row(), existing], rowcounts=[0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ArtifactPayloadConflictError, match="payload_hash 冲突"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=_canonical_payload(),
            payload_hash=_PAYLOAD_HASH,
        )

    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.closed is True


def test_put_or_verify_rejects_missing_task_before_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """task 不存在时锁定查询返回空，应在 fencing 阶段失败并回滚。"""

    cursor = RecordingCursor(ones=[None], rowcounts=[0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(Exception, match="不存在|有效 attempt lease"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=_canonical_payload(),
            payload_hash=_PAYLOAD_HASH,
        )

    assert connection.rollback_count == 1


def test_put_or_verify_rejects_wrong_token_before_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """token 不匹配的 processing task 不能写 Artifact。"""

    locked = _locked_task_row(attempt_token="attempt-old", lease_is_active=1)
    cursor = RecordingCursor(ones=[locked], rowcounts=[0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(Exception, match="有效 attempt lease"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-current",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=_canonical_payload(),
            payload_hash=_PAYLOAD_HASH,
        )

    assert len(cursor.executed) == 1
    assert connection.rollback_count == 1


def test_put_or_verify_rejects_expired_lease_before_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lease 已过期的 processing task 不能写 Artifact。"""

    locked = _locked_task_row(lease_is_active=0)
    cursor = RecordingCursor(ones=[locked], rowcounts=[0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(Exception, match="有效 attempt lease"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=_canonical_payload(),
            payload_hash=_PAYLOAD_HASH,
        )

    assert connection.rollback_count == 1


def test_put_or_verify_rejects_non_processing_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非 processing 状态（lease_is_active=0）应在 fencing 阶段拒绝。"""

    locked = _locked_task_row(
        sync_status="success",
        attempt_token=None,
        lease_until=None,
        lease_is_active=0,
    )
    cursor = RecordingCursor(ones=[locked], rowcounts=[0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(Exception, match="有效 attempt lease"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=_canonical_payload(),
            payload_hash=_PAYLOAD_HASH,
        )


def test_put_or_verify_rejects_forbidden_top_level_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """payload_json 顶层包含未授权键时 INSERT 前白名单校验 fail-closed。"""

    bad_payload = json.loads(_canonical_payload())
    bad_payload["raw_source_row"] = {"id": 901, "company_name": "敏感"}
    bad_payload["payload_hash"] = _PAYLOAD_HASH

    cursor = RecordingCursor(ones=[_locked_task_row(), None], rowcounts=[0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ArtifactPayloadValidationError, match="未授权顶层键"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=json.dumps(bad_payload, ensure_ascii=False),
            payload_hash=_PAYLOAD_HASH,
        )

    assert connection.rollback_count == 1


def test_put_or_verify_rejects_forbidden_contract_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract 内出现 embedding/vector/previous_state/action/stale_vector_doc_ids 拒绝。"""

    bad_payload = json.loads(_canonical_payload())
    bad_payload["contracts"][0]["embedding"] = [0.1, 0.2]
    bad_payload["payload_hash"] = _PAYLOAD_HASH

    cursor = RecordingCursor(ones=[_locked_task_row(), None], rowcounts=[0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ArtifactPayloadValidationError, match="禁止字段"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=json.dumps(bad_payload, ensure_ascii=False),
            payload_hash=_PAYLOAD_HASH,
        )

    assert connection.rollback_count == 1


def test_put_or_verify_rejects_forbidden_scope_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """impact_scope 包含未授权键时拒绝。"""

    bad_payload = json.loads(_canonical_payload())
    bad_payload["impact_scope"]["raw_mysql_row"] = "敏感"
    bad_payload["payload_hash"] = _PAYLOAD_HASH

    cursor = RecordingCursor(ones=[_locked_task_row(), None], rowcounts=[0, 0])
    connection = RecordingConnection(cursor)
    _patch_connection(monkeypatch, connection)

    with pytest.raises(ArtifactPayloadValidationError, match="未授权键"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=json.dumps(bad_payload, ensure_ascii=False),
            payload_hash=_PAYLOAD_HASH,
        )


def test_put_or_verify_rejects_invalid_json_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """payload_json 非合法 JSON 时参数校验失败，不建立数据库连接。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("payload 校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.artifact.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError, match="不是合法 JSON"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json="{not valid json",
            payload_hash=_PAYLOAD_HASH,
        )


def test_put_or_verify_rejects_bad_payload_hash_format_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """payload_hash 非 64 位小写十六进制时参数校验失败。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("payload_hash 校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.artifact.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError, match="64 位小写 SHA-256"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0",
            payload_json=_canonical_payload(),
            payload_hash="Z" * 64,
        )


def test_put_or_verify_rejects_unsafe_artifact_version_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """artifact_version 含业务字符时参数校验失败。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("artifact_version 校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.artifact.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError, match="安全机器标识"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="contract_payload",
            artifact_version="4.1.0 has spaces",
            payload_json=_canonical_payload(),
            payload_hash=_PAYLOAD_HASH,
        )


def test_put_or_verify_rejects_unsupported_artifact_type_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """artifact_type 不在 012 CHECK 范围时拒绝。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("artifact_type 校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.artifact.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError, match="不支持的 artifact_type"):
        _repository().put_or_verify_contract_artifact(
            801,
            "attempt-801",
            artifact_type="embedding_cache",
            artifact_version="4.1.0",
            payload_json=_canonical_payload(),
            payload_hash=_PAYLOAD_HASH,
        )


def test_get_contract_artifact_rejects_unsafe_inputs_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查询路径同样校验 artifact_type/version，不建立数据库连接。"""

    def unexpected_connect(**_kwargs: object) -> None:
        raise AssertionError("参数校验失败时不应连接 MySQL")

    monkeypatch.setattr(
        "expert_index.repositories.artifact.pymysql.connect",
        unexpected_connect,
    )

    with pytest.raises(ValueError, match="不支持的 artifact_type"):
        _repository().get_contract_artifact(
            801,
            artifact_type="unknown",
            artifact_version="4.1.0",
        )
