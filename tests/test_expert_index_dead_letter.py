"""P6.3 DLQ 模型和迁移的最小契约测试。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from expert_index.models import DeadLetterRecord

_NOW = datetime(2026, 8, 26, 12, 0, 0)  # noqa: DTZ001 -- 固定领域模型测试时间
_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "sql"
    / "016_extend_dead_letter_resolution_audit.sql"
).read_text(encoding="utf-8")


def _record(**overrides: object) -> DeadLetterRecord:
    values: dict[str, object] = {
        "dead_letter_id": 1,
        "task_id": 2,
        "batch_id": 3,
        "source_table": "experts",
        "source_record_id": 4,
        "expert_id": 5,
        "failure_stage": "receipt",
        "reason_code": "receipt_invalid",
        "reason_message": "safe",
        "retry_count": 1,
        "artifact_id": None,
        "resolution_status": "pending",
        "resolution_note": None,
        "resolution_operator": None,
        "resolution_request_id": None,
        "resolution_version": 0,
        "resolved_at": None,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    values.update(overrides)
    return DeadLetterRecord(**values)


def test_dead_letter_record_rejects_unsafe_resolution_status() -> None:
    with pytest.raises(ValueError):
        _record(resolution_status="processing")


def test_resolution_audit_migration_is_idempotent_and_local_only() -> None:
    assert "resolution_operator" in _MIGRATION
    assert "resolution_request_id" in _MIGRATION
    assert "resolution_version" in _MIGRATION
    assert "information_schema.COLUMNS" in _MIGRATION
    assert "DROP TABLE" not in _MIGRATION.upper()
    assert "TRUNCATE" not in _MIGRATION.upper()
