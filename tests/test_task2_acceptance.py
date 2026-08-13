from __future__ import annotations

from scripts.task2_acceptance import _safe_error


def test_task2_acceptance_redacts_mysql_password(monkeypatch) -> None:
    monkeypatch.setenv("MYSQL_PASSWORD", "mysql-secret")

    error = _safe_error(RuntimeError("connection failed for mysql-secret"))

    assert "mysql-secret" not in error["message"]
    assert "<redacted>" in error["message"]
