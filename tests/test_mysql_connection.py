from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from expert_index.config import DatabaseConfig
from rag_vector_store.config import MySQLSettings
from rag_vector_store.mysql import MySQLConnection, MySQLConnectionError


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_mysql_settings_reads_environment_without_exposing_password() -> None:
    settings = MySQLSettings.from_env(
        {
            "MYSQL_HOST": "mysql.local",
            "MYSQL_PORT": "3307",
            "MYSQL_DATABASE": "chunk_db",
            "MYSQL_USER": "readonly",
            "MYSQL_PASSWORD": "secret-value",
        }
    )

    assert settings.host == "mysql.local"
    assert settings.port == 3307
    assert settings.database == "chunk_db"
    assert settings.user == "readonly"
    assert settings.password == SecretStr("secret-value")
    assert "secret-value" not in repr(settings)
    assert "secret-value" not in str(settings)
    assert settings.connector_kwargs["password"] == "secret-value"


def test_mysql_connection_manages_sync_connection_lifecycle() -> None:
    settings = MySQLSettings(database="chunk_db", user="readonly")
    created: list[dict[str, Any]] = []
    fake_connection = FakeConnection()

    def connector(**kwargs: Any) -> FakeConnection:
        created.append(kwargs)
        return fake_connection

    with MySQLConnection(settings, connector_factory=connector) as connection:
        assert connection is fake_connection

    assert created == [{
        "host": "127.0.0.1",
        "port": 3306,
        "database": "chunk_db",
        "user": "readonly",
    }]
    assert fake_connection.closed is True


def test_mysql_connection_reuses_connection_and_close_is_idempotent() -> None:
    settings = MySQLSettings(database="chunk_db", user="readonly")
    fake_connection = FakeConnection()
    calls = 0

    def connector(**_: Any) -> FakeConnection:
        nonlocal calls
        calls += 1
        return fake_connection

    connection = MySQLConnection(settings, connector_factory=connector)
    assert connection.connect() is fake_connection
    assert connection.connection is fake_connection
    assert calls == 1

    connection.close()
    connection.close()
    assert fake_connection.closed is True


def test_mysql_connection_accepts_teammate_database_config() -> None:
    settings = DatabaseConfig("mysql.local", 3306, "readonly", "", "expert_db")
    created: list[dict[str, Any]] = []
    fake_connection = FakeConnection()

    def connector(**kwargs: Any) -> FakeConnection:
        created.append(kwargs)
        return fake_connection

    with MySQLConnection(settings, connector_factory=connector):
        pass

    assert created[0]["host"] == "mysql.local"
    assert created[0]["database"] == "expert_db"
    assert created[0]["user"] == "readonly"


def test_mysql_connection_failure_does_not_expose_secret() -> None:
    settings = MySQLSettings(
        database="chunk_db",
        user="readonly",
        password=SecretStr("secret-value"),
    )

    def connector(**_: Any) -> FakeConnection:
        raise RuntimeError("driver received secret-value")

    with pytest.raises(MySQLConnectionError) as error:
        MySQLConnection(settings, connector_factory=connector).connect()

    assert "secret-value" not in str(error.value)
    assert "RuntimeError" in str(error.value)


def test_mysql_settings_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="MYSQL_PORT"):
        MySQLSettings.from_env({"MYSQL_PORT": "not-a-port"})
