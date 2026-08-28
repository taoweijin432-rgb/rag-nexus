"""验证 P7.0/P7.1 的本机 Canal 事实基线和安全门禁。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_index.cdc import (
    CANAL_DESTINATION,
    CanalProtocolBaseline,
    CanalProtocolNotFrozenError,
)
from expert_index.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def _baseline() -> CanalProtocolBaseline:
    """读取固定基线并转换为领域门禁模型。"""

    payload = json.loads(
        (ROOT / "fixtures/canal_protocol_baseline_v1.json").read_text(encoding="utf-8")
    )
    server = payload["server"]
    probe = payload["probe"]
    return CanalProtocolBaseline(
        server_version=server["version"],
        server_mode=server["mode"],
        tcp_port=server["port"],
        destination=server["destination"],
        subscription=server["subscription"],
        source_connected=payload["source_connected"],
        probe_batch_id=probe["batch_id"],
        probe_entry_count=probe["entry_count"],
        ack_called=probe["ack_called"],
        production_ready=payload["production_ready"],
        unresolved=tuple(payload["unresolved"]),
    )


def test_local_canal_baseline_records_real_batch_but_remains_partial() -> None:
    """真实 batch 已记录到基线，但未确认重投语义时仍不得开启生产 CDC。"""

    baseline = _baseline()
    assert baseline.destination == CANAL_DESTINATION
    assert baseline.server_version == "1.1.8"
    assert baseline.source_connected is True
    assert baseline.probe_batch_id == 1
    assert baseline.probe_entry_count == 6
    assert baseline.ack_called is False
    assert baseline.production_ready is False

    with pytest.raises(CanalProtocolNotFrozenError):
        baseline.require_production_ready()

    payload = json.loads(
        (ROOT / "fixtures/canal_protocol_baseline_v1.json").read_text(encoding="utf-8")
    )
    real_delete = payload["real_batch"]["entries"][4]
    assert real_delete["operation_type"] == "DELETE"
    assert {"id", "expert_id"}.issubset(real_delete["before_columns"])


def test_cdc_runtime_defaults_to_disabled_and_rejects_ambiguous_switch() -> None:
    """正式配置缺省关闭 CDC，且不能用含义模糊的文本绕过门禁。"""

    environment = {
        "SOURCE_DB_PASSWORD": "source-secret",
        "INDEX_DB_HOST": "127.0.0.1",
        "INDEX_DB_PORT": "3306",
        "INDEX_DB_USER": "index-writer",
        "INDEX_DB_PASSWORD": "index-secret",
        "INDEX_DB_NAME": "expert_index",
    }
    assert Settings.from_env(environment).cdc_runtime.enabled is False

    environment["CDC_ENABLED"] = "maybe"
    with pytest.raises(ValueError, match="CDC_ENABLED"):
        Settings.from_env(environment)


def test_enabled_cdc_requires_stable_client_id() -> None:
    """开启 CDC 时必须显式配置稳定 client_id，保证 ACK 失败后可重连重投。"""

    environment = {
        "SOURCE_DB_PASSWORD": "source-secret",
        "INDEX_DB_HOST": "127.0.0.1",
        "INDEX_DB_PORT": "3306",
        "INDEX_DB_USER": "index-writer",
        "INDEX_DB_PASSWORD": "index-secret",
        "INDEX_DB_NAME": "expert_index",
        "CDC_ENABLED": "1",
    }

    with pytest.raises(ValueError, match="CANAL_CLIENT_ID"):
        Settings.from_env(environment)

    environment["CANAL_CLIENT_ID"] = "2001"
    assert Settings.from_env(environment).cdc_runtime.client_id == "2001"

    environment["CANAL_CLIENT_ID"] = "32768"
    with pytest.raises(ValueError, match="1 到 32767"):
        Settings.from_env(environment)
