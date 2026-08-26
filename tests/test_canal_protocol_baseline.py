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
