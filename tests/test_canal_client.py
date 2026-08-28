"""验证真实 Canal SDK 适配器的最小传输行为。"""

from __future__ import annotations

import pytest

from expert_index.cdc import (
    CANAL_DESTINATION,
    CanalBatch,
    CanalPythonConnector,
    build_real_canal_client,
    decode_canal_message,
)


class _SdkClient:
    """模拟 canal-python 常见的 get/ack/rollback 接口。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def connect(self) -> None:
        self.calls.append(("connect", None))

    def disconnect(self) -> None:
        self.calls.append(("disconnect", None))

    def subscribe(self, expression: str) -> None:
        self.calls.append(("subscribe", expression))

    def get(self, size: int) -> object:
        self.calls.append(("get", size))
        return {"id": 1}

    def rollback(self, batch_id: int) -> None:
        self.calls.append(("rollback", batch_id))

    def ack(self, batch_id: int) -> None:
        self.calls.append(("ack", batch_id))


def test_real_adapter_wraps_sdk_without_auto_ack() -> None:
    """真实适配器只转发 transport 调用，ACK 仍由上层显式决定。"""

    sdk = _SdkClient()
    batch = CanalBatch(-1, (), CANAL_DESTINATION)
    client = build_real_canal_client(
        host="127.0.0.1",
        port=11111,
        sdk_client=sdk,
        decoder=lambda _raw, _destination: batch,
    )

    client.connect()
    assert client.fetch_without_ack(10).is_empty
    client.rollback(1)
    client.ack(1)
    client.disconnect()

    assert sdk.calls[0][0] == "connect"
    assert ("get", 10) in sdk.calls
    assert sdk.calls[-1][0] == "disconnect"


def test_adapter_rolls_back_when_decoder_fails() -> None:
    """解码异常时立即回滚原始 batch，避免未确认消息悬挂。"""

    sdk = _SdkClient()
    client = build_real_canal_client(
        host="127.0.0.1",
        port=11111,
        sdk_client=sdk,
        decoder=lambda _raw, _destination: (_ for _ in ()).throw(ValueError("bad payload")),
    )

    client.connect()
    with pytest.raises(ValueError, match="bad payload"):
        client.fetch_without_ack(10)

    assert ("rollback", 1) in sdk.calls


def test_decoder_maps_dict_message_to_domain_batch() -> None:
    """解码器只构造无 SDK 领域模型，保留 INSERT 的定位字段。"""

    batch = decode_canal_message(
        {
            "id": 3,
            "entries": [
                {
                    "entry_type": "ROWDATA",
                    "header": {
                        "schema_name": "expert",
                        "table_name": "experts",
                        "logfile_name": "binlog.000012",
                        "logfile_offset": 9,
                        "server_id": 1,
                        "execute_time": 1,
                    },
                    "store_value": {
                        "event_type": "INSERT",
                        "row_datas": [
                            {"after": [{"name": "id", "value": 7}]},
                        ],
                    },
                }
            ],
        }
    )

    assert batch.batch_id == 3
    assert batch.entries[0].operation_type == "INSERT"
    assert batch.entries[0].rows[0].after[0].name == "id"


def test_connector_supports_constructor_endpoint_signature() -> None:
    """兼容标准 canal-python 在 Client 构造函数中接收 host/port 的形式。"""

    captured: dict[str, object] = {}

    class Client:
        def __init__(self, host: str, port: int) -> None:
            captured.update(host=host, port=port)

        def connect(self) -> None:
            pass

        def subscribe(self, client_filter: str) -> None:
            captured["filter"] = client_filter

        def get(self, batch_size: int) -> object:
            return {"id": batch_size}

        def disconnect(self) -> None:
            pass

        def rollback(self, batch_id: int) -> None:
            pass

        def ack(self, batch_id: int) -> None:
            pass

    connector = CanalPythonConnector(
        "canal.example",
        11111,
        sdk_client=None,
        client_factory=Client,
    )
    connector.connect()
    connector.subscribe("expert.*")
    assert captured == {"host": "canal.example", "port": 11111, "filter": "expert.*"}
