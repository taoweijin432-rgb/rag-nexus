"""CDC 正式入口的配置、身份和 CLI 门禁测试。"""

from dataclasses import replace

from expert_index.cdc import (
    CANAL_DESTINATION,
    CANAL_SUBSCRIPTION,
    CanalChangeEvent,
    CdcBatchRunResult,
    CdcRuntime,
    build_cdc_event_id,
    build_cdc_protocol_baseline,
)
from expert_index.cli import build_parser
from expert_index.config import DatabaseConfig, DatabaseRole, IncrementalScanConfig, Settings


def _settings(**cdc_values: object) -> Settings:
    """构造不连接数据库的 CDC 配置。"""

    source = DatabaseConfig("source", 3306, "reader", "secret", "expert", DatabaseRole.SOURCE)
    index = DatabaseConfig("index", 3306, "writer", "secret", "expert_index", DatabaseRole.INDEX)
    from expert_index.config import CdcRuntimeConfig

    config = CdcRuntimeConfig(
        enabled=True,
        client_id="2001",
        source_connected=True,
        probe_batch_id=1,
        probe_entry_count=13,
        ack_confirmed=True,
        protocol_ready=True,
    )
    config = replace(config, **cdc_values)
    return Settings(source, index, incremental_scan=IncrementalScanConfig(enabled=True), cdc_runtime=config)


def test_event_identity_is_stable_across_rebatching_and_row_specific() -> None:
    """重分批导致 entry_index 变化时身份稳定，不同行序号仍不会碰撞。"""

    event = CanalChangeEvent(
        "expert",
        "experts_work_experience",
        10,
        20,
        "binlog.000012",
        76094382,
        1,
        1787747052000,
        1,
        0,
        "delete",
    )
    assert build_cdc_event_id(event) == build_cdc_event_id(event)
    assert build_cdc_event_id(event) == build_cdc_event_id(replace(event, entry_index=9))
    assert build_cdc_event_id(event) != build_cdc_event_id(replace(event, row_index=1))


def test_protocol_baseline_requires_explicit_ack_gate() -> None:
    """真实 batch 信息齐全但未确认 ACK 时，基线仍保持关闭。"""

    settings = _settings(ack_confirmed=False)
    baseline = build_cdc_protocol_baseline(settings)
    assert baseline.destination == CANAL_DESTINATION
    assert baseline.subscription == CANAL_SUBSCRIPTION
    assert baseline.probe_entry_count == 13
    assert baseline.production_ready is False
    assert "ack_replay_unconfirmed" in baseline.unresolved


def test_cli_exposes_cdc_worker_command() -> None:
    """正式入口要求显式 Worker 和重试/租约参数。"""

    args = build_parser().parse_args(
        [
            "cdc-worker",
            "--worker-id",
            "cdc-test",
            "--max-attempts",
            "3",
            "--retry-delay-seconds",
            "30",
            "--task-lease-seconds",
            "180",
            "--max-cycles",
            "1",
        ]
    )
    assert args.command == "cdc-worker"
    assert args.batch_size == 100
    assert args.task_lease_seconds == 180


def test_runtime_reuses_canal_session_until_stopped() -> None:
    """连续消费只订阅一次，异常时才释放会话。"""

    calls: list[str] = []

    class Client:
        def connect(self) -> None:
            calls.append("connect")

        def disconnect(self) -> None:
            calls.append("disconnect")

    class Coordinator:
        def consume_once(self, client: object, *, batch_size: int, local_batch_id: int) -> CdcBatchRunResult:
            assert isinstance(client, Client)
            assert batch_size == 10
            assert local_batch_id > 0
            calls.append("consume")
            return CdcBatchRunResult("empty", -1, local_batch_id, 0)

    runtime = CdcRuntime(Client(), Coordinator())
    runtime.start()
    runtime.consume_once(batch_size=10, local_batch_id=1)
    runtime.consume_once(batch_size=10, local_batch_id=2)
    runtime.stop()
    runtime.stop()

    assert calls == ["connect", "consume", "consume", "disconnect"]
