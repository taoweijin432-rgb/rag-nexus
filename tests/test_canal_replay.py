"""验证 P7.2 固定 Canal Fixture 的分类、去重和安全回滚。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_index.cdc import (
    CanalBatch,
    CanalEntry,
    FakeCanalClient,
    replay_batch,
    replay_next_batch,
)

ROOT = Path(__file__).resolve().parents[1]
REPLAY_FIXTURE = json.loads(
    (ROOT / "fixtures/canal_p7_replay_v1.json").read_text(encoding="utf-8")
)


def _batch(case: dict) -> CanalBatch:
    """把脱敏 JSON case 转为领域层 batch，不引入任何列值。"""

    return CanalBatch(
        batch_id=case["batch_id"],
        entries=tuple(
            CanalEntry(
                entry_type=entry["entry_type"],
                source_schema=entry.get("schema", ""),
                source_table=entry.get("table", ""),
                operation_type=entry.get("operation_type"),
                logfile_name=entry.get("logfile_name"),
                logfile_offset=entry.get("logfile_offset"),
                server_id=entry.get("server_id"),
                execute_time=entry.get("execute_time"),
                before_columns=tuple(entry.get("before_columns", ())),
                after_columns=tuple(entry.get("after_columns", ())),
            )
            for entry in case["entries"]
        ),
    )


@pytest.mark.parametrize("case", REPLAY_FIXTURE["cases"], ids=lambda case: case["name"])
def test_replay_fixture_matches_expected_classification(case: dict) -> None:
    """固定样例覆盖空 batch、忽略更新、四表删除、畸形和重复投递。"""

    expected = case["expected"]
    result = replay_batch(_batch(case))

    assert len(result.accepted_deletes) == expected["accepted_deletes"]
    assert result.ignored_count == expected["ignored"]
    assert result.duplicate_count == expected["duplicates"]
    assert result.malformed_count == expected["malformed"]
    assert result.rollback_required is expected["rollback"]


def test_fake_client_rolls_back_malformed_batch_without_ack() -> None:
    """畸形 batch 必须回滚，Fake Client 不应被重放流程确认。"""

    case = next(item for item in REPLAY_FIXTURE["cases"] if item["name"] == "malformed_delete")
    client = FakeCanalClient((_batch(case),))

    result = replay_next_batch(client)

    assert result.rollback_required is True
    assert client.rollback_count == 1
    assert client.ack_count == 0
