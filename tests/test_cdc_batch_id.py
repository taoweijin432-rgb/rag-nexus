"""验证 CDC 本地批次号避开已有 Poll/全量批次。"""

from __future__ import annotations

from typing import Self

from expert_index.cdc.batch_id import MySQLCdcLocalBatchIdProvider
from expert_index.config import DatabaseConfig, DatabaseRole


class _Cursor:
    def __init__(self) -> None:
        self.result: dict[str, int] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, _params: tuple[object, ...] = ()) -> None:
        normalized = " ".join(sql.split()).lower()
        if "get_lock" in normalized:
            self.result = {"acquired": 1}
        elif "max(local_batch_id)" in normalized or "last_local_batch_id" in normalized:
            self.result = {"max_id": 0}
        elif "max(id)" in normalized and "expert_index_batch" in normalized:
            self.result = {"max_id": 7}
        elif "max(batch_id)" in normalized and "expert_index_batch_source" in normalized:
            self.result = {"max_id": 6}
        elif "max(batch_id)" in normalized and "expert_index_sync_task" in normalized:
            self.result = {"max_id": 5}
        elif "release_lock" in normalized:
            self.result = {"released": 1}

    def fetchone(self) -> dict[str, int]:
        return self.result


class _Connection:
    def __init__(self) -> None:
        self.cursor_obj = _Cursor()

    def cursor(self) -> _Cursor:
        return self.cursor_obj

    def close(self) -> None:
        return None


def test_provider_starts_after_existing_index_batch_ids(monkeypatch) -> None:
    """CDC 首个本地批次号不得复用已有全量/Poll 批次号。"""

    connection = _Connection()
    monkeypatch.setattr(
        "expert_index.cdc.batch_id.pymysql.connect",
        lambda **_kwargs: connection,
    )
    config = DatabaseConfig("127.0.0.1", 3306, "index", "", "expert_test", DatabaseRole.INDEX)

    provider = MySQLCdcLocalBatchIdProvider(config, destination="company_expert")
    assert provider() == 8
    provider.close()
