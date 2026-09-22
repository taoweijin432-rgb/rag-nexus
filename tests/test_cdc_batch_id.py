"""验证 CDC 本地批次号避开已有 Poll/全量批次。"""

from __future__ import annotations

from typing import Self

import pytest

from expert_index.cdc.batch_id import MySQLCdcLocalBatchIdProvider
from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.constants import CDC_LOCAL_BATCH_ID_BASE, MAX_UNSIGNED_BIGINT


class _Cursor:
    def __init__(self, *, persisted_max_id: int = 0) -> None:
        self.result: dict[str, int] = {}
        self.persisted_max_id = persisted_max_id

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, _params: tuple[object, ...] = ()) -> None:
        normalized = " ".join(sql.split()).lower()
        if "get_lock" in normalized:
            self.result = {"acquired": 1}
        elif "max(local_batch_id)" in normalized or "last_local_batch_id" in normalized:
            self.result = {"max_id": self.persisted_max_id}
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
    def __init__(self, *, persisted_max_id: int = 0) -> None:
        self.cursor_obj = _Cursor(persisted_max_id=persisted_max_id)

    def cursor(self) -> _Cursor:
        return self.cursor_obj

    def close(self) -> None:
        return None


def test_provider_uses_high_namespace_after_legacy_batch_ids(monkeypatch) -> None:
    """历史低位编号存在时，CDC 必须切换到独立高位命名空间。"""

    connection = _Connection()
    monkeypatch.setattr(
        "expert_index.cdc.batch_id.pymysql.connect",
        lambda **_kwargs: connection,
    )
    config = DatabaseConfig("127.0.0.1", 3306, "index", "", "expert_test", DatabaseRole.INDEX)

    provider = MySQLCdcLocalBatchIdProvider(config, destination="company_expert")
    assert provider() == CDC_LOCAL_BATCH_ID_BASE
    provider.close()


def test_provider_resumes_after_persisted_high_batch_id(monkeypatch) -> None:
    """Worker 重启后必须从已落库的高位 CDC 编号继续递增。"""

    persisted_max_id = CDC_LOCAL_BATCH_ID_BASE + 17
    connection = _Connection(persisted_max_id=persisted_max_id)
    monkeypatch.setattr(
        "expert_index.cdc.batch_id.pymysql.connect",
        lambda **_kwargs: connection,
    )
    config = DatabaseConfig("127.0.0.1", 3306, "index", "", "expert_test", DatabaseRole.INDEX)

    provider = MySQLCdcLocalBatchIdProvider(config, destination="company_expert")
    assert provider() == persisted_max_id + 1
    provider.close()


def test_provider_rejects_unsigned_bigint_overflow(monkeypatch) -> None:
    """CDC 高位序列耗尽时必须失败关闭，不能产生数据库不可写编号。"""

    connection = _Connection(persisted_max_id=MAX_UNSIGNED_BIGINT)
    monkeypatch.setattr(
        "expert_index.cdc.batch_id.pymysql.connect",
        lambda **_kwargs: connection,
    )
    config = DatabaseConfig("127.0.0.1", 3306, "index", "", "expert_test", DatabaseRole.INDEX)

    provider = MySQLCdcLocalBatchIdProvider(config, destination="company_expert")
    with pytest.raises(OverflowError, match="无符号 BIGINT"):
        provider()
    provider.close()
