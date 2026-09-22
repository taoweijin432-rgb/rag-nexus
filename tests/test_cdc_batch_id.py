"""验证 CDC 使用全链路共享 Batch ID 序列。"""

from __future__ import annotations

from typing import Self

import pytest

from expert_index.cdc.batch_id import MySQLCdcLocalBatchIdProvider
from expert_index.config import DatabaseConfig, DatabaseRole
from expert_index.constants import GLOBAL_BATCH_ID_SEQUENCE_NAME, MAX_UNSIGNED_BIGINT
from expert_index.repositories.batch_id import BatchIdAllocationError


class _Cursor:
    """模拟命名锁和统一序列所需的最小 Cursor。"""

    def __init__(
        self,
        *,
        current_value: int = 70_193,
        sequence_available: bool = True,
    ) -> None:
        self.result: dict[str, int] = {}
        self.current_value = current_value
        self.sequence_available = sequence_available
        self.lastrowid = 0
        self.rowcount = 0
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        self.executed.append((" ".join(sql.split()), params))
        normalized = " ".join(sql.split()).lower()
        if "get_lock" in normalized:
            self.result = {"acquired": 1}
            return
        if "expert_index_batch_id_sequence" in normalized:
            if self.sequence_available and self.current_value < MAX_UNSIGNED_BIGINT:
                self.current_value += 1
                self.lastrowid = self.current_value
                self.rowcount = 1
            else:
                self.lastrowid = 0
                self.rowcount = 0
            return
        if "release_lock" in normalized:
            self.result = {"released": 1}

    def fetchone(self) -> dict[str, int]:
        return self.result


class _Connection:
    """记录 CDC 分配器的事务边界。"""

    def __init__(
        self,
        *,
        current_value: int = 70_193,
        sequence_available: bool = True,
    ) -> None:
        self.cursor_obj = _Cursor(
            current_value=current_value,
            sequence_available=sequence_available,
        )
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> _Cursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def _config() -> DatabaseConfig:
    """返回只供测试使用的 Index 数据库配置。"""

    return DatabaseConfig(
        "127.0.0.1",
        3306,
        "index",
        "",
        "expert_test",
        DatabaseRole.INDEX,
    )


def test_provider_allocates_consecutive_ids_from_shared_sequence(monkeypatch) -> None:
    """CDC 应与其他入口一样从统一序列原子取得批次号。"""

    connection = _Connection()
    monkeypatch.setattr(
        "expert_index.cdc.batch_id.pymysql.connect",
        lambda **_kwargs: connection,
    )
    provider = MySQLCdcLocalBatchIdProvider(_config(), destination="company_expert")

    assert provider() == 70_194
    assert provider() == 70_195
    assert connection.commit_count == 2
    sequence_calls = [
        call
        for call in connection.cursor_obj.executed
        if "expert_index_batch_id_sequence" in call[0]
    ]
    assert sequence_calls[0][1] == (
        GLOBAL_BATCH_ID_SEQUENCE_NAME,
        MAX_UNSIGNED_BIGINT,
    )

    provider.close()
    assert connection.closed is True


def test_provider_rolls_back_when_shared_sequence_is_missing(monkeypatch) -> None:
    """统一序列未迁移时必须停止 CDC，不能回退到自行推算编号。"""

    connection = _Connection(sequence_available=False)
    monkeypatch.setattr(
        "expert_index.cdc.batch_id.pymysql.connect",
        lambda **_kwargs: connection,
    )
    provider = MySQLCdcLocalBatchIdProvider(_config(), destination="company_expert")

    with pytest.raises(BatchIdAllocationError, match="未初始化或已耗尽"):
        provider()

    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    provider.close()


def test_provider_rejects_unsigned_bigint_overflow(monkeypatch) -> None:
    """共享序列耗尽时必须失败，不能产生数据库不可写编号。"""

    connection = _Connection(current_value=MAX_UNSIGNED_BIGINT)
    monkeypatch.setattr(
        "expert_index.cdc.batch_id.pymysql.connect",
        lambda **_kwargs: connection,
    )
    provider = MySQLCdcLocalBatchIdProvider(_config(), destination="company_expert")

    with pytest.raises(BatchIdAllocationError, match="未初始化或已耗尽"):
        provider()

    assert connection.rollback_count == 1
    provider.close()
