"""安全 Checkpoint 数据库行的唯一映射实现，不访问数据库。"""

from __future__ import annotations

from typing import Any

from ..models import SourceCheckpoint


def source_checkpoint_from_row(
    row: dict[str, Any],
    *,
    expected_source_table: str,
) -> SourceCheckpoint:
    """把已校验表名的数据库行转换为安全 Checkpoint 模型。"""

    if row.get("source_table") != expected_source_table:
        raise RuntimeError("Checkpoint 查询结果与请求来源表不一致")
    return SourceCheckpoint(
        source_table=expected_source_table,
        last_updated_at=row["last_updated_at"],
        last_id=int(row["last_id"]),
        last_batch_id=(
            int(row["last_batch_id"])
            if row.get("last_batch_id") is not None
            else None
        ),
        last_success_time=row.get("last_success_time"),
        lock_version=int(row["lock_version"]),
        lease_owner=row.get("lease_owner"),
        lease_until=row.get("lease_until"),
    )
