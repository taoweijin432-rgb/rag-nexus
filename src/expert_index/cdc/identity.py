"""CDC 事件稳定身份生成。

事件身份只使用 Canal Header 和行序号，不包含业务正文、Before/After Image 或 PII。
同一 Binlog 事件在重投、重连和进程重启后会得到相同的 ``cdc_event_id``。
"""

from __future__ import annotations

import hashlib
import json

from .parser import CanalChangeEvent
from .protocol import CANAL_DESTINATION


def build_cdc_event_id(event: CanalChangeEvent) -> str:
    """按已确认的 destination、server、Binlog 位点和行序号生成 SHA-256 身份。"""

    if not isinstance(event, CanalChangeEvent):
        raise TypeError("event 必须是 CanalChangeEvent")
    if event.server_id is None or event.server_id < 0:
        raise ValueError("CDC 事件缺少有效 server_id")
    identity = {
        "destination": CANAL_DESTINATION,
        "server_id": event.server_id,
        "logfile_name": event.logfile_name,
        "logfile_offset": event.logfile_offset,
        "entry_index": event.entry_index,
        "row_index": event.row_index,
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = ["build_cdc_event_id"]
