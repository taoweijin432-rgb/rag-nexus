"""MySQL 专家数据读取层。

本模块只负责连接数据库和分批读取原始记录，不负责业务抽取。这样测试 chunk
逻辑时不需要连接数据库，未来也可以把 MySQL 替换成 JSON、API 或消息队列。
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymysql
from pymysql.connections import Connection

ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(env_path: Path | None = None) -> None:
    """读取项目根目录的简单 ``.env`` 文件。

    支持 ``KEY=value``、``export KEY=value``、首尾引号、空行和注释。已经由
    shell 或部署平台设置的变量优先级更高，不会被本地文件覆盖。
    """

    path = env_path or Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"{path} 第 {line_number} 行格式错误，应为 KEY=value")

        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if not ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"{path} 第 {line_number} 行环境变量名无效：{name}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def create_connection() -> Connection:
    """根据环境变量创建 MySQL 连接，密码缺失时给出明确错误。"""

    load_env_file()
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise RuntimeError("缺少 DB_PASSWORD，请在 .env 或 shell 环境中配置数据库密码")

    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=password,
        database=os.getenv("DB_NAME", "expert_demo"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def get_persons(limit: int = 100, after_id: int = 0) -> list[dict[str, Any]]:
    """读取一批未删除专家，使用主键游标分页而非 OFFSET 分页。"""

    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    # 只取 chunk 实际需要的字段，避免联系方式和合同信息进入处理链路。
    sql = """
        SELECT
            id, serial_number, name, company, job_name, job_status,
            country, province, city, background, work_experience,
            project_experience, personal_intro, self_appraisal, certifications,
            source_type, updated_at
        FROM experts
        WHERE id > %s AND deleted_at IS NULL
        ORDER BY id ASC
        LIMIT %s
    """

    connection = create_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (after_id, limit))
            return list(cursor.fetchall())
    finally:
        connection.close()


def iter_person_batches(
    batch_size: int = 100,
    after_id: int = 0,
    max_records: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """以批次形式迭代数据，避免全量专家一次性占满内存。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records 必须大于 0 或为 None")

    current_id, emitted = after_id, 0
    while True:
        current_limit = batch_size
        if max_records is not None:
            current_limit = min(current_limit, max_records - emitted)
            if current_limit <= 0:
                return

        persons = get_persons(limit=current_limit, after_id=current_id)
        if not persons:
            return
        yield persons

        emitted += len(persons)
        current_id = int(persons[-1]["id"])
        if len(persons) < current_limit:
            return
