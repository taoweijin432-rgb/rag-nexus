#!/usr/bin/env python3
"""V15 增量向量化持久化层迁移执行器（仅限本地 MySQL）。

用法::

    # 建表模式（默认）：目标表必须尚不存在
    uv run python scripts/apply_v15_persistence_migration.py [sql_file]

    # 升级模式：目标表必须已存在；按 information_schema 逐对象跳过已应用项，
    # 重复运行安全。用于把旧版本 011 建出的库补齐到当前终态。
    uv run python scripts/apply_v15_persistence_migration.py --upgrade [sql_file]

``sql_file`` 缺省：建表模式为 ``sql/011_add_v15_vectorization_persistence.sql``；
升级模式为 ``sql/011a_upgrade_v15_work_item_version_fields.sql``。

安全门（任一不满足立即终止，不执行任何 DDL）：

1. 连接目标必须是 127.0.0.1/localhost，禁止远程库与 SSH Tunnel 目标；
2. ``@@hostname`` 必须是本机 hostname，排除经隧道落到远程库的情况；
3. ``DATABASE()`` 必须等于配置的 index_db 名，禁止在其他数据库操作；
4. ``@@read_only`` 必须为 0；
5. 建表模式：每张目标表必须尚不存在；升级模式：每张目标表必须已存在；
6. 建表模式禁止 DROP / ALTER / TRUNCATE 等语句；升级模式仅允许
   ``ALTER TABLE ... ADD COLUMN/KEY/CONSTRAINT``，其余语句类型
   （含 DROP COLUMN / MODIFY / CHANGE 等）一律拒绝。

连接配置全部来自 Settings/env（``INDEX_DB_*`` / ``DB_PASSWORD``），
脚本内不保存任何凭据。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pymysql

from expert_index.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQL = PROJECT_ROOT / "sql" / "011_add_v15_vectorization_persistence.sql"
DEFAULT_UPGRADE_SQL = PROJECT_ROOT / "sql" / "011a_upgrade_v15_work_item_version_fields.sql"

EXPECTED_HOSTS = {"127.0.0.1", "localhost"}
# 本机 hostname 小写形式；SSH Tunnel 到远程库时 @@hostname 会暴露远端主机名。
EXPECTED_HOSTNAME = "yukademac-mini.local"

# 硬禁令：两种模式下都拒绝的语句类型。
_FORBIDDEN_STATEMENTS = re.compile(
    r"^\s*(DROP|TRUNCATE|RENAME|GRANT|CREATE\s+(?:DATABASE|USER))\b",
    re.IGNORECASE,
)

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`?(\w+)`?", re.IGNORECASE
)
_ALTER_TARGET_RE = re.compile(r"ALTER\s+TABLE\s+`?(\w+)`?", re.IGNORECASE)
_ALTER_ADD_COLUMN_RE = re.compile(
    r"^ALTER\s+TABLE\s+`?(\w+)`?\s+ADD\s+COLUMN\s+`?(\w+)`?", re.IGNORECASE
)
_ALTER_ADD_INDEX_RE = re.compile(
    r"^ALTER\s+TABLE\s+`?(\w+)`?\s+ADD\s+(?:UNIQUE\s+)?(?:KEY|INDEX)\s+`?(\w+)`?",
    re.IGNORECASE,
)
_ALTER_ADD_CONSTRAINT_RE = re.compile(
    r"^ALTER\s+TABLE\s+`?(\w+)`?\s+ADD\s+CONSTRAINT\s+`?(\w+)`?", re.IGNORECASE
)


def _extract_create_tables(sql_text: str) -> list[str]:
    """从迁移文本提取 CREATE TABLE 的目标表名。"""

    return _CREATE_TABLE_RE.findall(sql_text)


def _strip_comment_lines(statement: str) -> str:
    """去掉以 ``--`` 开头的注释行，返回有效 SQL 文本。"""

    return "\n".join(
        line for line in statement.splitlines() if not line.strip().startswith("--")
    ).strip()


def _classify_alter_add(statement: str) -> tuple[str, str, str] | None:
    """识别 ``ALTER TABLE <t> ADD COLUMN/KEY/CONSTRAINT <name>`` 语句。

    返回 (table, 对象类型, 对象名)；不匹配白名单形态时返回 None。
    """

    for regex, kind in (
        (_ALTER_ADD_COLUMN_RE, "column"),
        (_ALTER_ADD_INDEX_RE, "index"),
        (_ALTER_ADD_CONSTRAINT_RE, "constraint"),
    ):
        match = regex.match(statement)
        if match:
            return match.group(1), kind, match.group(2)
    return None


def _object_exists(cursor: Any, table: str, kind: str, name: str) -> bool:
    """按 information_schema 判断列 / 索引 / CHECK 约束是否已存在。"""

    if kind == "column":
        cursor.execute(
            "SELECT COUNT(*) AS n FROM information_schema.columns "
            "WHERE table_schema=DATABASE() AND table_name=%s AND column_name=%s",
            (table, name),
        )
    elif kind == "index":
        cursor.execute(
            "SELECT COUNT(DISTINCT index_name) AS n FROM information_schema.statistics "
            "WHERE table_schema=DATABASE() AND table_name=%s AND index_name=%s",
            (table, name),
        )
    else:  # constraint（CHECK）
        cursor.execute(
            "SELECT COUNT(*) AS n FROM information_schema.table_constraints "
            "WHERE table_schema=DATABASE() AND table_name=%s "
            "AND constraint_type='CHECK' AND constraint_name=%s",
            (table, name),
        )
    return int(cursor.fetchone()["n"]) > 0


# 升级模式允许的目标表白名单：防止自定义 SQL 文件对任意本地表执行 ALTER。
_UPGRADE_ALLOWED_TABLES = frozenset({"v15_vectorization_work_item"})


def _prepare_upgrade_statements(sql_text: str) -> tuple[list[str], list[str]]:
    """解析并校验升级 SQL 文本，返回 (statements, 目标表列表)。

    校验失败抛 ``ValueError``，由调用方转为 [ABORT] 输出：

    - 含硬禁令语句（DROP/TRUNCATE/RENAME/GRANT/CREATE DATABASE|USER）；
    - 存在白名单形态之外的语句（仅允许 ADD COLUMN/KEY/CONSTRAINT）；
    - 未找到任何 ALTER TABLE 语句；
    - 目标表超出白名单。
    """

    statements = [
        stripped
        for raw in sql_text.split(";")
        if (stripped := _strip_comment_lines(raw))
    ]
    for i, stmt in enumerate(statements, 1):
        head = " ".join(stmt.split())[:60]
        if _FORBIDDEN_STATEMENTS.search(stmt):
            raise ValueError(f"第 {i} 段含被禁止的语句类型: {head}")
        if _classify_alter_add(stmt) is None:
            raise ValueError(
                f"第 {i} 段不是允许的 ALTER TABLE ... ADD COLUMN/KEY/CONSTRAINT: "
                f"{head}"
            )
    targets = sorted({*_ALTER_TARGET_RE.findall(sql_text)})
    if not targets:
        raise ValueError("迁移文件中未找到 ALTER TABLE 语句")
    outside = sorted(set(targets) - _UPGRADE_ALLOWED_TABLES)
    if outside:
        raise ValueError(f"升级目标表不在白名单内: {', '.join(outside)}")
    return statements, targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V15 持久化层迁移执行器（仅限本地 MySQL）"
    )
    parser.add_argument("sql_file", nargs="?", default=None, help="迁移 SQL 文件路径")
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="升级模式：目标表必须已存在，按对象跳过已应用项（幂等）",
    )
    args = parser.parse_args(argv)

    default_path = DEFAULT_UPGRADE_SQL if args.upgrade else DEFAULT_SQL
    sql_path = Path(args.sql_file or default_path).resolve()
    if not sql_path.exists():
        print(f"[ABORT] 迁移文件不存在: {sql_path}")
        return 1

    db = Settings.from_env().index_db
    print(f"目标配置: host={db.host} port={db.port} database={db.database} user={db.user}")
    if db.host not in EXPECTED_HOSTS:
        print(f"[ABORT] 连接目标不是本地 MySQL（禁止远程库/SSH Tunnel）: {db.host}")
        return 1

    sql_text = sql_path.read_text(encoding="utf-8")

    alter_targets: list[str] = []
    tables: list[str] = []
    if args.upgrade:
        try:
            statements, alter_targets = _prepare_upgrade_statements(sql_text)
        except ValueError as exc:
            print(f"[ABORT] {exc}")
            return 1
    else:
        statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
        for i, stmt in enumerate(statements, 1):
            if _FORBIDDEN_STATEMENTS.search(stmt) or _ALTER_TARGET_RE.match(stmt):
                head = " ".join(stmt.split())[:60]
                print(f"[ABORT] 迁移包含被禁止的语句类型（第 {i} 段）: {head} ...")
                return 1
        tables = _extract_create_tables(sql_text)
        if not tables:
            print("[ABORT] 迁移文件中未找到 CREATE TABLE 语句")
            return 1

    conn = pymysql.connect(**db.as_kwargs(autocommit=True))
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT @@hostname AS hostname, @@port AS port, "
                "DATABASE() AS db, @@read_only AS read_only, @@version AS version"
            )
            info = cur.fetchone()
            print(
                "数据库安全检查: "
                f"hostname={info['hostname']} port={info['port']} "
                f"database={info['db']} read_only={info['read_only']} "
                f"version={info['version']}"
            )
            if str(info["hostname"]).lower() != EXPECTED_HOSTNAME:
                print(
                    f"[ABORT] @@hostname 不是本机（疑似远程库/SSH Tunnel）: "
                    f"{info['hostname']}"
                )
                return 1
            if info["read_only"] != 0:
                print("[ABORT] 数据库处于只读模式")
                return 1
            if info["db"] != db.database:
                print(f"[ABORT] DATABASE() 与配置不一致: {info['db']} != {db.database}")
                return 1

            if args.upgrade:
                # 升级模式：目标表必须已存在（先有 011 基础结构）。
                for table in alter_targets:
                    cur.execute(
                        "SELECT COUNT(*) AS n FROM information_schema.tables "
                        "WHERE table_schema = DATABASE() AND table_name = %s",
                        (table,),
                    )
                    if not cur.fetchone()["n"]:
                        print(
                            f"[ABORT] 表 {table} 不存在，请先应用基础迁移 011 "
                            "（不要用 --upgrade 建表）"
                        )
                        return 1

                print(
                    f"\n升级模式：检查 {len(statements)} 条 ALTER 语句"
                    f"（{sql_path.name}）..."
                )
                executed_count = 0
                for i, stmt in enumerate(statements, 1):
                    head = " ".join(stmt.split())[:60]
                    target = _classify_alter_add(stmt)
                    assert target is not None  # 前置白名单校验已保证
                    table, kind, name = target
                    if _object_exists(cur, table, kind, name):
                        print(f"  [{i}/{len(statements)}] 跳过（{kind} {name} 已存在）")
                        continue
                    cur.execute(stmt)
                    executed_count += 1
                    print(f"  [{i}/{len(statements)}] OK: {head} ...")
                print(
                    f"\n升级完成：执行 {executed_count} 条，跳过 "
                    f"{len(statements) - executed_count} 条。"
                )
            else:
                # 建表模式：目标表必须尚不存在。
                for table in tables:
                    cur.execute(
                        "SELECT COUNT(*) AS n FROM information_schema.tables "
                        "WHERE table_schema = DATABASE() AND table_name = %s",
                        (table,),
                    )
                    if cur.fetchone()["n"]:
                        print(f"[ABORT] 表 {table} 已存在，拒绝重复应用")
                        return 1

                print(f"\n开始执行 {len(statements)} 条 DDL 语句（{sql_path.name}）...")
                for i, stmt in enumerate(statements, 1):
                    head = " ".join(stmt.split())[:60]
                    cur.execute(stmt)
                    print(f"  [{i}/{len(statements)}] OK: {head} ...")

        conn.commit()
        print("\nDDL 全部执行成功。")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
