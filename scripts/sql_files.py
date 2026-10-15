"""隔离库验证脚本共用的 SQL 文件读取，不执行 SQL。"""

from pathlib import Path


def load_sql_statements(path: Path) -> list[str]:
    """返回 SQL 文件内按顺序排列的可执行语句。"""

    statements: list[str] = []
    for raw_statement in path.read_text(encoding="utf-8").split(";"):
        statement = "\n".join(
            line
            for line in raw_statement.splitlines()
            if not line.lstrip().startswith("--")
        ).strip()
        if statement:
            statements.append(statement)
    return statements
