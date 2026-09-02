"""Resume 独立命令行入口。

当前阶段只开放来源结构预检，不提供建表、业务读取、Chunk 构建或向量写入命令。
Expert CLI 不导入本模块，Resume 入口也不调用任何 Expert 运行命令。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import ResumeSettings
from .preflight import run_resume_source_preflight
from .schema_initializer import MySQLResumeSchemaInitializer


def build_parser() -> argparse.ArgumentParser:
    """创建 Resume 来源预检和独立 Index 初始化命令。"""

    parser = argparse.ArgumentParser(description="Resume 数据处理独立入口")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "preflight-source",
        help="只读检查四张 Resume 来源表和 Expert 隔离边界",
    )
    subcommands.add_parser(
        "init-db",
        help="在 RESUME_INDEX_DB_* 创建 Resume 控制表",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """执行 Resume 命令并输出不含凭据或正文的 JSON 摘要。

    :param argv: 可选命令参数；不传时由 ``argparse`` 读取进程参数。
    """

    args = build_parser().parse_args(argv)
    if args.command == "preflight-source":
        settings = ResumeSettings.from_env()
        result = run_resume_source_preflight(settings)
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "init-db":
        settings = ResumeSettings.from_env()
        result = MySQLResumeSchemaInitializer(settings).initialize_schema()
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
