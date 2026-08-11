"""专家索引流水线命令行入口。

常用顺序：先 ``init-db``，再 ``build --expert-id ...``，最后 ``export`` 和
``inspect`` 验收。命令输出只包含数量、状态和文件路径，不打印 Chunk 正文。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import Settings
from .exporter import export_jsonl
from .pipeline import ExpertIndexPipeline
from .repository import MySQLIndexRepository
from .test_fixture import ensure_complete_test_expert


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DDL = PROJECT_ROOT / "sql" / "001_create_expert_index_tables.sql"


def build_parser() -> argparse.ArgumentParser:
    """定义四个边界明确的子命令。"""

    parser = argparse.ArgumentParser(description="专家索引 Chunk 数据加工")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-db", help="在派生索引库创建 Batch/Stage/Chunk 表")
    init.add_argument("--ddl", type=Path, default=DEFAULT_DDL)

    build = sub.add_parser("build", help="完整重建一位专家")
    build.add_argument("--expert-id", type=int, required=True)
    build.add_argument("--data-version", default=datetime.now().strftime("%Y%m%d_01"))
    build.add_argument("--batch-no")

    export = sub.add_parser("export", help="导出符合向量化输入契约的 JSONL")
    export.add_argument("--expert-id", type=int, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--eligible-only",
        action="store_true",
        help="只导出 source_evidence/validated 的 A、B 级 active Chunk",
    )

    inspect = sub.add_parser("inspect", help="查看不含正文的 Chunk 验收统计")
    inspect.add_argument("--expert-id", type=int, required=True)

    # 该命令是显式测试数据写入入口，不会在 init-db/build 中自动运行。
    sub.add_parser(
        "seed-test-expert",
        help="在测试源库幂等创建主档、任职、项目完整关联的虚构专家",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """执行命令并打印机器可读 JSON 摘要。"""

    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    repository = MySQLIndexRepository(settings.index_db)
    if args.command == "init-db":
        repository.initialize_schema(args.ddl)
        print(json.dumps({"status": "ok", "schema": settings.index_db.database}, ensure_ascii=False))
    elif args.command == "build":
        result = ExpertIndexPipeline(settings, repository=repository).build_one(
            args.expert_id,
            data_version=args.data_version,
            batch_no=args.batch_no,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "export":
        rows = repository.fetch_active_chunks(args.expert_id, eligible_only=args.eligible_only)
        count = export_jsonl(rows, args.output, settings.timezone)
        print(json.dumps({"expert_id": args.expert_id, "exported": count, "output": str(args.output)}, ensure_ascii=False))
    elif args.command == "inspect":
        print(json.dumps(repository.summary(args.expert_id), ensure_ascii=False, default=str, sort_keys=True))
    elif args.command == "seed-test-expert":
        fixture = ensure_complete_test_expert(settings.source_db)
        print(json.dumps(asdict(fixture), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
