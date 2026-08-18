"""专家索引流水线命令行入口。

常用顺序：先 ``init-db``，再用 ``build`` 重建单专家、用 ``build-batch`` 运行全量，
或用 ``resume-batch`` 从一个中断批次的可靠页游标继续。
两类构建命令都会以内存方法调用把 Chunk 交给下游向量服务，不再写入或读取
``expert_chunk``。命令输出只包含数量和状态，不打印 Chunk 正文。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import Settings
from .pipeline import ExpertIndexPipeline
from .repository import MySQLIndexRepository
from .test_fixture import ensure_complete_test_expert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DDL = PROJECT_ROOT / "sql" / "001_create_expert_index_tables.sql"


def build_parser() -> argparse.ArgumentParser:
    """定义数据库初始化、加工、导出和验收子命令。"""

    parser = argparse.ArgumentParser(description="专家索引 Chunk 数据加工")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-db", help="在派生索引库创建 Batch/Stage 表")
    init.add_argument("--ddl", type=Path, default=DEFAULT_DDL)

    build = sub.add_parser("build", help="完整重建一位专家")
    build.add_argument("--expert-id", type=int, required=True)
    build.add_argument(
        "--data-version",
        default=datetime.now().astimezone().strftime("%Y%m%d_01"),
    )
    build.add_argument("--batch-no")

    build_batch = sub.add_parser(
        "build-batch",
        help="按 experts.id 分页执行第一版全量 Chunk 加工",
    )
    build_batch.add_argument(
        "--data-version",
        default=datetime.now().astimezone().strftime("%Y%m%d_01"),
        help="本批次使用的业务数据版本",
    )
    build_batch.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="每页候选专家数，范围 1~1000，默认 200",
    )
    build_batch.add_argument(
        "--max-experts",
        type=int,
        help="本批次最多处理的候选专家数；不传时处理固定上界内全部专家",
    )
    build_batch.add_argument("--batch-no", help="可选业务批次号；不传时自动生成")

    resume_batch = sub.add_parser(
        "resume-batch",
        help="从中断 full 批次最后完整提交页之后继续加工",
    )
    resume_batch.add_argument(
        "--batch-id",
        type=int,
        required=True,
        help="要恢复的 expert_index_batch.id",
    )
    resume_batch.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="续跑每页候选专家数，范围 1~1000，默认 200",
    )

    # 该命令是显式测试数据写入入口，不会在 init-db/build 中自动运行。
    sub.add_parser(
        "seed-test-expert",
        help="在测试源库幂等创建主档及工作、项目、教育经历完整关联的虚构专家",
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
        result = _build_pipeline(settings, repository).build_one(
            args.expert_id,
            data_version=args.data_version,
            batch_no=args.batch_no,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "build-batch":
        # build_many 内部完成全局互斥、固定扫描上界、分页、专家失败隔离、批次明细和
        # 生命周期处理；CLI 只负责接收参数并输出不含专家正文的数量摘要。
        result = _build_pipeline(settings, repository).build_many(
            data_version=args.data_version,
            page_size=args.page_size,
            max_experts=args.max_experts,
            batch_no=args.batch_no,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "resume-batch":
        # data_version、扫描上界、批次号和累计计数均从原批次恢复，CLI 只允许调整
        # 本次读取页大小，防止续跑意外改变原任务范围。
        result = _build_pipeline(settings, repository).resume_many(
            args.batch_id,
            page_size=args.page_size,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "seed-test-expert":
        fixture = ensure_complete_test_expert(settings.source_db)
        print(json.dumps(asdict(fixture), ensure_ascii=False, sort_keys=True))


def _build_pipeline(
    settings: Settings,
    repository: MySQLIndexRepository,
) -> ExpertIndexPipeline:
    """组装只使用内存 Contract 连接下游的正式 Pipeline。"""

    from rag_vector_store import create_vector_ingestion_service

    from .vector_sync import ExpertVectorSynchronizer

    vector_service = create_vector_ingestion_service()
    return ExpertIndexPipeline(
        settings,
        repository=repository,
        vector_sync=ExpertVectorSynchronizer(vector_service),
    )


if __name__ == "__main__":
    main()
