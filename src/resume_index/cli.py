"""Resume 独立命令行入口。

所有业务命令只组装 Resume Source、Resume Index 状态库和 Resume Milvus Collection。
Expert CLI 不导入本模块，Resume 入口也不调用任何 Expert 运行命令。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from .config import ResumeSettings
from .preflight import run_resume_source_preflight
from .schema_initializer import MySQLResumeSchemaInitializer

if TYPE_CHECKING:
    from .runtime import ResumeRuntime


def build_parser() -> argparse.ArgumentParser:
    """创建 Resume 预检、初始化、单人物和全量批次命令。"""

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

    build = subcommands.add_parser(
        "build",
        help="按 resume_id 完整重建一位人物",
    )
    build.add_argument("--resume-id", type=int, required=True)
    build.add_argument("--data-version", required=True)
    build.add_argument("--batch-no")

    build_batch = subcommands.add_parser(
        "build-batch",
        help="按 resume_profile.id 固定范围执行全量构建",
    )
    build_batch.add_argument("--data-version", required=True)
    build_batch.add_argument("--page-size", type=int, default=200)
    build_batch.add_argument("--max-resumes", type=int)
    build_batch.add_argument("--batch-no")

    resume_batch = subcommands.add_parser(
        "resume-batch",
        help="从原全量批次最后可靠物理游标继续执行",
    )
    resume_batch.add_argument("--batch-id", type=int, required=True)
    resume_batch.add_argument("--page-size", type=int, default=200)

    retry_batch = subcommands.add_parser(
        "retry-batch",
        help="只重试 partial 全量批次中的失败 Resume 人物",
    )
    retry_batch.add_argument("--batch-id", type=int, required=True)
    retry_batch.add_argument("--page-size", type=int, default=200)
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
    elif args.command == "build":
        settings = ResumeSettings.from_env()
        result = _build_runtime(settings).pipeline.build_single_resume(
            args.resume_id,
            data_version=args.data_version,
            batch_no=args.batch_no,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "build-batch":
        settings = ResumeSettings.from_env()
        result = _build_runtime(settings).pipeline.build_resume_batch(
            data_version=args.data_version,
            page_size=args.page_size,
            max_resumes=args.max_resumes,
            batch_no=args.batch_no,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "resume-batch":
        settings = ResumeSettings.from_env()
        result = _build_runtime(settings).pipeline.resume_batch(
            args.batch_id,
            page_size=args.page_size,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "retry-batch":
        settings = ResumeSettings.from_env()
        result = _build_runtime(settings).pipeline.retry_failed_resumes(
            args.batch_id,
            page_size=args.page_size,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))


def _build_runtime(settings: ResumeSettings) -> ResumeRuntime:
    """延迟导入并组装正式运行时，预检/初始化命令不创建 Milvus 客户端。"""

    from .runtime import build_resume_runtime

    return build_resume_runtime(settings)


if __name__ == "__main__":
    main()
