"""Resume 独立命令行入口。

所有业务命令只组装 Resume Source、Resume Index 状态库和 Resume Milvus Collection。
Expert CLI 不导入本模块，Resume 入口也不调用任何 Expert 运行命令。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from expert_index.incremental_worker import IncrementalWorker, IncrementalWorkerConfig

from .checkpoint_initializer import initialize_resume_checkpoints
from .config import ResumeSettings
from .preflight import run_resume_source_preflight
from .schema_initializer import MySQLResumeSchemaInitializer

if TYPE_CHECKING:
    from .incremental_runtime import ResumeIncrementalRuntime
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
    init_checkpoints = subcommands.add_parser(
        "init-checkpoints",
        help="显式初始化 Resume Poll 四表安全水位，不覆盖已有 Checkpoint",
    )
    init_checkpoints.add_argument(
        "--position",
        choices=("beginning", "current"),
        required=True,
        help="beginning=安全补扫全部历史；current=仅用于暂停来源写入的受控全量切换",
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
        help="按 resume.id 固定范围执行全量构建",
    )
    build_batch.add_argument("--data-version", required=True)
    build_batch.add_argument("--page-size", type=int, default=200)
    build_batch.add_argument("--max-resumes", type=int)
    build_batch.add_argument("--batch-no")
    build_batch.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="页内来源读取和 Chunk 准备线程数，范围 1~16，默认 1",
    )

    resume_batch = subcommands.add_parser(
        "resume-batch",
        help="从原全量批次最后可靠物理游标继续执行",
    )
    resume_batch.add_argument("--batch-id", type=int, required=True)
    resume_batch.add_argument("--page-size", type=int, default=200)
    resume_batch.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="续跑时的页内准备线程数，不改变原批次游标和范围",
    )

    retry_batch = subcommands.add_parser(
        "retry-batch",
        help="只重试 partial 全量批次中的失败 Resume 人物",
    )
    retry_batch.add_argument("--batch-id", type=int, required=True)
    retry_batch.add_argument("--page-size", type=int, default=200)

    incremental_once = subcommands.add_parser(
        "incremental-once",
        help="执行一轮 Resume Poll：扫描、处理并提交安全 Checkpoint",
    )
    incremental_once.add_argument("--worker-id", required=True)
    incremental_once.add_argument("--cycle-id", required=True)
    incremental_once.add_argument("--max-attempts", type=int, required=True)
    incremental_once.add_argument("--retry-delay-seconds", type=int, required=True)
    incremental_once.add_argument("--task-lease-seconds", type=int, required=True)
    incremental_once.add_argument(
        "--worker-count",
        type=int,
        choices=range(1, 5),
        default=1,
        help="不同来源表之间的并发度，范围 1~4；默认 1",
    )

    incremental_worker = subcommands.add_parser(
        "incremental-worker",
        help="常驻执行 Resume Poll；不包含 CDC/Canal",
    )
    incremental_worker.add_argument("--worker-id", required=True)
    incremental_worker.add_argument("--interval-seconds", type=float, default=60.0)
    incremental_worker.add_argument("--error-sleep-seconds", type=float, default=30.0)
    incremental_worker.add_argument("--max-attempts", type=int, required=True)
    incremental_worker.add_argument("--retry-delay-seconds", type=int, required=True)
    incremental_worker.add_argument("--task-lease-seconds", type=int, required=True)
    incremental_worker.add_argument(
        "--worker-count",
        type=int,
        choices=range(1, 5),
        default=1,
        help="不同来源表之间的并发度，范围 1~4；默认 1",
    )
    incremental_worker.add_argument("--max-cycles", type=int)
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
    elif args.command == "init-checkpoints":
        settings = ResumeSettings.from_env()
        result = initialize_resume_checkpoints(settings, position=args.position)
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
            worker_count=args.worker_count,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "resume-batch":
        settings = ResumeSettings.from_env()
        result = _build_runtime(settings).pipeline.resume_batch(
            args.batch_id,
            page_size=args.page_size,
            worker_count=args.worker_count,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "retry-batch":
        settings = ResumeSettings.from_env()
        result = _build_runtime(settings).pipeline.retry_failed_resumes(
            args.batch_id,
            page_size=args.page_size,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "incremental-once":
        settings = ResumeSettings.from_env()
        result = _build_poll_runtime(
            settings,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            task_lease_seconds=args.task_lease_seconds,
            source_worker_count=args.worker_count,
        ).run_once(args.worker_id, cycle_id=args.cycle_id)
        print(json.dumps(result.to_summary(), ensure_ascii=False, sort_keys=True))
    elif args.command == "incremental-worker":
        settings = ResumeSettings.from_env()
        runtime = _build_poll_runtime(
            settings,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            task_lease_seconds=args.task_lease_seconds,
            source_worker_count=args.worker_count,
        )
        result = IncrementalWorker(
            runtime,
            args.worker_id,
            config=IncrementalWorkerConfig(
                interval_seconds=args.interval_seconds,
                error_sleep_seconds=args.error_sleep_seconds,
                max_cycles=args.max_cycles,
            ),
        ).run()
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))


def _build_runtime(settings: ResumeSettings) -> ResumeRuntime:
    """延迟导入并组装正式运行时，预检/初始化命令不创建 Milvus 客户端。"""

    from .runtime import build_resume_runtime

    return build_resume_runtime(settings)


def _build_poll_runtime(
    settings: ResumeSettings,
    *,
    max_attempts: int,
    retry_delay_seconds: int,
    task_lease_seconds: int,
    source_worker_count: int,
) -> ResumeIncrementalRuntime:
    """延迟组装 Resume Poll，并显式传入来源表级并发度。"""

    from .incremental_runtime import build_resume_incremental_runtime

    return build_resume_incremental_runtime(
        settings,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        task_lease_seconds=task_lease_seconds,
        source_worker_count=source_worker_count,
    )


if __name__ == "__main__":
    main()
