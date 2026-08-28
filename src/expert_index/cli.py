"""专家索引流水线命令行入口。

常用顺序：先 ``init-db``，再用 ``build`` 重建单专家、用 ``build-batch`` 运行全量、
用 ``resume-batch`` 继续中断批次，或用 ``retry-batch`` 重试 partial 批次失败专家。
两类构建命令都会以内存方法调用把 Chunk 交给下游向量服务，不再写入或读取
``expert_chunk``。CDC 入口额外要求真实协议基线和 ACK 测试已配置；命令输出只包含数量
和状态，不打印 Chunk 正文、CDC Payload 或凭据。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import Settings
from .incremental_runtime import build_incremental_runtime
from .incremental_worker import IncrementalWorker, IncrementalWorkerConfig
from .pipeline import ExpertIndexPipeline
from .schema_initializer import MySQLSchemaInitializer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DDL = PROJECT_ROOT / "sql" / "001_create_expert_index_tables.sql"


def build_parser() -> argparse.ArgumentParser:
    """定义数据库初始化、加工、导出和验收子命令。"""

    parser = argparse.ArgumentParser(description="专家索引 Chunk 数据加工")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-db", help="在派生索引库创建批次状态表")
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

    retry_batch = sub.add_parser(
        "retry-batch",
        help="只重试 partial full 批次中的失败专家",
    )
    retry_batch.add_argument(
        "--batch-id",
        type=int,
        required=True,
        help="要重试的 expert_index_batch.id",
    )
    retry_batch.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="每页失败专家数，范围 1~1000，默认 200",
    )

    incremental_once = sub.add_parser(
        "incremental-once",
        help="执行一轮普通 Poll 增量：扫描、处理任务并提交安全 Checkpoint",
    )
    incremental_once.add_argument("--worker-id", required=True)
    incremental_once.add_argument("--cycle-id", required=True)
    incremental_once.add_argument("--max-attempts", type=int, required=True)
    incremental_once.add_argument("--retry-delay-seconds", type=int, required=True)
    incremental_once.add_argument("--task-lease-seconds", type=int, required=True)

    incremental_worker = sub.add_parser(
        "incremental-worker",
        help="常驻执行普通 Poll 增量；按间隔扫描四张来源表",
    )
    incremental_worker.add_argument("--worker-id", required=True)
    incremental_worker.add_argument("--interval-seconds", type=float, default=60.0)
    incremental_worker.add_argument("--error-sleep-seconds", type=float, default=30.0)
    incremental_worker.add_argument("--max-attempts", type=int, required=True)
    incremental_worker.add_argument("--retry-delay-seconds", type=int, required=True)
    incremental_worker.add_argument("--task-lease-seconds", type=int, required=True)
    incremental_worker.add_argument("--max-cycles", type=int)

    cdc_worker = sub.add_parser(
        "cdc-worker",
        help="常驻消费真实 Canal batch；业务成功后提交 Checkpoint 并 ACK",
    )
    cdc_worker.add_argument("--worker-id", required=True)
    cdc_worker.add_argument("--batch-size", type=int, default=100)
    cdc_worker.add_argument("--idle-sleep-seconds", type=float, default=5.0)
    cdc_worker.add_argument("--error-sleep-seconds", type=float, default=10.0)
    cdc_worker.add_argument("--max-attempts", type=int, required=True)
    cdc_worker.add_argument("--retry-delay-seconds", type=int, required=True)
    cdc_worker.add_argument("--task-lease-seconds", type=int, required=True)
    cdc_worker.add_argument("--max-cycles", type=int)

    return parser


def main(argv: list[str] | None = None) -> None:
    """执行命令并打印机器可读 JSON 摘要。"""

    args = build_parser().parse_args(argv)
    if args.command == "cdc-worker":
        # canal-python 0.4 的 protobuf 生成代码需要 Python 实现；必须在 SDK 导入前设置。
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    settings = Settings.from_env()
    # 所有正式命令都会创建或更新本地状态；在加载下游组件或连接 MySQL 前先验证边界。
    settings.validate_index_write_boundary()
    if args.command == "init-db":
        MySQLSchemaInitializer(settings.index_db).initialize_schema(args.ddl)
        print(json.dumps({"status": "ok", "schema": settings.index_db.database}, ensure_ascii=False))
    elif args.command == "build":
        result = _build_pipeline(settings).build_single_expert(
            args.expert_id,
            data_version=args.data_version,
            batch_no=args.batch_no,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "build-batch":
        # build_expert_batch 内部完成全局互斥、固定扫描上界、分页、专家失败隔离、
        # 批次明细和生命周期处理；CLI 只负责接收参数并输出不含专家正文的数量摘要。
        result = _build_pipeline(settings).build_expert_batch(
            data_version=args.data_version,
            page_size=args.page_size,
            max_experts=args.max_experts,
            batch_no=args.batch_no,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "resume-batch":
        # data_version、扫描上界、批次号和累计计数均从原批次恢复，CLI 只允许调整
        # 本次读取页大小，防止续跑意外改变原任务范围。
        result = _build_pipeline(settings).resume_expert_batch(
            args.batch_id,
            page_size=args.page_size,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "retry-batch":
        result = _build_pipeline(settings).retry_failed_batch_experts(
            args.batch_id,
            page_size=args.page_size,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    elif args.command == "incremental-once":
        result = build_incremental_runtime(
            settings,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            task_lease_seconds=args.task_lease_seconds,
        ).run_once(args.worker_id, cycle_id=args.cycle_id)
        print(json.dumps(result.to_summary(), ensure_ascii=False, sort_keys=True))
    elif args.command == "incremental-worker":
        runtime = build_incremental_runtime(
            settings,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            task_lease_seconds=args.task_lease_seconds,
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
    elif args.command == "cdc-worker":
        result = _run_cdc_worker(settings, args)
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))


def _build_pipeline(settings: Settings) -> ExpertIndexPipeline:
    """组装只使用内存 Contract 连接下游的正式 Pipeline。"""

    from rag_vector_store import create_vector_ingestion_service

    from .vector_sync import ExpertVectorSynchronizer

    vector_service = create_vector_ingestion_service()
    return ExpertIndexPipeline(
        settings,
        vector_sync=ExpertVectorSynchronizer(vector_service),
    )


def _run_cdc_worker(settings: Settings, args: argparse.Namespace):
    """组装并运行真实 CDC Worker，退出时释放 destination 单实例锁。"""

    from .cdc import (
        CdcWorkerConfig,
        MySQLCdcLocalBatchIdProvider,
        build_cdc_event_id,
        build_cdc_protocol_baseline,
        build_cdc_runtime,
        build_real_canal_client,
        decode_canal_message,
        run_cdc_worker,
    )

    config = settings.cdc_runtime
    baseline = build_cdc_protocol_baseline(settings)
    # 在构造 Vector/Embedding/Canal Client 前先执行协议门禁，避免半启动服务。
    baseline.require_production_ready()
    client = build_real_canal_client(
        host=config.host,
        port=config.port,
        destination=config.destination,
        client_id=config.client_id,
        subscription=config.subscription,
        username=config.username,
        password=config.password,
        decoder=decode_canal_message,
    )
    runtime = build_cdc_runtime(
        settings,
        client=client,
        protocol_baseline=baseline,
        identity_provider=build_cdc_event_id,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        task_lease_seconds=args.task_lease_seconds,
    )
    batch_ids = MySQLCdcLocalBatchIdProvider(
        settings.index_db,
        destination=config.destination,
    )
    try:
        return run_cdc_worker(
            runtime,
            batch_ids,
            config=CdcWorkerConfig(
                batch_size=args.batch_size,
                idle_sleep_seconds=args.idle_sleep_seconds,
                error_sleep_seconds=args.error_sleep_seconds,
                max_cycles=args.max_cycles,
            ),
        )
    finally:
        batch_ids.close()


if __name__ == "__main__":
    main()
