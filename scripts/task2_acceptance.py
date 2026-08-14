"""执行正式 Task 2 DEV 的 Export Contract 4.0.0 ingestion 真实验收。

正式链路（本脚本只编排已有正式组件，不复制业务逻辑）::

    MySQL expert_chunk
      → MySqlExpertChunkReader（只读固定列 SELECT）
      → ExpertChunkEmbeddingInput 4.0.0（严格 Contract）
      → ContractValidator / EmbeddingEligibilityValidator（valid / eligible / skipped）
      → 阿里云百炼 text-embedding-v4（1024 维）
      → ContractBatchIngestionService（批量 upsert）
      → ContractMilvusVectorRepository（v14 Contract Collection）

全部身份统一使用 ``vector_doc_id``（唯一 PK，auto_id=False），脚本不再依赖
``chunk_id`` / ``source_record_id`` / ``metadata_json`` 等已废弃字段。目标
Collection 必须已经存在且通过只读 Schema/Index 兼容性检查；脚本不会创建、修改、
删除 Collection，也不会清理历史数据。旧 v13 Collection 只用只读快照确认验收前后
行数不变；保护名单会直接拒绝任何指向旧 Collection 的写入。

示例::

    set -a; source .env; set +a
    uv run --no-sync python scripts/task2_acceptance.py \
        --collection expert_candidates_v14_contract_dev \
        --report reports/task2_acceptance.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn
from zoneinfo import ZoneInfo

from expert_contracts import ExpertChunkEmbeddingInput
from rag_vector_store import (
    EXPECTED_EMBEDDING_DIMENSION,
    CompanyCloudEmbeddingModel,
    ContractBatchIngestionResult,
    ContractBatchIngestionService,
    ContractEmbeddingWorker,
    ContractMilvusVectorRepository,
    ContractValidationError,
    ContractValidator,
    EmbeddingEligibilityValidator,
    MilvusSettings,
)
from rag_vector_store.config import MySQLSettings
from rag_vector_store.milvus.compatibility import datatype_name
from rag_vector_store.model_space import FORMAL_EMBEDDING_MODEL_SPACE_ID
from rag_vector_store.mysql import MySqlExpertChunkReader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_COLLECTION = "expert_candidates_v14_contract_dev"
_DEFAULT_V13_COLLECTION = "expert_candidates_v13_contract_dev"
_DEFAULT_REPORT = _PROJECT_ROOT / "reports" / "task2_acceptance.json"
_DEFAULT_EXPECTED_COUNT = 7
_DEFAULT_DATABASE_TIMEZONE = "Asia/Shanghai"
_MISSING_VECTOR_DOC_ID = "task2-acceptance:999641217:does-not-exist"
_PROTECTED_COLLECTIONS = frozenset(
    {
        "expert_candidates_v1",
        "resume_candidates_v1",
        "expert_candidates_v12_dev",
        "expert_candidates_v12_contract_dev",
        "expert_candidates_v13_contract_dev",
    }
)
# v14 Schema / 行中不得再出现的旧字段；v13 保留这些字段是历史事实，但新链路
# 不得写入或回查它们。source_field 不是废弃字段，它由 source_ext 完整承载。
_DEPRECATED_FIELD_NAMES = (
    "chunk_id",
    "source_item_key",
    "data_version",
    "dictionary_version",
    "project_role",
    "company_normalization_status",
    "metadata_json",
    "source_record_id",
    "token_count_estimate",
)
_STAGES = (
    "mysql_read",
    "collection_precondition",
    "contract_validation",
    "eligibility",
    "embedding",
    "milvus_upsert",
    "vector_doc_id_query",
    "expert_id_query",
    "missing_id_query",
    "no_deprecated_query",
    "v13_integrity",
)
_QUERY_FIELDS = [
    "vector_doc_id",
    "expert_id",
    "chunk_type",
    "chunk_text",
    "content_hash",
    "metadata_hash",
    "metadata_ext",
    "source_ext",
    "versions_ext",
    "embedding_model_alias",
    "embedding_model_version",
    "embedding_dim",
    "embedding",
    "is_active",
    "verification_status",
    "eligible_for_recommendation",
]


class AcceptanceFailure(RuntimeError):
    """验收阶段失败。"""


def main(argv: Sequence[str] | None = None) -> int:
    """执行验收并输出终端摘要及 JSON 报告。"""

    args = _parse_args(argv)
    report = _new_report(args)
    repository: ContractMilvusVectorRepository | None = None

    try:
        records = _read_mysql_records(args, report)
        settings = _load_settings(args.collection, report)
        repository = _verify_collection(settings, report)
        _snapshot_v13(repository.client, args.v13_collection, report)
        model, worker = _build_embedding_worker(report)
        stats = _count_valid_and_eligible(records, report)
        _ingest_records(records, model, worker, repository, report, stats)
        _verify_queries(records, repository, report, model)
        _check_no_deprecated_fields(report)
        _check_v13_unchanged(repository.client, report)
        if report["failures"]:
            raise AcceptanceFailure("Task 2 验收存在失败项")
        report["overall"] = "PASS"
    except AcceptanceFailure:
        report["overall"] = "FAIL"
    except Exception as exc:  # noqa: BLE001
        stage = str(report.get("current_stage") or "runtime")
        _record_failure(report, stage, exc)
        report["overall"] = "FAIL"
    finally:
        if repository is not None:
            repository.close()
        report["current_stage"] = None

    report_path = _write_report(args.report, report)
    _print_summary(report, report_path)
    return 0 if report["overall"] == "PASS" and report_path is not None else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help=(
            "已存在的 DEV Milvus Collection 名称 "
            f"（默认：{_DEFAULT_COLLECTION}）"
        ),
    )
    parser.add_argument(
        "--v13-collection",
        default=_DEFAULT_V13_COLLECTION,
        help=(
            "只读保护名单中的旧 Collection 名称，验收前后会做只读行数快照对比 "
            f"（默认：{_DEFAULT_V13_COLLECTION}）"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_DEFAULT_REPORT,
        help="JSON 验收报告路径（默认：reports/task2_acceptance.json）",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=_DEFAULT_EXPECTED_COUNT,
        help=f"期望 MySQL expert_chunk 读取记录数（默认：{_DEFAULT_EXPECTED_COUNT}）",
    )
    parser.add_argument(
        "--database-timezone",
        default=_DEFAULT_DATABASE_TIMEZONE,
        help=(
            "MySQL 无时区 DATETIME 解释时区（Reader fail-closed 要求显式提供；"
            f"默认：{_DEFAULT_DATABASE_TIMEZONE}）"
        ),
    )
    return parser.parse_args(argv)


def _new_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/task2_acceptance.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": "FAIL",
        "current_stage": None,
        "stages": {stage: "NOT_RUN" for stage in _STAGES},
        "input": {
            "source": "mysql:expert_chunk",
            "expected_count": args.expected_count,
            "input_count": 0,
            "valid_count": 0,
            "eligible_count": 0,
            "skipped_count": 0,
            "records": [],
            "skipped": [],
            "read_errors": [],
            "contract_versions": [],
            "chunk_type_counts": {},
        },
        "collection": {
            "name": args.collection,
            "server_version": None,
            "actual_vector_dimension": None,
            "schema": None,
            "schema_comparison": None,
            "indexes": [],
            "index_comparison": None,
        },
        "embedding": {
            "model": None,
            "model_alias": None,
            "model_version": None,
            "model_space_id": None,
            "formal_model_space_id": FORMAL_EMBEDDING_MODEL_SPACE_ID,
            "configured_dimension": None,
            "observed_dimensions": [],
        },
        "ingestion": {
            "service": "ContractBatchIngestionService",
            "embedded_count": 0,
            "skipped_count": 0,
            "affected_count": 0,
            "upsert_ids": [],
            "skipped_ids": [],
        },
        "queries": {
            "vector_doc_id": [],
            "expert_id": {},
            "missing_id": {
                "vector_doc_id": _MISSING_VECTOR_DOC_ID,
                "is_empty": False,
            },
            "deprecated_fields": {"schema": [], "rows": []},
        },
        "v13": {
            "collection": args.v13_collection,
            "exists": None,
            "pk": None,
            "before_count": None,
            "after_count": None,
            "unchanged": None,
        },
        "failures": [],
    }


def _read_mysql_records(
    args: argparse.Namespace,
    report: dict[str, Any],
) -> list[ExpertChunkEmbeddingInput]:
    """通过正式 Reader 只读读取真实 ``expert_chunk`` 并映射为 Contract 4.0.0。"""

    _begin(report, "mysql_read")
    settings = MySQLSettings.from_env()
    if not settings.database or not settings.user:
        _fail(
            report,
            "MYSQL_DATABASE / MYSQL_USER 未配置，无法读取真实 expert_chunk",
            "mysql_read",
        )
    try:
        timezone_name = args.database_timezone
        timezone_zone = ZoneInfo(timezone_name)
    except Exception as exc:  # noqa: BLE001
        _fail(
            report,
            f"database-timezone 不是合法 IANA 时区: {_error_message(exc)}",
            "mysql_read",
        )

    reader = MySqlExpertChunkReader(
        settings,
        database_timezone=timezone_zone,
    )
    try:
        result = reader.read()
    except Exception as exc:  # noqa: BLE001
        _fail(
            report,
            f"MySQL expert_chunk 读取失败: {_error_message(exc)}",
            "mysql_read",
        )

    report["input"]["input_count"] = result.input_count
    report["input"]["read_errors"] = [
        {
            "row_id": error.row_id,
            "vector_doc_id": error.vector_doc_id,
            "code": error.code,
            "message": error.message,
        }
        for error in result.errors
    ]
    if result.errors:
        _fail(
            report,
            f"{len(result.errors)} 条 MySQL 行无法映射为 Contract 4.0.0",
            "mysql_read",
        )
    if result.input_count != args.expected_count:
        _fail(
            report,
            (
                f"真实 expert_chunk 应读取 {args.expected_count} 条，"
                f"实际为 {result.input_count} 条"
            ),
            "mysql_read",
        )

    records = result.records
    report["input"]["records"] = [
        {
            "vector_doc_id": record.vector_doc_id,
            "expert_id": record.expert_id,
            "chunk_type": record.chunk_type,
            "chunk_index": record.chunk_index,
            "chunk_total": record.chunk_total,
            "is_active": record.is_active,
            "verification_status": record.metadata.verification_status,
            "origin_type": record.metadata.origin_type,
            "source_field_id": record.source.source_field_id,
            "content_hash": record.content_hash,
            "metadata_hash": record.metadata_hash,
        }
        for record in records
    ]
    report["input"]["contract_versions"] = sorted(
        {record.contract_version for record in records}
    )
    report["input"]["chunk_type_counts"] = dict(
        sorted(Counter(record.chunk_type for record in records).items())
    )
    _pass(report, "mysql_read")
    return records


def _load_settings(collection: str, report: dict[str, Any]) -> MilvusSettings:
    """加载正式 Milvus 配置，并拒绝任何受保护旧 Collection 写入。"""

    if not isinstance(collection, str) or not collection.strip():
        _fail(report, "Collection 名称不能为空", "collection_precondition")
    if collection in _PROTECTED_COLLECTIONS:
        _fail(report, f"受保护 Collection 不允许写入: {collection}", "collection_precondition")
    settings = MilvusSettings.from_env().model_copy(update={"collection_name": collection})
    if settings.schema_mode != "contract":
        _fail(
            report,
            "Task 2 必须使用 MILVUS_SCHEMA_MODE=contract",
            "collection_precondition",
        )
    return settings


def _verify_collection(
    settings: MilvusSettings,
    report: dict[str, Any],
) -> ContractMilvusVectorRepository:
    """只读确认 v14 Collection；Repository 禁止自动创建。"""

    _begin(report, "collection_precondition")
    repository = ContractMilvusVectorRepository(
        settings,
        vector_dimension=EXPECTED_EMBEDDING_DIMENSION,
        allow_collection_creation=False,
    )
    try:
        client = repository.client
        collection = report["collection"]
        collection["server_version"] = client.get_server_version()
        if not client.has_collection(collection_name=settings.collection_name):
            _fail(
                report,
                f"目标 Collection 不存在: {settings.collection_name}",
                "collection_precondition",
            )
        description = client.describe_collection(collection_name=settings.collection_name)
        actual_dimension = _actual_vector_dimension(description)
        collection["actual_vector_dimension"] = actual_dimension
        collection["schema"] = _schema_summary(description)
        collection["expected_schema"] = repository.contract_schema_summary(
            EXPECTED_EMBEDDING_DIMENSION
        )
        schema_comparison = repository.compare_contract_schema(
            description,
            EXPECTED_EMBEDDING_DIMENSION,
        )
        collection["schema_comparison"] = schema_comparison

        index_descriptions = []
        for index_name in sorted(
            str(name)
            for name in client.list_indexes(collection_name=settings.collection_name)
        ):
            index_description = client.describe_index(
                collection_name=settings.collection_name,
                index_name=index_name,
            )
            index_descriptions.append(_json_value(index_description))
        collection["indexes"] = index_descriptions
        collection["expected_indexes"] = repository.expected_contract_indexes()
        index_comparison = repository.compare_contract_indexes(client)
        collection["index_comparison"] = index_comparison

        if actual_dimension != EXPECTED_EMBEDDING_DIMENSION:
            _fail(
                report,
                (
                    f"Collection embedding 维度为 {actual_dimension}，"
                    f"正式 Contract 期望 {EXPECTED_EMBEDDING_DIMENSION}"
                ),
                "collection_precondition",
            )
        if not schema_comparison["compatible"]:
            _fail(report, "Collection Schema 与正式 Contract 不兼容", "collection_precondition")
        if not index_comparison["compatible"]:
            _fail(report, "Collection Index 与正式 Contract 不兼容", "collection_precondition")
        if collection["schema"]["auto_id"]:
            _fail(report, "Collection auto_id 必须为 False", "collection_precondition")
        pk_names = [
            str(field["name"])
            for field in collection["schema"]["fields"]
            if field.get("primary_key")
        ]
        if pk_names != ["vector_doc_id"]:
            _fail(
                report,
                f"Collection 主键必须是唯一的 vector_doc_id，实际为 {pk_names}",
                "collection_precondition",
            )

        # 已存在 Collection 的 initialize 只执行正式 Repository 的 gate，不会创建或修改
        # Schema/Index；allow_collection_creation=False 作为额外保护。
        repository.initialize(EXPECTED_EMBEDDING_DIMENSION)
        _pass(report, "collection_precondition")
        return repository
    except Exception:
        repository.close()
        raise


def _build_embedding_worker(
    report: dict[str, Any],
) -> tuple[CompanyCloudEmbeddingModel, ContractEmbeddingWorker]:
    """构造正式云端 Embedding Adapter 和已有 Contract Worker，并核对模型空间。"""

    _begin(report, "embedding")
    model = CompanyCloudEmbeddingModel.from_env()
    embedding_report = report["embedding"]
    embedding_report["model"] = model.settings.model
    embedding_report["model_alias"] = model.model_alias
    embedding_report["model_version"] = model.model_version
    embedding_report["model_space_id"] = model.model_space_id
    embedding_report["configured_dimension"] = model.settings.expected_dimension
    actual_dimension = report["collection"]["actual_vector_dimension"]
    if model.settings.expected_dimension not in (None, actual_dimension):
        _fail(
            report,
            "Embedding 配置维度与 DEV Collection 维度不一致",
            "embedding",
        )
    if model.model_space_id != FORMAL_EMBEDDING_MODEL_SPACE_ID:
        _fail(
            report,
            (
                f"Embedding model-space 不是正式空间: "
                f"actual={model.model_space_id!r}, "
                f"expected={FORMAL_EMBEDDING_MODEL_SPACE_ID!r}"
            ),
            "embedding",
        )
    worker = ContractEmbeddingWorker(
        model,
        embedding_model_alias=model.model_alias,
        embedding_model_version=model.model_version,
        expected_dimension=EXPECTED_EMBEDDING_DIMENSION,
    )
    _pass(report, "embedding")
    return model, worker


def _count_valid_and_eligible(
    records: Sequence[ExpertChunkEmbeddingInput],
    report: dict[str, Any],
) -> dict[str, int]:
    """预检统计 valid / eligible / skipped 数量，不调用 Embedding 服务。

    只做 ContractValidator 与 EmbeddingEligibilityValidator 的纯计数；真正消费
    仍由后续 ContractBatchIngestionService 执行，两处数量会在 _ingest_records
    中交叉核对，任何不一致都视为失败。
    """

    _begin(report, "contract_validation")
    validator = ContractValidator()
    eligibility_validator = EmbeddingEligibilityValidator()

    valid_records: list[ExpertChunkEmbeddingInput] = []
    skipped_entries: list[dict[str, Any]] = []
    invalid_entries: list[dict[str, Any]] = []
    for record in records:
        try:
            prepared = validator.validate(record)
        except ContractValidationError as exc:
            invalid_entries.append(
                {
                    "vector_doc_id": record.vector_doc_id,
                    "error": _safe_error(exc),
                }
            )
            continue
        valid_records.append(record)
        reason = eligibility_validator.get_skip_reason(prepared)
        if reason is not None:
            skipped_entries.append(
                {"vector_doc_id": prepared.vector_doc_id, "reason": reason}
            )

    report["input"]["valid_count"] = len(valid_records)
    report["input"]["eligible_count"] = (
        len(valid_records) - len(skipped_entries)
    )
    report["input"]["skipped_count"] = len(skipped_entries)
    report["input"]["skipped"] = skipped_entries
    if invalid_entries:
        report["failures"].extend(
            {
                "stage": "contract_validation",
                "vector_doc_id": entry["vector_doc_id"],
                "error": entry["error"],
            }
            for entry in invalid_entries
        )
        _fail(
            report,
            f"{len(invalid_entries)} 条记录未通过 Contract 4.0.0 校验",
            "contract_validation",
        )
    _pass(report, "contract_validation")
    _pass(report, "eligibility")
    return {
        "valid_count": len(valid_records),
        "eligible_count": len(valid_records) - len(skipped_entries),
        "skipped_count": len(skipped_entries),
    }


def _ingest_records(
    records: Sequence[ExpertChunkEmbeddingInput],
    model: CompanyCloudEmbeddingModel,
    worker: ContractEmbeddingWorker,
    repository: ContractMilvusVectorRepository,
    report: dict[str, Any],
    stats: Mapping[str, int],
) -> None:
    """通过正式批量 ingestion 一次完成校验、向量化和 v14 upsert。"""

    _begin(report, "embedding")
    service = ContractBatchIngestionService(worker, repository)
    try:
        result = service.ingest_batch(records)
    except Exception as exc:
        _record_failure(report, "milvus_upsert", exc)
        raise

    report["ingestion"]["embedded_count"] = result.embedded_count
    report["ingestion"]["skipped_count"] = result.skipped_count
    report["ingestion"]["affected_count"] = result.affected_count
    report["ingestion"]["upsert_ids"] = [
        item.vector_doc_id
        for item in result.results
        if item.status == "embedded"
    ]
    report["ingestion"]["skipped_ids"] = [
        item.vector_doc_id
        for item in result.results
        if item.status == "skipped"
    ]

    mismatch = _batch_count_mismatch(result, stats)
    if mismatch is not None:
        _fail(report, f"批量 ingestion 数量与预检不一致: {mismatch}", "milvus_upsert")
    if result.embedded_count < 1:
        _fail(
            report,
            "批量 ingestion 没有任何 embedded 记录，验收必须产生真实写入",
            "milvus_upsert",
        )

    embedding_report = report["embedding"]
    embedding_report["observed_dimensions"] = [model.dimension]
    _pass(report, "embedding")
    _pass(report, "milvus_upsert")


def _verify_queries(
    records: Sequence[ExpertChunkEmbeddingInput],
    repository: ContractMilvusVectorRepository,
    report: dict[str, Any],
    model: CompanyCloudEmbeddingModel,
) -> None:
    """执行主键、专家过滤、模型空间和不存在 ID 的只读精确回查。"""

    record_by_vector_id = {record.vector_doc_id: record for record in records}
    client = repository.client
    collection_name = repository.settings.collection_name

    _begin(report, "vector_doc_id_query")
    vector_results: list[dict[str, Any]] = []
    for record in records:
        result: dict[str, Any] = {
            "vector_doc_id": record.vector_doc_id,
            "found": False,
        }
        try:
            row = repository.get_by_vector_doc_id(record.vector_doc_id)
            if row is None:
                raise LookupError("vector_doc_id 查询返回空")
            checks = _matching_fields(row, record)
            checks.update(_model_space_checks(row, model))
            result["found"] = True
            result["embedding_dimension"] = _row_dimension(row)
            result["fields_match"] = checks
            if not all(checks.values()):
                raise ValueError(
                    "回查字段不一致: "
                    + ", ".join(name for name, passed in checks.items() if not passed)
                )
            deprecated = _deprecated_fields_present(list(row.keys()))
            if deprecated:
                report["queries"]["deprecated_fields"]["rows"].append(
                    {
                        "vector_doc_id": record.vector_doc_id,
                        "fields": deprecated,
                    }
                )
                raise ValueError("回查行包含已废弃字段: " + ",".join(deprecated))
        except Exception as exc:  # noqa: BLE001
            result["error"] = _safe_error(exc)
            _record_failure(report, "vector_doc_id_query", exc, record.vector_doc_id)
        vector_results.append(result)
    report["queries"]["vector_doc_id"] = vector_results
    if any("error" in result for result in vector_results):
        report["stages"]["vector_doc_id_query"] = "FAIL"
    else:
        _pass(report, "vector_doc_id_query")

    _begin(report, "expert_id_query")
    expert_results: dict[str, Any] = {}
    for expert_id in sorted({record.expert_id for record in records}):
        result: dict[str, Any] = {
            "expert_id": expert_id,
            "row_count": 0,
            "expected_count": sum(record.expert_id == expert_id for record in records),
        }
        try:
            rows = _query_rows(client, collection_name, f"expert_id == {expert_id}")
            result["row_count"] = len(rows)
            result["vector_doc_ids"] = sorted(
                str(row.get("vector_doc_id")) for row in rows
            )
            result["expected_vector_doc_ids"] = sorted(
                record.vector_doc_id
                for record in records
                if record.expert_id == expert_id
            )
            result["exact_set_match"] = (
                len(rows) == result["expected_count"]
                and set(result["vector_doc_ids"]) == set(result["expected_vector_doc_ids"])
            )
            field_checks = []
            for row in rows:
                vector_doc_id = row.get("vector_doc_id")
                expected_record = record_by_vector_id.get(vector_doc_id)
                if expected_record is None:
                    field_checks.append(False)
                else:
                    field_checks.append(
                        all(_matching_fields(row, expected_record).values())
                    )
            result["all_fields_match"] = all(field_checks) and len(field_checks) == len(rows)
            if not result["exact_set_match"] or not result["all_fields_match"]:
                raise ValueError("expert_id 查询结果与本次输入记录集合不一致")
        except Exception as exc:  # noqa: BLE001
            result["error"] = _safe_error(exc)
            _record_failure(report, "expert_id_query", exc, str(expert_id))
        expert_results[str(expert_id)] = result
    report["queries"]["expert_id"] = expert_results
    if any("error" in result for result in expert_results.values()):
        report["stages"]["expert_id_query"] = "FAIL"
    else:
        _pass(report, "expert_id_query")

    _begin(report, "missing_id_query")
    missing_result = {
        "vector_doc_id": _MISSING_VECTOR_DOC_ID,
        "is_empty": False,
    }
    try:
        missing_result["is_empty"] = (
            repository.get_by_vector_doc_id(_MISSING_VECTOR_DOC_ID) is None
        )
        if not missing_result["is_empty"]:
            raise ValueError("不存在 ID 查询未返回空")
    except Exception as exc:  # noqa: BLE001
        missing_result["error"] = _safe_error(exc)
        _record_failure(report, "missing_id_query", exc)
    report["queries"]["missing_id"] = missing_result
    if "error" in missing_result:
        report["stages"]["missing_id_query"] = "FAIL"
    else:
        _pass(report, "missing_id_query")


def _check_no_deprecated_fields(report: dict[str, Any]) -> None:
    """确认 v14 Schema 与服务端回查行中不存在任何已废弃字段。"""

    _begin(report, "no_deprecated_query")
    schema_fields = [
        str(field.get("name"))
        for field in report["collection"]["schema"]["fields"]
    ]
    schema_deprecated = _deprecated_fields_present(schema_fields)
    report["queries"]["deprecated_fields"]["schema"] = schema_deprecated
    row_deprecated = report["queries"]["deprecated_fields"]["rows"]
    if schema_deprecated or row_deprecated:
        _fail(
            report,
            "v14 Collection 的 Schema 或回查行包含已废弃字段",
            "no_deprecated_query",
        )
    _pass(report, "no_deprecated_query")


def _snapshot_v13(
    client: Any,
    collection_name: str,
    report: dict[str, Any],
) -> None:
    """ingestion 前对旧 v13 Collection 做只读行数快照。"""

    snapshot = _v13_snapshot(client, collection_name)
    report["v13"]["exists"] = snapshot["exists"]
    report["v13"]["pk"] = snapshot.get("pk")
    report["v13"]["before_count"] = snapshot.get("row_count")


def _check_v13_unchanged(client: Any, report: dict[str, Any]) -> None:
    """ingestion 后再次快照旧 Collection，确认行数前后一致。"""

    _begin(report, "v13_integrity")
    if not report["v13"].get("exists"):
        report["v13"]["unchanged"] = None
        _pass(report, "v13_integrity")
        return
    after = _v13_snapshot(client, report["v13"]["collection"])
    report["v13"]["after_count"] = after.get("row_count")
    state = _compare_v13_state(report["v13"], after)
    report["v13"]["unchanged"] = not state["changed"]
    if state["changed"]:
        _fail(
            report,
            (
                f"旧 Collection {report['v13']['collection']} 行数在验收前后变化: "
                f"before={state['before_count']}, after={state['after_count']}"
            ),
            "v13_integrity",
        )
    _pass(report, "v13_integrity")


def _v13_snapshot(client: Any, collection_name: str) -> dict[str, Any]:
    """只读统计旧 Collection 行数；不存在时返回 exists=False。

    行数来自 ``get_collection_stats``，不做无过滤的全量实体查询，
    避免空表达式 query 在服务端被拒绝。
    """

    if not client.has_collection(collection_name=collection_name):
        return {"exists": False, "row_count": None, "pk": None}
    description = client.describe_collection(collection_name=collection_name)
    pk = None
    for field in description.get("fields", []):
        if isinstance(field, Mapping) and field.get("is_primary"):
            pk = str(field.get("name"))
            break
    if pk is None:
        raise RuntimeError(f"旧 Collection {collection_name!r} 缺少主键字段")
    stats = client.get_collection_stats(collection_name=collection_name)
    if not isinstance(stats, Mapping):
        raise TypeError(f"旧 Collection {collection_name!r} 行数统计返回非法结果")
    row_count = stats.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool):
        raise TypeError(
            f"旧 Collection {collection_name!r} 行数统计缺少合法 row_count: {row_count!r}"
        )
    return {"exists": True, "row_count": row_count, "pk": pk}


def _compare_v13_state(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """比较旧 Collection 前后行数快照，返回是否发生变化。"""

    before_count = before.get("before_count", before.get("row_count"))
    after_count = after.get("row_count")
    if before_count is None or after_count is None:
        return {
            "changed": True,
            "before_count": before_count,
            "after_count": after_count,
        }
    return {
        "changed": before_count != after_count,
        "before_count": before_count,
        "after_count": after_count,
    }


def _query_rows(
    client: Any,
    collection_name: str,
    filter_expression: str,
) -> list[Mapping[str, Any]]:
    """用非空过滤表达式执行只读回查。

    调用方必须传入非空 ``filter_expression``；空表达式会被服务端拒绝，
    行数统计请改用 ``get_collection_stats``。
    """

    rows = client.query(
        collection_name=collection_name,
        filter=filter_expression,
        output_fields=_QUERY_FIELDS,
        consistency_level="Strong",
    )
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("Milvus query 返回的 Entity 列表格式不正确")
    return rows


def _matching_fields(
    row: Mapping[str, Any],
    record: ExpertChunkEmbeddingInput,
) -> dict[str, bool]:
    """比较回查行与输入 Contract 的完整字段，只使用 vector_doc_id 定位。"""

    return {
        "vector_doc_id": row.get("vector_doc_id") == record.vector_doc_id,
        "expert_id": row.get("expert_id") == record.expert_id,
        "chunk_type": row.get("chunk_type") == record.chunk_type,
        "content_hash": row.get("content_hash") == record.content_hash,
        "metadata_hash": row.get("metadata_hash") == record.metadata_hash,
        "metadata_ext": row.get("metadata_ext")
        == record.metadata.model_dump(mode="json"),
        "source_ext": row.get("source_ext") == record.source.model_dump(mode="json"),
        "versions_ext": row.get("versions_ext") == record.versions.model_dump(mode="json"),
        "is_active": row.get("is_active") == record.is_active,
        "verification_status": row.get("verification_status")
        == record.metadata.verification_status,
        "eligible_for_recommendation": row.get("eligible_for_recommendation")
        == record.metadata.eligible_for_recommendation,
    }


def _model_space_checks(
    row: Mapping[str, Any],
    model: CompanyCloudEmbeddingModel,
) -> dict[str, bool]:
    """核对回查行的模型空间与当前 Embedding 配置一致。"""

    return {
        "embedding_model_alias": row.get("embedding_model_alias") == model.model_alias,
        "embedding_model_version": row.get("embedding_model_version")
        == model.model_version,
        "embedding_dim": row.get("embedding_dim") == model.dimension,
        "embedding_dimension": _row_dimension(row) == model.dimension,
    }


def _batch_count_mismatch(
    result: ContractBatchIngestionResult,
    stats: Mapping[str, int],
) -> str | None:
    """返回批量 ingestion 与预检数量不一致的原因；一致时返回 None。"""

    if result.embedded_count != stats["eligible_count"]:
        return (
            f"embedded_count={result.embedded_count} "
            f"与预检 eligible_count={stats['eligible_count']} 不一致"
        )
    if result.skipped_count != stats["skipped_count"]:
        return (
            f"skipped_count={result.skipped_count} "
            f"与预检 skipped_count={stats['skipped_count']} 不一致"
        )
    if result.affected_count != result.embedded_count:
        return (
            f"affected_count={result.affected_count} "
            f"与 embedded_count={result.embedded_count} 不一致"
        )
    return None


def _deprecated_fields_present(
    field_names: Sequence[object],
) -> list[str]:
    """返回字段名列表中属于已废弃名单的字段（排序去重）。"""

    names = {str(name) for name in field_names}
    return sorted(names & set(_DEPRECATED_FIELD_NAMES))


def _actual_vector_dimension(description: Mapping[str, Any]) -> int:
    for field in description.get("fields", []):
        if not isinstance(field, Mapping) or field.get("name") != "embedding":
            continue
        params = field.get("params")
        if not isinstance(params, Mapping):
            break
        dimension = params.get("dim")
        if isinstance(dimension, str) and dimension.isdigit():
            dimension = int(dimension)
        if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
            return dimension
        break
    raise ValueError("Collection Schema 缺少合法 embedding dim")


def _schema_summary(description: Mapping[str, Any]) -> dict[str, Any]:
    fields = []
    for field in description.get("fields", []):
        if not isinstance(field, Mapping):
            continue
        fields.append(
            {
                "name": field.get("name"),
                "datatype": datatype_name(field.get("type")),
                "nullable": bool(field.get("nullable", False)),
                "primary_key": bool(field.get("is_primary", False)),
                "params": _json_value(field.get("params") or {}),
            }
        )
    return {
        "collection_name": description.get("collection_name"),
        "auto_id": bool(description.get("auto_id", False)),
        "enable_dynamic_field": bool(description.get("enable_dynamic_field", False)),
        "fields": fields,
    }


def _begin(report: dict[str, Any], stage: str) -> None:
    report["current_stage"] = stage
    report["stages"][stage] = "RUNNING"


def _pass(report: dict[str, Any], stage: str) -> None:
    report["stages"][stage] = "PASS"
    report["current_stage"] = None


def _fail(report: dict[str, Any], message: str, stage: str) -> NoReturn:
    report["stages"][stage] = "FAIL"
    report["current_stage"] = stage
    report["failures"].append(
        {
            "stage": stage,
            "error": {"type": "AcceptanceFailure", "message": message},
        }
    )
    raise AcceptanceFailure(message)


def _record_failure(
    report: dict[str, Any],
    stage: str,
    exc: Exception,
    identifier: str | None = None,
) -> None:
    report["stages"][stage] = "FAIL"
    failure: dict[str, Any] = {"stage": stage, "error": _safe_error(exc)}
    if identifier is not None:
        failure["identifier"] = identifier
    report["failures"].append(failure)


def _safe_error(exc: Exception) -> dict[str, str]:
    message = " ".join(str(exc).split())
    for env_name in (
        "MILVUS_TOKEN",
        "MILVUS_PASSWORD",
        "MYSQL_PASSWORD",
        "EMBEDDING_API_KEY",
    ):
        secret = os.environ.get(env_name)
        if secret:
            message = message.replace(secret, "<redacted>")
    return {
        "type": type(exc).__name__,
        "message": message[:1000],
    }


def _error_message(exc: Exception) -> str:
    return _safe_error(exc)["message"]


def _row_dimension(row: Mapping[str, Any]) -> int | None:
    embedding = row.get("embedding")
    return len(embedding) if isinstance(embedding, list) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_report(path: Path, report: Mapping[str, Any]) -> Path | None:
    try:
        resolved = path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(_json_value(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"FAIL 无法写入 JSON 报告: {_error_message(exc)}", file=sys.stderr)
        return None
    return resolved


def _print_summary(report: Mapping[str, Any], report_path: Path | None) -> None:
    overall = report["overall"]
    print(f"Task 2 DEV acceptance: {overall}")
    for stage, status in report["stages"].items():
        print(f"[{status}] {stage}")
    collection = report["collection"]
    embedding = report["embedding"]
    input_summary = report["input"]
    ingestion = report["ingestion"]
    v13 = report["v13"]
    print(f"collection: {collection['name']}")
    print(
        "input: "
        f"mysql_rows={input_summary['input_count']} "
        f"valid={input_summary['valid_count']} "
        f"eligible={input_summary['eligible_count']} "
        f"skipped={input_summary['skipped_count']}"
    )
    print(f"embedding_model: {embedding['model']}")
    print(f"embedding_model_space: {embedding['model_space_id']}")
    print(f"embedding_dimension: {embedding['observed_dimensions']}")
    print(
        "upsert: "
        f"embedded={ingestion['embedded_count']} "
        f"skipped={ingestion['skipped_count']} "
        f"affected={ingestion['affected_count']}"
    )
    for vector_doc_id in ingestion["upsert_ids"]:
        print(f"write: {vector_doc_id}")
    print(
        "v13: "
        f"{v13['collection']} exists={v13.get('exists')} "
        f"before={v13.get('before_count')} after={v13.get('after_count')} "
        f"unchanged={v13.get('unchanged')}"
    )
    if report["failures"]:
        print(f"failures: {len(report['failures'])}")
        for failure in report["failures"]:
            print(f"- {failure}")
    if report_path is not None:
        print(f"report: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())