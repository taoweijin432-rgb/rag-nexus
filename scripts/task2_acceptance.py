"""执行 Task 2 DEV 的真实 Contract ingestion 验收。

脚本只编排已有的 Shared Contract、Embedding Worker、ContractIngestionService 和
正式 Milvus Repository。目标 Collection 必须已经存在且通过只读 Schema/Index
兼容性检查；脚本不会创建、修改、删除 Collection，也不会清理历史数据。

示例::

    set -a; source .env; set +a
    uv run --no-sync python scripts/task2_acceptance.py \
        --input fixtures/expert_999641217.jsonl \
        --collection expert_candidates_v13_contract_dev \
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

from expert_contracts import (
    ExpertChunkEmbeddingInput,
    adapt_expert_chunk_mapping,
)
from rag_vector_store import (
    EXPECTED_EMBEDDING_DIMENSION,
    CompanyCloudEmbeddingModel,
    ContractEmbeddingWorker,
    ContractIngestionResult,
    ContractIngestionService,
    ContractMilvusVectorRepository,
    MilvusSettings,
)
from rag_vector_store.milvus.compatibility import datatype_name

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT = _PROJECT_ROOT / "fixtures" / "expert_999641217.jsonl"
_DEFAULT_COLLECTION = "expert_candidates_v13_contract_dev"
_DEFAULT_REPORT = _PROJECT_ROOT / "reports" / "task2_acceptance.json"
_DEFAULT_EXPECTED_COUNT = 7
_MISSING_VECTOR_DOC_ID = "task2-acceptance:999641217:does-not-exist"
_PROTECTED_COLLECTIONS = frozenset(
    {
        "expert_candidates_v1",
        "resume_candidates_v1",
        "expert_candidates_v12_dev",
    }
)
_STAGES = (
    "contract_bridge",
    "collection_precondition",
    "embedding",
    "milvus_upsert",
    "vector_doc_id_query",
    "chunk_id_query",
    "expert_id_query",
    "missing_id_query",
)
_QUERY_FIELDS = [
    "vector_doc_id",
    "chunk_id",
    "expert_id",
    "chunk_type",
    "source_field_id",
    "metadata_ext",
    "source_ext",
    "versions_ext",
]


class AcceptanceFailure(RuntimeError):
    """验收阶段失败。"""


def main(argv: Sequence[str] | None = None) -> int:
    """执行验收并输出终端摘要及 JSON 报告。"""

    args = _parse_args(argv)
    report = _new_report(args)
    repository: ContractMilvusVectorRepository | None = None

    try:
        records = _load_contracts(args.input, args.expected_count, report)
        settings = _load_settings(args.collection, report)
        repository = _verify_collection(settings, report)
        model, worker = _build_embedding_worker(report)
        _ingest_records(records, model, worker, repository, report)
        _verify_queries(records, repository, report)
        if report["failures"]:
            raise AcceptanceFailure("Task 2 查询阶段存在失败项")
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
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="真实上游 JSONL 路径（默认：fixtures/expert_999641217.jsonl）",
    )
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help=(
            "已存在的 DEV Milvus Collection 名称 "
            f"（默认：{_DEFAULT_COLLECTION}）"
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
        help=f"期望输入记录数（默认：{_DEFAULT_EXPECTED_COUNT}）",
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
            "path": str(Path(args.input).resolve()),
            "expected_count": args.expected_count,
            "record_count": 0,
            "contract_versions": [],
            "chunk_type_counts": {},
            "records": [],
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
            "configured_dimension": None,
            "observed_dimensions": [],
        },
        "writes": [],
        "queries": {
            "vector_doc_id": [],
            "chunk_id": [],
            "expert_id": {},
            "missing_id": {
                "vector_doc_id": _MISSING_VECTOR_DOC_ID,
                "is_empty": False,
            },
        },
        "failures": [],
    }


def _load_contracts(
    path: Path,
    expected_count: int,
    report: dict[str, Any],
) -> list[ExpertChunkEmbeddingInput]:
    """逐行解析 JSONL，并通过正式 Shared Contract Adapter。"""

    _begin(report, "contract_bridge")
    if expected_count < 1:
        _fail(report, "expected-count 必须是正整数", "contract_bridge")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _fail(report, f"无法读取输入 JSONL: {_error_message(exc)}", "contract_bridge")

    records: list[ExpertChunkEmbeddingInput] = []
    contract_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    non_empty_line_count = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        non_empty_line_count += 1
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError("JSONL 记录必须是 JSON object")
            record = adapt_expert_chunk_mapping(payload)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "line_number": line_number,
                    "error": _safe_error(exc),
                }
            )
            continue

        metadata_vector_doc_id = record.metadata.vector_doc_id
        contract_rows.append(
            {
                "line_number": line_number,
                "chunk_id": record.chunk_id,
                "vector_doc_id": record.vector_doc_id,
                "expert_id": record.expert_id,
                "chunk_type": record.chunk_type,
                "metadata_vector_doc_id": metadata_vector_doc_id,
                "source_field_id": record.source.source_field_id,
                "vector_doc_id_promoted": record.vector_doc_id == metadata_vector_doc_id,
            }
        )
        records.append(record)

    report["input"]["record_count"] = non_empty_line_count
    report["input"]["records"] = contract_rows
    report["input"]["contract_versions"] = sorted(
        {record.contract_version for record in records}
    )
    report["input"]["chunk_type_counts"] = dict(
        sorted(Counter(record.chunk_type for record in records).items())
    )
    if non_empty_line_count != expected_count:
        failures.append(
            {
                "error": {
                    "type": "InputCountError",
                    "message": (
                        f"输入记录数应为 {expected_count}，实际为 {non_empty_line_count}"
                    ),
                }
            }
        )
    if len(records) != non_empty_line_count:
        failures.append(
            {
                "error": {
                    "type": "ContractBridgeError",
                    "message": "存在未能转换为 ExpertChunkEmbeddingInput v2 的记录",
                }
            }
        )
    report["failures"].extend(failures)
    if failures:
        _fail(report, "JSONL → ExpertChunkEmbeddingInput 转换失败", "contract_bridge")
    _pass(report, "contract_bridge")
    return records


def _load_settings(collection: str, report: dict[str, Any]) -> MilvusSettings:
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
    """只读确认 Collection；Repository 禁止自动创建。"""

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
    """构造正式云端 Embedding Adapter 和已有 Contract Worker。"""

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
    worker = ContractEmbeddingWorker(
        model,
        embedding_model_alias=model.model_alias,
        embedding_model_version=model.model_version,
        expected_dimension=EXPECTED_EMBEDDING_DIMENSION,
    )
    return model, worker


def _ingest_records(
    records: Sequence[ExpertChunkEmbeddingInput],
    model: CompanyCloudEmbeddingModel,
    worker: ContractEmbeddingWorker,
    repository: ContractMilvusVectorRepository,
    report: dict[str, Any],
) -> None:
    """逐条调用正式单条 ingestion，不复制 Worker 或 Mapper 逻辑。"""

    _begin(report, "embedding")
    service = ContractIngestionService(worker, repository)
    failures: list[dict[str, Any]] = []
    for record in records:
        result_entry: dict[str, Any] = {
            "chunk_id": record.chunk_id,
            "vector_doc_id": record.vector_doc_id,
            "status": "FAIL",
        }
        try:
            result = service.ingest_one(record)
        except Exception as exc:  # noqa: BLE001
            result_entry["error"] = _safe_error(exc)
            failures.append(
                {
                    "stage": "embedding_and_upsert",
                    "chunk_id": record.chunk_id,
                    "error": result_entry["error"],
                }
            )
        else:
            result_entry.update(_ingestion_result(result))
            if result.status != "embedded" or result.affected_count != 1:
                failures.append(
                    {
                        "stage": "embedding_and_upsert",
                        "chunk_id": record.chunk_id,
                        "error": {
                            "type": "UnexpectedIngestionResult",
                            "message": "Task 2 每条记录都必须 embedded 且 affected_count=1",
                        },
                    }
                )
        report["writes"].append(result_entry)

    report["embedding"]["observed_dimensions"] = sorted(
        {
            len(result_entry.get("embedding", []))
            for result_entry in report["writes"]
            if isinstance(result_entry.get("embedding"), list)
        }
    )
    report["failures"].extend(failures)
    if failures:
        report["stages"]["milvus_upsert"] = "FAIL"
        _fail(report, "存在未成功 embedded/upsert 的记录", "embedding")
    observed_dimension = model.dimension
    report["embedding"]["observed_dimensions"] = [observed_dimension]
    _pass(report, "embedding")
    _pass(report, "milvus_upsert")


def _verify_queries(
    records: Sequence[ExpertChunkEmbeddingInput],
    repository: ContractMilvusVectorRepository,
    report: dict[str, Any],
) -> None:
    """执行主键、业务 ID 和不存在 ID 的只读精确回查。"""

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
            result["found"] = True
            result["embedding_dimension"] = _row_dimension(row)
            result["fields_match"] = checks
            if not all(checks.values()):
                raise ValueError(
                    "回查字段不一致: "
                    + ", ".join(name for name, passed in checks.items() if not passed)
                )
        except Exception as exc:  # noqa: BLE001
            result["error"] = _safe_error(exc)
            _record_failure(report, "vector_doc_id_query", exc, record.chunk_id)
        vector_results.append(result)
    report["queries"]["vector_doc_id"] = vector_results
    if any("error" in result for result in vector_results):
        report["stages"]["vector_doc_id_query"] = "FAIL"
    else:
        _pass(report, "vector_doc_id_query")

    _begin(report, "chunk_id_query")
    chunk_results: list[dict[str, Any]] = []
    for record in records:
        result: dict[str, Any] = {
            "chunk_id": record.chunk_id,
            "row_count": 0,
            "exactly_one": False,
        }
        try:
            rows = _query_rows(
                client,
                collection_name,
                f"chunk_id == '{_quote_milvus_string(record.chunk_id)}'",
            )
            result["row_count"] = len(rows)
            result["vector_doc_ids"] = sorted(
                str(row.get("vector_doc_id")) for row in rows
            )
            result["exactly_one"] = len(rows) == 1
            if len(rows) != 1:
                raise ValueError(f"chunk_id 查询应返回 1 条，实际 {len(rows)} 条")
            checks = _matching_fields(rows[0], record)
            result["fields_match"] = checks
            if not all(checks.values()):
                raise ValueError(
                    "回查字段不一致: "
                    + ", ".join(name for name, passed in checks.items() if not passed)
                )
        except Exception as exc:  # noqa: BLE001
            result["error"] = _safe_error(exc)
            _record_failure(report, "chunk_id_query", exc, record.chunk_id)
        chunk_results.append(result)
    report["queries"]["chunk_id"] = chunk_results
    if any("error" in result for result in chunk_results):
        report["stages"]["chunk_id_query"] = "FAIL"
    else:
        _pass(report, "chunk_id_query")

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


def _query_rows(
    client: Any,
    collection_name: str,
    filter_expression: str,
) -> list[Mapping[str, Any]]:
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
    return {
        "vector_doc_id": row.get("vector_doc_id") == record.vector_doc_id,
        "chunk_id": row.get("chunk_id") == record.chunk_id,
        "expert_id": row.get("expert_id") == record.expert_id,
        "chunk_type": row.get("chunk_type") == record.chunk_type,
        "source_field_id": row.get("source_field_id") == record.source.source_field_id,
        "metadata": row.get("metadata_ext") == record.metadata.model_dump(mode="json"),
        "source": row.get("source_ext") == record.source.model_dump(mode="json"),
        "versions": row.get("versions_ext") == record.versions.model_dump(mode="json"),
    }


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


def _ingestion_result(result: ContractIngestionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "vector_doc_id": result.vector_doc_id,
        "affected_count": result.affected_count,
        "skip_reason": result.skip_reason,
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


def _quote_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


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
    print(f"collection: {collection['name']}")
    print(f"embedding_model: {embedding['model']}")
    print(f"embedding_dimension: {embedding['observed_dimensions']}")
    for item in report["writes"]:
        print(
            "write: "
            f"{item.get('chunk_id')} status={item.get('status')} "
            f"affected_count={item.get('affected_count', 0)}"
        )
    if report["failures"]:
        print(f"failures: {len(report['failures'])}")
        for failure in report["failures"]:
            print(f"- {failure}")
    if report_path is not None:
        print(f"report: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
