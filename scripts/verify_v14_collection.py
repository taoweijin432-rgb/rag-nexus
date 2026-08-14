"""只创建并结构验证新的 DEV Milvus Collection（不写任何业务数据）。

目标 Collection: ``expert_candidates_v14_contract_dev``。

本脚本只使用正式 Repository / Schema Builder（``ContractMilvusVectorRepository``
→ ``schemas.contract.build_contract_schema`` + ``indexes.index_specs``），不手写
第二套 Schema。创建前先只读检查目标是否已存在：

- 已存在：只做 Schema / Index / auto_id / PK / dim 全面对比，**不重建**，差异以
  FAIL 报告。
- 不存在：仅在 ``ALLOW_CREATE=1`` 时通过 Repository 创建（默认只读）。

创建完成后只做结构验证，不写入任何 ``expert_chunk`` 数据、不调用 Embedding。

示例::

    set -a; source .env; set +a
    uv run --no-sync python scripts/verify_v14_collection.py   # 只读检查
    ALLOW_CREATE=1 uv run --no-sync python scripts/verify_v14_collection.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from rag_vector_store import (
    CONTRACT_SCALAR_INDEX_FIELDS,
    EXPECTED_EMBEDDING_DIMENSION,
    ContractMilvusVectorRepository,
    MilvusSettings,
    contract_milvus_field_specs,
)
from rag_vector_store.milvus.compatibility import datatype_name

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_COLLECTION = "expert_candidates_v14_contract_dev"
_DEFAULT_REPORT = _PROJECT_ROOT / "reports" / "v14_collection_verify.json"
_FROZEN_EXPECTED_INDEX_COUNT = 4  # 1 vector + 3 scalar
_FROZEN_VECTOR_INDEX = {
    "index_name": "embedding_vector",
    "field_name": "embedding",
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
}
_FROZEN_SCALAR_INDEX_FIELDS = CONTRACT_SCALAR_INDEX_FIELDS  # expert_id/chunk_type/eligible_for_recommendation
_FROZEN_EXPECTED_FIELD_COUNT = 25
_PROTECTED_V13 = "expert_candidates_v13_contract_dev"


class VerificationFailure(RuntimeError):
    """结构验证失败。"""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = _new_report(args)
    repository: ContractMilvusVectorRepository | None = None
    try:
        repository = _run(args, report)
        if report["failures"]:
            raise VerificationFailure("v14 结构验证存在失败项")
        report["overall"] = "PASS"
    except VerificationFailure:
        report["overall"] = "FAIL"
    except Exception as exc:  # noqa: BLE001 - 顶层统一上报
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


def _run(
    args: argparse.Namespace,
    report: dict[str, Any],
) -> ContractMilvusVectorRepository:
    """执行只读预检 → 按需创建 → 结构验证 → v13 只读确认。"""

    settings = _load_settings(args.collection, report)
    allow_create = (
        os.environ.get("ALLOW_CREATE", "").strip().lower() in {"1", "true", "yes"}
    )
    repository = ContractMilvusVectorRepository(
        settings,
        vector_dimension=EXPECTED_EMBEDDING_DIMENSION,
        allow_collection_creation=allow_create,
    )
    client = repository.client
    report["collection"]["server_version"] = client.get_server_version()

    existed = client.has_collection(collection_name=args.collection)
    report["collection"]["existed_before"] = existed
    # 无论创建还是只读路径，都先做一次旧 v13 快照，供前后对比。
    _snapshot_v13(client, report)
    if existed:
        _existing_collection_report(client, repository, report)
    else:
        _create_collection(repository, client, report)

    _verify_structure(client, repository, report)
    _verify_v13_unchanged(client, report)
    return repository


def _load_settings(collection: str, report: dict[str, Any]) -> MilvusSettings:
    if not isinstance(collection, str) or not collection.strip():
        _fail(report, "Collection 名称不能为空", "precondition")
    if collection == _PROTECTED_V13:
        _fail(
            report,
            f"受保护旧 Collection 不允许创建或修改: {collection}",
            "precondition",
        )
    settings = MilvusSettings.from_env().model_copy(update={"collection_name": collection})
    if settings.schema_mode != "contract":
        _fail(report, "必须使用 MILVUS_SCHEMA_MODE=contract", "precondition")

    expected_params = {"M": 16, "efConstruction": 200}
    if settings.index_params != expected_params:
        _fail(
            report,
            (
                f"MILVUS_INDEX_PARAMS 必须等于冻结方案 {{M:16, efConstruction:200}}，"
                f"实际为 {settings.index_params}"
            ),
            "precondition",
        )
    if settings.index_type.upper() != "HNSW" or settings.metric_type.upper() != "COSINE":
        _fail(
            report,
            "MILVUS_INDEX_TYPE 必须为 HNSW、MILVUS_METRIC_TYPE 必须为 COSINE",
            "precondition",
        )
    report["collection"]["settings"] = {
        "uri_host": settings.uri,
        "collection_name": settings.collection_name,
        "schema_mode": settings.schema_mode,
        "index_type": settings.index_type,
        "metric_type": settings.metric_type,
        "index_params": dict(settings.index_params),
    }
    return settings


def _existing_collection_report(
    client: Any,
    repository: ContractMilvusVectorRepository,
    report: dict[str, Any],
) -> None:
    """目标已存在：只读对比 Schema/Index，不重建。"""

    _begin(report, "existence_check")
    collection_name = repository.settings.collection_name
    description = client.describe_collection(collection_name=collection_name)
    report["collection"]["auto_id"] = bool(description.get("auto_id", False))
    report["collection"]["enable_dynamic_field"] = bool(
        description.get("enable_dynamic_field", False)
    )
    report["collection"]["schema"] = _schema_summary(description)
    report["collection"]["field_count"] = len(report["collection"]["schema"]["fields"])

    expected_specs = contract_milvus_field_specs(EXPECTED_EMBEDDING_DIMENSION)
    report["collection"]["expected_schema"] = [spec.summary() for spec in expected_specs]

    schema_comparison = repository.compare_contract_schema(
        description,
        EXPECTED_EMBEDDING_DIMENSION,
    )
    report["collection"]["schema_comparison"] = schema_comparison
    if not schema_comparison["compatible"]:
        _fail(report, "已存在 Collection 的 Schema 与冻结方案不一致", "existence_check")

    index_names = sorted(str(name) for name in client.list_indexes(collection_name=collection_name))
    index_descriptions = {
        name: client.describe_index(collection_name=collection_name, index_name=name)
        for name in index_names
    }
    report["collection"]["indexes"] = _index_descriptions(index_descriptions)
    index_comparison = repository.compare_contract_indexes(client)
    report["collection"]["index_comparison"] = index_comparison
    if not index_comparison["compatible"]:
        _fail(report, "已存在 Collection 的 Index 与冻结方案不一致", "existence_check")
    _pass(report, "existence_check")


def _create_collection(
    repository: ContractMilvusVectorRepository,
    client: Any,
    report: dict[str, Any],
) -> None:
    """目标不存在：通过正式 Repository 创建（需要 ALLOW_CREATE=1）。"""

    _begin(report, "existence_check")
    if os.environ.get("ALLOW_CREATE", "").strip().lower() not in {"1", "true", "yes"}:
        _fail(
            report,
            (
                "Collection 不存在，且未设置 ALLOW_CREATE=1；"
                "脚本默认只读，不会自动创建"
            ),
            "existence_check",
        )
    try:
        repository.initialize(EXPECTED_EMBEDDING_DIMENSION)
    except Exception as exc:
        _record_failure(report, "creation", exc)
        raise
    report["collection"]["created_now"] = True

    # 结构验证前重新读取服务端描述（不信任本地构造结果）。
    description = client.describe_collection(
        collection_name=repository.settings.collection_name
    )
    report["collection"]["auto_id"] = bool(description.get("auto_id", False))
    report["collection"]["enable_dynamic_field"] = bool(
        description.get("enable_dynamic_field", False)
    )
    report["collection"]["schema"] = _schema_summary(description)
    report["collection"]["field_count"] = len(report["collection"]["schema"]["fields"])
    expected_specs = contract_milvus_field_specs(EXPECTED_EMBEDDING_DIMENSION)
    report["collection"]["expected_schema"] = [spec.summary() for spec in expected_specs]
    report["collection"]["schema_comparison"] = repository.compare_contract_schema(
        description,
        EXPECTED_EMBEDDING_DIMENSION,
    )
    index_names = sorted(
        str(name) for name in client.list_indexes(collection_name=repository.settings.collection_name)
    )
    index_descriptions = {
        name: client.describe_index(
            collection_name=repository.settings.collection_name, index_name=name
        )
        for name in index_names
    }
    report["collection"]["indexes"] = _index_descriptions(index_descriptions)
    report["collection"]["index_comparison"] = repository.compare_contract_indexes(client)
    _pass(report, "existence_check")


def _verify_structure(
    client: Any,
    repository: ContractMilvusVectorRepository,
    report: dict[str, Any],
) -> None:
    """只读结构验证：字段数、PK、auto_id、dim、JSON、废弃字段、索引集合。"""

    _begin(report, "structure_validation")
    collection = report["collection"]
    collection_name = repository.settings.collection_name
    if not client.has_collection(collection_name=collection_name):
        _fail(report, "Collection 不存在", "structure_validation")

    description = client.describe_collection(collection_name=collection_name)
    actual_dimension = _actual_vector_dimension(description)
    collection["actual_vector_dimension"] = actual_dimension
    fields = collection["schema"]["fields"]

    if collection["field_count"] != _FROZEN_EXPECTED_FIELD_COUNT:
        _fail(
            report,
            (
                f"字段数应为 {_FROZEN_EXPECTED_FIELD_COUNT}，"
                f"实际为 {collection['field_count']}"
            ),
            "structure_validation",
        )
    if not collection.get("auto_id") is False:
        _fail(report, "auto_id 必须为 False", "structure_validation")
    pk_names = sorted(
        str(field["name"]) for field in fields if field.get("primary_key")
    )
    if pk_names != ["vector_doc_id"]:
        _fail(
            report,
            f"主键必须是唯一 vector_doc_id，实际为 {pk_names}",
            "structure_validation",
        )
    if actual_dimension != EXPECTED_EMBEDDING_DIMENSION:
        _fail(
            report,
            (
                f"embedding dim 应为 {EXPECTED_EMBEDDING_DIMENSION}，"
                f"实际为 {actual_dimension}"
            ),
            "structure_validation",
        )
    field_names = {str(field["name"]) for field in fields}
    for json_field in ("metadata_ext", "source_ext", "versions_ext"):
        datatype = next(
            (
                str(field["datatype"])
                for field in fields
                if str(field["name"]) == json_field
            ),
            None,
        )
        if datatype != "JSON":
            _fail(
                report,
                f"{json_field} 必须是 JSON，实际为 {datatype}",
                "structure_validation",
            )
    if "chunk_id" in field_names:
        _fail(report, "Schema 不得包含已废弃的 chunk_id", "structure_validation")

    index_names = sorted(
        str(name) for name in client.list_indexes(collection_name=collection_name)
    )
    index_descriptions = {
        name: client.describe_index(collection_name=collection_name, index_name=name)
        for name in index_names
    }
    collection["indexes"] = _index_descriptions(index_descriptions)
    _verify_index_set(client, report)
    _pass(report, "structure_validation")


def _verify_index_set(client: Any, report: dict[str, Any]) -> None:
    """验证索引集合：1 vector (HNSW/COSINE/M=16/efConstruction=200) + 3 scalar。"""

    expected = {
        "index_name": "embedding_vector",
        "field_name": "embedding",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    index_info = report["collection"]["indexes"]
    index_names = sorted(index_info)
    if len(index_names) != _FROZEN_EXPECTED_INDEX_COUNT:
        _fail(
            report,
            (
                f"索引总数应为 {_FROZEN_EXPECTED_INDEX_COUNT}，"
                f"实际为 {len(index_names)}: {index_names}"
            ),
            "structure_validation",
        )

    vector_field = None
    for index_name in index_names:
        info = index_info[index_name]
        if info.get("field_name") == "embedding":
            vector_field = info
            break
    if vector_field is None:
        _fail(report, "缺少 embedding 向量索引", "structure_validation")
        return
    # describe_index 的 index_name 字段与服务端列名一致；params 按字符串数字归一化。
    actual_vector = {
        "index_name": vector_field.get("index_name"),
        "field_name": vector_field.get("field_name"),
        "index_type": vector_field.get("index_type"),
        "metric_type": vector_field.get("metric_type"),
        "params": {
            str(k): _normalized_index_param(v)
            for k, v in (vector_field.get("params") or {}).items()
        },
    }
    _assert_index_matches(
        actual_vector,
        expected,
        "embedding 向量索引",
        report,
    )

    scalar_fields = set()
    for index_name in index_names:
        info = index_info[index_name]
        field_name = info.get("field_name")
        if field_name == "embedding":
            continue
        scalar_fields.add(field_name)
        if info.get("index_type") != "INVERTED":
            _fail(
                report,
                f"标量索引 {field_name} 类型必须是 INVERTED",
                "structure_validation",
            )
    if scalar_fields != set(_FROZEN_SCALAR_INDEX_FIELDS):
        _fail(
            report,
            (
                "Scalar Index 必须恰好为 "
                f"{sorted(_FROZEN_SCALAR_INDEX_FIELDS)}，实际为 {sorted(scalar_fields)}"
            ),
            "structure_validation",
        )


def _assert_index_matches(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
    report: dict[str, Any],
) -> None:
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key == "params" and isinstance(actual_value, dict) and isinstance(expected_value, dict):
            actual_value = {k: _normalized_index_param(v) for k, v in actual_value.items()}
            expected_value = {k: _normalized_index_param(v) for k, v in expected_value.items()}
            if actual_value != expected_value:
                mismatches.append(f"{key}={actual_value} (expected {expected_value})")
        elif actual_value != expected_value:
            mismatches.append(f"{key}={actual_value} (expected {expected_value})")
    if mismatches:
        _fail(
            report,
            f"{label} 不一致: " + "; ".join(mismatches),
            "structure_validation",
        )


def _snapshot_v13(client: Any, report: dict[str, Any]) -> None:
    """创建 v14 前对旧 v13 做只读 Schema/Index/行数快照。"""

    report["v13"]["before"] = _v13_state(client)


def _verify_v13_unchanged(client: Any, report: dict[str, Any]) -> None:
    """只读确认旧 v13 仍存在且本脚本未改动它。"""

    _begin(report, "v13_integrity")
    after = _v13_state(client)
    report["v13"].update(after)
    before = report["v13"].get("before")
    if before is not None and before != after:
        _fail(
            report,
            f"旧 Collection {_PROTECTED_V13} 在创建前后结构发生变化",
            "v13_integrity",
        )
    _pass(report, "v13_integrity")


def _v13_state(client: Any) -> dict[str, Any]:
    """返回 v13 的只读结构摘要（exists / schema / index / row_count）。"""

    if not client.has_collection(collection_name=_PROTECTED_V13):
        return {"exists": False, "auto_id": None, "field_count": None, "pk": [], "index_names": [], "row_count": None}
    description = client.describe_collection(collection_name=_PROTECTED_V13)
    fields = [
        field
        for field in description.get("fields", [])
        if isinstance(field, Mapping) and field.get("name")
    ]
    pk = sorted(
        str(field.get("name"))
        for field in description.get("fields", [])
        if isinstance(field, Mapping) and field.get("is_primary")
    )
    index_names = sorted(
        str(name) for name in client.list_indexes(collection_name=_PROTECTED_V13)
    )
    row_count = None
    if pk:
        rows = client.query(
            collection_name=_PROTECTED_V13,
            filter="",
            output_fields=pk,
            limit=16384,
            consistency_level="Strong",
        )
        if isinstance(rows, list):
            row_count = len(rows)
    return {
        "exists": True,
        "auto_id": bool(description.get("auto_id", False)),
        "enable_dynamic_field": bool(description.get("enable_dynamic_field", False)),
        "field_count": len(fields),
        "pk": pk,
        "index_names": index_names,
        "row_count": row_count,
    }


def _index_descriptions(
    descriptions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(descriptions):
        description = descriptions[name]
        params = description.get("params")
        params = params if isinstance(params, Mapping) else {}
        # PyMilvus describe_index 对向量索引把 M/efConstruction 放在顶层
        # （部分版本放进 params）；统一收集到 params 后再归一化比较。
        merged: dict[str, object] = {str(k): v for k, v in params.items()}
        for key in ("M", "efConstruction"):
            if key in description and key not in merged:
                merged[key] = description[key]
        params = {str(k): _json_value(v) for k, v in merged.items()}
        result[str(name)] = {
            "field_name": str(description.get("field_name", "")),
            "index_name": str(description.get("index_name", "") or ""),
            "index_type": str(description.get("index_type", "")).upper(),
            "metric_type": str(description.get("metric_type", "")).upper(),
            "params": params,
        }
    return result


def _normalized_index_param(value: object) -> object:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


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


def _schema_summary(description: Mapping[str, Any]) -> dict[str, dict[str, object]]:
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
    return {"fields": fields}


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
            "error": {"type": "VerificationFailure", "message": message},
        }
    )
    raise VerificationFailure(message)


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
    for secret in (
        os.environ.get("MILVUS_TOKEN"),
        os.environ.get("MILVUS_PASSWORD"),
        os.environ.get("MYSQL_PASSWORD"),
        os.environ.get("EMBEDDING_API_KEY"),
    ):
        if secret:
            message = message.replace(secret, "<redacted>")
    return {"type": type(exc).__name__, "message": message[:1000]}


def _error_message(exc: Exception) -> str:
    return _safe_error(exc)["message"]


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


def _new_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/verify_v14_collection.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": "FAIL",
        "current_stage": None,
        "stages": {
            "precondition": "NOT_RUN",
            "existence_check": "NOT_RUN",
            "creation": "NOT_RUN",
            "structure_validation": "NOT_RUN",
            "v13_integrity": "NOT_RUN",
        },
        "collection": {
            "name": args.collection,
            "server_version": None,
            "existed_before": None,
            "created_now": False,
            "settings": None,
            "auto_id": None,
            "enable_dynamic_field": None,
            "schema": None,
            "expected_schema": None,
            "field_count": None,
            "actual_vector_dimension": None,
            "schema_comparison": None,
            "indexes": {},
            "index_comparison": None,
        },
        "v13": {
            "collection": _PROTECTED_V13,
            "before": None,
            "exists": False,
            "auto_id": None,
            "enable_dynamic_field": None,
            "field_count": None,
            "pk": [],
            "index_names": [],
            "row_count": None,
        },
        "failures": [],
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help=f"目标 DEV Milvus Collection（默认：{_DEFAULT_COLLECTION}）",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_DEFAULT_REPORT,
        help="JSON 验收报告路径（默认：reports/v14_collection_verify.json）",
    )
    return parser.parse_args(argv)


def _print_summary(report: Mapping[str, Any], report_path: Path | None) -> None:
    overall = report["overall"]
    print(f"\nv14 Collection verification: {overall}")
    for stage, status in report["stages"].items():
        print(f"[{status}] {stage}")
    collection = report["collection"]
    print("--- v14 ---")
    print(f"collection: {collection['name']}")
    print(f"existed_before: {collection.get('existed_before')}")
    print(f"created_now: {collection.get('created_now')}")
    print(f"field_count: {collection.get('field_count')}")
    print(f"auto_id: {collection.get('auto_id')}")
    print(f"vector_dim: {collection.get('actual_vector_dimension')}")
    print(f"indexes: {sorted((collection.get('indexes') or {}).keys())}")
    print("--- v13 ---")
    print(f"exists: {report['v13'].get('exists')}")
    print(f"field_count: {report['v13'].get('field_count')}")
    print(f"pk: {report['v13'].get('pk')}")
    print(f"index_names: {report['v13'].get('index_names')}")
    print(f"row_count: {report['v13'].get('row_count')}")
    if report["failures"]:
        print(f"failures: {len(report['failures'])}")
        for failure in report["failures"]:
            print(f"- {failure}")
    if report_path is not None:
        print(f"report: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())