"""Milvus Schema、Index 和模型空间的纯兼容性判断。

这里刻意不持有 Milvus Client，也不执行创建、修改或删除。模块只把服务端只读
描述归一化后与本地 Contract 定义比较，生成可解释的差异；Repository/脚本据此
执行 fail-closed 的 zero-write gate。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .indexes import IndexSpec
from .schemas.specs import MilvusFieldSpec


@dataclass(frozen=True)
class CollectionCompatibilityResult:
    """不执行 Milvus 操作的兼容性判断结果。

    ``details`` 保存缺失、意外、类型、索引或模型空间差异，便于安全报告说明
    为什么阻止写入，而不是把环境问题伪装成业务成功。
    """

    compatible: bool
    details: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """返回现有脚本和公共方法使用的稳定字典结构。"""

        return {"compatible": self.compatible, **dict(self.details)}


def compare_schema_specs(
    description: Mapping[str, Any],
    specs: Sequence[MilvusFieldSpec],
) -> CollectionCompatibilityResult:
    """比较服务端字段描述与一组 Schema 定义。

    字段集合、类型、nullable、VARCHAR 容量、向量维度和 Collection 选项任一不
    一致都会返回 incompatible；该函数不尝试修复或迁移现有 Collection。
    """

    expected_specs = {spec.name: spec for spec in specs}
    actual_fields = field_map(description)
    missing_fields = sorted(set(expected_specs) - set(actual_fields))
    unexpected_fields = sorted(set(actual_fields) - set(expected_specs))
    changed_fields: dict[str, dict[str, object]] = {}

    for field_name in sorted(set(expected_specs).intersection(actual_fields)):
        expected = expected_specs[field_name].summary()
        actual = field_summary(actual_fields[field_name])
        if actual != expected:
            changed_fields[field_name] = {
                "expected": expected,
                "actual": actual,
            }

    collection_options = collection_option_mismatches(description)
    return CollectionCompatibilityResult(
        compatible=not (
            missing_fields
            or unexpected_fields
            or changed_fields
            or collection_options
        ),
        details={
            "missing_fields": missing_fields,
            "unexpected_fields": unexpected_fields,
            "changed_fields": changed_fields,
            "collection_options": collection_options,
        },
    )


def compare_contract_indexes(
    index_names: Sequence[str],
    descriptions: Mapping[str, Mapping[str, Any]],
    expected_specs: Sequence[IndexSpec],
) -> CollectionCompatibilityResult:
    """比较 Contract 向量索引和完整 scalar index set。

    向量索引比较 metric/params，scalar 索引比较字段与类型；缺少或多出索引都
    被报告，避免 Repository 在未知索引集合上继续写入。
    """

    expected = {
        str(item["index_name"]): item
        for item in (spec.summary(include_name=True) for spec in expected_specs)
        if item.get("index_name") is not None
    }
    actual_names = set(index_names)
    missing_indexes = sorted(set(expected) - actual_names)
    unexpected_indexes = sorted(actual_names - set(expected))
    changed_indexes: dict[str, dict[str, object]] = {}

    for index_name in sorted(set(expected).intersection(actual_names)):
        description = descriptions[index_name]
        expected_index = expected[index_name]
        if expected_index["field_name"] == "embedding":
            actual = index_summary(description, expected_index["params"])
            comparable_expected = {
                "field_name": expected_index["field_name"],
                "index_type": expected_index["index_type"],
                "metric_type": expected_index["metric_type"],
                "params": expected_index["params"],
            }
        else:
            actual = {
                "field_name": str(description.get("field_name", "")),
                "index_type": str(description.get("index_type", "")).upper(),
                "metric_type": "",
                "params": {},
            }
            comparable_expected = {
                "field_name": expected_index["field_name"],
                "index_type": expected_index["index_type"],
                "metric_type": "",
                "params": {},
            }
        if actual != comparable_expected:
            changed_indexes[index_name] = {
                "expected": comparable_expected,
                "actual": actual,
            }

    return CollectionCompatibilityResult(
        compatible=not (missing_indexes or unexpected_indexes or changed_indexes),
        details={
            "missing_indexes": missing_indexes,
            "unexpected_indexes": unexpected_indexes,
            "changed_indexes": changed_indexes,
        },
    )


def compare_model_space(
    rows: Sequence[Mapping[str, Any]],
    *,
    embedding_model_alias: str,
    embedding_model_version: str,
    embedding_dimension: int,
) -> CollectionCompatibilityResult:
    """判断已读取 Entity 是否属于同一个 alias/version/dimension 空间。

    维度相同不等于向量空间相同，因此 alias、version、dimension 必须同时匹配；
    这是防止两个同维模型混写同一 Collection 的只读保护。
    """

    mismatched_rows = [
        index
        for index, row in enumerate(rows)
        if (
            row.get("embedding_model_alias") != embedding_model_alias
            or row.get("embedding_model_version") != embedding_model_version
            or row.get("embedding_dim") != embedding_dimension
        )
    ]
    return CollectionCompatibilityResult(
        compatible=not mismatched_rows,
        details={"mismatched_rows": mismatched_rows},
    )


def field_map(description: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """将 Collection description 的字段列表归一化为名称映射。"""

    return {
        field.get("name"): field
        for field in description.get("fields", [])
        if isinstance(field, Mapping) and field.get("name")
    }


def field_summary(field: Mapping[str, Any]) -> dict[str, object]:
    """归一化 PyMilvus 字段描述。"""

    params = field.get("params")
    normalized_params = params if isinstance(params, Mapping) else {}
    datatype = datatype_name(field.get("type"))
    return {
        "field": str(field.get("name")),
        "datatype": datatype,
        "nullable": bool(field.get("nullable", False)),
        "primary_key": bool(field.get("is_primary", False)),
        "max_length": (
            optional_int(normalized_params.get("max_length"))
            if datatype == "VARCHAR"
            else None
        ),
        "dim": (
            optional_int(normalized_params.get("dim"))
            if datatype == "FLOAT_VECTOR"
            else None
        ),
    }


def collection_option_mismatches(
    description: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """检查当前所有正式 Schema 都要求的 Collection 选项。"""

    options: dict[str, dict[str, object]] = {}
    actual_auto_id = bool(description.get("auto_id", False))
    if actual_auto_id:
        options["auto_id"] = {"expected": False, "actual": actual_auto_id}
    actual_dynamic = bool(description.get("enable_dynamic_field", False))
    if actual_dynamic:
        options["enable_dynamic_field"] = {
            "expected": False,
            "actual": actual_dynamic,
        }
    return options


def datatype_name(value: Any) -> str:
    """兼容 DataType 枚举、整数和字符串形式的字段类型。"""

    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            from pymilvus import DataType

            return DataType(value).name
        except (ImportError, ValueError):
            return str(value)
    return str(value).rsplit(".", maxsplit=1)[-1].upper()


def index_summary(
    description: Mapping[str, Any],
    expected_params: object,
) -> dict[str, object]:
    """归一化 describe_index 返回值，并仅比较已配置参数。"""

    nested_params = description.get("params")
    param_source: dict[str, Any] = (
        dict(nested_params) if isinstance(nested_params, Mapping) else {}
    )
    param_source.update(description)
    configured_params = expected_params if isinstance(expected_params, Mapping) else {}
    actual_params = {
        str(name): normalized_index_param(param_source.get(name))
        for name in configured_params
    }
    return {
        "field_name": str(description.get("field_name", "")),
        "index_type": str(description.get("index_type", "")).upper(),
        "metric_type": str(description.get("metric_type", "")).upper(),
        "params": actual_params,
    }


def normalized_index_param(value: Any) -> object:
    """将服务端字符串数字转换为整数，便于与配置比较。"""

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def optional_int(value: Any) -> int | None:
    """将可选的 Schema 数字参数转换为整数。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
