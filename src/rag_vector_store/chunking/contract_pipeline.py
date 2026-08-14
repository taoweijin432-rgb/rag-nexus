"""专家 Chunk 4.0.0 契约的校验、资格判断与向量化消费链。

本模块负责把正式 Contract 校验成可消费记录、执行 Eligibility 判断，并把唯一的
``chunk_text`` 交给 EmbeddingModel。它不负责输入 Mapping 适配、MySQL 连接、任务
状态持久化、增量游标或 Milvus 写入。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Any

from pydantic import ValidationError

from expert_contracts.models import (
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
)

from ..embedding import EmbeddingModel
from ..milvus.contract import contract_milvus_field_specs

EXPECTED_EMBEDDING_DIMENSION = 1024

_CHUNK_TEXT_LIMITS = {
    "profile_summary": 600,
    "employment": 900,
    "project": 1000,
}
_CHUNK_SUBTYPES = {
    "profile_summary": {None},
    "employment": {None, "responsibility", "achievement", "segment"},
    "project": {None, "context", "action", "result", "segment"},
}
_FORBIDDEN_TEXT_PATTERN = re.compile(
    "[\\x00\\u200b-\\u200f\\u202a-\\u202e\\u2066-\\u2069\\ufeff]"
)
_PII_PATTERNS = (
    ("邮箱", re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")),
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("身份证号", re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")),
    ("银行卡号", re.compile(r"(?<!\d)\d{13,19}(?!\d)")),
    ("原始 URL", re.compile(r"(?i)\bhttps?://[^\s]+")),
    (
        "IP 地址",
        re.compile(
            r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"
        ),
    ),
    (
        "微信号",
        re.compile(
            r"(?:微信|wechat|wx)\s*[:：]\s*[A-Za-z][-_A-Za-z0-9]{5,19}",
            re.IGNORECASE,
        ),
    ),
)


class ContractValidationError(ValueError):
    """输入记录不符合 4.0.0 数据契约。"""


class EmbeddingEligibilityError(ValueError):
    """记录通过契约校验，但不允许进入正式向量化链路。"""


@dataclass(frozen=True)
class SkippedContractChunk:
    """未进入向量消费的契约记录及安全原因（以 vector_doc_id 定位）。"""

    vector_doc_id: str
    reason: str


@dataclass(frozen=True)
class ContractEmbeddingResult:
    """一批正式输入契约的向量化结果。"""

    embedded_chunks: list[EmbeddedContractPreparedChunk]
    skipped_chunks: list[SkippedContractChunk]

    @property
    def eligible_count(self) -> int:
        """返回实际进入 Embedding 调用的记录数。"""

        return len(self.embedded_chunks)

    @property
    def skipped_count(self) -> int:
        """返回因资格条件未满足而跳过的记录数。"""

        return len(self.skipped_chunks)


class ContractValidator:
    """执行结构约束和无法由 Schema 表达的跨字段业务校验。

    输入可以是原始 Mapping 或严格的 ``ExpertChunkEmbeddingInput``；成功时返回
    ``ContractPreparedChunk``，失败则抛出不携带正文的 ``ContractValidationError``。
    ``content_hash`` 会在消费端重新按 ``chunk_text`` 校验，但 ``source_hash``
    不在这里重算，因为它属于上游来源事实的责任边界。
    """

    def validate_mapping(self, value: Mapping[str, Any]) -> ContractPreparedChunk:
        """先执行 Pydantic 生成的 JSON Schema 等价校验，再执行跨字段校验。"""

        try:
            record = ExpertChunkEmbeddingInput.model_validate(value)
        except ValidationError as exc:
            raise ContractValidationError(_safe_validation_message(exc)) from exc
        return self.validate(record)

    def validate(
        self,
        value: ExpertChunkEmbeddingInput,
    ) -> ContractPreparedChunk:
        """校验一条已完成结构解析的契约记录。"""

        if not isinstance(value, ExpertChunkEmbeddingInput):
            raise TypeError("ContractValidator 只接受 ExpertChunkEmbeddingInput")

        errors: list[str] = []
        text = value.chunk_text
        metadata = value.metadata

        if value.chunk_index >= value.chunk_total:
            errors.append("chunk_index 必须小于 chunk_total")
        if value.chunk_subtype not in _CHUNK_SUBTYPES[value.chunk_type]:
            errors.append("chunk_subtype 与 chunk_type 不兼容")
        if not text.strip():
            errors.append("chunk_text 去除首尾空白后不能为空")
        if "\r" in text:
            errors.append("chunk_text 不允许包含回车符")
        if _FORBIDDEN_TEXT_PATTERN.search(text):
            errors.append("chunk_text 包含禁止的控制字符")
        if len(text) != value.char_count:
            errors.append("char_count 与 chunk_text 实际字符数不一致")
        if len(text) > _CHUNK_TEXT_LIMITS[value.chunk_type]:
            errors.append(f"{value.chunk_type} 的 chunk_text 超过字符上限")
        if len(text.encode("utf-8")) > 8192:
            errors.append("chunk_text 的 UTF-8 字节数超过 8192")

        # 这是下游防御性校验：即使上游已经填充 content_hash，也必须用实际文本
        # 重新验证，防止错误版本进入 Embedding 和幂等状态。
        actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_hash != value.content_hash:
            errors.append("content_hash 与 chunk_text 的 SHA-256 不一致")

        if metadata.is_current is True and metadata.end_month is not None:
            errors.append("is_current=true 时 end_month 必须为 null")

        # origin_type × chunk_type 与 verification_status × chunk_type 填充规则。
        # verification_status=rejected 是 profile_summary 的合法值，不属于
        # Contract 非法，因此这里允许；由 Eligibility 决定是否进入向量消费。
        if value.chunk_type == "profile_summary":
            if metadata.origin_type != "compressed_from_source":
                errors.append(
                    "profile_summary 的 origin_type 必须是 compressed_from_source"
                )
            if metadata.verification_status not in {"source_summary", "rejected"}:
                errors.append(
                    "profile_summary 的 verification_status "
                    "必须是 source_summary 或 rejected"
                )
        else:
            if metadata.origin_type != "source_parsed":
                errors.append(
                    f"{value.chunk_type} 的 origin_type 必须是 source_parsed"
                )
            if metadata.verification_status != "verified":
                errors.append(
                    f"{value.chunk_type} 的 verification_status 必须是 verified"
                )
            if metadata.eligible_for_recommendation is not None:
                errors.append(
                    "eligible_for_recommendation 只能出现在 profile_summary"
                )

        # 正式表里 start_month / is_current / company_name_raw / company_name_norm /
        # role_name_raw / project_name 均为 NULLABLE，文档未规定 employment/project
        # 必须因所属 chunk_type 强制这些字段非空。因此这里不校验旧 v2 的必填规则。

        # PII 检查在 Embedding 前执行，避免敏感信息进入外部模型；错误只返回
        # 安全摘要，不把命中的原文写入日志或异常。
        for pii_name, pattern in _PII_PATTERNS:
            if pattern.search(text):
                errors.append(f"chunk_text 疑似包含{pii_name}")

        if errors:
            raise ContractValidationError("；".join(errors))
        return ContractPreparedChunk.model_validate(value.model_dump())


class EmbeddingEligibilityValidator:
    """决定一条合法 Contract 是否进入正常向量消费。

    Eligibility skip 与 Contract invalid 是两类不同结果：前者是记录结构合法但
    当前不应进入正式向量索引（``is_active=false`` 或
    ``verification_status=rejected``），后者表示输入契约本身不可信。这里不执行
    删除或 tombstone，生命周期动作由上层后续流程处理。
    """

    def get_skip_reason(self, record: ContractPreparedChunk) -> str | None:
        """返回跳过原因；返回 None 表示允许进入向量消费。

        只保留两个规则：``is_active=false`` 与 ``verification_status=rejected``。
        ``weight_tier`` 是检索权重字段、``eligible_for_recommendation`` 是 profile
        推荐资格字段，均不影响是否向量化。
        """

        if not record.is_active:
            return "is_active=false"
        if record.metadata.verification_status == "rejected":
            return "verification_status=rejected 不允许进入正常向量消费"
        return None

    def is_eligible(self, record: ContractPreparedChunk) -> bool:
        """判断记录是否允许进入 Embedding。"""

        return self.get_skip_reason(record) is None

    def validate(self, record: ContractPreparedChunk) -> None:
        """不满足资格时抛出明确异常。"""

        reason = self.get_skip_reason(record)
        if reason is not None:
            raise EmbeddingEligibilityError(reason)


class ContractEmbeddingWorker:
    """只把通过校验且符合资格的 ``chunk_text`` 交给 EmbeddingModel。

    Worker 负责批量调用和响应防御性校验，输出带模型空间并保留上游
    ``vector_doc_id`` 的 Embedded Contract。它不负责任务级幂等；是否跳过调用由
    ``ContractIngestionService`` 在调用 Worker 之前决定。
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        embedding_model_alias: str | None = None,
        embedding_model_version: str | None = None,
        expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
        validator: ContractValidator | None = None,
        eligibility_validator: EmbeddingEligibilityValidator | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(expected_dimension, bool) or expected_dimension < 1:
            raise ValueError("expected_dimension 必须是正整数")
        self.embedding_model = embedding_model
        self.embedding_model_alias = embedding_model_alias or _required_model_text(
            embedding_model,
            "model_alias",
        )
        self.embedding_model_version = embedding_model_version or _required_model_text(
            embedding_model,
            "model_version",
        )
        self.expected_dimension = expected_dimension
        self.validator = validator or ContractValidator()
        self.eligibility_validator = (
            eligibility_validator or EmbeddingEligibilityValidator()
        )
        self.clock = clock or _utc_now

    def embed(
        self,
        records: Iterable[ExpertChunkEmbeddingInput],
    ) -> ContractEmbeddingResult:
        """校验、筛选并批量向量化；不会改写或拼接 chunk_text。"""

        eligible: list[ContractPreparedChunk] = []
        skipped: list[SkippedContractChunk] = []
        for record in records:
            prepared = self.validator.validate(record)
            reason = self.eligibility_validator.get_skip_reason(prepared)
            if reason is None:
                eligible.append(prepared)
            else:
                skipped.append(
                    SkippedContractChunk(
                        vector_doc_id=prepared.vector_doc_id, reason=reason
                    )
                )

        # 没有 eligible 记录时不触碰外部 Embedding 服务，保证 inactive/不合格
        # 输入不会产生无意义调用或成本。
        if not eligible:
            return ContractEmbeddingResult(
                embedded_chunks=[],
                skipped_chunks=skipped,
            )

        # Embedding 的语义输入严格只有 chunk_text；Metadata、Source、Versions
        # 在 Milvus 行映射中保存，不能拼接进文本导致向量语义和过滤字段耦合。
        texts = [record.chunk_text for record in eligible]
        raw_embeddings = list(self.embedding_model.embed(texts))
        if len(raw_embeddings) != len(eligible):
            raise ValueError(
                "Embedding 返回数量与符合资格的输入记录数量不一致"
            )

        embedded_at = _format_embedded_at(self.clock())
        embedded: list[EmbeddedContractPreparedChunk] = []
        # zip 保持 Reader/模型返回顺序一一对应；数量已在上方检查，避免静默错配。
        for record, raw_embedding in zip(eligible, raw_embeddings):
            embedding = _validate_embedding(
                raw_embedding,
                expected_dimension=self.expected_dimension,
                vector_doc_id=record.vector_doc_id,
            )
            embedded.append(
                EmbeddedContractPreparedChunk(
                    **record.model_dump(),
                    embedding=embedding,
                    embedding_dim=len(embedding),
                    embedding_model_alias=self.embedding_model_alias,
                    embedding_model_version=self.embedding_model_version,
                    embedded_at=embedded_at,
                )
            )

        return ContractEmbeddingResult(
            embedded_chunks=embedded,
            skipped_chunks=skipped,
        )


def expected_contract_output_fields(
    vector_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
) -> dict[str, dict[str, object]]:
    """返回完整保存契约输入与向量化输出所需的最小 Milvus 字段。"""

    return {
        spec.name: {
            "datatype": spec.datatype,
            "nullable": spec.nullable,
            "primary_key": spec.primary_key,
            "max_length": spec.max_length,
            "dim": spec.dim,
        }
        for spec in contract_milvus_field_specs(vector_dimension)
    }


def compare_contract_output_schema(
    description: Mapping[str, Any],
    vector_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
) -> dict[str, object]:
    """只读比较现有 Collection 是否能无损承载正式契约输出。

    该比较是写入前的 fail-closed gate：缺字段、类型/nullable/dimension 不兼容
    都只返回差异，由 Repository 或验证脚本阻止写入，不尝试自动迁移现有 Collection。
    """

    expected = expected_contract_output_fields(vector_dimension)
    actual = {
        field.get("name"): field
        for field in description.get("fields", [])
        if isinstance(field, Mapping) and field.get("name")
    }
    missing_fields = sorted(set(expected) - set(actual))
    incompatible_fields: dict[str, dict[str, object]] = {}

    for field_name in sorted(set(expected).intersection(actual)):
        expected_field = expected[field_name]
        actual_field = actual[field_name]
        differences: dict[str, object] = {}
        actual_datatype = _milvus_datatype_name(actual_field.get("type"))
        if actual_datatype != expected_field["datatype"]:
            differences["datatype"] = {
                "expected": expected_field["datatype"],
                "actual": actual_datatype,
            }

        expected_nullable = bool(expected_field.get("nullable", False))
        actual_nullable = bool(actual_field.get("nullable", False))
        if actual_nullable != expected_nullable:
            differences["nullable"] = {
                "expected": expected_nullable,
                "actual": actual_nullable,
            }

        params = actual_field.get("params") or {}
        expected_dim = expected_field.get("dim")
        if expected_dim is not None and _as_int(params.get("dim")) != expected_dim:
            differences["dim"] = {
                "expected": expected_dim,
                "actual": _as_int(params.get("dim")),
            }

        expected_max_length = expected_field.get("max_length")
        actual_max_length = _as_int(params.get("max_length"))
        if (
            expected_max_length is not None
            and actual_max_length is not None
            and actual_max_length < expected_max_length
        ):
            differences["max_length"] = {
                "minimum": expected_max_length,
                "actual": actual_max_length,
            }
        if differences:
            incompatible_fields[field_name] = differences

    return {
        "compatible": not missing_fields and not incompatible_fields,
        "missing_fields": missing_fields,
        "incompatible_fields": incompatible_fields,
    }


def _validate_embedding(
    raw_embedding: Sequence[float],
    *,
    expected_dimension: int,
    vector_doc_id: str,
) -> list[float]:
    """校验单条向量的类型、有限性和固定维度。

    模型 API 的响应不视为可信 Contract；这里在构造输出对象前再次拒绝字符串、
    bool、NaN/Infinity、空向量和错误维度，避免非法值进入 Milvus。
    """

    if isinstance(raw_embedding, (str, bytes)) or not isinstance(
        raw_embedding,
        Sequence,
    ):
        raise TypeError(f"vector_doc_id {vector_doc_id!r} 的 embedding 必须是数值序列")
    if not raw_embedding:
        raise ValueError(f"vector_doc_id {vector_doc_id!r} 的 embedding 不能为空")

    embedding: list[float] = []
    for value in raw_embedding:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"vector_doc_id {vector_doc_id!r} 的 embedding 元素必须是数值"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"vector_doc_id {vector_doc_id!r} 的 embedding 包含 NaN 或 Infinity"
            )
        embedding.append(number)
    if len(embedding) != expected_dimension:
        raise ValueError(
            f"vector_doc_id {vector_doc_id!r} 的 embedding dimension 不匹配: "
            f"expected {expected_dimension}, got {len(embedding)}"
        )
    return embedding


def _required_model_text(model: EmbeddingModel, attribute: str) -> str:
    """从模型读取非空别名或版本，不猜测正式版本值。"""

    value = getattr(model, attribute, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"必须显式提供 embedding_{attribute}")
    return value.strip()


def _format_embedded_at(value: datetime) -> str:
    """把带时区时间转换为 ISO 8601 字符串。"""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("embedded_at 时钟必须返回带时区 datetime")
    return value.isoformat()


def _utc_now() -> datetime:
    """返回带时区的当前 UTC 时间。"""

    return datetime.now(timezone.utc)


def _safe_validation_message(exc: ValidationError) -> str:
    """生成不包含输入值和完整 Chunk 正文的结构校验摘要。"""

    messages = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(item) for item in error["loc"])
        messages.append(f"{location}: {error['msg']}")
    return "；".join(messages)


def _milvus_datatype_name(value: object) -> str:
    """兼容 PyMilvus 枚举、整数和字符串形式的数据类型。"""

    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            from pymilvus import DataType

            return DataType(value).name.upper()
        except (ImportError, ValueError):
            return str(value)
    text = str(value).upper()
    for datatype in ("FLOAT_VECTOR", "VARCHAR", "INT64", "BOOL", "JSON"):
        if datatype in text:
            return datatype
    return text


def _as_int(value: object) -> int | None:
    """将 Milvus Schema 参数安全转换为整数。"""

    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
