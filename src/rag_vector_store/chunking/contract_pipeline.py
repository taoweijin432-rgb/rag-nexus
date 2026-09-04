"""专家 Chunk 4.1.0 契约的校验、资格判断与向量化消费链。

本模块负责把正式 Contract 校验成可消费记录、执行 Eligibility 判断，并把唯一的
``chunk_text`` 交给 EmbeddingModel。它不负责输入 Mapping 适配、MySQL 连接、任务
状态持久化、增量游标或 Milvus 写入。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from expert_contracts.models import (
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    EmploymentMetadata,
    ExpertChunkEmbeddingInput,
)

from ..constants import FORMAL_EMBEDDING_DIMENSION
from ..core.embedding import DomainEmbeddingWorker
from ..domains import ExpertContractAdapter
from ..embedding import EmbeddingModel
from ..milvus.compatibility import compare_schema_specs
from ..milvus.schemas.contract_v15 import v15_milvus_field_specs

EXPECTED_EMBEDDING_DIMENSION = FORMAL_EMBEDDING_DIMENSION

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
# PII 检查已下沉到上游 expert_index/normalize.py 的 clean_text，
# Contract 层不再重复校验 PII（避免误判标准编号等业务数字，且职责归上游清洗）。


class ContractValidationError(ValueError):
    """输入记录不符合 4.1.0 数据契约。"""


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

    调用时机：
        ``ContractEmbeddingWorker.embed`` 在每条记录进入 Embedding 前调用；
        上游内存同步的预检统计也复用本类。

    输入：
        可以是原始 Mapping 或严格的 ``ExpertChunkEmbeddingInput``。

    输出：
        校验通过时返回 ``ContractPreparedChunk``（由输入转换而来，携带全部
        Contract 字段，是 Eligibility 与 Milvus 投影的下一站）。

    失败：
        抛出不携带正文的 ``ContractValidationError``（invalid）。

    职责：
        判断“数据是否合法/可信”，不是判断“是否应该 Embedding”——后者属于
        ``EmbeddingEligibilityValidator``。``content_hash`` 会在消费端重新按
        ``chunk_text`` 校验，但 ``source_hash`` 不在这里重算，因为它属于上游
        来源事实的责任边界。
    """

    def validate_mapping(self, value: Mapping[str, Any]) -> ContractPreparedChunk:
        """先执行 Pydantic 生成的 JSON Schema 等价校验，再执行跨字段校验。

        调用方：需要从原始 dict/Mapping 构造正式记录的外部适配层（如测试 Fixture）。
        """

        try:
            record = ExpertChunkEmbeddingInput.model_validate(value)
        except ValidationError as exc:
            raise ContractValidationError(_safe_validation_message(exc)) from exc
        return self.validate(record)

    def validate(
        self,
        value: ExpertChunkEmbeddingInput,
    ) -> ContractPreparedChunk:
        """校验一条已完成结构解析的契约记录。

        “合法”的范围（当前正式校验类别，具体规则见函数体）：

        - chunk 组织关系：chunk_index < chunk_total、chunk_subtype × chunk_type 兼容；
        - chunk_text：非空、无回车/控制字符、按 chunk_type 上限、char_count 一致、
          UTF-8 字节量；
        - content_hash：按实际 chunk_text 重算 SHA-256 并比对；
        - PII / 控制字符：PII 模式与禁止字符检测（Embedding 前执行，避免敏感
          信息进入外部模型）；
        - 时间跨字段约束：is_current=true 时 end_month 必须为 null；
        - origin_type × chunk_type、verification_status × chunk_type 填充规则；
        - recommendation 字段适用性：eligible_for_recommendation 只能出现在
          profile_summary。

        输出类型：
            ``ContractPreparedChunk`` —— “已通过消费端 Validator”的阶段类型，
            下一站是 ``EmbeddingEligibilityValidator``。

        分支：
            - invalid：任一规则不满足则抛 ``ContractValidationError``，
              不会进入 Embedding；
            - valid：返回 ContractPreparedChunk。
        """

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

        # 时间跨字段约束：is_current 只存在于 employment Metadata，
        # 只有 employment 需要执行 is_current=true 时 end_month 必须为 null。
        # 用 isinstance 收窄类型，等价于 chunk_type == "employment"
        # （ExpertChunkEmbeddingInput 已强制 metadata 类型匹配 chunk_type）。
        if (
            isinstance(metadata, EmploymentMetadata)
            and metadata.is_current is True
            and metadata.end_month is not None
        ):
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
            # per-type 后 employment/project Metadata 已不含该字段（结构上不可能
            # 存在）；保留 getattr 防御性检查，避免未来回归成 superset。
            if getattr(metadata, "eligible_for_recommendation", None) is not None:
                errors.append(
                    "eligible_for_recommendation 只能出现在 profile_summary"
                )

        # 正式表里 start_month / is_current / company_name_raw / company_name_norm /
        # role_name_raw / project_name 均为 NULLABLE，文档未规定 employment/project
        # 必须因所属 chunk_type 强制这些字段非空。因此这里不校验旧 v2 的必填规则。

        # PII 校验已下沉到上游 expert_index/normalize.py 的 clean_text，
        # Contract 层不再重复校验 PII（避免误判标准编号等业务数字，如 IEC 标准号）。

        if errors:
            raise ContractValidationError("；".join(errors))
        return ContractPreparedChunk.model_validate(value.model_dump())


class EmbeddingEligibilityValidator:
    """决定一条合法 Contract 是否进入正常向量消费。

    调用时机：
        ``ContractEmbeddingWorker.embed`` 对每条已通过 ``ContractValidator`` 的
        ``ContractPreparedChunk`` 调用 ``get_skip_reason``。

    输入：
        ``ContractPreparedChunk``（合法但尚未向量化）。

    输出：
        ``None``（允许 Embedding）或 skip reason 字符串（跳过）。

    当前正式 Eligibility 规则：
        - ``is_active=false`` → skip；
        - ``verification_status=rejected`` → skip。

    边界：
        - rejected 是 profile_summary 的合法 Contract 状态，不是
          ContractValidationError；它只是不进入正式向量索引。
        - ``weight_tier``（检索权重）与 ``eligible_for_recommendation``（推荐资格）
          不参与 Embedding Eligibility。
        - Eligibility skip 与 Contract invalid 是两类不同结果；这里不执行删除或
          tombstone，生命周期动作由上层后续流程处理。
    """

    def get_skip_reason(self, record: ExpertChunkEmbeddingInput) -> str | None:
        """返回跳过原因；返回 None 表示允许进入向量消费。

        参数放宽为 ``ExpertChunkEmbeddingInput``：本方法只访问 ``is_active`` 与
        ``metadata.verification_status``，二者均在基类上；``ContractPreparedChunk``
        作为子类调用者仍有效。正式范围 Consumer 通过 ``ContractConsumerCore`` 注入
        本方法时对 ``ExpertChunkEmbeddingInput`` 直接判定，避免 duck-typing 的类型臆断。
        只保留两个规则：``is_active=false`` 与 ``verification_status=rejected``。
        ``weight_tier`` 是检索权重字段、``eligible_for_recommendation`` 是 profile
        推荐资格字段，均不影响是否向量化。
        """

        if not record.is_active:
            return "is_active=false"
        if record.metadata.verification_status == "rejected":
            return "verification_status=rejected 不允许进入正常向量消费"
        return None

    def is_eligible(self, record: ExpertChunkEmbeddingInput) -> bool:
        """判断记录是否允许进入 Embedding（get_skip_reason 的布尔视图）。"""

        return self.get_skip_reason(record) is None

    def validate(self, record: ExpertChunkEmbeddingInput) -> None:
        """不满足资格时抛出明确异常（供需要 fail-fast 的场景）。"""

        reason = self.get_skip_reason(record)
        if reason is not None:
            raise EmbeddingEligibilityError(reason)


class ContractEmbeddingWorker:
    """只把通过校验且符合资格的 ``chunk_text`` 交给 EmbeddingModel。

    调用时机：
        - ``ContractIngestionService.ingest_one``（单条路径）；
        - ``ContractBatchIngestionService.ingest_batch``（批量路径）；
        - 上游内存同步的预检复用其内部 Validator。

    输入：
        ``Iterable[ExpertChunkEmbeddingInput]`` —— 尚未经过消费端校验/资格判断的记录。

    主要调用：
        1. ``ContractValidator.validate``（每条记录）
        2. ``EmbeddingEligibilityValidator.get_skip_reason``（每条 valid 记录）
        3. ``EmbeddingModel.embed(texts)``（只传 eligible 的 chunk_text）

    输出：
        ``ContractEmbeddingResult`` —— embedded_chunks（``EmbeddedContractPreparedChunk``
        列表，携带 embedding 与模型空间）+ skipped_chunks（``SkippedContractChunk``
        列表，携带 vector_doc_id 与原因）。

    职责边界：
        Worker 只负责批量调用与响应防御性校验，输出带模型空间并保留上游
        ``vector_doc_id`` 的 Embedded Contract；它不负责任务级幂等，也不负责
        Milvus 写入（写入由 Ingestion 层交给 Repository）。
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
        # 专家入口保留原有公开属性和结果模型；实际校验、资格判断、Embedding
        # 与回填委托给不绑定领域的核心，未来领域只需提供自己的 Adapter。
        self._domain_worker = DomainEmbeddingWorker(
            embedding_model,
            ExpertContractAdapter(self.validator, self.eligibility_validator),
            embedding_model_alias=self.embedding_model_alias,
            embedding_model_version=self.embedding_model_version,
            expected_dimension=self.expected_dimension,
            clock=self.clock,
        )

    def embed(
        self,
        records: Iterable[ExpertChunkEmbeddingInput],
    ) -> ContractEmbeddingResult:
        """校验、筛选并批量向量化；不会改写或拼接 chunk_text。

        调用链：
            ExpertChunkEmbeddingInput
            -> ContractValidator.validate()            # invalid 在此抛错
            -> ContractPreparedChunk
            -> EmbeddingEligibilityValidator.get_skip_reason()
            -> eligible / skipped
            -> 提取 eligible.chunk_text               # Embedding 唯一输入
            -> embedding_model.embed(texts)           # 外部模型调用
            -> EmbeddedContractPreparedChunk          # 逐个构造
            -> ContractEmbeddingResult

        关键约束：
            - Embedding 的语义输入严格只有 ``chunk_text``；Metadata、Source、
              Versions、vector_doc_id 一律不拼入 Embedding 文本（它们在 Milvus
              行映射中保存，避免向量语义与过滤字段耦合）。
            - invalid（Contract 不合法）与 skipped（合法但不符合资格）是两种
              不同结果：invalid 在这里直接抛 ``ContractValidationError``，
              skipped 进入结果列表。
            - 没有 eligible 记录时不触碰外部 Embedding 服务。
            - 模型返回数量/维度/有限性在构造输出对象前 fail-closed 校验。
        """

        result = self._domain_worker.embed(records)
        return ContractEmbeddingResult(
            embedded_chunks=result.embedded_chunks,
            skipped_chunks=[
                SkippedContractChunk(
                    vector_doc_id=item.vector_doc_id,
                    reason=item.reason,
                )
                for item in result.skipped_chunks
            ],
        )


def expected_contract_output_fields(
    vector_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
) -> dict[str, dict[str, object]]:
    """返回完整保存契约输入与向量化输出所需的最小 Milvus 字段。

    V14 25 字段形态已退役；当前以 V15 精简 4 字段物理 Schema 为正式契约
    输出基线（``vector_doc_id`` / ``chunk_text`` / ``embedding`` / ``metadata``）。
    """

    return {
        spec.name: {
            "datatype": spec.datatype,
            "nullable": spec.nullable,
            "primary_key": spec.primary_key,
            "max_length": spec.max_length,
            "dim": spec.dim,
        }
        for spec in v15_milvus_field_specs(vector_dimension)
    }


def compare_contract_output_schema(
    description: Mapping[str, Any],
    vector_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
) -> dict[str, object]:
    """只读比较现有 Collection 是否能无损承载正式契约输出。

    该比较是写入前的 fail-closed gate：缺字段、类型/nullable/dimension 不兼容
    都只返回差异，由 Repository 或验证脚本阻止写入，不尝试自动迁移现有 Collection。
    """

    comparison = compare_schema_specs(
        description,
        v15_milvus_field_specs(vector_dimension),
    )
    result = comparison.as_dict()
    changed_fields = result.get("changed_fields")
    result["incompatible_fields"] = _legacy_incompatible_fields(changed_fields)
    return result


def _legacy_incompatible_fields(
    changed_fields: object,
) -> dict[str, dict[str, object]]:
    """保留旧比较入口的字段差异键，同时复用正式 Gate 的判断结果。"""

    if not isinstance(changed_fields, Mapping):
        return {}
    incompatible: dict[str, dict[str, object]] = {}
    for field_name, change in changed_fields.items():
        if not isinstance(field_name, str) or not isinstance(change, Mapping):
            continue
        expected = change.get("expected")
        actual = change.get("actual")
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            continue
        differences: dict[str, object] = {}
        for key in ("datatype", "nullable", "primary_key", "dim"):
            if expected.get(key) != actual.get(key):
                differences[key] = {
                    "expected": expected.get(key),
                    "actual": actual.get(key),
                }
        if expected.get("max_length") != actual.get("max_length"):
            differences["max_length"] = {
                "minimum": expected.get("max_length"),
                "actual": actual.get("max_length"),
            }
        if differences:
            incompatible[field_name] = differences
    return incompatible


def _required_model_text(model: EmbeddingModel, attribute: str) -> str:
    """从模型读取非空别名或版本，不猜测正式版本值。"""

    value = getattr(model, attribute, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"必须显式提供 embedding_{attribute}")
    return value.strip()


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
