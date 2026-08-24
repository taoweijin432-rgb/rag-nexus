"""Work Item 到 V15 正式 Contract 的薄适配器（Phase 2）。

职责边界：
    只做“结构搬运 + 严格校验”：把持久化层取出的 Work Item 行映射为
    ``ExpertChunkEmbeddingInput``（或 disable 目标集合），并复用现有的
    Contract Validator 与 Eligibility 校验。本模块：

    - 不生成、不改写、不截断 ``chunk_text``（文本组装始终由上游 expert_index
      完成，这里只做非空、长度与 Contract 一致性校验）；
    - 不计算任何业务 Hash（``content_hash`` 由 Contract Validator 按
      ``chunk_text`` 重算比对；``metadata_hash`` / ``source_hash`` 只透传并校验
      格式，重算属于上游责任边界）；
    - 不调用 Embedding、不连接 MySQL / Milvus；
    - 不是 Chunk Builder，也不重新实现三类 Chunk 组装。

metadata 载荷布局（Phase 2 冻结的传输边界）：
    Work Item 的 ``metadata`` JSON 列同时承载两类信息：

    1. 结构保留键（适配器消费后不进入业务 Metadata）：
       - ``chunk_subtype``：str | None，缺省 None；
       - ``chunk_index`` / ``chunk_total``：int，必填；
       - ``char_count``：int，必填（上游声明值，Validator 与实际文本比对）；
       - ``serial_number``：str | None，缺省 None；
       - ``updated_at``：带时区 ISO 8601 字符串，必填；
       - ``source_field``：str，必填（"+" 拼接的源列名）；
       - ``source_version``：str，必填；
       - ``source_hash``：SHA-256 十六进制，必填；
       - ``metadata_schema_version``：str，必填；
       - ``source_updated_at``：带时区 ISO 8601 字符串 | None，可选；
         缺省时回退 Work Item 顶层 ``source_updated_at`` 列。
    2. chunk_type 私有业务字段：其余全部键原样交给
       ``metadata_model_for_chunk_type`` 选出的 per-type Metadata 模型
       （extra=forbid，未知字段 fail-closed）。禁止出现 ``source_record_id``
       （统一使用 ``source_field_id``）；禁止携带 embedding/model-space 字段
       （向量字段只能由本模块 Embedding Worker 在校验之后产生）。

操作路由：
    - ``upsert`` → 完整 Contract（``adapt_upsert``）；
    - ``disable`` → 最小输入契约：只产出待下线的 vector_doc_id 集合
      （``adapt_disable``），不构造伪 Contract；
    - ``metadata_update`` → 本阶段 fail-closed 拒绝：Work Item 契约不允许携带
      chunk_text，而 Contract 与 update_metadata 一致性校验必须使用 Milvus 已存
      原文；Consumer 应在执行期取回原文后再走完整 Contract 路径。

模型空间说明：
    适配器不做需要访问 Milvus 的 model space 校验；该安全门禁复用既有
    ingestion / repository 层逻辑（``_validate_existing_model_space`` 等），
    对适配产物透明生效。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import tzinfo
from typing import Any, NoReturn

from pydantic import ValidationError

from expert_contracts.models import (
    EXPERT_CHUNK_CONTRACT_VERSION,
    EXPERT_CHUNK_METADATA_BY_TYPE,
    ContractPreparedChunk,
    ExpertChunkEmbeddingInput,
    metadata_model_for_chunk_type,
)

from ..chunking.contract_pipeline import (
    ContractValidationError,
    ContractValidator,
    EmbeddingEligibilityValidator,
)
from .models import V15_VECTOR_DOC_ID_MAX_BYTES, V15VectorizationWorkItem

__all__ = [
    "DisableWorkItem",
    "V15WorkItemContractAdapter",
    "WorkItemContractAdaptation",
    "WorkItemContractAdapterError",
]

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# 结构保留键：适配器按契约消费，不会进入 per-type 业务 Metadata。
_STRUCTURE_KEYS: frozenset[str] = frozenset(
    {
        "chunk_subtype",
        "chunk_index",
        "chunk_total",
        "char_count",
        "serial_number",
        "updated_at",
        "source_field",
        "source_version",
        "source_hash",
        "source_updated_at",
        "metadata_schema_version",
    }
)

# upsert 必填的结构保留键；缺失时一次性报出全部缺口（fail-closed）。
_REQUIRED_STRUCTURE_KEYS: tuple[str, ...] = (
    "chunk_index",
    "chunk_total",
    "char_count",
    "updated_at",
    "source_field",
    "source_version",
    "source_hash",
    "metadata_schema_version",
)

_METADATA_UPDATE_BLOCK_MESSAGE = (
    "operation=metadata_update 无法在 Phase 2 适配器中直接构造完整 Contract："
    "Work Item 契约不允许携带 chunk_text，而 Contract 校验与 "
    "VectorIngestionService.update_metadata 的一致性门禁都必须使用 Milvus 已存"
    "chunk_text。Consumer 应在执行期通过 get_by_vector_doc_id(include_embedding="
    "True) 取回已存原文后走 upsert 形态适配（安全门禁失败时再自动降级为全量 "
    "upsert）；适配器禁止伪造、占位或重新生成 chunk_text。"
)


class WorkItemContractAdapterError(ValueError):
    """Work Item 无法无损适配为正式 Contract（fail-closed）。

    所有由本适配器拒绝的输入（缺字段、hash 不一致、非法 chunk_type、身份冲突、
    禁用字段出现等）都抛本异常；它属于“确定性 rejected”类别，不是可重试错误。
    """


@dataclass(frozen=True)
class WorkItemContractAdaptation:
    """upsert Work Item 的适配产物。

    - ``contract``：已通过结构与跨字段校验的正式输入，可交给向量化链路；
    - ``prepared``：``ContractValidator.validate`` 的返回值（同一记录的阶段视图）；
    - ``eligibility_skip_reason``：复用 ``EmbeddingEligibilityValidator`` 的预判
      结果（None 表示允许 Embedding）；是否跳过由 Consumer 决策，适配器不执行。
    """

    contract: ExpertChunkEmbeddingInput
    prepared: ContractPreparedChunk
    eligibility_skip_reason: str | None


@dataclass(frozen=True)
class DisableWorkItem:
    """disable Work Item 的最小输入契约产物。

    只包含待下线的 vector_doc_id 集合（主 id + stale 子块 id，去重且保持
    出现顺序）；Milvus 实体保留、仅置 is_active=false，由既有
    ``VectorIngestionService.disable_vectors`` 执行。
    """

    vector_doc_ids: tuple[str, ...]


def _validation_summary(exc: ValidationError) -> str:
    """提取 pydantic 校验错误的简明摘要，不把输入正文带进错误消息。"""

    parts = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", ()))
        parts.append(f"{loc or '<顶层>'}: {error.get('msg', '校验失败')}")
    return "；".join(parts) or str(exc)


class V15WorkItemContractAdapter:
    """把 V15 Work Item 行薄适配为正式 Contract 或 disable 目标集合。

    参数：
        ``business_timezone``：Work Item 顶层 ``source_updated_at`` 为 naive
        datetime 时用于补齐时区的业务时区（如 ``ZoneInfo("Asia/Shanghai")``）；
        未配置且遇到 naive 值时 fail-closed，绝不静默假设时区。
        ``validator`` / ``eligibility``：注入点，默认使用正式实现。
    """

    def __init__(
        self,
        *,
        business_timezone: tzinfo | None = None,
        validator: ContractValidator | None = None,
        eligibility: EmbeddingEligibilityValidator | None = None,
    ) -> None:
        self._business_timezone = business_timezone
        self._validator = validator if validator is not None else ContractValidator()
        self._eligibility = (
            eligibility if eligibility is not None else EmbeddingEligibilityValidator()
        )

    # ------------------------------------------------------------------
    # 操作分发
    # ------------------------------------------------------------------
    def adapt(
        self,
        work_item: V15VectorizationWorkItem,
    ) -> WorkItemContractAdaptation | DisableWorkItem:
        """按 operation 分发到对应适配路径。"""

        if work_item.operation == "upsert":
            return self.adapt_upsert(work_item)
        if work_item.operation == "disable":
            return self.adapt_disable(work_item)
        self.adapt_metadata_update(work_item)

    def adapt_metadata_update(
        self,
        work_item: V15VectorizationWorkItem,
    ) -> NoReturn:
        """metadata_update 在本阶段固定 fail-closed（见模块 docstring）。"""

        raise WorkItemContractAdapterError(_METADATA_UPDATE_BLOCK_MESSAGE)

    # ------------------------------------------------------------------
    # upsert：完整 Contract
    # ------------------------------------------------------------------
    def adapt_upsert(
        self,
        work_item: V15VectorizationWorkItem,
    ) -> WorkItemContractAdaptation:
        """把 upsert Work Item 映射为 ``ExpertChunkEmbeddingInput`` 并完成校验。"""

        if work_item.operation != "upsert":
            raise WorkItemContractAdapterError(
                f"adapt_upsert 只接受 operation=upsert，收到 {work_item.operation!r}"
            )
        chunk_type = self._require_chunk_type(work_item)
        expert_id = self._require_expert_id(work_item)
        metadata_hash = self._require_metadata_hash(work_item)
        payload_md = self._require_metadata_payload(work_item)

        structure = self._extract_structure(payload_md)
        metadata_model = self._build_business_metadata(chunk_type, payload_md)
        source_updated_at = self._resolve_source_updated_at(work_item, structure)
        chunk_policy_version = self._require_chunk_policy_version(work_item)

        payload: dict[str, Any] = {
            "contract_version": EXPERT_CHUNK_CONTRACT_VERSION,
            "vector_doc_id": work_item.vector_doc_id,
            "expert_id": expert_id,
            "serial_number": structure["serial_number"],
            "chunk_type": chunk_type,
            "chunk_subtype": structure["chunk_subtype"],
            "chunk_index": structure["chunk_index"],
            "chunk_total": structure["chunk_total"],
            # chunk_text 原样透传：只读不写，长度/字符/Hash 由 Validator 校验。
            "chunk_text": work_item.chunk_text,
            "char_count": structure["char_count"],
            "content_hash": work_item.content_hash,
            "metadata_hash": metadata_hash,
            "is_active": work_item.is_active,
            "updated_at": structure["updated_at"],
            "metadata": metadata_model,
            "source": {
                "source_table": self._require_source_table(work_item),
                "source_field": structure["source_field"],
                "source_field_id": self._require_source_field_id(work_item),
                "source_version": structure["source_version"],
                "source_updated_at": source_updated_at,
                "source_hash": structure["source_hash"],
            },
            "versions": {
                "chunk_policy_version": chunk_policy_version,
                "metadata_schema_version": structure["metadata_schema_version"],
            },
        }

        try:
            contract = ExpertChunkEmbeddingInput.model_validate(payload)
        except ValidationError as exc:
            raise WorkItemContractAdapterError(
                "upsert Work Item 结构校验失败: "
                f"{_validation_summary(exc)}"
            ) from exc

        try:
            prepared = self._validator.validate(contract)
        except ContractValidationError as exc:
            raise WorkItemContractAdapterError(
                f"upsert Work Item 跨字段校验失败: {exc}"
            ) from exc

        return WorkItemContractAdaptation(
            contract=contract,
            prepared=prepared,
            eligibility_skip_reason=self._eligibility.get_skip_reason(prepared),
        )

    # ------------------------------------------------------------------
    # disable：最小输入契约
    # ------------------------------------------------------------------
    def adapt_disable(
        self,
        work_item: V15VectorizationWorkItem,
    ) -> DisableWorkItem:
        """把 disable Work Item 解析为待下线 id 集合（不构造伪 Contract）。

        规则：
            - 主 ``vector_doc_id`` 必须符合正式公式，并在 Work Item 携带身份
              字段时与之交叉核对；
            - ``stale_vector_doc_ids`` 必须同属一个公式家族（同 expert + 同
              chunk_type）；source_field_id 允许不同，不做强制；
            - 输出去重并保持出现顺序。
        """

        if work_item.operation != "disable":
            raise WorkItemContractAdapterError(
                f"adapt_disable 只接受 operation=disable，收到 {work_item.operation!r}"
            )
        # V15WorkItemInput 的模型校验器已强制 disable 的 is_active=False；
        # 这里保留防御性检查（例如经 model_construct 绕过校验构造的对象）。
        if work_item.is_active is not False:
            raise WorkItemContractAdapterError("disable Work Item 必须 is_active=False")

        ordered_ids: list[str] = [work_item.vector_doc_id]
        ordered_ids.extend(work_item.stale_vector_doc_ids)

        seen: set[str] = set()
        resolved: list[str] = []
        for index, vector_doc_id in enumerate(ordered_ids):
            label = "主 vector_doc_id" if index == 0 else "stale_vector_doc_id"
            parsed = _parse_vector_doc_id(vector_doc_id, label)
            if index == 0:
                _cross_check_identity(parsed, work_item, label)
            else:
                # stale 子块必须与 Work Item 属于同一专家 + 同一 chunk_type 家族。
                if (
                    work_item.expert_id is not None
                    and parsed.expert_id != work_item.expert_id
                ):
                    raise WorkItemContractAdapterError(
                        f"{label} {vector_doc_id!r} 的 expert_id 与 Work Item "
                        f"不一致（{parsed.expert_id} != {work_item.expert_id}）"
                    )
                if (
                    work_item.chunk_type is not None
                    and parsed.chunk_type != work_item.chunk_type
                ):
                    raise WorkItemContractAdapterError(
                        f"{label} {vector_doc_id!r} 的 chunk_type 与 Work Item "
                        f"不一致（{parsed.chunk_type} != {work_item.chunk_type}）"
                    )
            if vector_doc_id not in seen:
                seen.add(vector_doc_id)
                resolved.append(vector_doc_id)

        return DisableWorkItem(vector_doc_ids=tuple(resolved))

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _require_chunk_type(self, work_item: V15VectorizationWorkItem) -> str:
        chunk_type = work_item.chunk_type
        if chunk_type is None:
            raise WorkItemContractAdapterError(
                "upsert Work Item 缺少 chunk_type（只允许 profile_summary / "
                "employment / project；education 不生成独立 Chunk，只能作为 "
                "profile_summary 的 Metadata 来源）"
            )
        if chunk_type not in EXPERT_CHUNK_METADATA_BY_TYPE:
            raise WorkItemContractAdapterError(
                f"非法 chunk_type: {chunk_type!r}（只允许 profile_summary / "
                "employment / project）"
            )
        return chunk_type

    @staticmethod
    def _require_expert_id(work_item: V15VectorizationWorkItem) -> int:
        expert_id = work_item.expert_id
        if expert_id is None or expert_id < 1:
            raise WorkItemContractAdapterError(
                f"upsert Work Item 必须携带有效 expert_id（>=1），收到 {expert_id!r}"
            )
        return expert_id

    @staticmethod
    def _require_metadata_hash(work_item: V15VectorizationWorkItem) -> str:
        if work_item.metadata_hash is None:
            raise WorkItemContractAdapterError(
                "upsert Work Item 缺少 metadata_hash"
                "（Contract 必填，用于非内容字段变化判断）"
            )
        return work_item.metadata_hash

    @staticmethod
    def _require_chunk_policy_version(work_item: V15VectorizationWorkItem) -> str:
        value = work_item.chunk_policy_version
        if value is None or not value.strip():
            raise WorkItemContractAdapterError(
                "upsert Work Item 缺少 chunk_policy_version（versions 必填）"
            )
        return value

    @staticmethod
    def _require_source_table(work_item: V15VectorizationWorkItem) -> str:
        value = work_item.source_table
        if value is None or not value.strip():
            raise WorkItemContractAdapterError("upsert Work Item 缺少 source_table")
        return value

    @staticmethod
    def _require_source_field_id(work_item: V15VectorizationWorkItem) -> str:
        value = work_item.source_field_id
        if value is None or not value.strip():
            raise WorkItemContractAdapterError(
                "upsert Work Item 缺少 source_field_id"
                "（来源定位统一使用 source_field_id，不存在 source_record_id）"
            )
        return value

    @staticmethod
    def _require_metadata_payload(
        work_item: V15VectorizationWorkItem,
    ) -> dict[str, Any]:
        payload = work_item.metadata
        if not isinstance(payload, dict) or not payload:
            required = "、".join(_REQUIRED_STRUCTURE_KEYS)
            raise WorkItemContractAdapterError(
                "upsert Work Item 缺少 metadata 载荷；Phase 2 传输边界要求其中"
                f"至少包含结构保留键: {required}"
            )
        if "source_record_id" in payload:
            raise WorkItemContractAdapterError(
                "metadata 载荷出现被禁止的字段 source_record_id；"
                "来源定位统一使用 source_field_id"
            )
        return dict(payload)

    @staticmethod
    def _extract_structure(payload_md: dict[str, Any]) -> dict[str, Any]:
        """取出结构保留键并做基础类型检查；业务键留在载荷里等待模型校验。"""

        missing = [key for key in _REQUIRED_STRUCTURE_KEYS if key not in payload_md]
        if missing:
            required = "、".join(_REQUIRED_STRUCTURE_KEYS)
            raise WorkItemContractAdapterError(
                f"metadata 载荷缺少结构保留键: {'、'.join(missing)}"
                f"（Phase 2 传输边界要求的完整集合: {required}）"
            )

        structure: dict[str, Any] = {}

        for int_key in ("chunk_index", "chunk_total", "char_count"):
            value = payload_md.pop(int_key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise WorkItemContractAdapterError(
                    f"结构保留键 {int_key} 必须是整数，收到 {type(value).__name__}"
                )
            structure[int_key] = value

        for str_key in ("updated_at", "source_field", "source_version",
                        "source_hash", "metadata_schema_version"):
            value = payload_md.pop(str_key)
            if not isinstance(value, str) or not value.strip():
                raise WorkItemContractAdapterError(
                    f"结构保留键 {str_key} 必须是非空字符串"
                )
            structure[str_key] = value

        if not _SHA256_HEX_PATTERN.fullmatch(structure["source_hash"]):
            raise WorkItemContractAdapterError(
                "结构保留键 source_hash 必须是 SHA-256 十六进制（64 位小写）"
            )

        for optional_key in ("chunk_subtype", "serial_number",
                             "source_updated_at"):
            optional_value = payload_md.pop(optional_key, None)
            if optional_value is not None and not isinstance(optional_value, str):
                raise WorkItemContractAdapterError(
                    f"结构保留键 {optional_key} 必须是字符串或 None，"
                    f"收到 {type(optional_value).__name__}"
                )
            structure[optional_key] = optional_value

        return structure

    def _resolve_source_updated_at(
        self,
        work_item: V15VectorizationWorkItem,
        structure: dict[str, Any],
    ) -> str | None:
        """优先使用载荷中的显式字符串；否则回退 Work Item 顶层列。"""

        if structure["source_updated_at"] is not None:
            return str(structure["source_updated_at"])
        column_value = work_item.source_updated_at
        if column_value is None:
            return None
        if column_value.tzinfo is None:
            if self._business_timezone is None:
                raise WorkItemContractAdapterError(
                    "source_updated_at 为不带时区的 naive datetime 且未配置 "
                    "business_timezone，拒绝静默假设时区"
                )
            column_value = column_value.replace(tzinfo=self._business_timezone)
        return column_value.isoformat()

    @staticmethod
    def _build_business_metadata(
        chunk_type: str,
        payload_md: dict[str, Any],
    ) -> Any:
        """剩余键全部交给 per-type Metadata 模型严格校验（extra=forbid）。"""

        model_cls = metadata_model_for_chunk_type(chunk_type)
        try:
            return model_cls.model_validate(payload_md)
        except ValidationError as exc:
            raise WorkItemContractAdapterError(
                f"{chunk_type} 业务 Metadata 校验失败: {_validation_summary(exc)}"
            ) from exc


class _ParsedVectorDocId:
    """vector_doc_id 正式公式的解析结果。"""

    __slots__ = ("chunk_index", "chunk_type", "expert_id", "source_field_id")

    def __init__(
        self,
        expert_id: int,
        chunk_type: str,
        source_field_id: str,
        chunk_index: int,
    ) -> None:
        self.expert_id = expert_id
        self.chunk_type = chunk_type
        self.source_field_id = source_field_id
        self.chunk_index = chunk_index


def _parse_vector_doc_id(value: str, label: str) -> _ParsedVectorDocId:
    """解析 ``expert:{expert_id}:{chunk_type}:{source_field_id}:{chunk_index}``。

    只解析与核对，不生成新 id；不符合公式即 fail-closed。
    """

    if len(value.encode("utf-8")) > V15_VECTOR_DOC_ID_MAX_BYTES:
        raise WorkItemContractAdapterError(
            f"{label} 超过 V15 上限 {V15_VECTOR_DOC_ID_MAX_BYTES} 字节"
        )
    parts = value.split(":")
    if len(parts) != 5 or parts[0] != "expert":
        raise WorkItemContractAdapterError(
            f"{label} {value!r} 不符合正式公式 "
            "expert:{expert_id}:{chunk_type}:{source_field_id}:{chunk_index}"
        )
    try:
        expert_id = int(parts[1])
        chunk_index = int(parts[4])
    except ValueError as exc:
        raise WorkItemContractAdapterError(
            f"{label} {value!r} 的 expert_id/chunk_index 必须是整数"
        ) from exc
    chunk_type = parts[2]
    source_field_id = parts[3]
    if expert_id < 1 or chunk_index < 0 or not source_field_id:
        raise WorkItemContractAdapterError(
            f"{label} {value!r} 的取值不在合法范围内"
        )
    if chunk_type not in EXPERT_CHUNK_METADATA_BY_TYPE:
        raise WorkItemContractAdapterError(
            f"{label} {value!r} 含非法 chunk_type: {chunk_type!r}"
        )
    return _ParsedVectorDocId(expert_id, chunk_type, source_field_id, chunk_index)


def _cross_check_identity(
    parsed: _ParsedVectorDocId,
    work_item: V15VectorizationWorkItem,
    label: str,
) -> None:
    """Work Item 携带身份列时，与主 id 的解析结果逐一交叉核对。"""

    checks: tuple[tuple[int | str | None, int | str, str], ...] = (
        (work_item.expert_id, parsed.expert_id, "expert_id"),
        (work_item.chunk_type, parsed.chunk_type, "chunk_type"),
        (work_item.source_field_id, parsed.source_field_id, "source_field_id"),
    )
    for column_value, id_value, field_name in checks:
        if column_value is not None and column_value != id_value:
            raise WorkItemContractAdapterError(
                f"{label} 与 Work Item 的 {field_name} 不一致"
                f"（{id_value!r} != {column_value!r}）；"
                "vector_doc_id 必须与身份字段满足正式公式"
            )
