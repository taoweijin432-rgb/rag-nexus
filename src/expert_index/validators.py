"""Chunk 交给下游 Contract Bridge 前的契约与安全校验。

校验失败会抛出 ChunkValidationError，由流水线回滚这一位专家的整个事务，避免
profile 已更新但经历仍是旧版本的半成品状态。
"""

from __future__ import annotations

import re

from expert_contracts.models import vector_doc_id_prefix_for_source_table

from .chunk_builder import HARD_LIMITS, build_vector_doc_id, calculate_metadata_hash
from .dataset import EXPERT_SPEC, DatasetSpec
from .models import ChunkRecord
from .normalize import pii_hits, sha256_value

_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
_MONTH = re.compile(r"^\d{4}(0[1-9]|1[0-2])$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TYPES = {"profile_summary", "employment", "project"}
_SUBTYPES = {None, "responsibility", "achievement", "context", "action", "result", "segment"}


class ChunkValidationError(ValueError):
    """包含 vector_doc_id 的可定位校验错误，但不包含原始正文。"""


def validate_chunks(
    chunks: list[ChunkRecord],
    *,
    dataset_spec: DatasetSpec = EXPERT_SPEC,
) -> None:
    """校验完整集合，同时检查 vector_doc_id 唯一性。

    ``dataset_spec`` 只影响 ID 命名空间；默认值固定为 Expert，保持现有调用和
    生产链路的 ``expert:`` 公式不变。
    """

    ids = [chunk.vector_doc_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ChunkValidationError("同一专家生成了重复 vector_doc_id")
    for chunk in chunks:
        _validate_one(chunk, dataset_spec=dataset_spec)


def _validate_one(chunk: ChunkRecord, *, dataset_spec: DatasetSpec) -> None:
    """执行长度、PII、哈希、结构、Metadata 一致性检查。"""

    def fail(message: str) -> None:
        raise ChunkValidationError(f"{chunk.vector_doc_id}: {message}")

    if chunk.chunk_type not in _TYPES:
        fail("chunk_type 不合法")
    if chunk.chunk_subtype not in _SUBTYPES:
        fail("chunk_subtype 不合法")
    if chunk.chunk_index < 0 or chunk.chunk_total < 1 or chunk.chunk_index >= chunk.chunk_total:
        fail("chunk_index/chunk_total 不合法")
    if not chunk.chunk_text.strip() or "\r" in chunk.chunk_text:
        fail("chunk_text 为空或包含 CR")
    if len(chunk.chunk_text) != chunk.char_count:
        fail("char_count 与 Unicode 字符数不一致")
    if chunk.char_count > HARD_LIMITS[chunk.chunk_type]:
        fail("超过该类型字符硬上限")
    if len(chunk.chunk_text.encode("utf-8")) > 8192:
        fail("UTF-8 字节长度超过 8192")
    hits = pii_hits(chunk.chunk_text)
    if hits:
        fail(f"PII 检查未通过，命中类别={sorted(hits)}")
    if sha256_value(chunk.chunk_text) != chunk.content_hash:
        fail("content_hash 不一致")
    if not _SHA256.fullmatch(chunk.source_hash):
        fail("source_hash 格式错误")
    # 新溯源契约不再使用 source_record/source_record_id。source_field
    # 必须是真实源列名组合，source_field_id 必须能回查来源行。
    if not chunk.source_table.strip():
        fail("source_table 不能为空")
    if not chunk.source_field.strip() or any(
        not field_name.strip() for field_name in chunk.source_field.split("+")
    ):
        fail("source_field 必须是非空源列名组合")
    if not chunk.source_field_id.strip():
        fail("source_field_id 不能为空")
    # 旧调用未携带 DatasetSpec 时仍能处理 Resume：其来源标签是冻结契约的一部分。
    # 显式传入 ResumeSpec/其他规格时，以规格声明为权威，避免仅凭表名猜测。
    prefix = dataset_spec.vector_doc_id_prefix
    if dataset_spec is EXPERT_SPEC:
        prefix = vector_doc_id_prefix_for_source_table(chunk.source_table)
    expected_vector_doc_id = build_vector_doc_id(
        expert_id=chunk.expert_id,
        chunk_type=chunk.chunk_type,
        source_field_id=chunk.source_field_id,
        chunk_index=chunk.chunk_index,
        prefix=prefix,
    )
    if chunk.vector_doc_id != expected_vector_doc_id:
        fail("vector_doc_id 不符合统一生成规则")
    if not _SHA256.fullmatch(chunk.metadata_hash):
        fail("metadata_hash 格式错误")
    if calculate_metadata_hash(chunk) != chunk.metadata_hash:
        fail("metadata_hash 与扁平 Metadata 不一致")
    for name, values in (
        ("industry_codes", chunk.industry_codes),
        ("function_codes", chunk.function_codes),
        ("topic_codes", chunk.topic_codes),
    ):
        if list(values) != sorted(set(values)) or any(not _CODE.fullmatch(value) for value in values):
            fail(f"{name} 必须去重、排序并使用 snake_case")
    for name, value in (("start_month", chunk.start_month), ("end_month", chunk.end_month)):
        if value is not None and not _MONTH.fullmatch(value):
            fail(f"{name} 格式应为 YYYYMM")

    if chunk.chunk_type == "profile_summary":
        if chunk.origin_type != "compressed_from_source" or chunk.verification_status not in {
            "source_summary",
            "rejected",
        }:
            fail("profile_summary 的来源/核验枚举不合法")
    else:
        if chunk.eligible_for_recommendation is not None:
            fail("eligible_for_recommendation 只能出现在 profile_summary")
        if chunk.origin_type != "source_parsed" or chunk.verification_status != "verified":
            fail("employment/project 的来源/核验枚举不合法")

    if chunk.chunk_type == "employment" and not (
        chunk.company_name_raw or chunk.company_name_norm
    ):
        fail("employment 缺少公司")
    if chunk.chunk_type == "project" and not chunk.project_name:
        fail("project 缺少项目名")
