"""Chunk 落库前的契约与安全校验。

校验失败会抛出 ChunkValidationError，由流水线回滚这一位专家的整个事务，避免
profile 已更新但经历仍是旧版本的半成品状态。
"""

from __future__ import annotations

import re

from .chunk_builder import HARD_LIMITS
from .models import ChunkRecord
from .normalize import pii_hits, sha256_value


_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TYPES = {"profile_summary", "employment", "project"}
_SUBTYPES = {None, "responsibility", "achievement", "context", "action", "result", "segment"}


class ChunkValidationError(ValueError):
    """包含 chunk_id 的可定位校验错误，但不包含原始正文。"""


def validate_chunks(chunks: list[ChunkRecord]) -> None:
    """校验完整集合，同时检查 chunk_id 唯一性。"""

    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ChunkValidationError("同一专家生成了重复 chunk_id")
    for chunk in chunks:
        _validate_one(chunk)


def _validate_one(chunk: ChunkRecord) -> None:
    """执行长度、PII、哈希、结构、Metadata 一致性检查。"""

    def fail(message: str) -> None:
        raise ChunkValidationError(f"{chunk.chunk_id}: {message}")

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

    metadata = chunk.metadata
    required = {
        "company_name_raw", "company_name_norm", "company_normalization_status",
        "role_name_raw", "role_code", "role_level", "start_month", "end_month",
        "is_current", "industry_codes", "topic_codes", "origin_type",
        "verification_status", "weight_tier",
    }
    if not required.issubset(metadata):
        fail(f"metadata 缺少字段={sorted(required - set(metadata))}")
    if metadata["company_name_norm"] != chunk.company_name_norm:
        fail("metadata 与显式 company_name_norm 不一致")
    if metadata["role_code"] != chunk.role_code:
        fail("metadata 与显式 role_code 不一致")
    for name, values in (("industry_codes", chunk.industry_codes), ("topic_codes", chunk.topic_codes)):
        if metadata[name] != list(values):
            fail(f"metadata 与显式 {name} 不一致")
        if list(values) != sorted(set(values)) or any(not _CODE.fullmatch(value) for value in values):
            fail(f"{name} 必须去重、排序并使用 snake_case")
    for name, value in (("start_month", chunk.start_month), ("end_month", chunk.end_month)):
        if value is not None and not _MONTH.fullmatch(value):
            fail(f"{name} 格式应为 YYYY-MM")

    # partial 允许进入 expert_chunk 供质量修复和影子评测，但正式 Embedding 消费规则
    # 会排除它。source_evidence 的经历必须满足数据契约的条件必填字段。
    if chunk.chunk_type == "employment" and chunk.verification_status == "source_evidence":
        if not (metadata["company_name_raw"] or metadata["company_name_norm"]):
            fail("employment 缺少公司")
        if not metadata["role_name_raw"] or not chunk.start_month or chunk.is_current is None:
            fail("employment 缺少岗位、开始月份或任职状态")
    if chunk.chunk_type == "project" and chunk.verification_status == "source_evidence":
        if not metadata.get("project_name") or not metadata.get("project_role"):
            fail("project 缺少项目名或项目角色")

