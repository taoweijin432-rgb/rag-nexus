"""验证正式 JSONL Contract 到新 Milvus DEV Collection 的完整链路。

本脚本只允许创建和写入 ``expert_candidates_v12_contract_dev``，不会修改任何
已有旧 Collection，也不会执行 drop、alter 或 delete。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymilvus

from rag_vector_store import (
    CompanyCloudEmbeddingModel,
    ContractEmbeddingWorker,
    ContractMilvusVectorRepository,
    EmbeddedContractPreparedChunk,
    EmbeddingEligibilityValidator,
    JsonlExpertChunkReader,
    MilvusRowMapper,
    MilvusSettings,
    SearchHit,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _PROJECT_ROOT / "fixtures" / "expert_chunk_embedding_input_v1.jsonl"
_TARGET_COLLECTION = "expert_candidates_v12_contract_dev"
_MODEL = "text-embedding-v4"
_DIMENSION = 1024
_UNKNOWN_MODEL_VERSION = "dev-unknown"
_REQUIRED_INDEX_PARAMS = {"M": 16, "efConstruction": 200}
_PROTECTED_COLLECTIONS = frozenset(
    {
        "expert_candidates_v1",
        "resume_candidates_v1",
        "expert_candidates_v12_dev",
    }
)
_STAGES = (
    "contract_validation",
    "eligibility",
    "embedding",
    "collection_precondition",
    "collection_creation",
    "schema_verification",
    "first_upsert",
    "pk_query",
    "json_round_trip",
    "self_vector_search",
    "employment_filter",
    "second_upsert",
    "idempotency",
)


class VerificationBlocked(RuntimeError):
    """前置条件不满足，后续步骤不得继续。"""


@dataclass
class VerificationState:
    """记录每个验收阶段的 PASS、FAIL、BLOCKED 或 NOT_RUN。"""

    statuses: dict[str, str] = field(
        default_factory=lambda: {stage: "NOT_RUN" for stage in _STAGES}
    )
    details: dict[str, object] = field(default_factory=dict)
    current_stage: str | None = None
    message: str | None = None

    def begin(self, stage: str) -> None:
        """记录当前执行阶段。"""

        self.current_stage = stage

    def passed(self, stage: str) -> None:
        """标记阶段成功。"""

        self.statuses[stage] = "PASS"
        self.current_stage = None

    def blocked(self, stage: str, message: str) -> None:
        """标记前置条件阻塞并停止。"""

        self.statuses[stage] = "BLOCKED"
        self.message = message
        self.current_stage = None
        raise VerificationBlocked(message)


def main() -> None:
    """执行完整 DEV 验证，并始终输出安全的分阶段结果。"""

    state = VerificationState()
    exit_code = 0
    try:
        _run(state)
    except VerificationBlocked:
        exit_code = 2
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        if state.current_stage is not None:
            state.statuses[state.current_stage] = "FAIL"
        state.message = f"{type(exc).__name__}（详细异常未输出，避免泄露敏感内容）"
    _print_report(state)
    if exit_code:
        raise SystemExit(exit_code)


def _run(state: VerificationState) -> None:
    """按固定顺序执行契约、Embedding 和 Milvus 验收。"""

    state.begin("contract_validation")
    read_result = JsonlExpertChunkReader().read(_FIXTURE_PATH)
    state.details["input_count"] = read_result.input_count
    state.details["valid_count"] = read_result.valid_count
    state.details["failed_count"] = len(read_result.errors)
    if read_result.errors:
        state.statuses["contract_validation"] = "FAIL"
        state.message = "JSONL 中存在契约错误行"
        raise RuntimeError("JSONL contract validation failed")
    state.passed("contract_validation")

    state.begin("eligibility")
    eligibility = EmbeddingEligibilityValidator()
    eligible = [
        record for record in read_result.records if eligibility.is_eligible(record)
    ]
    state.details["eligible_count"] = len(eligible)
    state.details["skipped_count"] = len(read_result.records) - len(eligible)
    if len(eligible) != len(read_result.records):
        state.blocked("eligibility", "标准 DEV fixture 中存在不允许向量化的记录")
    state.passed("eligibility")

    state.begin("embedding")
    model = CompanyCloudEmbeddingModel.from_env()
    _validate_embedding_settings(model)
    model_version = model.model_version or _UNKNOWN_MODEL_VERSION
    embedding_result = ContractEmbeddingWorker(
        model,
        embedding_model_alias=model.model_alias,
        embedding_model_version=model_version,
        expected_dimension=_DIMENSION,
    ).embed(eligible)
    chunks = embedding_result.embedded_chunks
    if len(chunks) != len(eligible):
        raise RuntimeError("Embedding 输出数量不正确")
    if any(chunk.embedding_dim != _DIMENSION for chunk in chunks):
        raise RuntimeError("Embedding 输出维度不正确")
    state.details["embedded_count"] = len(chunks)
    state.details["model"] = model.model_alias
    state.details["model_version"] = model_version
    state.details["dimension"] = _DIMENSION
    state.details["production_model_version"] = (
        "BLOCKED" if model_version == _UNKNOWN_MODEL_VERSION else "PASS"
    )
    state.passed("embedding")

    state.begin("collection_precondition")
    settings = MilvusSettings.from_env()
    _validate_milvus_target(settings, state)
    repository = ContractMilvusVectorRepository(
        settings,
        vector_dimension=_DIMENSION,
    )
    mapper = MilvusRowMapper()
    _print_creation_plan(repository)

    try:
        client = repository.client
        state.details["pymilvus_version"] = pymilvus.__version__
        state.details["milvus_server_version"] = client.get_server_version()
        existed_before = client.has_collection(
            collection_name=settings.collection_name
        )
        state.details["collection_existed_before"] = existed_before

        if existed_before:
            description = client.describe_collection(
                collection_name=settings.collection_name
            )
            schema_difference = repository.compare_contract_schema(description)
            index_difference = repository.compare_contract_indexes(client)
            if not schema_difference["compatible"]:
                state.details["schema_difference"] = schema_difference
                state.blocked(
                    "collection_precondition",
                    "目标 DEV Collection 已存在但 Schema 不兼容",
                )
            if not index_difference["compatible"]:
                state.details["index_difference"] = index_difference
                state.blocked(
                    "collection_precondition",
                    "目标 DEV Collection 已存在但 Index 不兼容",
                )
            state.statuses["collection_creation"] = "NOT_RUN"
        state.passed("collection_precondition")

        if not existed_before:
            state.begin("collection_creation")
            repository.initialize(_DIMENSION)
            state.passed("collection_creation")
        else:
            repository.initialize(_DIMENSION)

        state.begin("schema_verification")
        description = client.describe_collection(
            collection_name=settings.collection_name
        )
        schema_difference = repository.compare_contract_schema(description)
        index_difference = repository.compare_contract_indexes(client)
        if not schema_difference["compatible"] or not index_difference["compatible"]:
            state.details["schema_difference"] = schema_difference
            state.details["index_difference"] = index_difference
            state.blocked(
                "schema_verification",
                "创建后的 Schema 或 Index 与预期不兼容",
            )
        state.details["json_fields"] = ["metadata_ext", "source_ext", "versions_ext"]
        state.details["nullable_supported"] = True
        state.details["scalar_index_supported"] = True
        state.details["index_count"] = len(client.list_indexes(
            collection_name=settings.collection_name
        ))
        state.passed("schema_verification")

        count_before = _logical_entity_count(client, settings.collection_name)
        state.details["entity_count_before"] = count_before
        if count_before not in (0, len(chunks)):
            state.blocked(
                "first_upsert",
                "目标 DEV Collection 包含非预期数量的已有 Entity",
            )

        state.begin("first_upsert")
        first_result = repository.upsert_batch(chunks)
        if first_result.affected_count != len(chunks):
            raise RuntimeError("第一次 upsert 影响行数不正确")
        count_after_first = _logical_entity_count(client, settings.collection_name)
        if count_after_first != len(chunks):
            raise RuntimeError("第一次 upsert 后 Entity 数量不正确")
        state.details["upserted_count"] = first_result.affected_count
        state.details["entity_count_after_first_upsert"] = count_after_first
        state.passed("first_upsert")

        state.begin("pk_query")
        entities = _query_and_validate_entities(repository, mapper, chunks)
        state.passed("pk_query")

        state.begin("json_round_trip")
        _validate_json_round_trip(chunks, entities)
        state.passed("json_round_trip")

        state.begin("self_vector_search")
        profile = next(chunk for chunk in chunks if chunk.chunk_type == "profile_summary")
        profile_entity = entities[profile.vector_doc_id]
        profile_embedding = _entity_embedding(profile_entity)
        self_hits = repository.search(profile_embedding, limit=3)
        if not _contains_id(self_hits, profile.vector_doc_id):
            raise RuntimeError("Self-vector search 未召回自身 Entity")
        state.details["self_vector_top_k"] = len(self_hits)
        state.passed("self_vector_search")

        state.begin("employment_filter")
        employment = next(chunk for chunk in chunks if chunk.chunk_type == "employment")
        employment_embedding = _entity_embedding(entities[employment.vector_doc_id])
        employment_hits = repository.search(
            employment_embedding,
            limit=3,
            chunk_type="employment",
        )
        if not employment_hits or not all(
            hit.chunk_type == "employment" for hit in employment_hits
        ):
            raise RuntimeError("chunk_type=employment 过滤结果不正确")
        if not _contains_id(employment_hits, employment.vector_doc_id):
            raise RuntimeError("employment 过滤未召回自身 Entity")
        state.details["employment_filter_hits"] = len(employment_hits)
        state.passed("employment_filter")

        state.begin("second_upsert")
        second_result = repository.upsert_batch(chunks)
        if second_result.affected_count != len(chunks):
            raise RuntimeError("第二次 upsert 影响行数不正确")
        state.passed("second_upsert")

        state.begin("idempotency")
        count_after_second = _logical_entity_count(client, settings.collection_name)
        if count_after_second != count_after_first:
            raise RuntimeError("第二次 upsert 后 Entity 数量增加")
        state.details["entity_count_after_second_upsert"] = count_after_second
        state.passed("idempotency")
    finally:
        repository.close()


def _validate_embedding_settings(model: CompanyCloudEmbeddingModel) -> None:
    """确认 DEV 验证使用已经确认的模型与向量维度。"""

    if model.settings.model != _MODEL:
        raise RuntimeError(f"EMBEDDING_MODEL 必须为 {_MODEL}")
    if model.dimension != _DIMENSION:
        raise RuntimeError(
            f"EMBEDDING_DIMENSION 必须为 {_DIMENSION}，实际为 {model.dimension}"
        )


def _validate_milvus_target(
    settings: MilvusSettings,
    state: VerificationState,
) -> None:
    """严格限制脚本只能创建和写入新的契约 DEV Collection。"""

    if settings.collection_name in _PROTECTED_COLLECTIONS:
        state.blocked(
            "collection_precondition",
            f"受保护 Collection 不允许写入: {settings.collection_name}",
        )
    if settings.collection_name != _TARGET_COLLECTION:
        state.blocked(
            "collection_precondition",
            "MILVUS_COLLECTION_NAME 必须严格等于 "
            f"{_TARGET_COLLECTION}",
        )
    if settings.index_type.upper() != "HNSW":
        state.blocked(
            "collection_precondition",
            "MILVUS_INDEX_TYPE 必须设置为 HNSW",
        )
    if settings.metric_type.upper() != "COSINE":
        state.blocked(
            "collection_precondition",
            "MILVUS_METRIC_TYPE 必须设置为 COSINE",
        )
    if settings.index_params != _REQUIRED_INDEX_PARAMS:
        state.blocked(
            "collection_precondition",
            "MILVUS_INDEX_PARAMS 必须为 HNSW DEV 参数 M=16、efConstruction=200",
        )


def _print_creation_plan(repository: ContractMilvusVectorRepository) -> None:
    """在创建前打印完整 Schema 与 Index，不输出连接凭据。"""

    print("Creation plan:")
    print(f"database: {repository.settings.db_name or 'default'}")
    print(f"collection: {repository.settings.collection_name}")
    print(
        "protected_collections: "
        "expert_candidates_v1, expert_candidates_v12_dev, resume_candidates_v1"
    )
    print("schema:")
    for field_spec in repository.contract_schema_summary(_DIMENSION):
        print(
            f"- {field_spec['field']}: {field_spec['datatype']}; "
            f"nullable={str(field_spec['nullable']).lower()}; "
            f"primary_key={str(field_spec['primary_key']).lower()}; "
            f"max_length={field_spec['max_length']}; dim={field_spec['dim']}"
        )
    print("indexes:")
    for index in repository.expected_contract_indexes():
        print(f"- {json.dumps(index, ensure_ascii=False, sort_keys=True)}")


def _logical_entity_count(client: Any, collection_name: str) -> int:
    """使用强一致 count(*) 读取逻辑 Entity 数量。"""

    rows = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["count(*)"],
        consistency_level="Strong",
    )
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
        or "count(*)" not in rows[0]
    ):
        raise RuntimeError("Milvus count(*) 返回格式不符合预期")
    count = rows[0]["count(*)"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("Milvus count(*) 未返回整数")
    return count


def _query_and_validate_entities(
    repository: ContractMilvusVectorRepository,
    mapper: MilvusRowMapper,
    chunks: Sequence[EmbeddedContractPreparedChunk],
) -> dict[str, dict[str, Any]]:
    """按主键回查并比较除浮点向量数值外的全部字段。"""

    entities: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        entity = repository.get_by_vector_doc_id(chunk.vector_doc_id)
        if entity is None:
            raise RuntimeError(f"按主键未找到 Entity: {chunk.vector_doc_id}")
        expected = mapper.to_row(chunk)
        for field_name, expected_value in expected.items():
            if field_name == "embedding":
                continue
            if field_name not in entity or entity[field_name] != expected_value:
                raise RuntimeError(
                    f"Entity {chunk.vector_doc_id} 的 {field_name} 回查不一致"
                )
        if len(_entity_embedding(entity)) != chunk.embedding_dim:
            raise RuntimeError("Milvus 回查向量维度不正确")
        entities[chunk.vector_doc_id] = entity
    return entities


def _validate_json_round_trip(
    chunks: Sequence[EmbeddedContractPreparedChunk],
    entities: Mapping[str, Mapping[str, Any]],
) -> None:
    """逐条确认 metadata、source 和 versions 完整往返。"""

    for chunk in chunks:
        entity = entities[chunk.vector_doc_id]
        if entity.get("metadata_ext") != chunk.metadata.model_dump(mode="json"):
            raise RuntimeError("metadata_ext round-trip 不一致")
        if entity.get("source_ext") != chunk.source.model_dump(mode="json"):
            raise RuntimeError("source_ext round-trip 不一致")
        if entity.get("versions_ext") != chunk.versions.model_dump(mode="json"):
            raise RuntimeError("versions_ext round-trip 不一致")


def _entity_embedding(entity: Mapping[str, Any]) -> list[float]:
    """读取 Entity 向量并校验 1024 维，不打印具体数值。"""

    embedding = entity.get("embedding")
    if not isinstance(embedding, list) or len(embedding) != _DIMENSION:
        raise RuntimeError("Milvus Entity embedding 不是 1024 维列表")
    return [float(value) for value in embedding]


def _contains_id(hits: Sequence[SearchHit], vector_doc_id: str) -> bool:
    """判断 SearchHit 是否包含指定主键。"""

    return any(hit.vector_doc_id == vector_doc_id for hit in hits)


def _print_report(state: VerificationState) -> None:
    """打印不含正文、完整向量和凭据的最终验收摘要。"""

    print("Verification summary:")
    for stage in _STAGES:
        print(f"{stage}: {state.statuses[stage]}")
    for key in (
        "pymilvus_version",
        "milvus_server_version",
        "input_count",
        "valid_count",
        "eligible_count",
        "embedded_count",
        "upserted_count",
        "skipped_count",
        "failed_count",
        "model",
        "model_version",
        "production_model_version",
        "dimension",
        "collection_existed_before",
        "json_fields",
        "nullable_supported",
        "scalar_index_supported",
        "index_count",
        "entity_count_before",
        "entity_count_after_first_upsert",
        "entity_count_after_second_upsert",
        "self_vector_top_k",
        "employment_filter_hits",
    ):
        if key in state.details:
            print(f"{key}: {state.details[key]}")
    if state.message:
        print(f"message: {state.message}")
    overall = (
        "PASS"
        if all(
            status == "PASS" or stage == "collection_creation" and status == "NOT_RUN"
            for stage, status in state.statuses.items()
        )
        else "BLOCKED"
        if "BLOCKED" in state.statuses.values()
        else "FAIL"
        if "FAIL" in state.statuses.values()
        else "NOT_RUN"
    )
    print(f"overall: {overall}")


if __name__ == "__main__":
    main()
