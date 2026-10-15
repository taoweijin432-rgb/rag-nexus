"""Expert V16 正式 Repository Profile 与一致性门禁。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import EXPERT_V16_COLLECTION_NAME, MilvusSettings
from ..core.hash_state import read_hash_field
from ..core.storage import VectorRepositoryProfile
from ..domains import ExpertContractAdapter
from .contract_v16 import MilvusV16RowMapper, validate_v16_mirrors
from .indexes import index_specs_v16
from .repository import ContractMilvusRepositoryBase
from .repository_metadata import MetadataModelSpaceRepository
from .schemas.contract_v16 import (
    V16_BM25_FUNCTION_NAME,
    build_v16_schema,
    v16_milvus_field_specs,
)
from .v16_analyzer import V16AnalyzerConfig, resolve_v16_analyzer

EXPERT_V16_PROFILE = VectorRepositoryProfile(
    schema_field_specs=v16_milvus_field_specs,
    schema_builder=build_v16_schema,
    index_specs_fn=index_specs_v16,
    row_mapper_cls=MilvusV16RowMapper,
    full_scan_filter='vector_doc_id != ""',
    model_space_fields=("metadata",),
    contract_adapter_cls=ExpertContractAdapter,
    owner_field="expert_id",
    vector_doc_id_prefix="expert:",
    owner_filter_fn=lambda owner_id: f"expert_id == {int(owner_id)}",
    owner_output_fields=("vector_doc_id",),
    dense_vector_field="dense_embedding",
    non_retrievable_fields=("sparse_embedding",),
)


class ContractMilvusV16VectorRepository(MetadataModelSpaceRepository):
    """复用通用 CRUD 和共享 metadata 模型空间采样的 Expert V16 Repository。"""

    def __init__(
        self,
        settings: MilvusSettings,
        vector_dimension: int | None = None,
        *,
        client: Any | None = None,
        allow_collection_creation: bool = False,
        profile: VectorRepositoryProfile[Any] = EXPERT_V16_PROFILE,
        analyzer_config: V16AnalyzerConfig | None = None,
    ) -> None:
        if (
            settings.expert_storage_profile != "v16"
            or settings.collection_name != EXPERT_V16_COLLECTION_NAME
            or profile is not EXPERT_V16_PROFILE
        ):
            raise ValueError(
                "Expert V16 Repository 必须绑定 "
                "EXPERT_MILVUS_STORAGE_PROFILE=v16 和 "
                f"Collection {EXPERT_V16_COLLECTION_NAME!r}"
            )
        self._analyzer_config = analyzer_config or resolve_v16_analyzer(settings)
        if (
            self._analyzer_config.profile != settings.expert_v16_analyzer
            or (
                settings.expert_v16_analyzer == "jieba_targeted"
                and (
                    self._analyzer_config.dictionary_version != settings.expert_v16_dictionary_version
                    or self._analyzer_config.dictionary_sha256 != settings.expert_v16_dictionary_sha256
                )
            )
        ):
            raise ValueError("V16 Analyzer 快照与构造配置不一致")
        super().__init__(
            settings,
            vector_dimension,
            client=client,
            allow_collection_creation=allow_collection_creation,
            profile=profile,
        )

    @property
    def analyzer_config(self) -> V16AnalyzerConfig:
        return self._analyzer_config

    def _build_schema(self, vector_dimension: int | None) -> Any:
        return build_v16_schema(
            self.settings, vector_dimension, analyzer_config=self.analyzer_config
        )

    def storage_manifest(self) -> dict[str, Any]:
        """离线 Schema/索引/Function/词典身份，用于创建报告和断点恢复。"""

        manifest = {
            "collection_name": self.settings.collection_name,
            "fields": self.contract_schema_summary(),
            "indexes": self.expected_contract_indexes(),
            "function": {
                "name": V16_BM25_FUNCTION_NAME,
                "type": "BM25",
                "input_field_names": ["chunk_text"],
                "output_field_names": ["sparse_embedding"],
            },
            "analyzer": self.analyzer_config.summary(),
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**manifest, "schema_fingerprint": hashlib.sha256(encoded).hexdigest()}

    def _validate_existing_collection(self, description: Mapping[str, Any]) -> None:
        ContractMilvusRepositoryBase._validate_existing_collection(self, description)
        fields = {
            str(field.get("name")): field
            for field in description.get("fields", [])
            if isinstance(field, Mapping)
        }
        text_params = fields.get("chunk_text", {}).get("params", {})
        if not isinstance(text_params, Mapping):
            raise TypeError("V16 chunk_text 缺少 Analyzer 参数")
        analyzer = text_params.get("analyzer_params")
        if isinstance(analyzer, str):
            try:
                analyzer = json.loads(analyzer)
            except json.JSONDecodeError as exc:
                raise ValueError("V16 chunk_text Analyzer 参数不是合法 JSON") from exc
        if (
            text_params.get("enable_analyzer") not in (True, "true")
            or analyzer != self.analyzer_config.params
        ):
            # 禁止打印大词典；只报告已固定的配置身份。
            raise ValueError(
                "V16 chunk_text Analyzer 或词典不兼容: "
                f"expected={json.dumps(self.analyzer_config.summary(), sort_keys=True)}"
            )

        functions = description.get("functions", [])
        matches = [
            item
            for item in functions
            if isinstance(item, Mapping)
            and item.get("name") == V16_BM25_FUNCTION_NAME
        ]
        if len(matches) != 1:
            raise ValueError("V16 Collection 缺少唯一的 chunk_text_bm25 Function")
        function = matches[0]
        function_type = function.get("type")
        type_name = str(getattr(function_type, "name", function_type)).upper()
        if (
            type_name != "BM25"
            or function.get("input_field_names") != ["chunk_text"]
            or function.get("output_field_names") != ["sparse_embedding"]
        ):
            raise ValueError("V16 BM25 Function 定义不兼容")

    def get_many_by_vector_doc_ids(
        self,
        vector_doc_ids: Sequence[str],
        *,
        include_embedding: bool = True,
    ) -> dict[str, dict[str, Any]]:
        rows = ContractMilvusRepositoryBase.get_many_by_vector_doc_ids(
            self,
            vector_doc_ids,
            include_embedding=include_embedding,
        )
        for row in rows.values():
            validate_v16_mirrors(row)
        return rows

    def query_by_owner_id(
        self,
        owner_id: int,
        *,
        output_fields: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        if isinstance(owner_id, bool) or not isinstance(owner_id, int) or owner_id < 1:
            raise ValueError("owner_id 必须是正整数")
        fields = list(
            output_fields
            if output_fields is not None
            else self.profile.owner_output_fields
        )
        if not fields or any(
            not isinstance(field, str) or not field.strip() for field in fields
        ):
            raise ValueError("output_fields 必须是非空字符串序列")
        forbidden = set(fields) & set(self.profile.non_retrievable_fields)
        if forbidden:
            raise ValueError(
                f"V16 Function 输出字段不能作为普通 query 字段: {sorted(forbidden)!r}"
            )
        validation_fields = {
            "vector_doc_id",
            "expert_id",
            "chunk_type",
            "is_active",
            "eligible_for_recommendation",
            "job_status_norm",
            "abroad_flag_norm",
            "work_years",
            "metadata",
        }
        rows = ContractMilvusRepositoryBase.query_by_owner_id(
            self,
            owner_id,
            output_fields=list(dict.fromkeys([*fields, *sorted(validation_fields)])),
        )
        for row in rows:
            validate_v16_mirrors(row)
        return [{field: row[field] for field in fields if field in row} for row in rows]

    def list_all_vector_states(
        self,
        *,
        output_fields: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        if output_fields is not None:
            forbidden = set(output_fields) & {"content_hash", "metadata_hash"}
            if forbidden:
                raise ValueError(
                    "V16 hash 位于 metadata JSON，不是一级字段: "
                    f"{sorted(forbidden)!r}"
                )
        fields = (
            list(output_fields)
            if output_fields is not None
            else ["vector_doc_id", "metadata"]
        )
        forbidden = set(fields) & set(self.profile.non_retrievable_fields)
        if forbidden:
            raise ValueError(
                f"V16 Function 输出字段不能作为普通 query 字段: {sorted(forbidden)!r}"
            )
        rows = self._scan_mirror_rows(fields)
        if output_fields is not None and "metadata" not in output_fields:
            return rows
        return [self._light_state(row, include_active=False) for row in rows]

    def list_all_vector_activity_states(self) -> list[dict[str, object]]:
        rows = self._scan_mirror_rows(
            ["vector_doc_id", "is_active", "metadata"]
        )
        return [self._light_state(row, include_active=True) for row in rows]

    def _scan_mirror_rows(self, fields: list[str]) -> list[dict[str, object]]:
        required = {
            "vector_doc_id",
            "expert_id",
            "chunk_type",
            "is_active",
            "eligible_for_recommendation",
            "job_status_norm",
            "abroad_flag_norm",
            "work_years",
            "metadata",
        }
        query_fields = list(dict.fromkeys([*fields, *sorted(required)]))
        iterator = self._get_client().query_iterator(
            collection_name=self.settings.collection_name,
            batch_size=1000,
            limit=-1,
            filter=self.profile.full_scan_filter,
            output_fields=query_fields,
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        result: list[dict[str, object]] = []
        seen: set[str] = set()
        try:
            while True:
                batch = iterator.next()
                if not isinstance(batch, list) or not all(
                    isinstance(row, Mapping) for row in batch
                ):
                    raise TypeError("Milvus V16 iterator 返回了非法 Entity 批次")
                if not batch:
                    break
                for raw in batch:
                    row = dict(raw)
                    vector_doc_id = row.get("vector_doc_id")
                    if not isinstance(vector_doc_id, str) or not vector_doc_id:
                        raise RuntimeError("Milvus V16 行缺少合法 vector_doc_id")
                    if vector_doc_id in seen:
                        raise RuntimeError(
                            f"Milvus V16 扫描返回重复 vector_doc_id: {vector_doc_id!r}"
                        )
                    seen.add(vector_doc_id)
                    validate_v16_mirrors(row)
                    result.append({key: row[key] for key in fields if key in row})
        finally:
            iterator.close()
        return result

    @staticmethod
    def _light_state(
        row: Mapping[str, object],
        *,
        include_active: bool,
    ) -> dict[str, object]:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("Milvus V16 行缺少合法 metadata")
        result = dict(row)
        result.pop("metadata", None)
        result["content_hash"] = read_hash_field(
            metadata, "content_hash", required=True
        )
        result["metadata_hash"] = read_hash_field(
            metadata, "metadata_hash", required=True
        )
        if include_active:
            is_active = row.get("is_active")
            if not isinstance(is_active, bool):
                raise RuntimeError("Milvus V16 顶层 is_active 必须是布尔值")
            result["is_active"] = is_active
        return result


__all__ = ["EXPERT_V16_PROFILE", "ContractMilvusV16VectorRepository"]
