"""外部 Jieba Expert 正式 Repository，复用通用 CRUD 和状态查询。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import MilvusSettings
from ..tokenization.jieba_index import JiebaIndexConfig, JiebaIndexTokenizer
from .contract_v16_external import MilvusV16ExternalRowMapper
from .repository import ContractMilvusRepositoryBase
from .repository_metadata import MetadataModelSpaceRepository
from .repository_v16 import EXPERT_V16_PROFILE, ContractMilvusV16VectorRepository
from .schemas.contract_v16 import V16_BM25_FUNCTION_NAME
from .schemas.contract_v16_external import (
    build_external_schema,
    external_field_specs,
    schema_description,
)

EXPERT_V16_EXTERNAL_PROFILE = replace(
    EXPERT_V16_PROFILE,
    schema_field_specs=external_field_specs,
    schema_builder=build_external_schema,
    row_mapper_cls=MilvusV16ExternalRowMapper,
)


def config_from_settings(settings: MilvusSettings) -> JiebaIndexConfig:
    if settings.expert_storage_profile != "v16_external_formal":
        raise ValueError("外部分词 Repository 必须绑定外部 Jieba Profile")
    if not all(
        (
            settings.expert_v16_dictionary_path,
            settings.expert_v16_dictionary_version,
            settings.expert_v16_dictionary_sha256,
        )
    ):
        raise ValueError("外部分词 Repository 缺少词典身份")
    return JiebaIndexConfig(
        Path(settings.expert_v16_dictionary_path),
        settings.expert_v16_dictionary_version,
        settings.expert_v16_dictionary_sha256,
        settings.expert_v16_external_hmm,
    )


class ContractMilvusV16ExternalVectorRepository(ContractMilvusV16VectorRepository):
    """继承 V16 mirrors/状态查询；原生 Analyzer 门禁替换为外部分词身份门禁。"""

    def __init__(
        self,
        settings: MilvusSettings,
        vector_dimension: int | None = None,
        *,
        client: Any | None = None,
        allow_collection_creation: bool = False,
        profile: Any = EXPERT_V16_EXTERNAL_PROFILE,
        tokenizer: JiebaIndexTokenizer | None = None,
    ) -> None:
        config = config_from_settings(settings)
        expected_names = {config.formal_collection_name}
        if (
            profile is not EXPERT_V16_EXTERNAL_PROFILE
            or settings.collection_name not in expected_names
        ):
            raise ValueError("外部分词集合名称必须匹配 HMM/词典/运行时的配置指纹")
        self.tokenizer = tokenizer or JiebaIndexTokenizer(config)
        if self.tokenizer.config != config:
            raise ValueError("Repository 与注入的分词器配置不一致")
        # 跳过原生 V16 的构造门禁；CRUD 仍由同一个通用基类提供。
        MetadataModelSpaceRepository.__init__(
            self,
            settings,
            vector_dimension,
            client=client,
            allow_collection_creation=allow_collection_creation,
            profile=profile,
            row_mapper=MilvusV16ExternalRowMapper(self.tokenizer),
        )

    def _build_schema(self, vector_dimension: int | None) -> Any:
        return build_external_schema(self.settings, vector_dimension, config=self.tokenizer.config)

    def storage_manifest(self) -> dict[str, Any]:
        manifest = {
            "collection_name": self.settings.collection_name,
            "fields": self.contract_schema_summary(),
            "indexes": self.expected_contract_indexes(),
            "tokenization": self.tokenizer.config.identity(),
            "term_count": self.tokenizer.term_count,
            "index_fingerprint": self.tokenizer.config.fingerprint,
            "function_input": "chunk_text_bm25",
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**manifest, "schema_fingerprint": hashlib.sha256(encoded).hexdigest()}

    def _validate_existing_collection(self, description: Mapping[str, Any]) -> None:
        ContractMilvusRepositoryBase._validate_existing_collection(self, description)
        if description.get("description") != schema_description(self.tokenizer.config):
            raise ValueError("集合持久化的分词身份不匹配，拒绝混用 token 流")
        fields = {field["name"]: field for field in description.get("fields", [])}
        params = fields.get("chunk_text_bm25", {}).get("params", {})
        analyzer = params.get("analyzer_params")
        if isinstance(analyzer, str):
            analyzer = json.loads(analyzer)
        if params.get("enable_analyzer") not in (True, "true") or analyzer != {
            "tokenizer": "whitespace"
        }:
            raise ValueError("外部分词集合必须使用 whitespace Analyzer")
        matches = [
            f for f in description.get("functions", []) if f.get("name") == V16_BM25_FUNCTION_NAME
        ]
        if len(matches) != 1:
            raise ValueError("外部分词集合必须有唯一的 BM25 Function")
        function = matches[0]
        function_type = function.get("type")
        if (
            str(getattr(function_type, "name", function_type)).upper() != "BM25"
            or function.get("input_field_names") != ["chunk_text_bm25"]
            or function.get("output_field_names") != ["sparse_embedding"]
        ):
            raise ValueError("外部分词 BM25 Function 不兼容")
