from __future__ import annotations

from typing import Any

import pytest

from rag_vector_store.config import MilvusSettings
from rag_vector_store.milvus.compatibility import (
    compare_model_space,
    compare_schema_specs,
)
from rag_vector_store.milvus.indexes import index_specs
from rag_vector_store.milvus.schemas import (
    CONTRACT_SCALAR_INDEX_FIELDS,
    contract_milvus_field_specs,
)

CONTRACT_FIELD_NAMES = (
    "vector_doc_id",
    "chunk_id",
    "expert_id",
    "serial_number",
    "chunk_type",
    "chunk_subtype",
    "chunk_index",
    "chunk_total",
    "chunk_text",
    "char_count",
    "content_hash",
    "embedding",
    "embedding_dim",
    "embedding_model_alias",
    "embedding_model_version",
    "company_name_norm",
    "company_normalization_status",
    "role_code",
    "role_level",
    "start_month",
    "end_month",
    "is_current",
    "origin_type",
    "verification_status",
    "weight_tier",
    "is_active",
    "source_table",
    "source_record_id",
    "source_item_key",
    "source_field",
    "source_hash",
    "contract_version",
    "chunk_policy_version",
    "metadata_schema_version",
    "data_version",
    "updated_at",
    "embedded_at",
    "metadata_ext",
    "source_ext",
    "versions_ext",
)


def _contract_settings() -> MilvusSettings:
    return MilvusSettings(
        collection_name="expert_candidates_v12_contract_dev",
        index_type="HNSW",
        metric_type="COSINE",
        index_params={"M": 16, "efConstruction": 200},
    )


def test_contract_schema_mode_is_the_only_runtime_mode() -> None:
    assert MilvusSettings().schema_mode == "contract"
    assert MilvusSettings.from_env({}).schema_mode == "contract"
    with pytest.raises(ValueError):
        MilvusSettings(schema_mode="legacy")
    with pytest.raises(ValueError):
        MilvusSettings(schema_mode="expert")


def _description_from_specs(specs: tuple[Any, ...]) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for spec in specs:
        params: dict[str, Any] = {}
        if spec.max_length is not None:
            params["max_length"] = spec.max_length
        if spec.dim is not None:
            params["dim"] = spec.dim
        fields.append(
            {
                "name": spec.name,
                "type": spec.datatype,
                "nullable": spec.nullable,
                "is_primary": spec.primary_key,
                "params": params,
            }
        )
    return {
        "auto_id": False,
        "enable_dynamic_field": False,
        "fields": fields,
    }


def test_contract_schema_provider_preserves_the_40_field_contract() -> None:
    specs = contract_milvus_field_specs(1024)

    assert len(specs) == 40
    assert tuple(spec.name for spec in specs) == CONTRACT_FIELD_NAMES
    assert specs[11].name == "embedding"
    assert specs[11].dim == 1024


def test_index_provider_preserves_contract_vector_and_scalar_indexes() -> None:
    specs = index_specs(_contract_settings())
    vector = specs[0]
    scalar = specs[1:]

    assert vector.summary(include_name=True) == {
        "index_name": "embedding_vector",
        "field_name": "embedding",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    assert len(scalar) == 11
    assert tuple(spec.field_name for spec in scalar) == CONTRACT_SCALAR_INDEX_FIELDS
    assert all(spec.index_type == "INVERTED" for spec in scalar)


def test_schema_compatibility_is_fail_closed_and_model_space_is_explicit() -> None:
    specs = contract_milvus_field_specs(1024)
    description = _description_from_specs(specs)
    description["fields"] = description["fields"][1:]

    schema_difference = compare_schema_specs(description, specs)
    assert schema_difference.compatible is False
    assert schema_difference.details["missing_fields"] == ["vector_doc_id"]

    dynamic_description = _description_from_specs(specs)
    dynamic_description["enable_dynamic_field"] = True
    dynamic_difference = compare_schema_specs(dynamic_description, specs)
    assert dynamic_difference.compatible is False
    assert "enable_dynamic_field" in dynamic_difference.details["collection_options"]

    model_difference = compare_model_space(
        [
            {
                "embedding_model_alias": "text-embedding-v4",
                "embedding_model_version": "another-revision",
                "embedding_dim": 1024,
            }
        ],
        embedding_model_alias="text-embedding-v4",
        embedding_model_version="v1",
        embedding_dimension=1024,
    )
    assert model_difference.compatible is False
    assert model_difference.details["mismatched_rows"] == [0]
