from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

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
    "expert_id",
    "serial_number",
    "chunk_type",
    "chunk_subtype",
    "chunk_index",
    "chunk_total",
    "chunk_text",
    "char_count",
    "content_hash",
    "metadata_hash",
    "contract_version",
    "verification_status",
    "eligible_for_recommendation",
    "weight_tier",
    "is_active",
    "updated_at",
    "embedded_at",
    "embedding",
    "embedding_dim",
    "embedding_model_alias",
    "embedding_model_version",
    "metadata_ext",
    "source_ext",
    "versions_ext",
)


def _contract_settings() -> MilvusSettings:
    return MilvusSettings(
        collection_name="expert_candidates_v14_contract_dev",
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


def test_milvus_defaults_match_v13_and_credentials_are_secret() -> None:
    settings = MilvusSettings.from_env(
        {
            "MILVUS_TOKEN": "token-value",
        }
    )

    assert settings.index_type == "HNSW"
    assert settings.metric_type == "COSINE"
    assert settings.index_params == {"M": 16, "efConstruction": 200}
    assert isinstance(settings.token, SecretStr)
    assert "token-value" not in repr(settings)
    assert settings.client_kwargs["token"] == "token-value"

    user_password_settings = MilvusSettings.from_env(
        {
            "MILVUS_USER": "user",
            "MILVUS_PASSWORD": "password-value",
        }
    )
    assert isinstance(user_password_settings.password, SecretStr)
    assert "password-value" not in repr(user_password_settings)
    assert user_password_settings.client_kwargs["password"] == "password-value"

    with pytest.raises(ValueError, match="either token"):
        MilvusSettings.from_env(
            {
                "MILVUS_TOKEN": "token-value",
                "MILVUS_USER": "user",
                "MILVUS_PASSWORD": "password-value",
            }
        )


def test_milvus_destructive_maintenance_collections_default_and_parse() -> None:
    # 默认空：不允许任何 Collection 执行 destructive cleanup（fail-closed）。
    assert MilvusSettings.from_env({}).destructive_maintenance_collections == ()
    assert MilvusSettings().destructive_maintenance_collections == ()

    settings = MilvusSettings.from_env(
        {
            "MILVUS_ALLOWED_DESTRUCTIVE_COLLECTIONS": (
                "expert_candidates_v15_contract_dev, stray ,  "
            ),
        }
    )
    assert settings.destructive_maintenance_collections == (
        "expert_candidates_v15_contract_dev",
        "stray",
    )
    assert settings.collection_name in settings.destructive_maintenance_collections


def test_milvus_destructive_maintenance_explicit_model_value() -> None:
    settings = MilvusSettings(
        destructive_maintenance_collections=["a", "b"],
    )
    assert settings.destructive_maintenance_collections == ("a", "b")


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


def test_contract_schema_provider_preserves_the_formal_field_contract() -> None:
    specs = contract_milvus_field_specs(1024)

    assert len(specs) == 25
    assert tuple(spec.name for spec in specs) == CONTRACT_FIELD_NAMES
    assert next(spec for spec in specs if spec.name == "embedding").dim == 1024
    assert next(spec for spec in specs if spec.name == "vector_doc_id").primary_key is True
    # 已废弃字段不得出现在正式 Schema
    for old_field in (
        "chunk_id",
        "source_item_key",
        "data_version",
        "company_normalization_status",
        "project_role",
        "dictionary_version",
    ):
        assert old_field not in {spec.name for spec in specs}
    # source_field 只进入 source_ext JSON，不保留独立 Scalar 列
    assert "source_field" not in {spec.name for spec in specs}
    source_ext = next(spec for spec in specs if spec.name == "source_ext")
    assert source_ext.datatype == "JSON"
    metadata_hash = next(spec for spec in specs if spec.name == "metadata_hash")
    assert metadata_hash.datatype == "VARCHAR"
    assert metadata_hash.max_length == 64



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
    assert len(scalar) == 3
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
