from __future__ import annotations

import pytest
from pydantic import SecretStr

from rag_vector_store.config import MilvusSettings


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


def test_milvus_destructive_maintenance_explicit_model_value() -> None:
    settings = MilvusSettings(
        destructive_maintenance_collections=["a", "b"],
    )
    assert settings.destructive_maintenance_collections == ("a", "b")
