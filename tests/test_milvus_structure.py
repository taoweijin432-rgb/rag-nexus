from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

import rag_vector_store
from rag_vector_store.config import DEFAULT_MILVUS_COLLECTION_NAME, MilvusSettings


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
        destructive_maintenance_collections=("a", "b"),
    )
    assert settings.destructive_maintenance_collections == ("a", "b")


def test_default_collection_is_the_formal_v15_contract() -> None:
    """正式运行默认目标必须是唯一正式 Collection。"""

    assert MilvusSettings().collection_name == DEFAULT_MILVUS_COLLECTION_NAME
    assert (
        MilvusSettings.from_env({}).collection_name
        == DEFAULT_MILVUS_COLLECTION_NAME
        == "expert_candidates_v15_contract"
    )
    # 环境变量显式覆盖仍然生效。
    overridden = MilvusSettings.from_env(
        {"MILVUS_COLLECTION_NAME": "custom_collection"}
    )
    assert overridden.collection_name == "custom_collection"


@pytest.mark.parametrize(
    "retired_name",
    [
        "expert_candidates_v15_contract_dev",
        "expert_candidates_v14_contract_dev",
        "expert_candidates_v14_contract_e2e",
    ],
)
def test_retired_collection_names_never_become_formal_default(
    retired_name: str,
) -> None:
    """历史 dev/v14 名称不得成为任何正式入口的默认值。"""

    assert MilvusSettings().collection_name != retired_name
    assert MilvusSettings.from_env({}).collection_name != retired_name


def test_no_retired_collection_names_hardcoded_in_production_code() -> None:
    """rag_vector_store 生产代码不得硬编码历史 Collection 名。

    唯一允许出现集合名的位置是 config.py 中指向正式名称的
    ``DEFAULT_MILVUS_COLLECTION_NAME``；历史 dev/v14/e2e 名称只能通过
    环境变量或显式参数出现在测试/脚本中。
    """

    forbidden = (
        "expert_candidates_v15_contract_dev",
        "expert_candidates_v14_contract_dev",
        "expert_candidates_v14_contract_e2e",
    )
    package_root = Path(rag_vector_store.__file__).resolve().parent
    hits: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            if name in text:
                hits.append(f"{path.relative_to(package_root)}: {name}")
    assert hits == [], f"生产代码中残留历史 Collection 名：{hits}"
