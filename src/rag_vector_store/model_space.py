"""正式 Embedding 模型空间的稳定标识定义。

这里的 ``v1`` 是项目侧模型空间版本，不是阿里云百炼供应商的内部 revision。
供应商没有提供可用 revision 时，项目仍使用可控且可审计的模型空间标识，禁止
使用 ``dev-unknown`` 这类占位值进入正式 Contract 写入。
"""

from __future__ import annotations

from typing import Final

FORMAL_EMBEDDING_PROVIDER: Final = "alibaba-bailian"
FORMAL_EMBEDDING_MODEL: Final = "text-embedding-v4"
FORMAL_EMBEDDING_MODEL_ALIAS: Final = (
    f"{FORMAL_EMBEDDING_PROVIDER}:{FORMAL_EMBEDDING_MODEL}"
)
FORMAL_EMBEDDING_DIMENSION: Final = 1024
FORMAL_EMBEDDING_MODEL_VERSION: Final = "v1"
FORMAL_EMBEDDING_MODEL_SPACE_ID: Final = (
    f"{FORMAL_EMBEDDING_MODEL_ALIAS}:{FORMAL_EMBEDDING_DIMENSION}:"
    f"{FORMAL_EMBEDDING_MODEL_VERSION}"
)
PLACEHOLDER_MODEL_VERSIONS: Final = frozenset({"dev-unknown"})


def build_model_space_id(
    *,
    model_alias: str,
    dimension: int,
    model_version: str,
) -> str:
    """由模型别名、向量维度和项目/供应商版本构造稳定空间标识。"""

    if not isinstance(model_alias, str) or not model_alias.strip():
        raise ValueError("embedding model alias 必须是非空字符串")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("embedding dimension 必须是正整数")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("embedding model version 必须是非空字符串")
    return f"{model_alias}:{dimension}:{model_version}"


def is_placeholder_model_version(model_version: str | None) -> bool:
    """判断模型版本是否仍是禁止写入正式空间的占位值。"""

    return model_version in PLACEHOLDER_MODEL_VERSIONS


__all__ = [
    "FORMAL_EMBEDDING_DIMENSION",
    "FORMAL_EMBEDDING_MODEL",
    "FORMAL_EMBEDDING_MODEL_ALIAS",
    "FORMAL_EMBEDDING_MODEL_SPACE_ID",
    "FORMAL_EMBEDDING_MODEL_VERSION",
    "FORMAL_EMBEDDING_PROVIDER",
    "PLACEHOLDER_MODEL_VERSIONS",
    "build_model_space_id",
    "is_placeholder_model_version",
]
