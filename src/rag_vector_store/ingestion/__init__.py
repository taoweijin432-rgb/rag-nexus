"""VectorIngestionService 公共契约包。

结果模型和异常保持轻量导入；Service、Factory 等依赖 PyMilvus 的实现按需加载，
避免只读取公共模型时初始化外部基础设施依赖。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .errors import (
    VectorContractError,
    VectorEmbeddingError,
    VectorInconsistentStateError,
    VectorIngestionError,
    VectorMilvusError,
)
from .models import (
    VectorDeletionResult,
    VectorIngestionResult,
    VectorItemResult,
    VectorItemStatus,
    VectorOwnerList,
    VectorQueryResult,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ResumeVectorIngestionService": (".vector_ingestion", "ResumeVectorIngestionService"),
    "VectorIngestionService": (".vector_ingestion", "VectorIngestionService"),
    "create_resume_vector_ingestion_service": (
        ".factory",
        "create_resume_vector_ingestion_service",
    ),
    "create_vector_ingestion_service": (
        ".factory",
        "create_vector_ingestion_service",
    ),
}

__all__ = [
    "ResumeVectorIngestionService",
    "VectorContractError",
    "VectorDeletionResult",
    "VectorEmbeddingError",
    "VectorInconsistentStateError",
    "VectorIngestionError",
    "VectorIngestionResult",
    "VectorIngestionService",
    "VectorItemResult",
    "VectorItemStatus",
    "VectorMilvusError",
    "VectorOwnerList",
    "VectorQueryResult",
    "create_resume_vector_ingestion_service",
    "create_vector_ingestion_service",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
