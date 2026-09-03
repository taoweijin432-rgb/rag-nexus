"""VectorIngestionService 公共契约包。

本包承载面向 rag_vector_store 调用方的稳定 Python 向量能力接口：
    - 公共结果模型（``models``）；
    - 公共异常层级（``errors``）；
    - ``VectorIngestionService`` Facade（upsert / update_metadata / delete /
      get / list 编排）；
    - ``create_vector_ingestion_service`` 生产配置工厂。
"""

from .errors import (
    VectorContractError,
    VectorEmbeddingError,
    VectorInconsistentStateError,
    VectorIngestionError,
    VectorMilvusError,
)
from .factory import (
    create_resume_vector_ingestion_service,
    create_vector_ingestion_service,
)
from .models import (
    VectorDeletionResult,
    VectorExpertList,
    VectorIngestionResult,
    VectorItemResult,
    VectorItemStatus,
    VectorQueryResult,
    VectorResumeList,
)
from .vector_ingestion import ResumeVectorIngestionService, VectorIngestionService

__all__ = [
    "ResumeVectorIngestionService",
    "VectorContractError",
    "VectorDeletionResult",
    "VectorEmbeddingError",
    "VectorExpertList",
    "VectorInconsistentStateError",
    "VectorIngestionError",
    "VectorIngestionResult",
    "VectorIngestionService",
    "VectorItemResult",
    "VectorItemStatus",
    "VectorMilvusError",
    "VectorQueryResult",
    "VectorResumeList",
    "create_resume_vector_ingestion_service",
    "create_vector_ingestion_service",
]
