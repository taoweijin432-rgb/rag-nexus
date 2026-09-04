"""正式向量化公共 API。

根包只登记导出，不在导入阶段加载 Expert Contract、Embedding Adapter 或
PyMilvus。具体能力在首次访问对应导出时按需加载。
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # expert_contracts
    "EXPERT_CHUNK_CONTRACT_VERSION": ("expert_contracts", "EXPERT_CHUNK_CONTRACT_VERSION"),
    "ContractPreparedChunk": ("expert_contracts", "ContractPreparedChunk"),
    "EmbeddedContractPreparedChunk": ("expert_contracts", "EmbeddedContractPreparedChunk"),
    "EmploymentMetadata": ("expert_contracts", "EmploymentMetadata"),
    "ExpertChunkEmbeddingInput": ("expert_contracts", "ExpertChunkEmbeddingInput"),
    "ExpertChunkSource": ("expert_contracts", "ExpertChunkSource"),
    "ExpertChunkVersions": ("expert_contracts", "ExpertChunkVersions"),
    "ProfileSummaryMetadata": ("expert_contracts", "ProfileSummaryMetadata"),
    "ProjectMetadata": ("expert_contracts", "ProjectMetadata"),
    # chunking
    "EXPECTED_EMBEDDING_DIMENSION": (
        ".chunking",
        "EXPECTED_EMBEDDING_DIMENSION",
    ),
    "ContractBatchIngestionResult": (".chunking", "ContractBatchIngestionResult"),
    "ContractBatchIngestionService": (".chunking", "ContractBatchIngestionService"),
    "ContractEmbeddingResult": (".chunking", "ContractEmbeddingResult"),
    "ContractEmbeddingWorker": (".chunking", "ContractEmbeddingWorker"),
    "ContractIngestionError": (".chunking", "ContractIngestionError"),
    "ContractIngestionResult": (".chunking", "ContractIngestionResult"),
    "ContractIngestionService": (".chunking", "ContractIngestionService"),
    "ContractRepository": (".chunking", "ContractRepository"),
    "ContractValidationError": (".chunking", "ContractValidationError"),
    "ContractValidator": (".chunking", "ContractValidator"),
    "EmbeddingEligibilityError": (".chunking", "EmbeddingEligibilityError"),
    "EmbeddingEligibilityValidator": (".chunking", "EmbeddingEligibilityValidator"),
    "SkippedContractChunk": (".chunking", "SkippedContractChunk"),
    "compare_contract_output_schema": (".chunking", "compare_contract_output_schema"),
    "expected_contract_output_fields": (".chunking", "expected_contract_output_fields"),
    # configuration and embedding
    "MilvusSettings": (".config", "MilvusSettings"),
    "EmbeddingModel": (".embedding", "EmbeddingModel"),
    "CompanyCloudEmbeddingError": (
        ".embedding_adapters",
        "CompanyCloudEmbeddingError",
    ),
    "CompanyCloudEmbeddingModel": (
        ".embedding_adapters",
        "CompanyCloudEmbeddingModel",
    ),
    "CompanyCloudEmbeddingSettings": (
        ".embedding_adapters",
        "CompanyCloudEmbeddingSettings",
    ),
    # ingestion
    "ResumeVectorIngestionService": (
        ".ingestion",
        "ResumeVectorIngestionService",
    ),
    "VectorContractError": (".ingestion", "VectorContractError"),
    "VectorDeletionResult": (".ingestion", "VectorDeletionResult"),
    "VectorEmbeddingError": (".ingestion", "VectorEmbeddingError"),
    "VectorExpertList": (".ingestion", "VectorExpertList"),
    "VectorInconsistentStateError": (
        ".ingestion",
        "VectorInconsistentStateError",
    ),
    "VectorIngestionError": (".ingestion", "VectorIngestionError"),
    "VectorIngestionResult": (".ingestion", "VectorIngestionResult"),
    "VectorIngestionService": (".ingestion", "VectorIngestionService"),
    "VectorItemResult": (".ingestion", "VectorItemResult"),
    "VectorItemStatus": (".ingestion", "VectorItemStatus"),
    "VectorMilvusError": (".ingestion", "VectorMilvusError"),
    "VectorOwnerList": (".ingestion", "VectorOwnerList"),
    "VectorQueryResult": (".ingestion", "VectorQueryResult"),
    "VectorResumeList": (".ingestion", "VectorResumeList"),
    "create_resume_vector_ingestion_service": (
        ".ingestion",
        "create_resume_vector_ingestion_service",
    ),
    "create_vector_ingestion_service": (
        ".ingestion",
        "create_vector_ingestion_service",
    ),
    # Milvus
    "V15_CONTRACT_VECTOR_DIMENSION": (
        ".milvus.contract_v15",
        "V15_CONTRACT_VECTOR_DIMENSION",
    ),
    "V15_VECTOR_DOC_ID_MAX_BYTES": (
        ".milvus.contract_v15",
        "V15_VECTOR_DOC_ID_MAX_BYTES",
    ),
    "ContractV15RowMapper": (".milvus.contract_v15", "ContractV15RowMapper"),
    "MilvusV15Entity": (".milvus.contract_v15", "MilvusV15Entity"),
    "MilvusV15RowMapper": (".milvus.contract_v15", "MilvusV15RowMapper"),
    "ResumeMilvusV15RowMapper": (
        ".milvus.contract_v15",
        "ResumeMilvusV15RowMapper",
    ),
    "ContractMilvusRepositoryBase": (
        ".milvus.repository",
        "ContractMilvusRepositoryBase",
    ),
    "ContractMilvusV15VectorRepository": (
        ".milvus.repository_v15",
        "ContractMilvusV15VectorRepository",
    ),
    "MilvusFieldSpec": (".milvus.schemas", "MilvusFieldSpec"),
    # model space and shared models
    "FORMAL_EMBEDDING_DIMENSION": (".model_space", "FORMAL_EMBEDDING_DIMENSION"),
    "FORMAL_EMBEDDING_MODEL": (".model_space", "FORMAL_EMBEDDING_MODEL"),
    "FORMAL_EMBEDDING_MODEL_ALIAS": (
        ".model_space",
        "FORMAL_EMBEDDING_MODEL_ALIAS",
    ),
    "FORMAL_EMBEDDING_MODEL_SPACE_ID": (
        ".model_space",
        "FORMAL_EMBEDDING_MODEL_SPACE_ID",
    ),
    "FORMAL_EMBEDDING_MODEL_VERSION": (
        ".model_space",
        "FORMAL_EMBEDDING_MODEL_VERSION",
    ),
    "FORMAL_EMBEDDING_PROVIDER": (".model_space", "FORMAL_EMBEDDING_PROVIDER"),
    "build_model_space_id": (".model_space", "build_model_space_id"),
    "MutationResult": (".models", "MutationResult"),
    # reconciliation
    "FullVectorReconciliationResult": (
        ".reconciliation",
        "FullVectorReconciliationResult",
    ),
    "FullVectorReconciliationService": (
        ".reconciliation",
        "FullVectorReconciliationService",
    ),
    "create_full_vector_reconciliation_service": (
        ".reconciliation",
        "create_full_vector_reconciliation_service",
    ),
}

_DEPRECATED_EXPORTS = {
    "VectorExpertList": "VectorExpertList 已弃用；请使用 VectorOwnerList",
    "VectorResumeList": "VectorResumeList 已弃用；请使用 VectorOwnerList",
}

__all__ = [
    "EXPECTED_EMBEDDING_DIMENSION",
    "EXPERT_CHUNK_CONTRACT_VERSION",
    "FORMAL_EMBEDDING_DIMENSION",
    "FORMAL_EMBEDDING_MODEL",
    "FORMAL_EMBEDDING_MODEL_ALIAS",
    "FORMAL_EMBEDDING_MODEL_SPACE_ID",
    "FORMAL_EMBEDDING_MODEL_VERSION",
    "FORMAL_EMBEDDING_PROVIDER",
    "V15_CONTRACT_VECTOR_DIMENSION",
    "V15_VECTOR_DOC_ID_MAX_BYTES",
    "CompanyCloudEmbeddingError",
    "CompanyCloudEmbeddingModel",
    "CompanyCloudEmbeddingSettings",
    "ContractBatchIngestionResult",
    "ContractBatchIngestionService",
    "ContractEmbeddingResult",
    "ContractEmbeddingWorker",
    "ContractIngestionError",
    "ContractIngestionResult",
    "ContractIngestionService",
    "ContractMilvusRepositoryBase",
    "ContractMilvusV15VectorRepository",
    "ContractPreparedChunk",
    "ContractRepository",
    "ContractV15RowMapper",
    "ContractValidationError",
    "ContractValidator",
    "EmbeddedContractPreparedChunk",
    "EmbeddingEligibilityError",
    "EmbeddingEligibilityValidator",
    "EmbeddingModel",
    "EmploymentMetadata",
    "ExpertChunkEmbeddingInput",
    "ExpertChunkSource",
    "ExpertChunkVersions",
    "FullVectorReconciliationResult",
    "FullVectorReconciliationService",
    "MilvusFieldSpec",
    "MilvusSettings",
    "MilvusV15Entity",
    "MilvusV15RowMapper",
    "MutationResult",
    "ProfileSummaryMetadata",
    "ProjectMetadata",
    "ResumeMilvusV15RowMapper",
    "ResumeVectorIngestionService",
    "SkippedContractChunk",
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
    "VectorOwnerList",
    "VectorQueryResult",
    "VectorResumeList",
    "build_model_space_id",
    "compare_contract_output_schema",
    "create_full_vector_reconciliation_service",
    "create_resume_vector_ingestion_service",
    "create_vector_ingestion_service",
    "expected_contract_output_fields",
]


def __getattr__(name: str) -> Any:
    """首次访问公共导出时加载其实际定义，并缓存到根包命名空间。"""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    warning = _DEPRECATED_EXPORTS.get(name)
    if warning is not None:
        warnings.warn(warning, DeprecationWarning, stacklevel=2)
    module_name, attribute = target
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让交互式补全看到全部懒加载公共名称。"""

    return sorted({*globals(), *__all__})


if TYPE_CHECKING:
    from expert_contracts import (
        EXPERT_CHUNK_CONTRACT_VERSION,
        ContractPreparedChunk,
        EmbeddedContractPreparedChunk,
        EmploymentMetadata,
        ExpertChunkEmbeddingInput,
        ExpertChunkSource,
        ExpertChunkVersions,
        ProfileSummaryMetadata,
        ProjectMetadata,
    )

    from .chunking import (
        EXPECTED_EMBEDDING_DIMENSION,
        ContractBatchIngestionResult,
        ContractBatchIngestionService,
        ContractEmbeddingResult,
        ContractEmbeddingWorker,
        ContractIngestionError,
        ContractIngestionResult,
        ContractIngestionService,
        ContractRepository,
        ContractValidationError,
        ContractValidator,
        EmbeddingEligibilityError,
        EmbeddingEligibilityValidator,
        SkippedContractChunk,
        compare_contract_output_schema,
        expected_contract_output_fields,
    )
    from .config import MilvusSettings
    from .embedding import EmbeddingModel
    from .embedding_adapters import (
        CompanyCloudEmbeddingError,
        CompanyCloudEmbeddingModel,
        CompanyCloudEmbeddingSettings,
    )
    from .ingestion import (
        ResumeVectorIngestionService,
        VectorContractError,
        VectorDeletionResult,
        VectorEmbeddingError,
        VectorExpertList,
        VectorInconsistentStateError,
        VectorIngestionError,
        VectorIngestionResult,
        VectorIngestionService,
        VectorItemResult,
        VectorItemStatus,
        VectorMilvusError,
        VectorOwnerList,
        VectorQueryResult,
        VectorResumeList,
        create_resume_vector_ingestion_service,
        create_vector_ingestion_service,
    )
    from .milvus.contract_v15 import (
        V15_CONTRACT_VECTOR_DIMENSION,
        V15_VECTOR_DOC_ID_MAX_BYTES,
        ContractV15RowMapper,
        MilvusV15Entity,
        MilvusV15RowMapper,
        ResumeMilvusV15RowMapper,
    )
    from .milvus.repository import ContractMilvusRepositoryBase
    from .milvus.repository_v15 import ContractMilvusV15VectorRepository
    from .milvus.schemas import MilvusFieldSpec
    from .model_space import (
        FORMAL_EMBEDDING_DIMENSION,
        FORMAL_EMBEDDING_MODEL,
        FORMAL_EMBEDDING_MODEL_ALIAS,
        FORMAL_EMBEDDING_MODEL_SPACE_ID,
        FORMAL_EMBEDDING_MODEL_VERSION,
        FORMAL_EMBEDDING_PROVIDER,
        build_model_space_id,
    )
    from .models import MutationResult
    from .reconciliation import (
        FullVectorReconciliationResult,
        FullVectorReconciliationService,
        create_full_vector_reconciliation_service,
    )
