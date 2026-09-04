"""FullVectorReconciliationService 的生产配置工厂。

调用方无需手工组装 ``ContractMilvusV15VectorRepository`` 与
``VectorIngestionService``，只调用 ``create_full_vector_reconciliation_service()``
即可从环境变量或注入组件组装完整服务。

一致性保证：
    本工厂复用 ``create_vector_ingestion_service()`` 构建 ``VectorIngestionService``，
    并直接以 ``ingestion.repository``（同一个对象）作为
    ``FullVectorReconciliationService.repository``。因此：

        - ``repository.list_all_vector_activity_states()``（读库 / 计算 stale 与终验）
        - ``ingestion.upsert_chunks / update_metadata / delete_vectors``（写库）

    必然作用于同一个 Milvus Collection，从结构上杜绝「读 Collection A、
    写 Collection B」的错配。

    若调用方显式注入 ``repository``，该对象会原样传回
    ``create_vector_ingestion_service(repository=...)``，因此 ingestion 内部
    与 reconciliation 仍共享同一 repository 对象，一致性不受影响。

    ``schema_profile``（仅 v15）控制 Repository 的持久化 profile，透传给
    ``create_vector_ingestion_service()``；reconciliation 与 ingestion 因此
    共享同一个 V15 Repository 实例。默认 ``v15``；V14 已退役，显式传 ``v14``
    会 fail-closed。
"""

from __future__ import annotations

from ..chunking.contract_pipeline import (
    EXPECTED_EMBEDDING_DIMENSION,
    ContractEmbeddingWorker,
    ContractValidator,
    EmbeddingEligibilityValidator,
)
from ..config import MilvusSettings
from ..embedding_adapters.company_cloud import CompanyCloudEmbeddingModel
from ..ingestion.factory import (
    SchemaProfile,
    _resolve_schema_profile,
    create_vector_ingestion_service,
)
from ..milvus.contract_v15 import MilvusV15RowMapper
from ..milvus.repository_v15 import ContractMilvusV15VectorRepository
from ..model_space import FORMAL_EMBEDDING_MODEL_SPACE_ID
from .full_reconciliation import (
    _DELETE_BATCH_SIZE,
    FullVectorReconciliationService,
)


def create_full_vector_reconciliation_service(
    *,
    embedding_model: CompanyCloudEmbeddingModel | None = None,
    milvus_settings: MilvusSettings | None = None,
    repository: ContractMilvusV15VectorRepository | None = None,
    embedding_worker: ContractEmbeddingWorker | None = None,
    validator: ContractValidator | None = None,
    row_mapper: MilvusV15RowMapper | None = None,
    eligibility_validator: EmbeddingEligibilityValidator | None = None,
    expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    formal_model_space_id: str = FORMAL_EMBEDDING_MODEL_SPACE_ID,
    delete_batch_size: int = _DELETE_BATCH_SIZE,
    schema_profile: SchemaProfile | None = None,
) -> FullVectorReconciliationService:
    """从环境配置（或注入组件）组装完整的 ``FullVectorReconciliationService``。

    依赖链：
        ``CompanyCloudEmbeddingModel.from_env()`` -> ``ContractEmbeddingWorker``
        -> ``ContractBatchIngestionService``；``MilvusSettings.from_env()`` ->
        ``schema_profile`` 对应的 Milvus Repository。全部经由
        ``create_vector_ingestion_service()`` 构建，随后以其内部同一 repository
        组装 ``FullVectorReconciliationService``。

    Args:
        schema_profile: 一个发布周期的兼容参数；省略时固定使用 ``v15``。
            显式传入 ``v15`` 仍可运行但会产生 ``DeprecationWarning``。
            V15 下 reconciliation 与 ingestion 共享同一个
            ``ContractMilvusV15VectorRepository`` 实例。
            显式传 ``v14`` 会 fail-closed（V14 已退役）。

    允许通过关键字注入部分或全部依赖用于测试 / 自定义组装，未注入项自动从环境读取；
    每个参数语义与 ``create_vector_ingestion_service()`` 一致。
    """

    _resolve_schema_profile(schema_profile)

    if validator is None:
        validator = ContractValidator()
    ingestion = create_vector_ingestion_service(
        embedding_model=embedding_model,
        milvus_settings=milvus_settings,
        repository=repository,
        embedding_worker=embedding_worker,
        validator=validator,
        row_mapper=row_mapper,
        expected_dimension=expected_dimension,
        formal_model_space_id=formal_model_space_id,
        schema_profile=None,
    )
    return FullVectorReconciliationService(
        ingestion.repository,
        ingestion=ingestion,
        validator=validator,
        eligibility_validator=eligibility_validator,
        delete_batch_size=delete_batch_size,
    )


__all__ = ["create_full_vector_reconciliation_service"]
