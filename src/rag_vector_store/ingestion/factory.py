"""VectorIngestionService 的生产配置工厂。

调用方无需直接构造内部分层依赖（Embedding Worker / Milvus Repository /
ContractBatchIngestionService / Validator / RowMapper），只调用
``create_vector_ingestion_service()`` 即可从环境变量组装完整服务。
"""

from __future__ import annotations

from ..chunking.contract_ingestion import ContractBatchIngestionService
from ..chunking.contract_pipeline import (
    EXPECTED_EMBEDDING_DIMENSION,
    ContractEmbeddingWorker,
    ContractValidator,
)
from ..config import MilvusSettings
from ..embedding_adapters.company_cloud import CompanyCloudEmbeddingModel
from ..milvus.contract import MilvusRowMapper
from ..milvus.repository import ContractMilvusVectorRepository
from ..model_space import FORMAL_EMBEDDING_MODEL_SPACE_ID
from .vector_ingestion import VectorIngestionService


def create_vector_ingestion_service(
    *,
    embedding_model: CompanyCloudEmbeddingModel | None = None,
    milvus_settings: MilvusSettings | None = None,
    repository: ContractMilvusVectorRepository | None = None,
    embedding_worker: ContractEmbeddingWorker | None = None,
    validator: ContractValidator | None = None,
    row_mapper: MilvusRowMapper | None = None,
    expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    formal_model_space_id: str = FORMAL_EMBEDDING_MODEL_SPACE_ID,
) -> VectorIngestionService:
    """从环境配置组装完整的 ``VectorIngestionService``。

    依赖链：
        ``CompanyCloudEmbeddingModel.from_env()`` -> ``ContractEmbeddingWorker``
        -> ``ContractBatchIngestionService``；``MilvusSettings.from_env()`` ->
        ``ContractMilvusVectorRepository``。全部汇聚到 ``VectorIngestionService``。

    允许通过关键字注入部分或全部依赖用于测试 / 自定义组装，未注入项自动从环境读取。
    """

    if validator is None:
        validator = ContractValidator()
    if embedding_model is None:
        embedding_model = CompanyCloudEmbeddingModel.from_env()
    if embedding_worker is None:
        embedding_worker = ContractEmbeddingWorker(
            embedding_model,
            validator=validator,
        )
    if milvus_settings is None:
        milvus_settings = MilvusSettings.from_env()
    if repository is None:
        repository = ContractMilvusVectorRepository(
            milvus_settings,
            vector_dimension=expected_dimension,
        )
    ingestion_service = ContractBatchIngestionService(
        embedding_worker,
        repository,
    )
    return VectorIngestionService(
        ingestion_service=ingestion_service,
        repository=repository,
        validator=validator,
        row_mapper=row_mapper,
        expected_dimension=expected_dimension,
        formal_model_space_id=formal_model_space_id,
    )


__all__ = ["create_vector_ingestion_service"]