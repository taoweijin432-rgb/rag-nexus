"""Expert/Resume 双领域向量服务工厂。

工厂在构造期绑定领域 Adapter、Repository Profile 和 RowMapper，运行时不允许
通过请求数据切换 Collection 或身份空间。Expert 可由构造配置选择 V15/V16，
Resume 仍固定使用 V15。
"""

from __future__ import annotations

from typing import Any

from ..chunking.contract_pipeline import (
    EXPECTED_EMBEDDING_DIMENSION,
    ContractValidator,
)
from ..config import (
    EXPERT_V16_COLLECTION_NAME,
    EXPERT_V16_EXTERNAL_FORMAL_COLLECTION_NAME,
    MilvusSettings,
)
from ..core.embedding import DomainEmbeddingWorker
from ..core.ingestion import DomainIngestionService
from ..domains import ExpertContractAdapter, ResumeContractAdapter
from ..embedding_adapters.company_cloud import CompanyCloudEmbeddingModel
from ..milvus.contract_v15 import MilvusV15RowMapper, ResumeMilvusV15RowMapper
from ..milvus.contract_v16 import MilvusV16RowMapper
from ..milvus.contract_v16_external import MilvusV16ExternalRowMapper
from ..milvus.repository import ContractMilvusRepositoryBase
from ..milvus.repository_v15 import (
    EXPERT_V15_PROFILE,
    RESUME_V15_PROFILE,
    ContractMilvusV15VectorRepository,
)
from ..milvus.repository_v16 import (
    EXPERT_V16_PROFILE,
    ContractMilvusV16VectorRepository,
)
from ..milvus.repository_v16_external import (
    EXPERT_V16_EXTERNAL_PROFILE,
    ContractMilvusV16ExternalVectorRepository,
)
from ..model_space import FORMAL_EMBEDDING_MODEL_SPACE_ID
from .vector_ingestion import ResumeVectorIngestionService, VectorIngestionService


def _require_profile(repository: Any, *, expected_profile: Any) -> None:
    """校验显式 Repository 已绑定目标 Profile。"""

    profile = getattr(repository, "profile", None)
    if profile is None:
        raise ValueError("显式 Repository 缺少 Profile，无法证明领域绑定")
    if (
        getattr(profile, "owner_field", None) != expected_profile.owner_field
        or getattr(profile, "vector_doc_id_prefix", None)
        != expected_profile.vector_doc_id_prefix
        or getattr(profile, "contract_adapter_cls", None)
        is not expected_profile.contract_adapter_cls
        or getattr(profile, "row_mapper_cls", None)
        is not expected_profile.row_mapper_cls
    ):
        domain = (
            "Resume"
            if expected_profile.owner_field == "resume_id"
            else "Expert"
        )
        raise ValueError(f"显式 Repository 与 {domain} Profile 不一致")


def _require_worker(
    worker: Any,
    *,
    domain_name: str,
    adapter: Any,
) -> None:
    """校验显式 Worker 暴露的 Adapter 与目标领域一致。"""

    worker_adapter = getattr(worker, "adapter", None)
    if worker_adapter is None:
        raise ValueError(
            f"{domain_name} Factory 要求显式 Worker 暴露 adapter，"
            "无法证明领域绑定"
        )
    if (
        worker_adapter.owner_field() != adapter.owner_field()
        or worker_adapter.id_prefix() != adapter.id_prefix()
    ):
        raise ValueError(f"{domain_name} Factory 拒绝跨领域 Embedding Worker")


def _require_mapper(
    mapper: Any,
    *,
    domain_name: str,
    expected_cls: type[Any],
    adapter: Any,
) -> None:
    """校验显式 RowMapper 的类型、Adapter 和身份空间。"""

    if not isinstance(mapper, expected_cls):
        raise TypeError(
            f"{domain_name} Factory 要求使用 {expected_cls.__name__}"
        )
    mapper_adapter = getattr(mapper, "adapter", None)
    if mapper_adapter is None:
        raise ValueError(
            f"{domain_name} RowMapper 缺少 adapter，无法证明领域绑定"
        )
    if (
        mapper_adapter.owner_field() != adapter.owner_field()
        or mapper_adapter.id_prefix() != adapter.id_prefix()
    ):
        raise ValueError(f"{domain_name} Factory 拒绝跨领域 RowMapper")


def _require_alignment(
    *,
    repository: Any,
    mapper: Any,
    adapter: Any,
    profile: Any,
) -> None:
    """校验 Repository/Profile/RowMapper/Adapter 四者一致。"""

    _require_profile(repository, expected_profile=profile)
    if (
        getattr(mapper, "adapter", None) is None
        or mapper.adapter.owner_field() != adapter.owner_field()
        or mapper.adapter.id_prefix() != adapter.id_prefix()
    ):
        raise ValueError("Repository、RowMapper 与 Adapter 领域身份不一致")
    repo_profile = repository.profile
    if repo_profile.owner_field != adapter.owner_field():
        raise ValueError("Repository owner 字段与 Adapter 不一致")
    if repo_profile.vector_doc_id_prefix != adapter.id_prefix():
        raise ValueError("Repository 主键前缀与 Adapter 不一致")


def _build_repository(
    *,
    milvus_settings: MilvusSettings,
    expected_dimension: int,
    profile: Any,
) -> ContractMilvusRepositoryBase:
    repository_cls = {
        MilvusV16RowMapper: ContractMilvusV16VectorRepository,
        MilvusV16ExternalRowMapper: ContractMilvusV16ExternalVectorRepository,
    }.get(profile.row_mapper_cls, ContractMilvusV15VectorRepository)
    return repository_cls(
        milvus_settings,
        vector_dimension=expected_dimension,
        allow_collection_creation=False,
        profile=profile,
    )


def _build_service(
    *,
    domain_name: str,
    embedding_model: CompanyCloudEmbeddingModel | None,
    milvus_settings: MilvusSettings | None,
    repository: ContractMilvusRepositoryBase | None,
    embedding_worker: Any | None,
    validator: ContractValidator | None,
    row_mapper: Any | None,
    expected_dimension: int,
    formal_model_space_id: str,
    adapter: Any,
    profile: Any,
    mapper_cls: type[Any],
    service_cls: type[VectorIngestionService],
) -> VectorIngestionService:
    if validator is None:
        validator = ContractValidator()
    if embedding_worker is None:
        if embedding_model is None:
            embedding_model = CompanyCloudEmbeddingModel.from_env()
        embedding_worker = DomainEmbeddingWorker(
            embedding_model,
            adapter,
            expected_dimension=expected_dimension,
        )
    else:
        _require_worker(embedding_worker, domain_name=domain_name, adapter=adapter)

    if repository is None:
        if milvus_settings is None:
            milvus_settings = MilvusSettings.from_env()
        repository = _build_repository(
            milvus_settings=milvus_settings,
            expected_dimension=expected_dimension,
            profile=profile,
        )
    if row_mapper is None:
        row_mapper = (
            repository.row_mapper
            if mapper_cls is MilvusV16ExternalRowMapper
            else mapper_cls(adapter)
        )
    else:
        _require_mapper(
            row_mapper,
            domain_name=domain_name,
            expected_cls=mapper_cls,
            adapter=adapter,
        )
    if (mapper_cls is MilvusV16ExternalRowMapper
            and row_mapper.tokenizer.config != repository.tokenizer.config):
        raise ValueError("生命周期与 Repository 的分词配置不一致")
    _require_alignment(
        repository=repository,
        mapper=row_mapper,
        adapter=adapter,
        profile=profile,
    )
    ingestion_service = DomainIngestionService(
        embedding_worker,
        repository,
        adapter,
    )
    return service_cls(
        ingestion_service=ingestion_service,
        repository=repository,
        validator=validator,
        row_mapper=row_mapper,
        expected_dimension=expected_dimension,
        formal_model_space_id=formal_model_space_id,
        adapter=adapter,
    )


def create_vector_ingestion_service(
    *,
    embedding_model: CompanyCloudEmbeddingModel | None = None,
    milvus_settings: MilvusSettings | None = None,
    repository: ContractMilvusRepositoryBase | None = None,
    embedding_worker: Any | None = None,
    validator: ContractValidator | None = None,
    row_mapper: Any | None = None,
    expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    formal_model_space_id: str = FORMAL_EMBEDDING_MODEL_SPACE_ID,
) -> VectorIngestionService:
    """按构造配置绑定 Expert V15 或 V16 Profile。"""

    validator = validator or getattr(embedding_worker, "validator", None) or ContractValidator()
    eligibility_validator = getattr(embedding_worker, "eligibility_validator", None)
    adapter = ExpertContractAdapter(validator, eligibility_validator)
    if repository is not None:
        profile = repository.profile
        if (
            profile.owner_field != "expert_id"
            or profile.vector_doc_id_prefix != "expert:"
            or profile.row_mapper_cls not in {
                MilvusV15RowMapper, MilvusV16RowMapper, MilvusV16ExternalRowMapper
            }
        ):
            raise ValueError("显式 Repository 与 Expert Profile 不一致")
    else:
        milvus_settings = milvus_settings or MilvusSettings.from_env()
        profile = {
            "v15": EXPERT_V15_PROFILE,
            "v16": EXPERT_V16_PROFILE,
            "v16_external_formal": EXPERT_V16_EXTERNAL_PROFILE,
        }[milvus_settings.expert_storage_profile]
    mapper_cls = profile.row_mapper_cls
    return _build_service(
        domain_name="Expert",
        embedding_model=embedding_model,
        milvus_settings=milvus_settings,
        repository=repository,
        embedding_worker=embedding_worker,
        validator=validator,
        row_mapper=row_mapper,
        expected_dimension=expected_dimension,
        formal_model_space_id=formal_model_space_id,
        adapter=adapter,
        profile=profile,
        mapper_cls=mapper_cls,
        service_cls=VectorIngestionService,
    )


def create_resume_vector_ingestion_service(
    *,
    embedding_model: CompanyCloudEmbeddingModel | None = None,
    milvus_settings: MilvusSettings | None = None,
    repository: ContractMilvusV15VectorRepository | None = None,
    embedding_worker: Any | None = None,
    validator: ContractValidator | None = None,
    row_mapper: Any | None = None,
    expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    formal_model_space_id: str = FORMAL_EMBEDDING_MODEL_SPACE_ID,
) -> ResumeVectorIngestionService:
    """构造固定 Resume Profile 的向量服务。"""

    if repository is None:
        milvus_settings = milvus_settings or MilvusSettings.from_env()
        if (milvus_settings.collection_name in {
                EXPERT_V16_COLLECTION_NAME, EXPERT_V16_EXTERNAL_FORMAL_COLLECTION_NAME,
            }
                or milvus_settings.collection_name.startswith("expert_candidates_v16_extjieba_")):
            raise ValueError("Resume Factory 拒绝绑定 Expert V16 Collection")
        if milvus_settings.expert_storage_profile != "v15":
            milvus_settings = milvus_settings.model_copy(
                update={"expert_storage_profile": "v15"}
            )
    validator = validator or getattr(embedding_worker, "validator", None) or ContractValidator()
    eligibility_validator = getattr(embedding_worker, "eligibility_validator", None)
    adapter = ResumeContractAdapter(validator, eligibility_validator)
    return _build_service(
        domain_name="Resume",
        embedding_model=embedding_model,
        milvus_settings=milvus_settings,
        repository=repository,
        embedding_worker=embedding_worker,
        validator=validator,
        row_mapper=row_mapper,
        expected_dimension=expected_dimension,
        formal_model_space_id=formal_model_space_id,
        adapter=adapter,
        profile=RESUME_V15_PROFILE,
        mapper_cls=ResumeMilvusV15RowMapper,
        service_cls=ResumeVectorIngestionService,
    )  # type: ignore[return-value]


__all__ = [
    "create_resume_vector_ingestion_service",
    "create_vector_ingestion_service",
]
