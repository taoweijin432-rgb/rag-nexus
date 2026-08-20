"""VectorIngestionService 的生产配置工厂。

调用方无需直接构造内部分层依赖（Embedding Worker / Milvus Repository /
ContractBatchIngestionService / Validator / RowMapper），只调用
``create_vector_ingestion_service()`` 即可从环境变量组装完整服务。

持久化 profile：
    ``schema_profile`` 控制 Repository 的物理 Schema profile（默认 ``v15``）：
        - ``v15``：``ContractMilvusV15VectorRepository``（4 字段 +
          ``MilvusV15RowMapper``）；
        - ``v14``：``ContractMilvusVectorRepository``（25 字段 +
          ``MilvusRowMapper``），保留为兼容���滚路径。
    profile 只影响持久化结构，不改变 ``collection_name``（仍由
    ``MilvusSettings`` 决定），也不影响公共 API 语义。
"""

from __future__ import annotations

from typing import Any, Literal

from ..chunking.contract_ingestion import ContractBatchIngestionService
from ..chunking.contract_pipeline import (
    EXPECTED_EMBEDDING_DIMENSION,
    ContractEmbeddingWorker,
    ContractValidator,
)
from ..config import MilvusSettings
from ..embedding_adapters.company_cloud import CompanyCloudEmbeddingModel
from ..milvus.contract import MilvusRowMapper
from ..milvus.contract_v15 import MilvusV15RowMapper
from ..milvus.repository import ContractMilvusVectorRepository
from ..milvus.repository_v15 import ContractMilvusV15VectorRepository
from ..model_space import FORMAL_EMBEDDING_MODEL_SPACE_ID
from .vector_ingestion import VectorIngestionService

SchemaProfile = Literal["v14", "v15"]
_VALID_PROFILES = ("v14", "v15")


def _repository_is_v15(repository: Any) -> bool:
    """判断 repository 对象是否属于 V15 持久化 profile。

    优先用类型判断（正式 V15 Repository 类型）；对 duck-typed fake 回退到
    V15 profile 特征（``_row_mapper_cls``）。无法识别的一律视为非 V15，
    保持现有 V14 兼容。
    """

    if isinstance(repository, ContractMilvusV15VectorRepository):
        return True
    return getattr(repository, "_row_mapper_cls", None) is MilvusV15RowMapper


def _validate_repository_profile(
    repository: Any,
    schema_profile: SchemaProfile,
) -> None:
    """显式注入 repository 与 schema_profile 明显冲突时 fail-closed。

    规则：
        - 显式注入的 repository 优先使用，不按 profile 偷偷替换；
        - 仅当能明确识别冲突时才 fail-closed：
            * ``schema_profile='v15'`` + 显式注入正式 V14 Repository → 冲突；
            * ``schema_profile='v14'`` + 显式注入 V15 Repository → 冲突；
        - 测试用 duck-typed fake（无法识别真实 profile）按「显式注入优先」
          放行，不误报。
    """

    if repository is None:
        return
    is_v15 = _repository_is_v15(repository)
    if schema_profile == "v15" and not is_v15 and isinstance(
        repository, ContractMilvusVectorRepository
    ):
        raise ValueError(
            "schema_profile='v15' 与显式注入的 V14 repository 冲突；"
            "请注入 ContractMilvusV15VectorRepository 或移除 repository 注入"
        )
    if schema_profile == "v14" and is_v15:
        raise ValueError(
            "schema_profile='v14' 与显式注入的 V15 repository 冲突；"
            "请将 schema_profile 设为 'v15' 或移除 repository 注入"
        )


def _validate_row_mapper_profile(
    row_mapper: Any,
    repository: Any,
    schema_profile: SchemaProfile,
) -> None:
    """显式注入 row_mapper 与 schema_profile / repository profile 冲突时 fail-closed。

    规则：
        - 正式 ``MilvusRowMapper``（V14）与正式 ``MilvusV15RowMapper``（V15）
          严格识别；
        - 无法可靠识别的 fake mapper 放行（不凭类名猜测）；
        - 冲突时直接 ``ValueError``，不静默替换调用方显式传入的 mapper。

    冲突组合（任一命中即失败）：
        - ``schema_profile='v15'`` + V14 mapper；
        - ``schema_profile='v14'`` + V15 mapper；
        - V15 repository + V14 mapper；
        - V14 repository + V15 mapper。
    """

    if row_mapper is None:
        return
    mapper_is_v15: bool | None
    if isinstance(row_mapper, MilvusV15RowMapper):
        mapper_is_v15 = True
    elif isinstance(row_mapper, MilvusRowMapper):
        mapper_is_v15 = False
    else:
        return  # 无法可靠识别，放行

    if mapper_is_v15:
        if schema_profile == "v14":
            raise ValueError(
                "显式传入 V15 row_mapper 与 schema_profile='v14' 冲突；"
                "请使用 MilvusRowMapper 或将 schema_profile 设为 'v15'"
            )
        if isinstance(repository, ContractMilvusVectorRepository) and not isinstance(
            repository, ContractMilvusV15VectorRepository
        ):
            raise ValueError(
                "显式传入 V15 row_mapper 与 V14 repository 冲突；"
                "请使用 MilvusRowMapper 或注入 V15 repository"
            )
    else:
        if schema_profile == "v15":
            raise ValueError(
                "显式传入 V14 row_mapper 与 schema_profile='v15' 冲突；"
                "请使用 MilvusV15RowMapper 或将 schema_profile 设为 'v14'"
            )
        if isinstance(repository, ContractMilvusV15VectorRepository):
            raise ValueError(
                "显式传入 V14 row_mapper 与 V15 repository 冲突；"
                "请使用 MilvusV15RowMapper 或注入 V14 repository"
            )


def _build_repository(
    *,
    milvus_settings: MilvusSettings,
    expected_dimension: int,
    schema_profile: SchemaProfile,
) -> ContractMilvusVectorRepository:
    """按 schema_profile 构造 Repository（V14 默认 / V15 显式）。"""

    if schema_profile == "v15":
        return ContractMilvusV15VectorRepository(
            milvus_settings,
            vector_dimension=expected_dimension,
        )
    return ContractMilvusVectorRepository(
        milvus_settings,
        vector_dimension=expected_dimension,
    )


def _resolve_row_mapper(
    row_mapper: MilvusRowMapper | MilvusV15RowMapper | None,
    repository: Any,
) -> MilvusRowMapper | MilvusV15RowMapper:
    """未显式传入 row_mapper 时，默认跟随 repository 的实际持久化 profile。

    保证 ``update_metadata`` 的 ``from_row`` 与 repository 物理 Schema 一致，
    杜绝「V15 repo + V14 mapper」的隐式错配。
    """

    if row_mapper is not None:
        return row_mapper
    if _repository_is_v15(repository):
        return MilvusV15RowMapper()
    return MilvusRowMapper()


def create_vector_ingestion_service(
    *,
    embedding_model: CompanyCloudEmbeddingModel | None = None,
    milvus_settings: MilvusSettings | None = None,
    repository: ContractMilvusVectorRepository | None = None,
    embedding_worker: ContractEmbeddingWorker | None = None,
    validator: ContractValidator | None = None,
    row_mapper: MilvusRowMapper | MilvusV15RowMapper | None = None,
    expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    formal_model_space_id: str = FORMAL_EMBEDDING_MODEL_SPACE_ID,
    schema_profile: SchemaProfile = "v15",
) -> VectorIngestionService:
    """从环境配置组装完整的 ``VectorIngestionService``。

    依赖链：
        ``CompanyCloudEmbeddingModel.from_env()`` -> ``ContractEmbeddingWorker``
        -> ``ContractBatchIngestionService``；``MilvusSettings.from_env()`` ->
        ``schema_profile`` 对应的 Milvus Repository。全部汇聚到
        ``VectorIngestionService``。

    Args:
        schema_profile: 持久化 profile，``v15``（默认）或 ``v14``。
            ``v15`` 时使用 ``ContractMilvusV15VectorRepository``，且未显式传入
            ``row_mapper`` 时自动跟随为 ``MilvusV15RowMapper``。
            ``v14`` 保留为兼容回滚路径（旧 25 字段 Collection）。
            该参数只控制持久化结构，不改变 ``collection_name``。
        repository: 显式注入的 Repository；注入时优先使用，不按 profile 替换,
            但若与 ``schema_profile`` 明显冲突则 fail-closed。
        row_mapper: 显式注入的 RowMapper；注入时优先使用，不按 profile 替换，
            但必须与 ``schema_profile`` 和 ``repository`` 的持久化 profile 一致，
            否则 fail-closed（不允许 V15 repo + V14 mapper 之类的串线）。

    其余关键字用于测试 / 自定义组装，未注入项自动从环境读取。
    """

    if schema_profile not in _VALID_PROFILES:
        raise ValueError(
            f"schema_profile 必须为 'v14' 或 'v15'，收到 {schema_profile!r}"
        )
    _validate_repository_profile(repository, schema_profile)
    _validate_row_mapper_profile(row_mapper, repository, schema_profile)

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
        repository = _build_repository(
            milvus_settings=milvus_settings,
            expected_dimension=expected_dimension,
            schema_profile=schema_profile,
        )
    resolved_row_mapper = _resolve_row_mapper(row_mapper, repository)
    ingestion_service = ContractBatchIngestionService(
        embedding_worker,
        repository,
    )
    return VectorIngestionService(
        ingestion_service=ingestion_service,
        repository=repository,
        validator=validator,
        row_mapper=resolved_row_mapper,
        expected_dimension=expected_dimension,
        formal_model_space_id=formal_model_space_id,
    )


__all__ = ["SchemaProfile", "create_vector_ingestion_service"]
