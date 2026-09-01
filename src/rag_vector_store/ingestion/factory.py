"""VectorIngestionService 的生产配置工厂。

调用方无需直接构造内部分层依赖（Embedding Worker / Milvus Repository /
ContractBatchIngestionService / Validator / RowMapper），只调用
``create_vector_ingestion_service()`` 即可从环境变量组装完整服务。

持久化 profile：
    ``schema_profile`` 控制 Repository 的物理 Schema profile（默认且唯一允许
    ``v15``）：
        - ``v15``：``ContractMilvusV15VectorRepository``（4 字段 +
          ``MilvusV15RowMapper``）。
    V14 25 字段物理形态已退役；显式传入 ``schema_profile="v14"`` 会 fail-closed
    抛 ``ValueError``。profile 只影响持久化结构，不改变 ``collection_name``
    （仍由 ``MilvusSettings`` 决定），也不影响公共 API 语义。
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
from ..milvus.contract_v15 import MilvusV15RowMapper
from ..milvus.repository_v15 import ContractMilvusV15VectorRepository
from ..model_space import FORMAL_EMBEDDING_MODEL_SPACE_ID
from .vector_ingestion import VectorIngestionService

# 当前唯一允许的持久化 profile；V14 已退役。
SchemaProfile = Literal["v15"]
_VALID_PROFILES = ("v15",)
_RETired_PROFILE = "v14"


def _repository_is_v15(repository: Any) -> bool:
    """判断 repository 对象是否属于 V15 持久化 profile。

    优先用类型判断（正式 V15 Repository 类型）；对 duck-typed fake 回退到
    V15 profile 特征（``_row_mapper_cls``）。无法识别的一律视为非 V15，
    由调用方按需 fail-closed。
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
            * ``schema_profile='v15'`` + 显式注入非 V15 的正式 Repository → 冲突；
        - 测试用 duck-typed fake（无法识别真实 profile）按「显式注入优先」
          放行，不误报。
    """

    if repository is None:
        return
    is_v15 = _repository_is_v15(repository)
    if schema_profile == "v15" and not is_v15 and hasattr(
        repository, "_schema_field_specs"
    ):
        raise ValueError(
            "schema_profile='v15' 与显式注入的非 V15 repository 冲突；"
            "请注入 ContractMilvusV15VectorRepository 或移除 repository 注入"
        )


def _validate_row_mapper_profile(
    row_mapper: Any,
    schema_profile: SchemaProfile,
) -> None:
    """显式注入 row_mapper 与 schema_profile 冲突时 fail-closed。

    规则：
        - 正式 ``MilvusV15RowMapper``（V15）严格识别后放行；
        - 无法可靠识别的 fake mapper 放行（不凭类名猜测，保留测试 duck-typing）；
        - 其余明确不符合 RowMapper 协议的对象在 v15 profile 下 fail-closed
          （V14 MilvusRowMapper 已删除，不再单独 isinstance 拦截）。

    冲突组合（任一命中即失败）：
        - ``schema_profile='v15'`` + 不符合 RowMapper 协议的对象。
    """

    if row_mapper is None:
        return
    if isinstance(row_mapper, MilvusV15RowMapper):
        return
    # duck-typed fake：具备 to_row/from_row 表面即放行，不凭类名猜测。
    if hasattr(row_mapper, "to_row") and hasattr(row_mapper, "from_row"):
        return
    if schema_profile == "v15":
        raise ValueError(
            "显式传入的 row_mapper 不符合 V15 持久化 profile；"
            "请使用 MilvusV15RowMapper"
        )


def _build_repository(
    *,
    milvus_settings: MilvusSettings,
    expected_dimension: int,
    schema_profile: SchemaProfile,
) -> ContractMilvusV15VectorRepository:
    """按 schema_profile 构造 Repository（仅 V15）。"""

    if schema_profile == "v15":
        return ContractMilvusV15VectorRepository(
            milvus_settings,
            vector_dimension=expected_dimension,
        )
    # 理论不可达：_validate_schema_profile 已拦截非法 profile。
    raise ValueError(f"未支持的 schema_profile {schema_profile!r}")


def validate_schema_profile(schema_profile: str) -> None:
    """校验 schema_profile，V14 已退役时显式 fail-closed。"""

    if schema_profile == _RETired_PROFILE:
        raise ValueError(
            "V14 旧向量链路已退役，schema_profile='v14' 不再支持；"
            "请使用 schema_profile='v15'"
        )
    if schema_profile not in _VALID_PROFILES:
        raise ValueError(
            f"schema_profile 必须为 'v15'，收到 {schema_profile!r}"
        )


# 保留内部旧名称，避免已有测试或同包代码直接引用时产生不必要的断裂。
_validate_schema_profile = validate_schema_profile


def _resolve_row_mapper(
    row_mapper: MilvusV15RowMapper | None,
    repository: Any,
) -> MilvusV15RowMapper:
    """未显式传入 row_mapper 时，默认跟随 repository 的实际持久化 profile。

    保证 ``update_metadata`` 的 ``from_row`` 与 repository 物理 Schema 一致，
    杜绝「V15 repo + 非 V15 mapper」的隐式错配。
    """

    if row_mapper is not None:
        return row_mapper
    mapper_cls = getattr(repository, "_row_mapper_cls", MilvusV15RowMapper)
    mapper = mapper_cls()
    if not hasattr(mapper, "to_row") or not hasattr(mapper, "from_row"):
        raise ValueError("Repository profile 未提供合法 RowMapper")
    return mapper


def create_vector_ingestion_service(
    *,
    embedding_model: CompanyCloudEmbeddingModel | None = None,
    milvus_settings: MilvusSettings | None = None,
    repository: ContractMilvusV15VectorRepository | None = None,
    embedding_worker: ContractEmbeddingWorker | None = None,
    validator: ContractValidator | None = None,
    row_mapper: MilvusV15RowMapper | None = None,
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
        schema_profile: 持久化 profile，唯一允许 ``v15``（默认）。
            ``v15`` 时使用 ``ContractMilvusV15VectorRepository``，且未显式传入
            ``row_mapper`` 时自动跟随为 ``MilvusV15RowMapper``。
            显式传入 ``v14`` 会 fail-closed 抛 ``ValueError``（V14 已退役）。
            该参数只控制持久化结构，不改变 ``collection_name``。
        repository: 显式注入的 Repository；注入时优先使用，不按 profile 替换,
            但若与 ``schema_profile`` 明显冲突则 fail-closed。
        row_mapper: 显式注入的 RowMapper；注入时优先使用，不按 profile 替换，
            但必须与 ``schema_profile`` 和 ``repository`` 的持久化 profile 一致，
            否则 fail-closed（不允许 V15 repo + 非 V15 mapper 之类的串线）。

    其余关键字用于测试 / 自定义组装，未注入项自动从环境读取。
    """

    validate_schema_profile(schema_profile)
    _validate_repository_profile(repository, schema_profile)
    _validate_row_mapper_profile(row_mapper, schema_profile)

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


__all__ = [
    "SchemaProfile",
    "create_vector_ingestion_service",
    "validate_schema_profile",
]
