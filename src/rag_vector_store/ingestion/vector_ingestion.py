"""VectorIngestionService：rag_vector_store 面向上游的唯一稳定 Python 向量能力入口。

职责：
    - ``upsert_chunks``：正式 Chunk 向量化并写入 Milvus（复用 ContractBatchIngestionService）。
    - ``disable_vectors``：按 vector_doc_id 逻辑下线向量（保留实体与向量，仅置 is_active=false）。
    - ``delete_vectors``：在线删除入口；自 T2-1 起统一委派给 ``disable_vectors``，
      不再触发物理删除（保持原签名与返回类型，上游 Port 无感知）。
    - ``get_vectors``：批量查询 Milvus 记录（不含 embedding）。
    - ``update_metadata``：metadata-only 更新，复用旧向量，不调用 Embedding。
    - ``list_vectors_by_expert``：按专家列出轻量 vector_doc_id。

不职责：
    - 不实现上游业务更新编排（sync_expert / update_item_data / 时间筛选 / Vector State / FastAPI）。
    - 不重新实现 Validator / Eligibility / Embedding / Milvus 主链。
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

from pymilvus.exceptions import MilvusException

from expert_contracts.models import ExpertChunkEmbeddingInput

from ..chunking.contract_ingestion import (
    ContractBatchIngestionService,
    ContractIngestionError,
)
from ..chunking.contract_pipeline import (
    EXPECTED_EMBEDDING_DIMENSION,
    ContractEmbeddingWorker,
    ContractValidationError,
    ContractValidator,
    EmbeddingEligibilityError,
)
from ..core.expert import ExpertContractAdapter
from ..core.hash_state import project_hash_fields
from ..core.lifecycle import DomainVectorLifecycleService
from ..embedding_adapters.company_cloud import CompanyCloudEmbeddingError
from ..milvus.contract_v15 import MilvusV15RowMapper
from ..milvus.repository import ContractMilvusRepositoryBase
from ..model_space import (
    FORMAL_EMBEDDING_MODEL_SPACE_ID,
)
from .errors import (
    VectorContractError,
    VectorEmbeddingError,
    VectorIngestionError,
    VectorMilvusError,
)
from .models import (
    VectorDeletionResult,
    VectorExpertList,
    VectorIngestionResult,
    VectorItemResult,
    VectorQueryResult,
)


def _safe_message(error: Exception) -> str:
    """构造不含内部细节的安全错误摘要（类型 + 截断消息）。"""

    text = str(error)
    return f"{type(error).__name__}: {text[:200]}"


# 底层异常分类（不含无边界 except Exception）：
#   - Contract 层：ContractValidationError / EmbeddingEligibilityError（-> VectorContractError）
#   - Ingestion 层：ContractIngestionError（-> VectorIngestionError）
#   - Embedding 层：CompanyCloudEmbeddingError（-> VectorEmbeddingError）
#   - Milvus 层：ValueError（含 pydantic.ValidationError）/ TypeError / RuntimeError /
#     MilvusException（-> VectorMilvusError）


def _guard_call(fn, *args: Any, **kwargs: Any) -> Any:
    """执行一次内部委派（ingestion / repository / row_mapper），按来源路由公共异常。

    - 已属 ``VectorIngestionError`` 的异常原样透传（不重复包装）。
    - Contract / Eligibility / Ingestion / Embedding 错误 -> 对应公共类型。
    - 其余底层错误（含 repository 透传的 ValueError/TypeError/RuntimeError/
      MilvusException）视为 Milvus 层故障 -> ``VectorMilvusError``。
    """

    try:
        return fn(*args, **kwargs)
    except VectorIngestionError:
        raise
    except (ContractValidationError, EmbeddingEligibilityError) as error:
        raise VectorContractError(_safe_message(error)) from error
    except ContractIngestionError as error:
        raise VectorIngestionError(_safe_message(error)) from error
    except CompanyCloudEmbeddingError as error:
        raise VectorEmbeddingError(_safe_message(error)) from error
    except (ValueError, TypeError, RuntimeError, MilvusException) as error:
        raise VectorMilvusError(_safe_message(error)) from error


def _guard_errors(func):
    """捕获公共方法体内非 Milvus 委派路径的异常，按来源路由公共异常。

    - 已属 ``VectorIngestionError`` 的异常原样透传（不重复包装）。
    - 输入校验 / Contract / 模型构造错误（ValueError/TypeError）->``VectorContractError``。
    - 意外基础设施故障（RuntimeError/MilvusException）->``VectorMilvusError``。
    """

    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except VectorIngestionError:
            raise
        except (ContractValidationError, EmbeddingEligibilityError, ValueError, TypeError) as error:
            raise VectorContractError(_safe_message(error)) from error
        except ContractIngestionError as error:
            raise VectorIngestionError(_safe_message(error)) from error
        except CompanyCloudEmbeddingError as error:
            raise VectorEmbeddingError(_safe_message(error)) from error
        except (RuntimeError, MilvusException) as error:
            raise VectorMilvusError(_safe_message(error)) from error

    return wrapper


class _GuardedLifecycleRepository:
    """把专家 Facade 的底层异常分类边界传给通用生命周期核心。"""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def get_by_vector_doc_id(
        self,
        vector_doc_id: str,
        *,
        include_embedding: bool = True,
    ) -> dict[str, object] | None:
        return _guard_call(
            self.repository.get_by_vector_doc_id,
            vector_doc_id,
            include_embedding=include_embedding,
        )

    def upsert_batch(self, records: Sequence[Any]) -> Any:
        return _guard_call(self.repository.upsert_batch, records)


class _GuardedLifecycleRowMapper:
    """把行还原失败归类为 Milvus 错误而非调用方 Contract 错误。"""

    def __init__(self, row_mapper: Any) -> None:
        self.row_mapper = row_mapper

    def from_row(self, row: dict[str, object]) -> Any:
        return _guard_call(self.row_mapper.from_row, row)


def _dedupe(vector_doc_ids: Sequence[str]) -> list[str]:
    """按输入顺序去重并校验主键，返回清理后的 ID 列表。"""

    cleaned: list[str] = []
    seen: set[str] = set()
    for vector_doc_id in vector_doc_ids:
        if not isinstance(vector_doc_id, str) or not vector_doc_id.strip():
            raise VectorContractError("vector_doc_id 必须是非空字符串")
        normalized = vector_doc_id.strip()
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


class VectorIngestionService:
    """统一编排 Chunk 向量化、写入、更新、删除与查询的 Facade。

    依赖：
        - ``ingestion_service``（或 ``embedding_worker`` + ``repository``）：upsert 编排。
        - ``repository``：Milvus 增删查。
        - ``validator``：update_metadata 的逐条 Contract 校验。
        - ``row_mapper``：update_metadata 复用旧向量。
    """

    def __init__(
        self,
        *,
        ingestion_service: ContractBatchIngestionService | None = None,
        embedding_worker: ContractEmbeddingWorker | None = None,
        repository: ContractMilvusRepositoryBase,
        validator: ContractValidator | None = None,
        row_mapper: MilvusV15RowMapper | None = None,
        expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
        formal_model_space_id: str = FORMAL_EMBEDDING_MODEL_SPACE_ID,
    ) -> None:
        if ingestion_service is None:
            if embedding_worker is None:
                raise ValueError(
                    "必须提供 ingestion_service 或 embedding_worker"
                )
            ingestion_service = ContractBatchIngestionService(
                embedding_worker,
                repository,
            )
        self.ingestion = ingestion_service
        self.repository = repository
        self.validator = validator or ContractValidator()
        # 默认 Mapper 固定为 V15，禁止静默回退到已退役的 V14 Mapper；
        # 正式组装应通过 factory 注入与 repository profile 一致的 Mapper。
        self.row_mapper = row_mapper if row_mapper is not None else MilvusV15RowMapper()
        self.expected_dimension = expected_dimension
        self.formal_model_space_id = formal_model_space_id
        eligibility_validator = getattr(
            getattr(ingestion_service, "embedding_worker", None),
            "eligibility_validator",
            None,
        )
        self._adapter = ExpertContractAdapter(
            self.validator,
            eligibility_validator,
        )
        self._lifecycle = DomainVectorLifecycleService(
            _GuardedLifecycleRepository(repository),
            _GuardedLifecycleRowMapper(self.row_mapper),
            self._adapter,
            expected_dimension=self.expected_dimension,
            formal_model_space_id=self.formal_model_space_id,
        )

    # ------------------------------------------------------------------
    # 1. upsert_chunks
    # ------------------------------------------------------------------
    @_guard_errors
    def upsert_chunks(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult:
        """正式 Chunk 全集向量化并写入 Milvus。

        内部复用 ``ContractBatchIngestionService.ingest_batch``：
        Contract 校验 → Eligibility → Embedding → Milvus upsert。
        非法公共输入（非 Contract 记录）直接判为 ``VectorContractError``。
        """

        if not all(isinstance(c, ExpertChunkEmbeddingInput) for c in chunks):
            raise VectorContractError("upsert_chunks 只接受 ExpertChunkEmbeddingInput")

        result = _guard_call(self.ingestion.ingest_batch, chunks)
        items = [
            _make_item_from_batch_item(item)
            for item in result.item_results
        ]
        return VectorIngestionResult(results=items)

    # ------------------------------------------------------------------
    # 2. disable_vectors / delete_vectors（统一逻辑下线）
    # ------------------------------------------------------------------
    @_guard_errors
    def disable_vectors(
        self,
        vector_doc_ids: Sequence[str],
    ) -> VectorDeletionResult:
        """按 vector_doc_id 集合逻辑下线 Milvus 向量（无物理删除）。

        逐条流程：
            1. ``get_by_vector_doc_id(include_embedding=True)`` 读取现有完整实体；
            2. 实体不存在 → 计入 ``not_found_count``（幂等，不报错）；
            3. 已处于 is_active=false → 幂等重放，跳过写入、计入下线数；
            4. 否则保留 vector_doc_id / chunk_text / embedding 与全部历史
               metadata，仅置 ``is_active=false`` 后以同一主键 upsert 回写。

        Repository 不提供物理删除能力；旧实体携带的 model_space 信息随行保留，若与 Collection 现行空间不一致由底层
        upsert 校验 fail-closed。

        返回的 ``deleted_count`` 语义为"处理后处于下线状态的数量"。
        """

        result = self._lifecycle.disable_vectors(vector_doc_ids)
        return VectorDeletionResult(
            requested_count=result.requested_count,
            deleted_count=result.deleted_count,
            not_found_count=result.not_found_count,
        )

    @_guard_errors
    def delete_vectors(
        self,
        vector_doc_ids: Sequence[str],
    ) -> VectorDeletionResult:
        """按 vector_doc_id 集合下线 Milvus 向量（在线删除入口）。

        自 T2-1 起在线删除语义统一为逻辑下线：本方法只委派
        ``disable_vectors``（保留实体与向量，仅置 is_active=false），
        Repository 已移除物理删除能力。
        签名与返回类型不变，上游 ``VectorStorePort`` 调用方无感知；
        空输入安全返回，去重保序，重复执行幂等。
        """

        return self.disable_vectors(vector_doc_ids)

    # ------------------------------------------------------------------
    # 3. get_vectors
    # ------------------------------------------------------------------
    @_guard_errors
    def get_vectors(
        self,
        vector_doc_ids: Sequence[str],
    ) -> VectorQueryResult:
        """按 vector_doc_id 集合批量查询 Milvus 记录（不含 embedding）。

        保留输入去重顺序；``records`` 为命中的记录，``missing`` 为未命中主键。

        hash 投影契约：本方法在 Service 边界统一投影三个哈希字段
        （``content_hash`` / ``metadata_hash`` / ``source_hash``，V15 把它们存进
        ``metadata`` JSON）到逻辑返回顶层——顶层已有合法哈希时保持不变，否则从
        ``metadata`` JSON 中提取。这是纯内存投影，不修改 Milvus 物理 Schema，
        不增加一级字段，不平铺其它 metadata 字段。

        下游消费差异：
        - 增量 ``ContractConsumerCore`` 读取全部三个哈希做三 Hash 动作矩阵；
        - 全量对账 ``ExpertVectorSynchronizer``（同事侧 expert_index）仍只消费
          ``content_hash`` / ``metadata_hash``，未升级为三 Hash，不在本次
          C1 范围；多投影出的 ``source_hash`` 对它是多余但无害的字段。

        缺失策略：顶层与 ``metadata`` 都缺失 ``content_hash`` / ``metadata_hash``
        时不补造，交调用方 fail-closed；``source_hash`` 缺失同样不补造，由
        ``ContractConsumerCore`` 按三 Hash 矩阵把已存缺失路由到 metadata-only
        （不误判为全同 Skip，不调用 Embedding）。
        """

        cleaned = _dedupe(vector_doc_ids)
        records: list[dict[str, object]] = []
        missing: list[str] = []
        for vid in cleaned:
            row = _guard_call(
                self.repository.get_by_vector_doc_id,
                vid,
                include_embedding=False,
            )
            if row is None:
                missing.append(vid)
            else:
                records.append(project_hash_fields(row))
        return VectorQueryResult(records=records, missing=missing)

    # ------------------------------------------------------------------
    # 4. update_metadata
    # ------------------------------------------------------------------
    @_guard_errors
    def update_metadata(
        self,
        chunks: Sequence[ExpertChunkEmbeddingInput],
    ) -> VectorIngestionResult:
        """仅更新 metadata / scalar，不重新 Embedding。

        流程（逐条）：
            1. Contract 校验；
            2. ``get_by_vector_doc_id`` 读取旧 Entity；
            3. metadata-only 一致性门：本次输入的 ``vector_doc_id`` /
               ``content_hash`` / ``chunk_text`` 必须与 existing 实体完全一致，
               否则属于调用方误用（正文/主键已变化），立即抛 ``VectorContractError``；
            4. 安全门：旧 Entity 存在 / model_space 一致 / embedding_dim 一致；
            5. 通过 → ``MilvusV15RowMapper.from_row`` 复用旧向量 + 新 metadata upsert；
            6. model_space/dim 安全门失败 → 该条 ``status=failed``、
               ``recommendation="call_upsert"``，不调用 Embedding、不静默降级。
        """
        result = self._lifecycle.update_metadata(chunks)
        return VectorIngestionResult(
            results=[
                VectorItemResult(
                    vector_doc_id=item.vector_doc_id,
                    status=item.status,
                    error_message=item.error_message,
                    recommendation=item.recommendation,
                )
                for item in result.results
            ]
        )

    # ------------------------------------------------------------------
    # 5. list_vectors_by_expert
    # ------------------------------------------------------------------
    @_guard_errors
    def list_vectors_by_expert(
        self,
        expert_id: int,
    ) -> VectorExpertList:
        """按专家列出该专家在 Milvus 的全部 vector_doc_id（仅轻量字段）。"""

        if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id < 1:
            raise VectorContractError(
                f"expert_id 必须是正整数，收到 {expert_id!r}"
            )

        rows = _guard_call(self.repository.query_by_expert_id, expert_id)
        vector_doc_ids = [str(row["vector_doc_id"]) for row in rows]
        return VectorExpertList(
            expert_id=expert_id,
            vector_doc_ids=vector_doc_ids,
            count=len(vector_doc_ids),
        )


def _make_item_from_batch_item(item: Any) -> VectorItemResult:
    """把 ``ContractIngestionResult`` 转换为公共 ``VectorItemResult``。"""

    if item.status == "embedded":
        return VectorItemResult(
            vector_doc_id=item.vector_doc_id,
            status="embedded",
        )
    return VectorItemResult(
        vector_doc_id=item.vector_doc_id,
        status="skipped",
        error_message=item.skip_reason,
    )


__all__ = ["VectorIngestionService"]
