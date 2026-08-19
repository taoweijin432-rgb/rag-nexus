"""VectorIngestionService：rag_vector_store 面向上游的唯一稳定 Python 向量能力入口。

职责：
    - ``upsert_chunks``：正式 Chunk 向量化并写入 Milvus（复用 ContractBatchIngestionService）。
    - ``delete_vectors``：按 vector_doc_id 删除 Milvus 向量。
    - ``get_vectors``：批量查询 Milvus 记录（不含 embedding）。
    - ``update_metadata``：metadata-only 更新，复用旧向量，不调用 Embedding。
    - ``list_vectors_by_expert``：按专家列出轻量 vector_doc_id。

不职责：
    - 不实现上游业务更新编排（sync_expert / update_item_data / 时间筛选 / Vector State / FastAPI）。
    - 不重新实现 Validator / Eligibility / Embedding / Milvus 主链。
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

from pymilvus.exceptions import MilvusException

from expert_contracts.models import (
    ContractPreparedChunk,
    EmbeddedContractPreparedChunk,
    ExpertChunkEmbeddingInput,
)

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
from ..embedding_adapters.company_cloud import CompanyCloudEmbeddingError
from ..milvus.contract import MilvusRowMapper
from ..milvus.repository import ContractMilvusVectorRepository
from ..model_space import (
    FORMAL_EMBEDDING_MODEL_SPACE_ID,
    build_model_space_id,
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


# 增量幂等读取需要投影到逻辑返回顶层的两个哈希字段名。V14 把它们存为物理
# 一级字段，V15 把它们移进 ``metadata`` JSON；本常量只服务于 get_vectors 的
# 逻辑投影，不代表物理 Schema 字段。
_HASH_PROJECTION_FIELDS = ("content_hash", "metadata_hash")


def _project_hash_fields(row: Mapping[str, object]) -> dict[str, object]:
    """把上游幂等判断需要的哈希投影到逻辑返回 record 顶层。

    语义：
        - 不修改原始物理 row（先 ``dict(row)`` 复制）；
        - 顶层已有合法哈希时保持不变（V14 物理 row 即此形态）；
        - 顶层缺失时，从 ``metadata`` JSON 中提取 ``content_hash`` /
          ``metadata_hash`` 到逻辑返回顶层（V15 物理 row 哈希位于 metadata）；
        - 只投影本模块明确声明需要的哈希字段，不平铺其它 metadata 字段；
        - 顶层与 ``metadata`` 均缺失时不补造哈希，交由调用方 fail-closed。

    Args:
        row: Repository ``get_by_vector_doc_id`` 返回的物理 row。

    Returns:
        逻辑返回 record（可能含投影后的顶层哈希）。
    """

    record = dict(row)
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        # metadata 不是对象时无法从其提取哈希；保持物理 row 形态，
        # 由下游 fail-closed。
        return record
    for field in _HASH_PROJECTION_FIELDS:
        top_value = record.get(field)
        if isinstance(top_value, str) and top_value:
            continue
        flat_value = metadata.get(field)
        if isinstance(flat_value, str) and flat_value:
            record[field] = flat_value
    return record


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
        repository: ContractMilvusVectorRepository,
        validator: ContractValidator | None = None,
        row_mapper: MilvusRowMapper | None = None,
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
        self.row_mapper = row_mapper if row_mapper is not None else MilvusRowMapper()
        self.expected_dimension = expected_dimension
        self.formal_model_space_id = formal_model_space_id

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
    # 2. delete_vectors
    # ------------------------------------------------------------------
    @_guard_errors
    def delete_vectors(
        self,
        vector_doc_ids: Sequence[str],
    ) -> VectorDeletionResult:
        """按 vector_doc_id 集合删除 Milvus 向量。

        空输入安全返回；保留去重顺序；删除幂等。
        单次 ``delete_by_vector_doc_ids`` 删除，按实际 ``deleted_count`` 推导
        ``not_found_count``（= requested_count - deleted_count）。
        pymilvus delete 只返回删除条数，不返回具体主键；因此公共结果只报告
        可靠的数量（requested / deleted / not_found_count），不伪造具体未命中 ID。
        """

        cleaned = _dedupe(vector_doc_ids)
        if not cleaned:
            return VectorDeletionResult(
                requested_count=0,
                deleted_count=0,
            )

        mutation = _guard_call(self.repository.delete_by_vector_doc_ids, cleaned)
        deleted_count = mutation.affected_count
        return VectorDeletionResult(
            requested_count=len(cleaned),
            deleted_count=deleted_count,
            not_found_count=len(cleaned) - deleted_count,
        )

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

        hash 投影契约：上游 ``ExpertVectorSynchronizer`` 增量幂等判断只读取
        逻辑 record 顶层的 ``content_hash`` / ``metadata_hash``（V14 把这两个
        哈希存为物理一级字段，V15 把它们移进了 ``metadata`` JSON）。这里在
        Service 边界做统一的逻辑投影——顶层已有合法哈希时保持不变，否则从
        ``metadata`` JSON 中提取这两个字段到逻辑返回顶层。这是纯内存投影，
        不修改 Milvus 物理 Schema，不增加一级字段，不平铺其它 metadata 字段；
        顶层与 ``metadata`` 都缺失时不在本层补造哈希，交由调用方 fail-closed。
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
                records.append(_project_hash_fields(row))
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
            5. 通过 → ``MilvusRowMapper.from_row`` 复用旧向量 + 新 metadata upsert；
            6. model_space/dim 安全门失败 → 该条 ``status=failed``、
               ``recommendation="call_upsert"``，不调用 Embedding、不静默降级。
        """

        items: list[VectorItemResult] = []
        for chunk in chunks:
            if not isinstance(chunk, ExpertChunkEmbeddingInput):
                raise VectorContractError(
                    "update_metadata 只接受 ExpertChunkEmbeddingInput"
                )
            vector_doc_id = chunk.vector_doc_id
            validated = self.validator.validate(chunk)

            old_row = _guard_call(
                self.repository.get_by_vector_doc_id,
                vector_doc_id,
                include_embedding=True,
            )
            if old_row is None:
                items.append(
                    _failed_item(
                        vector_doc_id,
                        "Milvus 实体不存在，无法复用旧向量",
                    )
                )
                continue

            old_chunk = _guard_call(self.row_mapper.from_row, old_row)

            # metadata-only 安全门：本次输入必须与 existing 实体的
            # vector_doc_id / content_hash / chunk_text 完全一致，否则属于调用方
            # 误用（正文/主键已变化，不应走 update_metadata）。任一不一致立即
            # fail-closed 抛 VectorContractError，不静默降级、不自动修正 hash、
            # 不重新生成 vector_doc_id、不调用 Embedding。
            metadata_only_error = self._metadata_only_inconsistency_error(
                validated,
                old_chunk,
            )
            if metadata_only_error is not None:
                raise VectorContractError(metadata_only_error)

            gate_error = self._metadata_gate_error(old_chunk)
            if gate_error is not None:
                items.append(_failed_item(vector_doc_id, gate_error))
                continue

            new_chunk = EmbeddedContractPreparedChunk(
                **validated.model_dump(),
                embedding=old_chunk.embedding,
                embedding_dim=old_chunk.embedding_dim,
                embedding_model_alias=old_chunk.embedding_model_alias,
                embedding_model_version=old_chunk.embedding_model_version,
                embedded_at=old_chunk.embedded_at,
            )
            _guard_call(self.repository.upsert_batch, [new_chunk])
            items.append(
                VectorItemResult(
                    vector_doc_id=vector_doc_id,
                    status="metadata_updated",
                )
            )
        return VectorIngestionResult(results=items)

    @staticmethod
    def _metadata_only_inconsistency_error(
        validated: ContractPreparedChunk,
        old_chunk: EmbeddedContractPreparedChunk,
    ) -> str | None:
        """校验本次输入是否为 metadata-only 更新，返回失败原因（None 表示通过）。

        ``update_metadata`` 复用旧向量，禁止任何正文 / 主键变化；任一不一致
        属于调用方误用，应改走 ``upsert_chunks`` 重新 Embedding。
        """

        if validated.vector_doc_id != old_chunk.vector_doc_id:
            return (
                f"update_metadata 只允许 metadata-only 更新；vector_doc_id 不一致 "
                f"(input={validated.vector_doc_id!r}, "
                f"existing={old_chunk.vector_doc_id!r})；"
                "正文/主键变化必须走 upsert_chunks 并重新 Embedding"
            )
        if validated.content_hash != old_chunk.content_hash:
            return (
                "update_metadata 只允许 metadata-only 更新；content_hash 不一致 "
                f"(input={validated.content_hash[:12]!r}, "
                f"existing={old_chunk.content_hash[:12]!r})；"
                "正文变化必须走 upsert_chunks 并重新 Embedding"
            )
        if validated.chunk_text != old_chunk.chunk_text:
            return (
                "update_metadata 只允许 metadata-only 更新；chunk_text 不一致；"
                "正文变化必须走 upsert_chunks 并重新 Embedding"
            )
        return None

    def _metadata_gate_error(
        self,
        old_chunk: EmbeddedContractPreparedChunk,
    ) -> str | None:
        """校验 metadata-only 安全门，返回失败原因（None 表示通过）。"""

        old_space = build_model_space_id(
            model_alias=old_chunk.embedding_model_alias,
            dimension=old_chunk.embedding_dim,
            model_version=old_chunk.embedding_model_version,
        )
        if old_space != self.formal_model_space_id:
            return (
                f"旧向量 model_space 不一致: {old_space} != "
                f"{self.formal_model_space_id}"
            )
        if old_chunk.embedding_dim != self.expected_dimension:
            return (
                f"旧向量 embedding_dim 不一致: {old_chunk.embedding_dim} != "
                f"{self.expected_dimension}"
            )
        return None

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


def _failed_item(
    vector_doc_id: str,
    message: str,
) -> VectorItemResult:
    """构造 metadata-only 安全门失败的公共结果。"""

    return VectorItemResult(
        vector_doc_id=vector_doc_id,
        status="failed",
        error_message=message,
        recommendation="call_upsert",
    )


__all__ = ["VectorIngestionService"]