"""Factory 对 V15 Repository 选择能力的接通测试（Fake/Mock，不联网）。

覆盖 Task5-3 要求：
    1. 默认 factory → V15 repository；
    2. schema_profile="v15" → V15 repository；
    3. ingestion V15 能正确组装（含 V15 row_mapper 跟随）；
    4. reconciliation V15 的 repository 与 ingestion.repository 是同一个对象（is）；
    5. V14 作为显式 schema_profile 时 fail-closed（已退役）；
    6. 非法 profile → fail-closed；
    7. 显式 repository 注入与 profile 冲突 → fail-closed；
    8. V15 update_metadata 使用 V15 mapper/repository 路径；
    9. 全程不连接真实 Milvus / Embedding 网络。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from expert_contracts import adapt_expert_chunk_mapping
from rag_vector_store import (
    ContractEmbeddingWorker,
    FullVectorReconciliationService,
    MilvusSettings,
    VectorIngestionService,
    create_full_vector_reconciliation_service,
    create_vector_ingestion_service,
)
from rag_vector_store.chunking.contract_ingestion import ContractBatchIngestionService
from rag_vector_store.core.ingestion import DomainIngestionService
from rag_vector_store.milvus import ContractMilvusV15VectorRepository
from rag_vector_store.milvus.contract_v15 import MilvusV15RowMapper

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v4.jsonl"
)


class FakeEmbeddingModel:
    """最小 fake：仅提供 Worker 构造所需的协议表面，不联网。"""

    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


def _worker() -> ContractEmbeddingWorker:
    return ContractEmbeddingWorker(
        FakeEmbeddingModel(),
        expected_dimension=1024,
    )


def _settings(
    collection_name: str = "expert_candidates_v15_contract_dev",
) -> MilvusSettings:
    return MilvusSettings(
        collection_name=collection_name,
        index_type="HNSW",
        metric_type="COSINE",
        index_params={"M": 16, "efConstruction": 200},
    )


def _input_records() -> list[Any]:
    """返回 Contract 输入记录（ExpertChunkEmbeddingInput，未嵌入）。"""

    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        return [
            adapt_expert_chunk_mapping(json.loads(line))
            for line in stream
            if line.strip()
        ]


# ---------------------------------------------------------------------------
# Fake V15 Milvus client（与 test_contract_milvus_repository_v15.py 同构）
# ---------------------------------------------------------------------------


class FakeIndexParams:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FakeQueryIterator:
    def __init__(self, batches: list[object]) -> None:
        self.batches = list(batches)

    def next(self) -> object:
        return self.batches.pop(0) if self.batches else []

    def close(self) -> None:
        pass


class FakeV15MilvusClient:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.description: dict[str, Any] | None = None
        self.index_params = FakeIndexParams()
        self.rows: dict[str, dict[str, Any]] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []

    def has_collection(self, collection_name: str) -> bool:
        return self.exists

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        assert self.description is not None
        return self.description

    def get_collection_stats(self, collection_name: str) -> dict[str, int]:
        return {"row_count": len(self.rows)}

    def prepare_index_params(self) -> FakeIndexParams:
        return self.index_params

    def list_indexes(self, collection_name: str) -> list[str]:
        return ["embedding_vector"]

    def describe_index(
        self,
        collection_name: str,
        index_name: str,
    ) -> dict[str, Any]:
        if index_name == "embedding_vector":
            return {
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "M": "16",
                "efConstruction": "200",
            }
        return {
            "field_name": index_name.removesuffix("_inverted"),
            "index_type": "INVERTED",
        }

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)
        self.exists = True

    def upsert(self, **kwargs: Any) -> dict[str, Any]:
        self.upsert_calls.append(kwargs)
        for row in kwargs["data"]:
            self.rows[str(row["vector_doc_id"])] = dict(row)
        return {
            "upsert_count": len(kwargs["data"]),
            "ids": [row["vector_doc_id"] for row in kwargs["data"]],
        }

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        ids = [str(item) for item in kwargs["ids"]]
        deleted = [item for item in ids if item in self.rows]
        for item in deleted:
            del self.rows[item]
        return {"delete_count": len(deleted)}

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append(kwargs)
        ids = kwargs.get("ids")
        if ids is None:
            rows = list(self.rows.values())
        else:
            rows = [
                self.rows[vid] for vid in ids if vid in self.rows
            ]
        output_fields = kwargs.get("output_fields")
        if output_fields:
            rows = [
                {f: row.get(f) for f in output_fields if f in row}
                for row in rows
            ]
        return rows

    def query_iterator(self, **kwargs: Any) -> FakeQueryIterator:
        rows = list(self.rows.values())
        if kwargs.get("filter", "").startswith("expert_id == "):
            expert_id = int(kwargs["filter"].split("== ")[1])
            rows = [
                row
                for row in rows
                if row.get("expert_id") == expert_id
                or str(row.get("expert_id")) == str(expert_id)
            ]
        output_fields = kwargs.get("output_fields")
        if output_fields:
            rows = [
                {f: row.get(f) for f in output_fields if f in row}
                for row in rows
            ]
        return FakeQueryIterator([rows])

    def search(self, **kwargs: Any) -> Any:
        raise AssertionError("factory V15 测试不应触发 search")


def _v15_description(repository: ContractMilvusV15VectorRepository) -> dict[str, Any]:
    return repository._build_schema(1024).to_dict()


# ---------------------------------------------------------------------------
# 1/5. 默认 V15 行为：factory 默认 schema_profile="v15"，V14 为显式回滚路径
# ---------------------------------------------------------------------------


def test_factory_default_uses_v15_repository() -> None:
    ingestion = create_vector_ingestion_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
    )
    assert isinstance(ingestion, VectorIngestionService)
    assert isinstance(ingestion.repository, ContractMilvusV15VectorRepository)
    assert isinstance(ingestion.row_mapper, MilvusV15RowMapper)
    assert ingestion.repository._allow_collection_creation is False


def test_factory_repository_refuses_to_create_missing_collection() -> None:
    ingestion = create_vector_ingestion_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
    )
    client = FakeV15MilvusClient(exists=False)
    ingestion.repository._client = client

    with pytest.raises(RuntimeError, match="禁止自动创建"):
        ingestion.repository.initialize(1024)

    assert client.create_calls == []


def test_factory_injected_worker_does_not_read_embedding_environment() -> None:
    with patch(
        "rag_vector_store.ingestion.factory.CompanyCloudEmbeddingModel.from_env",
        side_effect=AssertionError("不应读取 Embedding 环境配置"),
    ) as from_env:
        ingestion = create_vector_ingestion_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
        )

    assert isinstance(ingestion, VectorIngestionService)
    from_env.assert_not_called()


def test_factory_injected_repository_does_not_read_milvus_environment() -> None:
    repository = ContractMilvusV15VectorRepository(_settings(), 1024)
    with patch(
        "rag_vector_store.ingestion.factory.MilvusSettings.from_env",
        side_effect=AssertionError("不应读取 Milvus 环境配置"),
    ) as from_env:
        ingestion = create_vector_ingestion_service(
            repository=repository,
            embedding_worker=_worker(),
        )

    assert ingestion.repository is repository
    from_env.assert_not_called()


def test_explicit_schema_profile_warns_during_compatibility_period() -> None:
    with pytest.warns(DeprecationWarning, match="schema_profile 参数已弃用"):
        ingestion = create_vector_ingestion_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            schema_profile="v15",
        )

    assert isinstance(ingestion.repository, ContractMilvusV15VectorRepository)


def test_reconciliation_default_uses_v15_repository() -> None:
    service = create_full_vector_reconciliation_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
    )
    assert isinstance(service, FullVectorReconciliationService)
    assert isinstance(service.repository, ContractMilvusV15VectorRepository)
    assert isinstance(service.ingestion.row_mapper, MilvusV15RowMapper)
    # 同一 repository 对象
    assert service.repository is service.ingestion.repository


def test_factory_explicit_v14_profile_fail_closed() -> None:
    """V14 旧链路已退役：显式 schema_profile="v14" 必须 fail-closed。"""

    with pytest.raises(ValueError, match="V14"):
        create_vector_ingestion_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            schema_profile="v14",
        )
    with pytest.raises(ValueError, match="V14"):
        create_full_vector_reconciliation_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            schema_profile="v14",
        )


# ---------------------------------------------------------------------------
# 2/3. schema_profile="v15" → V15 repository / 组装
# ---------------------------------------------------------------------------


def test_factory_v15_uses_v15_repository_and_mapper() -> None:
    ingestion = create_vector_ingestion_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
        schema_profile="v15",
    )
    assert isinstance(ingestion.repository, ContractMilvusV15VectorRepository)
    # row_mapper 未显式传入时跟随 repository → V15 mapper
    assert isinstance(ingestion.row_mapper, MilvusV15RowMapper)


def test_ingestion_v15_assembles_chain() -> None:
    ingestion = create_vector_ingestion_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
        schema_profile="v15",
    )
    assert isinstance(ingestion.ingestion, DomainIngestionService)
    assert not isinstance(ingestion.ingestion, ContractBatchIngestionService)
    assert ingestion.ingestion.repository is ingestion.repository
    # 公共 API 语义保持不变
    for method in (
        "upsert_chunks",
        "update_metadata",
        "delete_vectors",
        "get_vectors",
        "list_vectors_by_expert",
    ):
        assert callable(getattr(ingestion, method))


def test_factory_v15_explicit_row_mapper_wins() -> None:
    """显式传入匹配的 row_mapper 时不被 profile 悄悄替换。"""

    explicit = MilvusV15RowMapper()
    ingestion = create_vector_ingestion_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
        row_mapper=explicit,
        schema_profile="v15",
    )
    assert ingestion.row_mapper is explicit


# ---------------------------------------------------------------------------
# 4. reconciliation V15 同一 repository 实例
# ---------------------------------------------------------------------------


def test_reconciliation_v15_shared_repository_instance() -> None:
    service = create_full_vector_reconciliation_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
        schema_profile="v15",
    )
    assert isinstance(service.repository, ContractMilvusV15VectorRepository)
    assert isinstance(service.ingestion.repository, ContractMilvusV15VectorRepository)
    # 对象身份一致（不只看 collection_name）
    assert service.repository is service.ingestion.repository


# ---------------------------------------------------------------------------
# 6. 非法 profile → fail-closed
# ---------------------------------------------------------------------------


def test_factory_invalid_profile_fail_closed() -> None:
    with pytest.raises(ValueError, match="schema_profile"):
        create_vector_ingestion_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            schema_profile="v16",
        )
    with pytest.raises(ValueError, match="schema_profile"):
        create_full_vector_reconciliation_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            schema_profile="v13",
        )


# ---------------------------------------------------------------------------
# 7. 显式 repository 注入与 profile 冲突 → fail-closed
# ---------------------------------------------------------------------------


def test_explicit_v15_repository_with_v14_profile_fail_closed() -> None:
    """schema_profile='v14' 先于 repository 校验 fail-closed（V14 已退役）。"""

    v15_repo = ContractMilvusV15VectorRepository(_settings(), 1024)
    with pytest.raises(ValueError, match="V14"):
        create_vector_ingestion_service(
            repository=v15_repo,
            embedding_worker=_worker(),
            schema_profile="v14",
        )
    with pytest.raises(ValueError, match="V14"):
        create_full_vector_reconciliation_service(
            repository=v15_repo,
            embedding_worker=_worker(),
            schema_profile="v14",
        )


def test_explicit_repository_is_used_without_replacement() -> None:
    """显式注入 V15 repository 时优先使用，不偷偷替换。"""

    v15_repo = ContractMilvusV15VectorRepository(_settings(), 1024)
    ingestion_v15 = create_vector_ingestion_service(
        repository=v15_repo,
        embedding_worker=_worker(),
        schema_profile="v15",
    )
    assert ingestion_v15.repository is v15_repo
    # row_mapper 跟随显式注入的 V15 repo
    assert isinstance(ingestion_v15.row_mapper, MilvusV15RowMapper)


# ---------------------------------------------------------------------------
# 8. V15 update_metadata 走 V15 mapper/repository 路径
# ---------------------------------------------------------------------------


def test_v15_update_metadata_uses_v15_mapper_and_repository() -> None:
    records = _input_records()
    client = FakeV15MilvusClient(exists=True)
    repository = ContractMilvusV15VectorRepository(
        _settings(),
        1024,
        client=client,
        allow_collection_creation=True,
    )
    client.description = _v15_description(repository)

    ingestion = create_vector_ingestion_service(
        repository=repository,
        embedding_worker=_worker(),
        schema_profile="v15",
    )

    # 初始写入（走 V15 upsert，内部由 worker 重新 Embedding）
    initial = ingestion.upsert_chunks(records)
    assert initial.embedded_count == len(records)
    assert isinstance(ingestion.row_mapper, MilvusV15RowMapper)
    initial_embedding = client.rows[records[0].vector_doc_id]["embedding"]

    # metadata-only 变更：仅 metadata_hash 变化
    new_record = records[0].model_copy(update={"metadata_hash": "1" * 64})
    result = ingestion.update_metadata([new_record])

    assert result.results[0].status == "metadata_updated"
    assert result.metadata_updated_count == 1
    # 写入的行是 4 字段 V15 行（含 flat metadata），且 metadata_hash 已更新
    written = client.rows[new_record.vector_doc_id]
    assert set(written) == {
        "vector_doc_id",
        "chunk_text",
        "embedding",
        "metadata",
    }
    # expert_id / content_hash / metadata_hash 已进入 flat metadata
    assert written["metadata"]["metadata_hash"] == "1" * 64
    assert written["metadata"]["content_hash"] == new_record.content_hash
    assert written["metadata"]["expert_id"] == new_record.expert_id
    # 复用旧向量：embedding 原样保留（update 不重新 Embedding）
    assert written["embedding"] == initial_embedding
    assert written["metadata"]["embedding_dim"] == 1024


# ---------------------------------------------------------------------------
# Final Fix：显式 RowMapper 与 Schema Profile 一致性校验
# ---------------------------------------------------------------------------


def test_mapper_v15_with_v14_profile_fail_closed() -> None:
    """schema_profile='v14' 先于 mapper 校验 fail-closed（V14 已退役）。"""

    with pytest.raises(ValueError, match="V14"):
        create_vector_ingestion_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            row_mapper=MilvusV15RowMapper(),
            schema_profile="v14",
        )


def test_non_mapper_object_with_v15_profile_fail_closed() -> None:
    """schema_profile='v15' + 显式传入不符合 RowMapper 协议的对象 → ValueError。"""

    with pytest.raises(ValueError, match="V15"):
        create_vector_ingestion_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            row_mapper=object(),  # type: ignore[arg-type]
            schema_profile="v15",
        )


def test_v15_mapper_with_explicit_v15_repo_allowed() -> None:
    """显式 V15 repo + V15 mapper → 允许。"""

    v15_repo = ContractMilvusV15VectorRepository(_settings(), 1024)
    ingestion = create_vector_ingestion_service(
        repository=v15_repo,
        embedding_worker=_worker(),
        row_mapper=MilvusV15RowMapper(),
        schema_profile="v15",
    )
    assert ingestion.repository is v15_repo
    assert isinstance(ingestion.row_mapper, MilvusV15RowMapper)


def test_mapper_matching_profile_allowed() -> None:
    """正常组合：v15 + V15 mapper → 允许。"""

    ingestion_v15 = create_vector_ingestion_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
        row_mapper=MilvusV15RowMapper(),
        schema_profile="v15",
    )
    assert isinstance(ingestion_v15.row_mapper, MilvusV15RowMapper)
    assert isinstance(ingestion_v15.repository, ContractMilvusV15VectorRepository)


def test_mapper_auto_follows_repository_when_not_given() -> None:
    """未显式 mapper 时默认跟随为 V15 mapper（V14 已退役）。"""

    ingestion_v15 = create_vector_ingestion_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
        schema_profile="v15",
    )
    assert isinstance(ingestion_v15.row_mapper, MilvusV15RowMapper)


def test_reconciliation_factory_mapper_profile_protected() -> None:
    """reconciliation factory 同样受显式 mapper 一致性保护。"""

    # v14 profile → fail-closed（V14 已退役，先于 mapper 校验）
    with pytest.raises(ValueError, match="V14"):
        create_full_vector_reconciliation_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            row_mapper=MilvusV15RowMapper(),
            schema_profile="v14",
        )
    # 非 RowMapper 对象 + v15 profile → fail-closed
    with pytest.raises(ValueError, match="V15"):
        create_full_vector_reconciliation_service(
            milvus_settings=_settings(),
            embedding_worker=_worker(),
            row_mapper=object(),  # type: ignore[arg-type]
            schema_profile="v15",
        )
    # 正常组合仍可组装，且同一 repository
    service = create_full_vector_reconciliation_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
        row_mapper=MilvusV15RowMapper(),
        schema_profile="v15",
    )
    assert isinstance(service.ingestion.row_mapper, MilvusV15RowMapper)
    assert service.repository is service.ingestion.repository
    assert isinstance(service.repository, ContractMilvusV15VectorRepository)


# ---------------------------------------------------------------------------
# 9. 不联网：上述测试仅构造对象 + 操作 fake client
# ---------------------------------------------------------------------------


def test_factory_v15_never_touches_network() -> None:
    """构造 V15 factory 组件不发起任何真实 Milvus / Embedding 网络调用。"""

    # 不注入 embedding_worker 会触发 CompanyCloudEmbeddingModel.from_env()，
    # 因此这里显式注入 fake worker 与本地 milvus_settings。
    ingestion = create_vector_ingestion_service(
        milvus_settings=_settings(),
        embedding_worker=_worker(),
        schema_profile="v15",
    )
    assert isinstance(ingestion.repository, ContractMilvusV15VectorRepository)
    # repository client 延迟创建：未调用任何 client 方法前为 None
    assert ingestion.repository._client is None
