"""FullVectorReconciliationService 生产工厂与公共导出测试。

重点验证：
    - factory 能成功组装 FullVectorReconciliationService；
    - repository 与 ingestion 使用同一个 repository 对象（即同一 Milvus
      Collection 配置），杜绝读集合 A / 写集合 B；
    - 配置错误 fail-closed（如 schema_mode 非 contract）；
    - 顶层公共 import 可用；
    - 现有 create_vector_ingestion_service() 行为不受影响。

全部使用注入的 fake 组件，不调用真实 Embedding / Milvus 网络服务。
"""

from __future__ import annotations

import pytest

from rag_vector_store import (
    FullVectorReconciliationResult,
    FullVectorReconciliationService,
    create_full_vector_reconciliation_service,
    create_vector_ingestion_service,
)
from rag_vector_store.chunking.contract_pipeline import ContractEmbeddingWorker
from rag_vector_store.config import MilvusSettings
from rag_vector_store.ingestion.vector_ingestion import VectorIngestionService
from rag_vector_store.milvus.repository_v15 import EXPERT_V15_PROFILE
from rag_vector_store.milvus.repository_v16 import ContractMilvusV16VectorRepository


class FakeEmbeddingModel:
    """最小 fake：仅提供 Worker 构造所需的协议表面，不联网。"""

    dimension = 1024
    model_alias = "fake-model"
    model_version = "fake-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


class FakeRepository:
    """最小 fake：持有与正式 Repository 相同的 ``settings`` 表面。"""

    def __init__(self, collection_name: str) -> None:
        self.settings = type(
            "FakeMilvusSettings", (), {"collection_name": collection_name}
        )()
        self.profile = EXPERT_V15_PROFILE


def _fake_worker() -> ContractEmbeddingWorker:
    model = FakeEmbeddingModel()
    return ContractEmbeddingWorker(model, expected_dimension=1024)


def test_factory_assembles_full_service() -> None:
    repo = FakeRepository("expert_candidates_v15_contract_dev")
    service = create_full_vector_reconciliation_service(
        repository=repo,  # type: ignore[arg-type]
        embedding_worker=_fake_worker(),
    )

    assert isinstance(service, FullVectorReconciliationService)
    assert service.repository is repo
    assert isinstance(service.ingestion, VectorIngestionService)
    assert service.delete_batch_size == 200
    # 三个入口方法仍然存在，未改名、未合并 dry_run 参数
    assert callable(service.reconcile_all)
    assert callable(service.apply_positive_changes)
    assert callable(service.apply_full_reconciliation)


def test_repository_and_ingestion_share_same_milvus_config() -> None:
    """核心安全约束：reader(读库)与写库共用同一 repository 对象与 Collection。"""

    repo = FakeRepository("expert_candidates_v15_contract_dev")
    service = create_full_vector_reconciliation_service(
        repository=repo,  # type: ignore[arg-type]
        embedding_worker=_fake_worker(),
    )

    # ingestion 与 reconciliation 全部引用同一个 repository 对象。
    assert service.ingestion.repository is service.repository
    assert service.repository is repo
    assert (
        service.ingestion.repository.settings.collection_name
        == service.repository.settings.collection_name
        == "expert_candidates_v15_contract_dev"
    )


def test_factory_consistent_config_without_explicit_repository() -> None:
    """不注入 repository 时，factory 复用由一个 milvus_settings 构建的 repository。"""

    service = create_full_vector_reconciliation_service(
        repository=None,
        embedding_worker=_fake_worker(),
        milvus_settings=MilvusSettings(
            collection_name="expert_candidates_v15_contract_dev"
        ),
    )

    assert service.ingestion.repository is service.repository
    assert service.repository.settings.collection_name == (
        "expert_candidates_v15_contract_dev"
    )


def test_factory_uses_v16_profile_for_full_reconciliation() -> None:
    service = create_full_vector_reconciliation_service(
        embedding_worker=_fake_worker(),
        milvus_settings=MilvusSettings(
            collection_name="expert_candidates_v16_contract",
            expert_storage_profile="v16",
        ),
    )

    assert isinstance(service.repository, ContractMilvusV16VectorRepository)
    assert service.ingestion.repository is service.repository


def test_factory_schema_mode_mismatch_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """schema_mode 非 contract 时 fail-closed（不产生任何组件）。"""

    monkeypatch.setenv("MILVUS_SCHEMA_MODE", "legacy")
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    with pytest.raises(ValueError, match="contract"):
        create_full_vector_reconciliation_service(
            repository=None,
            embedding_worker=_fake_worker(),
        )


def test_top_level_public_imports() -> None:
    from rag_vector_store import (
        FullVectorReconciliationService,
        create_full_vector_reconciliation_service,
    )

    assert FullVectorReconciliationResult is not None
    assert FullVectorReconciliationService is not None
    assert callable(create_full_vector_reconciliation_service)


def test_create_vector_ingestion_service_unaffected() -> None:
    """现有 create_vector_ingestion_service() 行为保持不变。"""

    repo = FakeRepository("expert_candidates_v15_contract_dev")
    ingestion = create_vector_ingestion_service(
        repository=repo,  # type: ignore[arg-type]
        embedding_worker=_fake_worker(),
    )

    assert isinstance(ingestion, VectorIngestionService)
    assert ingestion.repository is repo
    assert ingestion.ingestion.repository is repo


def test_delete_batch_size_forwarded() -> None:
    repo = FakeRepository("expert_candidates_v15_contract_dev")
    service = create_full_vector_reconciliation_service(
        repository=repo,  # type: ignore[arg-type]
        embedding_worker=_fake_worker(),
        delete_batch_size=5,
    )
    assert service.delete_batch_size == 5

    with pytest.raises(ValueError, match="正整数"):
        create_full_vector_reconciliation_service(
            repository=repo,  # type: ignore[arg-type]
            embedding_worker=_fake_worker(),
            delete_batch_size=0,
        )
