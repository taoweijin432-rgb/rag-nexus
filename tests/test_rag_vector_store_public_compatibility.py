"""根包懒加载与一个发布周期兼容入口测试。"""

from __future__ import annotations

import subprocess
import sys

import pytest

from rag_vector_store.core.ingestion import (
    DomainBatchIngestionService,
    DomainIngestionService,
)


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    """在全新解释器中验证模块加载状态，避免当前 pytest 进程污染。"""

    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def test_importing_generic_core_does_not_load_domain_or_milvus_dependencies() -> None:
    result = _run_isolated(
        """import sys
import rag_vector_store
assert set(rag_vector_store.__all__) == set(rag_vector_store._LAZY_EXPORTS)
import rag_vector_store.core
assert 'expert_contracts' not in sys.modules
assert 'pymilvus' not in sys.modules
"""
    )

    assert result.returncode == 0, result.stderr


def test_root_legacy_list_exports_warn_when_first_resolved() -> None:
    result = _run_isolated(
        """import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    from rag_vector_store import VectorExpertList, VectorResumeList
assert VectorExpertList.__name__ == 'VectorExpertList'
assert VectorResumeList.__name__ == 'VectorResumeList'
messages = [str(item.message) for item in caught]
assert any('VectorExpertList 已弃用' in message for message in messages)
assert any('VectorResumeList 已弃用' in message for message in messages)
"""
    )

    assert result.returncode == 0, result.stderr


def test_legacy_core_adapter_paths_warn() -> None:
    result = _run_isolated(
        """import warnings
import rag_vector_store.core as core
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    ExpertContractAdapter = core.ExpertContractAdapter
    from rag_vector_store.core.expert import ResumeContractAdapter
assert ExpertContractAdapter.__name__ == 'ExpertContractAdapter'
assert ResumeContractAdapter.__name__ == 'ResumeContractAdapter'
messages = [str(item.message) for item in caught]
assert any('core.ExpertContractAdapter 已弃用' in message for message in messages)
assert any('core.expert 已弃用' in message for message in messages)
"""
    )

    assert result.returncode == 0, result.stderr


def test_domain_batch_ingestion_service_warns_and_delegates() -> None:
    with pytest.warns(DeprecationWarning, match="DomainBatchIngestionService 已弃用"):
        service = DomainBatchIngestionService(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )

    assert isinstance(service, DomainIngestionService)


def test_removed_incremental_symbols_are_not_publicly_importable() -> None:
    from rag_vector_store import incremental

    removed = {
        "AdaptedContract",
        "ContractMetadataUpdateIntent",
        "ContractSynthesisError",
        "PreRejectedInput",
        "SourceEventOrderView",
        "resolve_winner_indices",
    }
    assert removed.isdisjoint(incremental.__all__)
    for name in removed:
        with pytest.raises(AttributeError):
            getattr(incremental, name)
