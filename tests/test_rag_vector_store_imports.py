"""根包懒加载与已删除入口测试。"""

from __future__ import annotations

import subprocess
import sys


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
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
import rag_vector_store.core
assert 'expert_contracts' not in sys.modules
assert 'pymilvus' not in sys.modules
"""
    )
    assert result.returncode == 0, result.stderr


def test_importing_result_models_does_not_load_milvus() -> None:
    result = _run_isolated(
        """import sys
from rag_vector_store import VectorOwnerList
assert VectorOwnerList.__name__ == 'VectorOwnerList'
assert 'pymilvus' not in sys.modules
"""
    )
    assert result.returncode == 0, result.stderr


def test_removed_legacy_symbols_are_not_publicly_importable() -> None:
    from rag_vector_store import incremental

    removed = {
        ("Adapted" "Contract"),
        "ContractMetadata" + "UpdateIntent",
        "ContractSynthesis" + "Error",
        "PreRejected" + "Input",
        "SourceEvent" + "OrderView",
        "resolve_winner" + "_indices",
        "CONSUMER_ACTION_ADAPT" + "_DISABLE",
    }
    assert removed.isdisjoint(incremental.__all__)
    for name in removed:
        assert not hasattr(incremental, name)
