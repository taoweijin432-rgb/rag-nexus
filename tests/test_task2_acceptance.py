"""scripts/task2_acceptance.py 的纯函数保护逻辑单元测试。

验收脚本编排真实 MySQL / Embedding / Milvus，本地测试只覆盖不依赖服务的
fail-closed 纯函数：旧字段检测、批量数量核对、旧 Collection 快照比较。
"""

from __future__ import annotations

from scripts.task2_acceptance import (
    _batch_count_mismatch,
    _compare_v13_state,
    _deprecated_fields_present,
    _safe_error,
    _v13_snapshot,
)


def test_task2_acceptance_redacts_mysql_password(monkeypatch) -> None:
    monkeypatch.setenv("MYSQL_PASSWORD", "mysql-secret")

    error = _safe_error(RuntimeError("connection failed for mysql-secret"))

    assert "mysql-secret" not in error["message"]
    assert "<redacted>" in error["message"]


def test_deprecated_fields_present_finds_old_schema_fields() -> None:
    field_names = [
        "vector_doc_id",
        "chunk_id",
        "expert_id",
        "metadata_json",
        "source_ext",
    ]

    assert _deprecated_fields_present(field_names) == ["chunk_id", "metadata_json"]


def test_deprecated_fields_present_is_empty_for_contract_4_0_0() -> None:
    field_names = [
        "vector_doc_id",
        "expert_id",
        "chunk_type",
        "metadata_ext",
        "source_ext",
        "versions_ext",
    ]

    assert _deprecated_fields_present(field_names) == []


def test_batch_count_mismatch_returns_none_when_consistent() -> None:
    result = _FakeBatchResult(embedded_count=6, skipped_count=1, affected_count=6)
    stats = {"eligible_count": 6, "skipped_count": 1}

    assert _batch_count_mismatch(result, stats) is None


def test_batch_count_mismatch_detects_embedded_difference() -> None:
    result = _FakeBatchResult(embedded_count=5, skipped_count=2, affected_count=5)
    stats = {"eligible_count": 6, "skipped_count": 1}

    mismatch = _batch_count_mismatch(result, stats)

    assert mismatch is not None
    assert "embedded_count=5" in mismatch
    assert "eligible_count=6" in mismatch


def test_batch_count_mismatch_detects_skipped_difference() -> None:
    result = _FakeBatchResult(embedded_count=6, skipped_count=2, affected_count=6)
    stats = {"eligible_count": 6, "skipped_count": 1}

    mismatch = _batch_count_mismatch(result, stats)

    assert mismatch is not None
    assert "skipped_count=2" in mismatch


def test_batch_count_mismatch_detects_affected_difference() -> None:
    result = _FakeBatchResult(embedded_count=6, skipped_count=1, affected_count=5)
    stats = {"eligible_count": 6, "skipped_count": 1}

    mismatch = _batch_count_mismatch(result, stats)

    assert mismatch is not None
    assert "affected_count=5" in mismatch


def test_v13_snapshot_uses_get_collection_stats(monkeypatch) -> None:
    """行数快照必须来自 get_collection_stats，不得发起无过滤实体查询。"""

    called = {"stats": 0, "query": 0}

    class _FakeClient:
        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return {
                "fields": [
                    {"name": "vector_doc_id", "is_primary": True},
                ]
            }

        def get_collection_stats(self, collection_name):
            called["stats"] += 1
            return {"row_count": 7}

        def query(self, **kwargs):
            called["query"] += 1
            return []

    snapshot = _v13_snapshot(_FakeClient(), "expert_candidates_v13_contract_dev")

    assert snapshot == {"exists": True, "row_count": 7, "pk": "vector_doc_id"}
    # 行数统计绝不走无过滤 query。
    assert called == {"stats": 1, "query": 0}


def test_v13_snapshot_returns_not_exists_for_missing_collection() -> None:
    class _FakeClient:
        def has_collection(self, collection_name):
            return False

        def get_collection_stats(self, collection_name):
            raise AssertionError("不存在的 Collection 不应查询行数")

    snapshot = _v13_snapshot(_FakeClient(), "expert_candidates_v13_contract_dev")

    assert snapshot == {"exists": False, "row_count": None, "pk": None}


def test_v13_snapshot_raises_when_stats_missing_row_count() -> None:
    class _FakeClient:
        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return {"fields": [{"name": "vector_doc_id", "is_primary": True}]}

        def get_collection_stats(self, collection_name):
            return {}

    try:
        _v13_snapshot(_FakeClient(), "expert_candidates_v13_contract_dev")
    except TypeError as exc:
        assert "row_count" in str(exc)
    else:  # pragma: no cover - 必须抛错
        raise AssertionError("缺少 row_count 的 stats 必须拒绝")


def test_v13_snapshot_raises_without_primary_key() -> None:
    class _FakeClient:
        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return {"fields": [{"name": "chunk_text"}]}

    try:
        _v13_snapshot(_FakeClient(), "expert_candidates_v13_contract_dev")
    except RuntimeError as exc:
        assert "主键" in str(exc)
    else:  # pragma: no cover - 必须抛错
        raise AssertionError("缺少主键的旧 Collection 必须拒绝")


def test_source_has_no_unlimited_empty_filter_query() -> None:
    """验收脚本不得存在无 limit 的空 filter query。"""

    import inspect

    import scripts.task2_acceptance as module

    source = inspect.getsource(module)
    # 不允许出现 filter 为空字符串的 query；行数统计一律走 get_collection_stats。
    assert 'filter=""' not in source
    assert 'filter=\'\'' not in source
    assert "filter_expression" in source  # 显式非空过滤表达式入口仍保留
    assert "get_collection_stats" in source


def test_compare_v13_state_unchanged_when_counts_equal() -> None:
    before = {"before_count": 7}
    after = {"row_count": 7}

    assert _compare_v13_state(before, after) == {
        "changed": False,
        "before_count": 7,
        "after_count": 7,
    }


def test_compare_v13_state_changed_when_counts_differ() -> None:
    before = {"before_count": 7}
    after = {"row_count": 8}

    state = _compare_v13_state(before, after)

    assert state["changed"] is True
    assert state["before_count"] == 7
    assert state["after_count"] == 8


def test_compare_v13_state_changed_when_snapshot_missing() -> None:
    before = {"before_count": None}
    after = {"row_count": 7}

    state = _compare_v13_state(before, after)

    assert state["changed"] is True


class _FakeBatchResult:
    """只含计数属性的最小替身，供 _batch_count_mismatch 使用。"""

    def __init__(
        self,
        *,
        embedded_count: int,
        skipped_count: int,
        affected_count: int,
    ) -> None:
        self.embedded_count = embedded_count
        self.skipped_count = skipped_count
        self.affected_count = affected_count