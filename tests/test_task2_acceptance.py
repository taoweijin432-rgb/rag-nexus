"""scripts/task2_acceptance.py 的纯函数保护逻辑单元测试。

验收脚本编排真实 MySQL / Embedding / Milvus，本地测试只覆盖不依赖服务的
fail-closed 纯函数：旧字段检测、批量数量核对、旧 Collection 快照比较。
"""

from __future__ import annotations

from expert_contracts.models import ExpertChunkEmbeddingInput
from scripts.task2_acceptance import (
    _PROJECT_ROOT,
    _batch_count_mismatch,
    _classify_validation_failure,
    _compare_v13_state,
    _deprecated_fields_present,
    _mean,
    _new_report,
    _safe_error,
    _v13_snapshot,
    _verify_queries,
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


def test_classify_validation_failure_known_categories() -> None:
    assert _classify_validation_failure("content_hash 与 chunk_text 的 SHA-256 不一致") == "content_hash"
    assert _classify_validation_failure("char_count 与 chunk_text 实际字符数不一致") == "char_count"
    assert _classify_validation_failure("chunk_text 疑似包含邮箱") == "PII"
    assert _classify_validation_failure("profile_summary 的 origin_type 必须是 compressed_from_source") == "origin_type"
    assert _classify_validation_failure("chunk_text 超过字符上限") == "chunk_text 超长"


def test_classify_validation_failure_unknown_returns_other() -> None:
    assert _classify_validation_failure("未知的校验错误") == "other"


def test_mean_empty_list_returns_zero() -> None:
    assert _mean([]) == 0.0


def test_mean_of_chars() -> None:
    assert _mean([100, 200, 300]) == 200.0


def test_quality_report_skeleton_has_all_sections() -> None:
    import argparse

    import scripts.task2_acceptance as module

    args = argparse.Namespace(
        collection="expert_candidates_v14_contract_dev",
        v13_collection="expert_candidates_v13_contract_dev",
        report=module._PROJECT_ROOT / "reports" / "task2_acceptance.json",
        expected_count=7,
        database_timezone="Asia/Shanghai",
    )
    report = module._new_report(args)
    quality = report["quality"]

    assert set(quality.keys()) == {
        "chunk_type_distribution",
        "chunk_text_length",
        "contract_validation_failures",
        "eligibility_skipped",
        "embedding",
        "milvus",
    }
    assert quality["chunk_text_length"]["count"] == 0
    assert quality["embedding"]["embedded_count"] == 0
    assert quality["milvus"]["written"] == 0
    assert quality["milvus"]["vector_doc_id_queried"] == 0
    assert quality["milvus"]["expert_id_queried"] == 0


# ---------------------------------------------------------------------------
# _verify_queries（skipped-aware）单元测试
# ---------------------------------------------------------------------------
# 验收脚本从“全部输入记录必然写入 Milvus”改为“只回查 embedded(eligible) 记录、
# 并要求 skipped 记录在 Milvus 中不存在”。以下是针对该分支逻辑的纯函数测试。

_VALID_HASH = "2bed81e2ab4c94e5e78cd9a7dccaefd5355b4039695ea6669cf2d0146339eb6e"


def _valid_record_payload(
    *, expert_id: int, vector_doc_id: str, chunk_type: str
) -> dict:
    """构造一条合法的 v4 ExpertChunkEmbeddingInput 输入（用于测试，不写数据库）。"""
    return {
        "contract_version": "4.0.0",
        "vector_doc_id": vector_doc_id,
        "expert_id": expert_id,
        "serial_number": "TEST",
        "chunk_type": chunk_type,
        "chunk_subtype": None,
        "chunk_index": 0,
        "chunk_total": 1,
        "chunk_text": f"虚构专家 {expert_id} 的 {chunk_type} 概述内容。",
        "char_count": 20,
        "content_hash": _VALID_HASH,
        "metadata_hash": _VALID_HASH,
        "metadata": {
            "company_name_raw": None,
            "company_name_norm": None,
            "role_name_raw": None,
            "role_code": None,
            "role_level": None,
            "job_status_norm": "active",
            "abroad_flag_norm": False,
            "work_years": "3.00",
            "project_name": None,
            "start_month": None,
            "end_month": None,
            "is_current": None,
            "industry_codes": ["artificial_intelligence"],
            "function_codes": ["engineering"],
            "topic_codes": ["search"],
            "school_names_raw": [],
            "major_names_raw": [],
            "eligible_for_recommendation": True,
            "origin_type": "compressed_from_source",
            "verification_status": "source_summary",
            "weight_tier": "B",
        },
        "source": {
            "source_table": "experts",
            "source_field": "background",
            "source_field_id": str(expert_id),
            "source_version": "2026-08-11T10:00:00+08:00",
            "source_updated_at": "2026-08-11T10:00:00+08:00",
            "source_hash": _VALID_HASH,
        },
        "versions": {"chunk_policy_version": "1.3", "metadata_schema_version": "3.0"},
        "is_active": True,
        "updated_at": "2026-08-11T10:05:00+08:00",
    }


def _build_record(*, expert_id: int, chunk_index: int) -> ExpertChunkEmbeddingInput:
    """按给定 expert/chunk_index 构造合法记录（v4 身份公式一致）。"""
    vector_doc_id = f"expert:{expert_id}:profile_summary:{expert_id}:{chunk_index}"
    return ExpertChunkEmbeddingInput.model_validate(
        _valid_record_payload(
            expert_id=expert_id,
            vector_doc_id=vector_doc_id,
            chunk_type="profile_summary",
        )
    )


def _row_for(record: ExpertChunkEmbeddingInput) -> dict:
    """根据记录构造一条与存储侧一致的 Milvus 回查行（维度 1024）。"""
    return {
        "vector_doc_id": record.vector_doc_id,
        "expert_id": record.expert_id,
        "chunk_type": record.chunk_type,
        "chunk_text": record.chunk_text,
        "char_count": record.char_count,
        "content_hash": record.content_hash,
        "metadata_hash": record.metadata_hash,
        "metadata_ext": record.metadata.model_dump(mode="json"),
        "source_ext": record.source.model_dump(mode="json"),
        "versions_ext": record.versions.model_dump(mode="json"),
        "is_active": record.is_active,
        "verification_status": record.metadata.verification_status,
        "eligible_for_recommendation": record.metadata.eligible_for_recommendation,
        "embedding_model_alias": "alibaba-bailian:text-embedding-v4",
        "embedding_model_version": "v1",
        "embedding_dim": 1024,
        "embedding": [0.0] * 1024,
    }


class _FakeModel:
    """模拟 CompanyCloudEmbeddingModel 的模型空间属性。"""

    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"
    dimension = 1024


class _FakeClient:
    """按 expert_id == N 过滤存储行，模拟 Milvus query。"""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def query(
        self,
        collection_name: str,
        filter: str,
        output_fields: list[str],
        consistency_level: str,
    ) -> list[dict]:
        expert_id = int(filter.split("==", 1)[1].strip())
        return [row for row in self._rows if row["expert_id"] == expert_id]


class _FakeSettings:
    collection_name = "test_collection"


class _FakeRepository:
    """最小 ContractMilvusVectorRepository 替身：只提供 _verify_queries 用到的接口。"""

    def __init__(self, stored_rows: list[dict]) -> None:
        self._stored = {row["vector_doc_id"]: row for row in stored_rows}
        self.client = _FakeClient(stored_rows)
        self.settings = _FakeSettings()

    def get_by_vector_doc_id(self, vector_doc_id: str) -> dict | None:
        return self._stored.get(vector_doc_id)


def _verify_report(
    upsert_ids: list[str], skipped_ids: list[str]
) -> dict:
    """构造 _verify_queries 需要的 report（ingestion 主键集合就绪）。"""
    import argparse

    report = _new_report(
        argparse.Namespace(
            collection="expert_candidates_v14_contract_dev",
            v13_collection="expert_candidates_v13_contract_dev",
            report=_PROJECT_ROOT / "reports" / "fake.json",
            expected_count=2,
            database_timezone="Asia/Shanghai",
        )
    )
    report["ingestion"]["upsert_ids"] = list(upsert_ids)
    report["ingestion"]["skipped_ids"] = list(skipped_ids)
    return report


def test_verify_queries_roundtrips_only_eligible_and_checks_skipped_absent() -> None:
    """skipped-aware：只回查 embedded 记录，skipped 记录要求 Milvus 缺席。"""
    rec_a = _build_record(expert_id=1, chunk_index=0)
    rec_b = _build_record(expert_id=2, chunk_index=0)
    skipped_c = _build_record(expert_id=99, chunk_index=0)
    skipped_d = _build_record(expert_id=100, chunk_index=0)

    # 存储侧只有 embedded 记录，skipped 记录不应存在。
    stored = [_row_for(rec_a), _row_for(rec_b)]
    repository = _FakeRepository(stored)
    report = _verify_report(
        upsert_ids=[rec_a.vector_doc_id, rec_b.vector_doc_id],
        skipped_ids=[skipped_c.vector_doc_id, skipped_d.vector_doc_id],
    )

    _verify_queries(
        [rec_a, rec_b, skipped_c, skipped_d], repository, report, _FakeModel()
    )

    # vector_doc_id_query 只覆盖 embedded 记录，无错误。
    vector_results = report["queries"]["vector_doc_id"]
    queried_ids = {r["vector_doc_id"] for r in vector_results}
    assert queried_ids == {rec_a.vector_doc_id, rec_b.vector_doc_id}
    assert skipped_c.vector_doc_id not in queried_ids
    assert skipped_d.vector_doc_id not in queried_ids
    assert all(r["found"] for r in vector_results)
    assert all("error" not in r for r in vector_results)

    # skipped_absent_query：skipped 记录均缺席。
    skipped_absent = report["queries"]["skipped_absent"]
    assert {s["vector_doc_id"] for s in skipped_absent} == {
        skipped_c.vector_doc_id,
        skipped_d.vector_doc_id,
    }
    assert all(s["absent"] for s in skipped_absent)
    assert all("error" not in s for s in skipped_absent)

    # expert_id_query 只统计 embedded 专家的记录（不含 99/100）。
    expert_results = report["queries"]["expert_id"]
    assert set(expert_results.keys()) == {"1", "2"}
    assert all(r["exact_set_match"] for r in expert_results.values())
    assert all(r["all_fields_match"] for r in expert_results.values())
    assert all("error" not in r for r in expert_results.values())

    # 所有相关 stage PASS。
    assert report["stages"]["vector_doc_id_query"] == "PASS"
    assert report["stages"]["skipped_absent_query"] == "PASS"
    assert report["stages"]["expert_id_query"] == "PASS"
    assert report["failures"] == []


def test_verify_queries_skipped_record_present_in_milvus_fails() -> None:
    """skipped 记录竟然存在于 Milvus 时，skipped_absent_query 必须 FAIL。"""
    rec_a = _build_record(expert_id=1, chunk_index=0)
    skipped_c = _build_record(expert_id=99, chunk_index=0)

    # 存储侧错误地同时包含 skipped 记录（不应发生）。
    stored = [_row_for(rec_a), _row_for(skipped_c)]
    repository = _FakeRepository(stored)
    report = _verify_report(
        upsert_ids=[rec_a.vector_doc_id],
        skipped_ids=[skipped_c.vector_doc_id],
    )

    _verify_queries([rec_a, skipped_c], repository, report, _FakeModel())

    skipped_absent = report["queries"]["skipped_absent"]
    # vector_doc_id_query 不要求 skipped 存在，因此 vector 失败为空。
    assert all("error" not in r for r in report["queries"]["vector_doc_id"])
    # skipped 存在 → 该行 absent=False 且有错误标记。
    assert not skipped_absent[0]["absent"]
    assert "error" in skipped_absent[0]
    assert report["stages"]["skipped_absent_query"] == "FAIL"
    assert any(
        f["stage"] == "skipped_absent_query" for f in report["failures"]
    )


def test_verify_queries_fails_when_embedded_record_missing() -> None:
    """embedded 记录未写入 Milvus 时，vector_doc_id_query 必须 FAIL。"""
    rec_a = _build_record(expert_id=1, chunk_index=0)
    rec_b = _build_record(expert_id=2, chunk_index=0)
    skipped_c = _build_record(expert_id=99, chunk_index=0)

    # 存储侧只写入了 rec_a，rec_b 虽被报告为 embedded 却缺失。
    stored = [_row_for(rec_a)]
    repository = _FakeRepository(stored)
    report = _verify_report(
        upsert_ids=[rec_a.vector_doc_id, rec_b.vector_doc_id],
        skipped_ids=[skipped_c.vector_doc_id],
    )

    _verify_queries([rec_a, rec_b, skipped_c], repository, report, _FakeModel())

    vector_results = report["queries"]["vector_doc_id"]
    failed = [r for r in vector_results if "error" in r]
    assert len(failed) == 1
    assert failed[0]["vector_doc_id"] == rec_b.vector_doc_id
    assert report["stages"]["vector_doc_id_query"] == "FAIL"
    assert any(
        f["stage"] == "vector_doc_id_query" and f.get("identifier") == rec_b.vector_doc_id
        for f in report["failures"]
    )