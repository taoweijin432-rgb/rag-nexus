"""VectorIngestionService 公共结果模型与异常层级测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_vector_store.ingestion import (
    VectorContractError,
    VectorDeletionResult,
    VectorEmbeddingError,
    VectorExpertList,
    VectorInconsistentStateError,
    VectorIngestionError,
    VectorIngestionResult,
    VectorItemResult,
    VectorMilvusError,
    VectorOwnerList,
    VectorQueryResult,
)


def test_vector_item_result_defaults() -> None:
    item = VectorItemResult(
        vector_doc_id="expert:1:profile_summary:1:0",
        status="embedded",
    )
    assert item.error_message is None
    assert item.recommendation is None


def test_vector_item_result_rejects_empty_vector_doc_id() -> None:
    with pytest.raises(ValidationError):
        VectorItemResult(vector_doc_id="", status="embedded")
    with pytest.raises(ValidationError):
        VectorItemResult(vector_doc_id="   ", status="embedded")


def test_vector_item_result_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        VectorItemResult(
            vector_doc_id="expert:1:profile_summary:1:0",
            status="unknown_status",
        )


def test_vector_item_result_accepts_all_statuses() -> None:
    for status in (
        "embedded",
        "metadata_updated",
        "skipped",
        "failed",
    ):
        item = VectorItemResult(
            vector_doc_id="expert:1:profile_summary:1:0",
            status=status,  # type: ignore[arg-type]
        )
        assert item.status == status


def test_vector_item_result_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        VectorItemResult(  # type: ignore[call-arg]
            vector_doc_id="expert:1:profile_summary:1:0",
            status="embedded",
            unrelated="x",
        )


def test_vector_ingestion_result_counts() -> None:
    result = VectorIngestionResult(
        results=[
            VectorItemResult(
                vector_doc_id=f"expert:1:profile_summary:{i}:0",
                status=status,
            )
            for i, status in enumerate(
                ["embedded", "metadata_updated", "skipped", "failed"]
            )
        ]
    )
    assert result.total_count == 4
    assert result.embedded_count == 1
    assert result.metadata_updated_count == 1
    assert result.skipped_count == 1
    assert result.failed_count == 1


def test_vector_ingestion_result_empty() -> None:
    assert VectorIngestionResult().total_count == 0


def test_vector_deletion_result_valid() -> None:
    result = VectorDeletionResult(
        requested_count=3,
        deleted_count=2,
        not_found_count=1,
    )
    assert result.requested_count == 3
    assert result.deleted_count == 2
    assert result.not_found_count == 1


def test_vector_deletion_result_defaults() -> None:
    result = VectorDeletionResult()
    assert result.requested_count == 0
    assert result.deleted_count == 0
    assert result.not_found_count == 0


def test_vector_deletion_result_inconsistent_count() -> None:
    # deleted + not_found_count != requested
    with pytest.raises(ValidationError):
        VectorDeletionResult(
            requested_count=3,
            deleted_count=1,
            not_found_count=0,
        )
    # deleted + not_found_count > requested
    with pytest.raises(ValidationError):
        VectorDeletionResult(
            requested_count=2,
            deleted_count=3,
            not_found_count=1,
        )
    # 不伪造具体 ID（现在只有数量字段，无 list）
    with pytest.raises(ValidationError):
        VectorDeletionResult(
            requested_count=1,
            deleted_count=0,
            not_found=["expert:9:profile_summary:9:0"],  # type: ignore[arg-type]
        )


def test_vector_query_result_rejects_embedding() -> None:
    with pytest.raises(ValidationError):
        VectorQueryResult(records=[{"vector_doc_id": "a", "embedding": [0.1]}])


def test_vector_query_result_accepts_no_embedding() -> None:
    result = VectorQueryResult(
        records=[
            {
                "vector_doc_id": "expert:1:profile_summary:1:0",
                "content_hash": "abc",
            }
        ],
        missing=["expert:9:profile_summary:9:0"],
    )
    assert len(result.records) == 1
    assert result.missing == ["expert:9:profile_summary:9:0"]


def test_vector_query_result_rejects_non_dict_record() -> None:
    with pytest.raises(ValidationError):
        VectorQueryResult(records=[["not-a-dict"]])


def test_vector_expert_list_valid() -> None:
    result = VectorExpertList(
        expert_id=900001,
        vector_doc_ids=["a", "b"],
        count=2,
    )
    assert result.expert_id == 900001
    assert result.vector_doc_ids == ["a", "b"]
    assert result.count == 2


def test_vector_expert_list_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        VectorExpertList(
            expert_id=900001,
            vector_doc_ids=["a", "b"],
            count=3,
        )


def test_vector_expert_list_defaults() -> None:
    result = VectorExpertList(expert_id=900001)
    assert result.vector_doc_ids == []
    assert result.count == 0


def test_vector_owner_list_validates_generic_owner_identity() -> None:
    result = VectorOwnerList(owner_id=900001, vector_doc_ids=["a"], count=1)
    assert result.owner_id == 900001
    with pytest.raises(ValidationError):
        VectorOwnerList(owner_id=900001, vector_doc_ids=["a"], count=0)


def test_vector_ingestion_error_is_base() -> None:
    assert issubclass(VectorContractError, VectorIngestionError)
    assert issubclass(VectorEmbeddingError, VectorIngestionError)
    assert issubclass(VectorMilvusError, VectorIngestionError)
    assert issubclass(VectorInconsistentStateError, VectorIngestionError)
    with pytest.raises(VectorIngestionError):
        raise VectorMilvusError("milvus down")


def test_vector_ingestion_error_normalizes_message() -> None:
    e = VectorContractError("line1\nline2  " + "x" * 500)
    message = str(e)
    assert "\n" not in message
    assert message.endswith("...")
    assert len(message) <= 200 + len("...")


def test_vector_ingestion_error_safe_large_payload() -> None:
    # 大对象（如 embedding / chunk_text）被截断，不整体泄漏
    huge = "0.1," * 100000
    e = VectorContractError(huge)
    assert "\n" not in str(e)
    assert str(e).endswith("...")
