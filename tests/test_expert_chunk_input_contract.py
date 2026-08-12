from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from expert_contracts import adapt_expert_chunk_mapping
from expert_contracts.models import ExpertChunkEmbeddingInput
from rag_vector_store import (
    ContractEmbeddingWorker,
    ContractValidationError,
    ContractValidator,
    EmbeddingEligibilityValidator,
    compare_contract_output_schema,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "expert_chunk_embedding_input_v2.jsonl"
)


class FakeContractEmbeddingModel:
    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.mode == "dimension_mismatch":
            return [[0.1] * 1023 for _ in texts]
        if self.mode == "nan":
            return [[float("nan"), *([0.1] * 1023)] for _ in texts]
        if self.mode == "infinity":
            return [[float("inf"), *([0.1] * 1023)] for _ in texts]
        return [[0.1] * 1024 for _ in texts]


def fixture_records() -> list[ExpertChunkEmbeddingInput]:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        return [
            adapt_expert_chunk_mapping(json.loads(line))
            for line in stream
            if line.strip()
        ]


def fixture_payload(chunk_type: str = "profile_summary") -> dict[str, object]:
    with FIXTURE_PATH.open(encoding="utf-8") as stream:
        payloads = [json.loads(line) for line in stream]
    return next(payload for payload in payloads if payload["chunk_type"] == chunk_type)


def update_text(payload: dict[str, object], text: str) -> dict[str, object]:
    updated = dict(payload)
    updated["chunk_text"] = text
    updated["char_count"] = len(text)
    updated["content_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return updated


def fixed_clock() -> datetime:
    return datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_three_contract_fixture_types_pass() -> None:
    records = fixture_records()

    assert len(records) == 3
    assert [record.chunk_type for record in records] == [
        "profile_summary",
        "employment",
        "project",
    ]


def test_generated_json_schema_is_strict_contract_v200() -> None:
    schema = ExpertChunkEmbeddingInput.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["contract_version"]["const"] == "2.0.0"
    assert "vector_doc_id" in schema["properties"]
    assert set(schema["required"]) >= {
        "metadata",
        "source",
        "versions",
        "serial_number",
        "vector_doc_id",
    }
    assert "vector_doc_id" in schema["$defs"]["ExpertChunkMetadata"]["required"]
    source_schema = schema["$defs"]["ExpertChunkSource"]
    assert "source_record_id" not in source_schema["properties"]
    assert "source_field_id" in source_schema["required"]


def test_contract_version_error_is_rejected() -> None:
    payload = fixture_payload()
    payload["contract_version"] = "1.0.0"

    with pytest.raises(ContractValidationError, match="contract_version"):
        ContractValidator().validate_mapping(payload)


def test_vector_doc_id_is_required_and_must_match_metadata() -> None:
    missing_top_level = fixture_payload()
    del missing_top_level["vector_doc_id"]

    with pytest.raises(ContractValidationError, match="vector_doc_id"):
        ContractValidator().validate_mapping(missing_top_level)

    mismatched = fixture_payload()
    mismatched["vector_doc_id"] = "expert:900001:profile_summary:other-source:0"

    with pytest.raises(ContractValidationError, match="metadata.vector_doc_id"):
        ContractValidator().validate_mapping(mismatched)

    missing_metadata = fixture_payload()
    del missing_metadata["metadata"]["vector_doc_id"]

    with pytest.raises(ContractValidationError, match="vector_doc_id"):
        ContractValidator().validate_mapping(missing_metadata)


def test_content_hash_error_is_rejected() -> None:
    payload = fixture_payload()
    payload["content_hash"] = "0" * 64

    with pytest.raises(ContractValidationError, match="content_hash"):
        ContractValidator().validate_mapping(payload)


def test_char_count_error_is_rejected() -> None:
    payload = fixture_payload()
    payload["char_count"] = int(payload["char_count"]) + 1

    with pytest.raises(ContractValidationError, match="char_count"):
        ContractValidator().validate_mapping(payload)


def test_chunk_index_must_be_less_than_chunk_total() -> None:
    payload = fixture_payload()
    payload["chunk_index"] = 1
    payload["chunk_total"] = 1

    with pytest.raises(ContractValidationError, match="chunk_index"):
        ContractValidator().validate_mapping(payload)


def test_employment_length_limit_is_rejected() -> None:
    payload = update_text(fixture_payload("employment"), "任" * 901)

    with pytest.raises(ContractValidationError, match="employment"):
        ContractValidator().validate_mapping(payload)


def test_project_length_limit_is_rejected() -> None:
    payload = update_text(fixture_payload("project"), "项" * 1001)

    with pytest.raises(ContractValidationError, match="chunk_text"):
        ContractValidator().validate_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("is_active", False),
        ("verification_status", "partial"),
        ("verification_status", "conflict"),
        ("verification_status", "rejected"),
        ("weight_tier", "C"),
    ],
)
def test_ineligible_records_are_skipped(field: str, value: object) -> None:
    record = fixture_records()[0]
    if field == "is_active":
        candidate = record.model_copy(update={field: value})
    else:
        candidate = record.model_copy(
            update={"metadata": record.metadata.model_copy(update={field: value})}
        )

    assert EmbeddingEligibilityValidator().is_eligible(candidate) is False
    model = FakeContractEmbeddingModel()
    result = ContractEmbeddingWorker(model, clock=fixed_clock).embed([candidate])
    assert result.embedded_chunks == []
    assert result.skipped_count == 1
    assert model.calls == []


def test_embedding_dimension_mismatch_is_rejected() -> None:
    worker = ContractEmbeddingWorker(
        FakeContractEmbeddingModel("dimension_mismatch"),
        clock=fixed_clock,
    )

    with pytest.raises(ValueError, match="dimension 不匹配"):
        worker.embed([fixture_records()[0]])


@pytest.mark.parametrize("mode", ["nan", "infinity"])
def test_non_finite_embedding_is_rejected(mode: str) -> None:
    worker = ContractEmbeddingWorker(
        FakeContractEmbeddingModel(mode),
        clock=fixed_clock,
    )

    with pytest.raises(ValueError, match="NaN 或 Infinity"):
        worker.embed([fixture_records()[0]])


def test_vector_doc_id_is_preserved_across_embedding_model_versions() -> None:
    records = fixture_records()
    first_model = FakeContractEmbeddingModel()
    first_worker = ContractEmbeddingWorker(first_model, clock=fixed_clock)
    second_model = FakeContractEmbeddingModel()
    second_model.model_version = "v2"
    second_worker = ContractEmbeddingWorker(second_model, clock=fixed_clock)

    first = first_worker.embed(records).embedded_chunks
    second = second_worker.embed(records).embedded_chunks

    assert [record.vector_doc_id for record in first] == [
        record.vector_doc_id for record in records
    ]
    assert all(
        embedded.vector_doc_id == source.vector_doc_id
        for source, embedded in zip(records, second)
    )
    assert {record.embedding_model_version for record in first} == {"v1"}
    assert {record.embedding_model_version for record in second} == {"v2"}


def test_metadata_source_versions_are_preserved_and_only_text_is_embedded() -> None:
    records = fixture_records()
    model = FakeContractEmbeddingModel()

    output = ContractEmbeddingWorker(model, clock=fixed_clock).embed(records)

    assert model.calls == [[record.chunk_text for record in records]]
    assert len(output.embedded_chunks) == 3
    for source, embedded in zip(records, output.embedded_chunks):
        assert embedded.metadata.model_dump() == source.metadata.model_dump()
        assert embedded.source.model_dump() == source.source.model_dump()
        assert embedded.versions.model_dump() == source.versions.model_dump()
        assert embedded.content_hash == source.content_hash
        assert embedded.vector_doc_id == source.vector_doc_id
        assert embedded.embedding_dim == 1024


def test_schema_gap_detects_missing_nested_contract_fields() -> None:
    difference = compare_contract_output_schema(
        {
            "fields": [
                {"name": "vector_doc_id", "type": "VARCHAR", "params": {"max_length": 256}},
                {"name": "chunk_id", "type": "VARCHAR", "params": {"max_length": 256}},
                {"name": "expert_id", "type": "INT64", "params": {}},
                {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": 1024}},
            ]
        }
    )

    assert difference["compatible"] is False
    assert "metadata_ext" in difference["missing_fields"]
    assert "source_ext" in difference["missing_fields"]
    assert "versions_ext" in difference["missing_fields"]
    assert "vector_doc_id" in difference["incompatible_fields"]
