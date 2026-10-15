"""正式外部分词的端到端增量回归；使用真实组件和 Fake I/O。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from expert_contracts import adapt_expert_chunk_mapping
from expert_index.chunk_builder import ChunkFactory
from rag_vector_store import ContractEmbeddingWorker, MilvusSettings
from rag_vector_store.ingestion.factory import (
    create_resume_vector_ingestion_service,
    create_vector_ingestion_service,
)
from rag_vector_store.milvus.indexes import index_specs_v16
from rag_vector_store.milvus.repository_v16_external import (
    ContractMilvusV16ExternalVectorRepository,
)
from rag_vector_store.tokenization.jieba_index import JiebaIndexConfig, JiebaIndexTokenizer
from tests import test_incremental_handoff_fake_e2e as e2e


class Model:
    dimension = 1024
    model_alias = "alibaba-bailian:text-embedding-v4"
    model_version = "v1"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[0.125] * 1024 for _ in texts]


class Iterator:
    def __init__(self, rows):
        self.rows, self.done, self.closed = rows, False, False

    def next(self):
        if self.done:
            return []
        self.done = True
        return self.rows

    def close(self):
        self.closed = True


class Client:
    def __init__(self):
        self.rows, self.writes, self.reads, self.iterators = {}, [], [], []
        self.schema = None
        self.fail_write = False

    def has_collection(self, **kwargs):
        self.reads.append(kwargs)
        return self.schema is not None

    def describe_collection(self, **kwargs):
        self.reads.append(kwargs)
        return self.schema

    def list_indexes(self, **kwargs):
        self.reads.append(kwargs)
        return [s.index_name for s in index_specs_v16(MilvusSettings())]

    def describe_index(self, index_name, **kwargs):
        self.reads.append(kwargs)
        return next(s.summary(include_name=True) for s in index_specs_v16(MilvusSettings())
                    if s.index_name == index_name)

    def get_collection_stats(self, **kwargs):
        self.reads.append(kwargs)
        return {"row_count": len(self.rows)}

    def query(self, collection_name, output_fields, ids=None, **kwargs):
        self.reads.append({"collection_name": collection_name, **kwargs})
        selected = ids if ids is not None else list(self.rows)
        return [{k: copy.deepcopy(v) for k, v in self.rows[key].items() if k in output_fields}
                for key in selected if key in self.rows]

    def query_iterator(self, collection_name, output_fields, **kwargs):
        rows = self.query(collection_name, output_fields, **kwargs)
        expression = kwargs.get("filter", "")
        if expression.startswith("expert_id == "):
            owner = int(expression.split(" == ")[1])
            rows = [r for r in rows if r["expert_id"] == owner]
        iterator = Iterator(rows)
        self.iterators.append(iterator)
        return iterator

    def upsert(self, collection_name, data, **kwargs):
        if self.fail_write:
            raise TimeoutError("synthetic timeout")
        self.writes.append({"collection_name": collection_name, "data": copy.deepcopy(data),
                            **kwargs})
        self.rows.update({r["vector_doc_id"]: copy.deepcopy(r) for r in data})
        return {"upsert_count": len(data), "ids": [r["vector_doc_id"] for r in data]}


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "dictionary.txt"
    path.write_text("新能源汽车\n工程师\n", encoding="utf-8")
    return JiebaIndexConfig(path, "fixture-v1", hashlib.sha256(path.read_bytes()).hexdigest(), True)


def assemble(config, timeout=None):
    settings = MilvusSettings(
        expert_storage_profile="v16_external_formal", collection_name=config.formal_collection_name,
        expert_v16_dictionary_path=str(config.dictionary_path),
        expert_v16_dictionary_version=config.dictionary_version,
        expert_v16_dictionary_sha256=config.dictionary_sha256,
        expert_v16_external_hmm=config.hmm, timeout=timeout,
    )
    client, model = Client(), Model()
    repo = ContractMilvusV16ExternalVectorRepository(settings, 1024, client=client)
    client.schema = repo._build_schema(1024).to_dict()
    service = create_vector_ingestion_service(repository=repo,
                                              embedding_worker=ContractEmbeddingWorker(model))
    return service, client, model


@pytest.fixture
def inputs():
    path = Path(__file__).resolve().parents[1] / "fixtures/expert_chunk_embedding_input_v4.jsonl"
    return [adapt_expert_chunk_mapping(json.loads(line)) for line in path.read_text().splitlines()
            if line]


@pytest.mark.parametrize("hmm", [True, False])
def test_formal_ingestion_metadata_and_disable_preserve_dense(config, inputs, hmm):
    service, client, model = assemble(replace(config, hmm=hmm))
    service.upsert_chunks(inputs)
    key = inputs[0].vector_doc_id
    old = copy.deepcopy(client.rows[key])
    assert old["chunk_text_bm25"] == service.repository.tokenizer.encode(old["chunk_text"])
    assert "sparse_embedding" not in old
    calls = len(model.calls)
    changed = inputs[0].model_copy(update={"metadata": inputs[0].metadata.model_copy(
        update={"metadata_hash": "b" * 64})})
    service.update_metadata([changed])
    service.disable_vectors([key])
    new = client.rows[key]
    assert len(model.calls) == calls
    assert new["dense_embedding"] == old["dense_embedding"]
    assert new["metadata"]["embedded_at"] == old["metadata"]["embedded_at"]
    assert new["chunk_text_bm25"] == old["chunk_text_bm25"]
    assert new["is_active"] is False and new["metadata"]["is_active"] is False
    assert all(w["collection_name"] == service.repository.settings.collection_name
               for w in client.writes)


@pytest.mark.parametrize("mutation", ["description", "analyzer", "function", "capacity"])
def test_drift_refused_before_embedding_or_write(config, inputs, mutation):
    service, client, model = assemble(config)
    if mutation == "description":
        client.schema["description"] = "wrong fingerprint"
    elif mutation == "function":
        client.schema["functions"][0]["input_field_names"] = ["chunk_text"]
    else:
        field = next(f for f in client.schema["fields"] if f["name"] == "chunk_text_bm25")
        field["params"]["analyzer_params" if mutation == "analyzer" else "max_length"] = (
            "{}" if mutation == "analyzer" else 8192)
    with pytest.raises((ValueError, RuntimeError)):
        service.repository.initialize(1024)
    assert client.writes == [] and model.calls == []


def test_missing_collection_never_created(config):
    service, client, _ = assemble(config)
    client.schema = None
    with pytest.raises(RuntimeError, match="禁止自动创建"):
        service.repository.initialize(1024)
    assert not client.writes


def test_dictionary_identity_and_capacity(config):
    tokenizer = JiebaIndexTokenizer(config)
    assert tokenizer.encode("C++/C#研发").split()[:2] == ["C++", "C#"]
    with pytest.raises(ValueError, match="字节"):
        tokenizer.encode("中" * 12000)
    config.dictionary_path.write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        JiebaIndexTokenizer(config)


def test_resume_and_test_profile_cannot_bind_formal_collection(config):
    service, _, model = assemble(config)
    with pytest.raises(ValueError, match="Resume"):
        create_resume_vector_ingestion_service(milvus_settings=service.repository.settings,
                                              embedding_worker=ContractEmbeddingWorker(model))
    values = service.repository.settings.model_dump()
    with pytest.raises(ValueError):
        MilvusSettings.model_validate({**values, "expert_storage_profile": "v16_external"})
    with pytest.raises(ValueError):
        MilvusSettings.model_validate({**values, "expert_v16_external_hmm": False})


@pytest.mark.parametrize("timeout", [None, 120.0])
def test_timeout_reaches_runtime_reads_and_writes(config, inputs, timeout):
    service, client, _ = assemble(config, timeout)
    service.upsert_chunks(inputs[:1])
    assert client.reads and client.writes
    for call in [*client.reads, *client.writes]:
        assert call.get("timeout") == timeout
        assert ("timeout" in call) == (timeout is not None)


def make_task_run(service):
    task = e2e._task()
    task_repo, artifacts = e2e.FakeTaskRepo(state=task), e2e.FakeArtifactRepo()
    source = e2e.FakeSourceAdapter()
    source.set_facts("employment", "21307", e2e._employment_facts(700001, 21307))
    return task, task_repo, artifacts, source, ChunkFactory(e2e._settings())


def test_task_to_formal_repository_and_idempotent_recovery(config):
    service, client, model = assemble(config)
    task, tasks, artifacts, source, factory = make_task_run(service)
    result = e2e._run(task.task_id, tasks, artifacts, service, source, factory)
    assert result.final_status == "success" and client.rows
    assert len(model.calls) == 1
    for row in client.rows.values():
        assert row["chunk_text_bm25"] == service.repository.tokenizer.encode(row["chunk_text"])
    writes = len(client.writes)
    tasks.state = e2e._task(processing_stage="handed_off", attempt_token="recovered")
    result = e2e._build_orchestrator(service, tasks, artifacts).process(task.task_id, "recovered")
    assert result.final_status == "success"
    assert len(model.calls) == 1 and len(client.writes) == writes
    assert all(iterator.closed for iterator in client.iterators)


def test_failed_write_retries_without_success_receipt(config):
    service, client, _ = assemble(config)
    task, tasks, artifacts, source, factory = make_task_run(service)
    client.fail_write = True
    result = e2e._run(task.task_id, tasks, artifacts, service, source, factory)
    assert result.final_status == "pending" and not tasks.finalize_calls
    assert not client.rows
    client.fail_write = False
    tasks.state = e2e._task(processing_stage="handed_off", attempt_token="retry")
    result = e2e._build_orchestrator(service, tasks, artifacts).process(task.task_id, "retry")
    assert result.final_status == "success" and client.rows


@pytest.mark.parametrize("change", ["metadata", "content", "delete"])
def test_incremental_task_changes_use_correct_vector_action(config, change):
    service, client, model = assemble(config)
    task, tasks, artifacts, source, factory = make_task_run(service)
    first = e2e._run(task.task_id, tasks, artifacts, service, source, factory)
    assert first.final_status == "success"
    old = copy.deepcopy(next(iter(client.rows.values())))
    kwargs = {"updated_at": e2e.UPDATED2}
    if change == "content":
        kwargs["company"] = "虚构的新公司"
    elif change == "delete":
        kwargs["deleted_at"] = e2e.UPDATED2
    source.set_facts("employment", "21307", e2e._employment_facts(700001, 21307, **kwargs))
    tasks = e2e.FakeTaskRepo(state=e2e._task(task_id=702, batch_id=9))
    result = e2e._run(702, tasks, e2e.FakeArtifactRepo(), service, source, factory)
    assert result.final_status == "success"
    row = next(iter(client.rows.values()))
    assert len(model.calls) == (2 if change == "content" else 1)
    if change != "content":
        assert row["dense_embedding"] == old["dense_embedding"]
        assert row["metadata"]["embedded_at"] == old["metadata"]["embedded_at"]
    if change == "delete":
        assert row["is_active"] is False
    elif change == "metadata":
        assert row["metadata"]["metadata_hash"] != old["metadata"]["metadata_hash"]
    else:
        assert row["chunk_text"] != old["chunk_text"]
        assert row["chunk_text_bm25"] == service.repository.tokenizer.encode(row["chunk_text"])


def test_external_dictionary_matches_existing_collection_identity():
    configured = os.environ.get("EXPERT_V16_DICTIONARY_PATH")
    if not configured:
        pytest.skip("正式词典为外部资源；设置 EXPERT_V16_DICTIONARY_PATH 后校验身份")
    raw = Path(configured).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "90d90b995659e55de2c0aa5d9074857a5ff184ce0e63b548529a988ea3afcac2")
