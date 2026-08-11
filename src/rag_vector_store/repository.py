"""Milvus repository 实现。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .chunking.contracts import EmbeddedPreparedChunk
from .config import MilvusSettings, SchemaMode
from .models import EmbeddedChunk, MutationResult
from .ports import RepositoryRecord


@dataclass(frozen=True)
class _ExpertFieldSpec:
    """V1.2 expert schema 单个字段的期望定义。"""

    name: str
    datatype: str
    nullable: bool = False
    primary_key: bool = False
    max_length: int | None = None
    dim: int | None = None

    def summary(self) -> dict[str, object]:
        """返回可用于展示和兼容性比较的稳定结构。"""

        return {
            "field": self.name,
            "datatype": self.datatype,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
            "max_length": self.max_length,
            "dim": self.dim,
        }


class MilvusVectorRepository:
    """封装 Milvus 连接、schema 管理和 CRUD 写操作。

    SDK client 会在第一次操作时延迟创建。传入 ``client`` 适合测试，或适合由
    应用自行管理 PyMilvus client 生命周期的场景。
    """

    def __init__(
        self,
        settings: MilvusSettings,
        vector_dimension: int | None = None,
        *,
        client: Any | None = None,
        schema_mode: SchemaMode | None = None,
        allow_collection_creation: bool = True,
    ) -> None:
        self.settings = settings
        self.schema_mode = settings.schema_mode if schema_mode is None else schema_mode
        if self.schema_mode not in ("legacy", "expert"):
            raise ValueError("schema_mode 必须是 legacy 或 expert")
        self._vector_dimension: int | None = None
        self._client = client
        self._initialized = False
        self._owns_client = client is None
        self._allow_collection_creation = allow_collection_creation
        self._lock = Lock()

        if vector_dimension is not None:
            self._set_vector_dimension(vector_dimension)

    @property
    def client(self) -> Any:
        """暴露延迟创建的 SDK client，供高级集成场景使用。"""

        return self._get_client()

    def expert_schema_summary(
        self,
        vector_dimension: int | None = None,
    ) -> list[dict[str, object]]:
        """返回 V1.2 expert schema 摘要，不连接或修改 Milvus。"""

        if self.schema_mode != "expert":
            raise ValueError("expert_schema_summary 只适用于 expert schema_mode")
        dimension = self._resolve_vector_dimension(vector_dimension)
        return [spec.summary() for spec in self._expert_field_specs(dimension)]

    def compare_expert_schema(
        self,
        description: Mapping[str, Any],
        vector_dimension: int | None = None,
    ) -> dict[str, object]:
        """将服务端 Collection Schema 与当前 V1.2 期望定义进行只读比较。"""

        if self.schema_mode != "expert":
            raise ValueError("compare_expert_schema 只适用于 expert schema_mode")
        dimension = self._resolve_vector_dimension(vector_dimension)
        expected_specs = {
            spec.name: spec for spec in self._expert_field_specs(dimension)
        }
        actual_fields = self._field_map(description)

        missing_fields = sorted(set(expected_specs) - set(actual_fields))
        unexpected_fields = sorted(set(actual_fields) - set(expected_specs))
        changed_fields: dict[str, dict[str, object]] = {}

        for field_name in sorted(set(expected_specs).intersection(actual_fields)):
            expected = expected_specs[field_name].summary()
            actual = _field_summary(actual_fields[field_name])
            if actual != expected:
                changed_fields[field_name] = {
                    "expected": expected,
                    "actual": actual,
                }

        collection_options: dict[str, dict[str, object]] = {}
        actual_auto_id = bool(description.get("auto_id", False))
        if actual_auto_id:
            collection_options["auto_id"] = {
                "expected": False,
                "actual": actual_auto_id,
            }
        actual_dynamic = bool(description.get("enable_dynamic_field", False))
        if actual_dynamic:
            collection_options["enable_dynamic_field"] = {
                "expected": False,
                "actual": actual_dynamic,
            }

        compatible = not (
            missing_fields
            or unexpected_fields
            or changed_fields
            or collection_options
        )
        return {
            "compatible": compatible,
            "missing_fields": missing_fields,
            "unexpected_fields": unexpected_fields,
            "changed_fields": changed_fields,
            "collection_options": collection_options,
        }

    def expected_index_summary(self) -> dict[str, object]:
        """返回当前配置要求的向量索引定义。"""

        field_name = (
            "embedding" if self.schema_mode == "expert" else self.settings.embedding_field
        )
        return {
            "field_name": field_name,
            "index_type": self.settings.index_type.upper(),
            "metric_type": self.settings.metric_type.upper(),
            "params": dict(self.settings.index_params),
        }

    def compare_expert_index(self, client: Any | None = None) -> dict[str, object]:
        """只读比较现有 expert 向量索引与当前配置。"""

        if self.schema_mode != "expert":
            raise ValueError("compare_expert_index 只适用于 expert schema_mode")
        active_client = self._get_client() if client is None else client
        collection_name = self.settings.collection_name
        index_names = list(
            active_client.list_indexes(
                collection_name=collection_name,
                field_name="embedding",
            )
        )
        expected = self.expected_index_summary()
        if not index_names:
            return {
                "compatible": False,
                "expected": expected,
                "actual": None,
                "reason": "embedding 字段缺少向量索引",
            }

        descriptions = [
            active_client.describe_index(
                collection_name=collection_name,
                index_name=index_name,
            )
            for index_name in index_names
        ]
        matching = next(
            (
                description
                for description in descriptions
                if description.get("field_name") == "embedding"
            ),
            descriptions[0],
        )
        actual = _index_summary(matching, expected["params"])
        return {
            "compatible": actual == expected,
            "expected": expected,
            "actual": actual,
            "reason": None if actual == expected else "向量索引配置不一致",
        }

    def initialize(self, vector_dimension: int) -> None:
        """建立连接，并创建或校验集合 schema。"""

        self._set_vector_dimension(vector_dimension)
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            client = self._get_client()
            collection_name = self.settings.collection_name
            if client.has_collection(collection_name=collection_name):
                description = client.describe_collection(collection_name=collection_name)
                self._validate_existing_collection(description)
                if self.schema_mode == "expert":
                    self._validate_existing_expert_index(client)
            else:
                if not self._allow_collection_creation:
                    raise RuntimeError(
                        f"Milvus collection {collection_name!r} 不存在，当前 repository "
                        "禁止自动创建"
                    )
                schema = self._build_schema(self._vector_dimension)
                index_params = self._build_index_params(client)
                client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    index_params=index_params,
                )
            self._initialized = True

    def insert_batch(self, records: Sequence[RepositoryRecord]) -> MutationResult:
        """插入一批已经完成向量化的 chunk。"""

        items = list(records)
        if not items:
            return MutationResult(affected_count=0)

        self._ensure_dimension_from_records(items)
        dimension = self._vector_dimension
        if dimension is None:  # 理论上不会发生，防止未来修改破坏维度约束
            raise ValueError("vector dimension is not configured")
        self.initialize(dimension)
        rows = self._to_rows(items)
        response = self._get_client().insert(
            collection_name=self.settings.collection_name,
            data=rows,
            **self._timeout_kwargs(),
        )
        return _mutation_result(
            response,
            "insert_count",
            len(items),
            self._primary_keys(items),
        )

    def upsert_batch(self, records: Sequence[RepositoryRecord]) -> MutationResult:
        """按当前 schema 的主键 upsert 一批记录。"""

        items = list(records)
        if not items:
            return MutationResult(affected_count=0)

        self._ensure_dimension_from_records(items)
        dimension = self._vector_dimension
        if dimension is None:  # 理论上不会发生，防止未来修改破坏维度约束
            raise ValueError("vector dimension is not configured")
        self.initialize(dimension)
        rows = self._to_rows(items)
        response = self._get_client().upsert(
            collection_name=self.settings.collection_name,
            data=rows,
            **self._timeout_kwargs(),
        )
        return _mutation_result(
            response,
            "upsert_count",
            len(items),
            self._primary_keys(items),
        )

    def get_by_vector_doc_id(
        self,
        vector_doc_id: str,
        *,
        include_embedding: bool = True,
    ) -> dict[str, Any] | None:
        """按 expert schema 主键回查单条 Entity，不执行向量检索。"""

        if self.schema_mode != "expert":
            raise ValueError("get_by_vector_doc_id 只适用于 expert schema_mode")
        if not isinstance(vector_doc_id, str) or not vector_doc_id.strip():
            raise ValueError("vector_doc_id 必须是非空字符串")

        dimension = self._resolve_vector_dimension(None)
        output_fields = [spec.name for spec in self._expert_field_specs(dimension)]
        if not include_embedding:
            output_fields.remove("embedding")

        rows = self._get_client().query(
            collection_name=self.settings.collection_name,
            ids=[vector_doc_id],
            output_fields=output_fields,
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        if len(rows) > 1:
            raise RuntimeError("同一个 vector_doc_id 回查到多条 Entity")
        if not rows:
            return None
        if not isinstance(rows[0], Mapping):
            raise TypeError("Milvus query 返回了非法 Entity 结构")
        return dict(rows[0])

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> MutationResult:
        """根据当前 schema 的主键删除记录，为后续清理流程保留扩展点。"""

        ids = list(chunk_ids)
        if not ids:
            return MutationResult(affected_count=0)
        if any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in ids):
            raise ValueError("chunk_ids must contain non-empty strings")

        if self._vector_dimension is None:
            raise ValueError(
                "delete_by_ids requires vector_dimension when the repository is first created"
            )
        self.initialize(self._vector_dimension)
        response = self._get_client().delete(
            collection_name=self.settings.collection_name,
            ids=ids,
            **self._timeout_kwargs(),
        )
        return _mutation_result(response, "delete_cnt", len(ids), ids, alternate_key="delete_count")

    def close(self) -> None:
        """关闭由当前 repository 创建的 PyMilvus client。"""

        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None
            self._initialized = False

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:  # pragma: no cover - 依赖已在 pyproject 中声明
                raise RuntimeError(
                    "pymilvus is required to use MilvusVectorRepository; run `uv sync` first"
                ) from exc
            self._client = MilvusClient(**self.settings.client_kwargs)
        return self._client

    def _set_vector_dimension(self, vector_dimension: int) -> None:
        if isinstance(vector_dimension, bool) or not isinstance(vector_dimension, int):
            # 保持原有异常类型，避免改变运行行为。
            raise ValueError(  # noqa: TRY004
                "vector_dimension must be a positive integer"
            )
        if vector_dimension < 1:
            raise ValueError("vector_dimension must be a positive integer")
        if self._vector_dimension is not None and self._vector_dimension != vector_dimension:
            raise ValueError(
                f"vector dimension is already configured as {self._vector_dimension}, "
                f"not {vector_dimension}"
            )
        self._vector_dimension = vector_dimension

    def _resolve_vector_dimension(self, vector_dimension: int | None) -> int:
        """解析期望向量维度，并复用 repository 的一致性约束。"""

        if vector_dimension is not None:
            self._set_vector_dimension(vector_dimension)
        if self._vector_dimension is None:
            raise ValueError("vector dimension is not configured")
        return self._vector_dimension

    def _ensure_dimension_from_records(self, records: Sequence[RepositoryRecord]) -> None:
        self._set_vector_dimension(len(records[0].embedding))
        for record in records:
            if len(record.embedding) != self._vector_dimension:
                raise ValueError(
                    f"embedding dimension mismatch for chunk {record.chunk_id!r}: "
                    f"expected {self._vector_dimension}, got {len(record.embedding)}"
                )

    def _build_schema(self, vector_dimension: int | None) -> Any:
        """根据 schema_mode 构造 legacy 或 expert schema。"""

        if self.schema_mode == "expert":
            return self._build_expert_schema(vector_dimension)
        return self._build_legacy_schema(vector_dimension)

    def _build_legacy_schema(self, vector_dimension: int | None) -> Any:
        if vector_dimension is None:  # 防御性检查，避免未来调用方遗漏向量维度
            raise ValueError("vector dimension is not configured")
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:  # pragma: no cover - 依赖已在 pyproject 中声明
            raise RuntimeError(
                "pymilvus is required to create a Milvus collection; run `uv sync` first"
            ) from exc

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name=self.settings.chunk_id_field,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self.settings.chunk_id_max_length,
        )
        schema.add_field(
            field_name=self.settings.text_field,
            datatype=DataType.VARCHAR,
            max_length=self.settings.text_max_length,
        )
        schema.add_field(
            field_name=self.settings.embedding_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=vector_dimension,
        )
        schema.add_field(
            field_name=self.settings.metadata_field,
            datatype=DataType.JSON,
        )
        return schema

    def _build_expert_schema(self, vector_dimension: int | None) -> Any:
        """构造 V1.2 专家索引 schema，不修改任何已有 Collection。"""

        if vector_dimension is None:  # 防御性检查，避免未来调用方遗漏向量维度
            raise ValueError("vector dimension is not configured")
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:  # pragma: no cover - 依赖已在 pyproject 中声明
            raise RuntimeError(
                "pymilvus is required to create a Milvus collection; run `uv sync` first"
            ) from exc

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        for spec in self._expert_field_specs(vector_dimension):
            field_kwargs: dict[str, Any] = {
                "field_name": spec.name,
                "datatype": getattr(DataType, spec.datatype),
            }
            if spec.primary_key:
                field_kwargs["is_primary"] = True
            if spec.nullable:
                field_kwargs["nullable"] = True
            if spec.max_length is not None:
                field_kwargs["max_length"] = spec.max_length
            if spec.dim is not None:
                field_kwargs["dim"] = spec.dim
            schema.add_field(**field_kwargs)
        return schema

    def _expert_field_specs(self, vector_dimension: int) -> tuple[_ExpertFieldSpec, ...]:
        """根据 PreparedChunk Python 类型生成完整 V1.2 字段定义。"""

        return (
            _ExpertFieldSpec(
                "vector_doc_id",
                "VARCHAR",
                primary_key=True,
                max_length=self.settings.expert_vector_doc_id_max_length,
            ),
            _ExpertFieldSpec(
                "chunk_id",
                "VARCHAR",
                max_length=self.settings.expert_chunk_id_max_length,
            ),
            _ExpertFieldSpec("expert_id", "INT64"),
            _ExpertFieldSpec("chunk_type", "VARCHAR", max_length=64),
            _ExpertFieldSpec("chunk_subtype", "VARCHAR", nullable=True, max_length=128),
            _ExpertFieldSpec(
                "chunk_text",
                "VARCHAR",
                max_length=self.settings.expert_chunk_text_max_length,
            ),
            _ExpertFieldSpec("parent_record_id", "VARCHAR", nullable=True, max_length=256),
            _ExpertFieldSpec("chunk_index", "INT64"),
            _ExpertFieldSpec("chunk_total", "INT64"),
            _ExpertFieldSpec("source_table", "VARCHAR", max_length=128),
            _ExpertFieldSpec("source_record_id", "VARCHAR", max_length=256),
            _ExpertFieldSpec("source_field", "VARCHAR", max_length=256),
            _ExpertFieldSpec("source_item_key", "VARCHAR", nullable=True, max_length=256),
            _ExpertFieldSpec("source_version", "INT64"),
            _ExpertFieldSpec("source_hash", "VARCHAR", max_length=128),
            _ExpertFieldSpec("content_hash", "VARCHAR", max_length=128),
            _ExpertFieldSpec("source_updated_at", "INT64", nullable=True),
            _ExpertFieldSpec(
                "company_name_norm",
                "VARCHAR",
                nullable=True,
                max_length=512,
            ),
            _ExpertFieldSpec("role_code", "VARCHAR", nullable=True, max_length=128),
            _ExpertFieldSpec("start_month", "INT64", nullable=True),
            _ExpertFieldSpec("end_month", "INT64", nullable=True),
            _ExpertFieldSpec("is_current", "BOOL", nullable=True),
            _ExpertFieldSpec("origin_type", "VARCHAR", nullable=True, max_length=64),
            _ExpertFieldSpec(
                "verification_status",
                "VARCHAR",
                nullable=True,
                max_length=64,
            ),
            _ExpertFieldSpec("weight_tier", "VARCHAR", nullable=True, max_length=32),
            _ExpertFieldSpec("embedding_model_alias", "VARCHAR", max_length=128),
            _ExpertFieldSpec("embedding_model_version", "VARCHAR", max_length=128),
            _ExpertFieldSpec("chunk_policy_version", "VARCHAR", max_length=64),
            _ExpertFieldSpec("metadata_schema_version", "VARCHAR", max_length=64),
            _ExpertFieldSpec("eligible_for_recommendation", "BOOL"),
            _ExpertFieldSpec("is_active", "BOOL"),
            _ExpertFieldSpec("metadata_ext", "JSON"),
            _ExpertFieldSpec("embedding", "FLOAT_VECTOR", dim=vector_dimension),
        )

    def _build_index_params(self, client: Any) -> Any:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding" if self.schema_mode == "expert" else self.settings.embedding_field,
            index_type=self.settings.index_type,
            metric_type=self.settings.metric_type,
            params=dict(self.settings.index_params),
        )
        return index_params

    def _validate_existing_collection(self, description: Mapping[str, Any]) -> None:
        """根据 schema_mode 校验已有集合，绝不自动迁移已有 schema。"""

        if self.schema_mode == "expert":
            self._validate_existing_expert_collection(description)
            return
        self._validate_existing_legacy_collection(description)

    def _validate_existing_legacy_collection(self, description: Mapping[str, Any]) -> None:
        if description.get("auto_id") is True:
            raise ValueError(
                f"Milvus collection {self.settings.collection_name!r} uses auto-generated IDs; "
                "it must be created with auto_id=False"
            )

        fields = {
            field.get("name"): field
            for field in description.get("fields", [])
            if isinstance(field, Mapping) and field.get("name")
        }
        required_fields = {
            self.settings.chunk_id_field,
            self.settings.text_field,
            self.settings.embedding_field,
            self.settings.metadata_field,
        }
        missing = sorted(required_fields - set(fields))
        if missing:
            raise ValueError(
                f"Milvus collection {self.settings.collection_name!r} is missing fields: {missing}"
            )

        primary_field = fields[self.settings.chunk_id_field]
        if not primary_field.get("is_primary", False):
            raise ValueError(
                f"Milvus field {self.settings.chunk_id_field!r} must be the primary key"
            )

        vector_field = fields[self.settings.embedding_field]
        vector_params = vector_field.get("params") or {}
        existing_dimension = vector_params.get("dim")
        if existing_dimension != self._vector_dimension:
            raise ValueError(
                f"Milvus collection vector dimension is {existing_dimension}, "
                f"but the embedding model returns {self._vector_dimension}"
            )

    def _validate_existing_expert_collection(self, description: Mapping[str, Any]) -> None:
        difference = self.compare_expert_schema(description)
        if not difference["compatible"]:
            raise ValueError(
                f"Milvus expert collection {self.settings.collection_name!r} "
                "与 V1.2 schema 不兼容: "
                f"{json.dumps(difference, ensure_ascii=False, sort_keys=True)}"
            )

    def _validate_existing_expert_index(self, client: Any) -> None:
        difference = self.compare_expert_index(client)
        if not difference["compatible"]:
            raise ValueError(
                f"Milvus expert collection {self.settings.collection_name!r} "
                "的向量索引不兼容: "
                f"{json.dumps(difference, ensure_ascii=False, sort_keys=True)}"
            )

    @staticmethod
    def _field_map(description: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        return {
            field.get("name"): field
            for field in description.get("fields", [])
            if isinstance(field, Mapping) and field.get("name")
        }

    def _to_rows(self, records: Sequence[RepositoryRecord]) -> list[dict[str, Any]]:
        if self.schema_mode == "expert":
            return self._to_expert_rows(self._as_expert_records(records))
        return self._to_legacy_rows(self._as_legacy_records(records))

    @staticmethod
    def _as_legacy_records(records: Sequence[RepositoryRecord]) -> list[EmbeddedChunk]:
        legacy_records = [record for record in records if isinstance(record, EmbeddedChunk)]
        if len(legacy_records) != len(records):
            raise TypeError("legacy schema_mode 只接受 EmbeddedChunk")
        return legacy_records

    @staticmethod
    def _as_expert_records(
        records: Sequence[RepositoryRecord],
    ) -> list[EmbeddedPreparedChunk]:
        expert_records = [
            record for record in records if isinstance(record, EmbeddedPreparedChunk)
        ]
        if len(expert_records) != len(records):
            raise TypeError("expert schema_mode 只接受 EmbeddedPreparedChunk")
        return expert_records

    def _to_legacy_rows(self, records: Sequence[EmbeddedChunk]) -> list[dict[str, Any]]:
        self._validate_batch_ids(records)
        rows: list[dict[str, Any]] = []
        for record in records:
            if len(record.chunk_id.encode("utf-8")) > self.settings.chunk_id_max_length:
                raise ValueError(
                    f"chunk_id {record.chunk_id!r} exceeds the configured UTF-8 byte limit "
                    f"({self.settings.chunk_id_max_length})"
                )
            if len(record.text.encode("utf-8")) > self.settings.text_max_length:
                raise ValueError(
                    f"text for chunk {record.chunk_id!r} exceeds the configured UTF-8 byte limit "
                    f"({self.settings.text_max_length})"
                )
            rows.append(
                {
                    self.settings.chunk_id_field: record.chunk_id,
                    self.settings.text_field: record.text,
                    self.settings.embedding_field: record.embedding,
                    self.settings.metadata_field: record.metadata,
                }
            )
        return rows

    def _to_expert_rows(
        self,
        records: Sequence[EmbeddedPreparedChunk],
    ) -> list[dict[str, Any]]:
        self._validate_batch_ids(records)
        rows: list[dict[str, Any]] = []
        for record in records:
            self._validate_expert_record_lengths(record)
            rows.append(
                {
                    "vector_doc_id": record.vector_doc_id,
                    "expert_id": record.expert_id,
                    "chunk_id": record.chunk_id,
                    "chunk_type": record.chunk_type,
                    "chunk_subtype": record.chunk_subtype,
                    "parent_record_id": record.parent_record_id,
                    "chunk_index": record.chunk_index,
                    "chunk_total": record.chunk_total,
                    "chunk_text": record.chunk_text,
                    "source_table": record.source_table,
                    "source_record_id": record.source_record_id,
                    "source_field": record.source_field,
                    "source_item_key": record.source_item_key,
                    "source_version": record.source_version,
                    "source_hash": record.source_hash,
                    "content_hash": record.content_hash,
                    "source_updated_at": record.source_updated_at,
                    "company_name_norm": record.company_name_norm,
                    "role_code": record.role_code,
                    "start_month": record.start_month,
                    "end_month": record.end_month,
                    "is_current": record.is_current,
                    "origin_type": record.origin_type,
                    "verification_status": record.verification_status,
                    "weight_tier": record.weight_tier,
                    "embedding_model_alias": record.embedding_model_alias,
                    "embedding_model_version": record.embedding_model_version,
                    "chunk_policy_version": record.chunk_policy_version,
                    "metadata_schema_version": record.metadata_schema_version,
                    "eligible_for_recommendation": record.eligible_for_recommendation,
                    "is_active": record.is_active,
                    "embedding": record.embedding,
                    "metadata_ext": record.metadata_ext,
                }
            )
        return rows

    def _validate_expert_record_lengths(self, record: EmbeddedPreparedChunk) -> None:
        dimension = self._resolve_vector_dimension(None)
        max_lengths = {
            spec.name: spec.max_length
            for spec in self._expert_field_specs(dimension)
            if spec.max_length is not None
        }
        values = {
            "vector_doc_id": record.vector_doc_id,
            "chunk_id": record.chunk_id,
            "chunk_type": record.chunk_type,
            "chunk_subtype": record.chunk_subtype,
            "parent_record_id": record.parent_record_id,
            "chunk_text": record.chunk_text,
            "source_table": record.source_table,
            "source_record_id": record.source_record_id,
            "source_field": record.source_field,
            "source_item_key": record.source_item_key,
            "source_hash": record.source_hash,
            "content_hash": record.content_hash,
            "company_name_norm": record.company_name_norm,
            "role_code": record.role_code,
            "origin_type": record.origin_type,
            "verification_status": record.verification_status,
            "weight_tier": record.weight_tier,
            "embedding_model_alias": record.embedding_model_alias,
            "embedding_model_version": record.embedding_model_version,
            "chunk_policy_version": record.chunk_policy_version,
            "metadata_schema_version": record.metadata_schema_version,
        }
        for field_name, value in values.items():
            if value is not None and len(value.encode("utf-8")) > max_lengths[field_name]:
                raise ValueError(
                    f"{field_name} 超过 UTF-8 字节长度限制 "
                    f"({max_lengths[field_name]})"
                )

    @staticmethod
    def _validate_batch_ids(records: Sequence[RepositoryRecord]) -> None:
        ids = [
            record.vector_doc_id
            if isinstance(record, EmbeddedPreparedChunk)
            else record.chunk_id
            for record in records
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("一个 mutation batch 不能包含重复的主键")

    def _primary_keys(self, records: Sequence[RepositoryRecord]) -> list[str]:
        if self.schema_mode == "expert":
            return [record.vector_doc_id for record in self._as_expert_records(records)]
        return [record.chunk_id for record in self._as_legacy_records(records)]

    def _timeout_kwargs(self) -> dict[str, Any]:
        return {} if self.settings.timeout is None else {"timeout": self.settings.timeout}


def _field_summary(field: Mapping[str, Any]) -> dict[str, object]:
    """将 PyMilvus describe_collection 字段归一化为稳定结构。"""

    params = field.get("params")
    normalized_params = params if isinstance(params, Mapping) else {}
    datatype = _datatype_name(field.get("type"))
    return {
        "field": str(field.get("name")),
        "datatype": datatype,
        "nullable": bool(field.get("nullable", False)),
        "primary_key": bool(field.get("is_primary", False)),
        "max_length": (
            _optional_int(normalized_params.get("max_length"))
            if datatype == "VARCHAR"
            else None
        ),
        "dim": (
            _optional_int(normalized_params.get("dim"))
            if datatype == "FLOAT_VECTOR"
            else None
        ),
    }


def _datatype_name(value: Any) -> str:
    """兼容 DataType 枚举、整数和字符串形式的字段类型。"""

    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            from pymilvus import DataType

            return DataType(value).name
        except (ImportError, ValueError):
            return str(value)
    return str(value).rsplit(".", maxsplit=1)[-1].upper()


def _index_summary(
    description: Mapping[str, Any],
    expected_params: object,
) -> dict[str, object]:
    """归一化 describe_index 返回值，并仅比较已配置参数。"""

    nested_params = description.get("params")
    param_source: dict[str, Any] = (
        dict(nested_params) if isinstance(nested_params, Mapping) else {}
    )
    param_source.update(description)
    configured_params = expected_params if isinstance(expected_params, Mapping) else {}
    actual_params = {
        str(name): _normalized_index_param(param_source.get(name))
        for name in configured_params
    }
    return {
        "field_name": str(description.get("field_name", "")),
        "index_type": str(description.get("index_type", "")).upper(),
        "metric_type": str(description.get("metric_type", "")).upper(),
        "params": actual_params,
    }


def _normalized_index_param(value: Any) -> object:
    """将服务端字符串数字转换为整数，便于与配置比较。"""

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _optional_int(value: Any) -> int | None:
    """将可选的 Schema 数字参数转换为整数。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mutation_result(
    response: Any,
    primary_key: str,
    fallback_count: int,
    fallback_ids: Sequence[str],
    *,
    alternate_key: str | None = None,
) -> MutationResult:
    if isinstance(response, Mapping):
        count = response.get(primary_key)
        if count is None and alternate_key is not None:
            count = response.get(alternate_key)
        if count is None:
            count = fallback_count
        ids = response.get("ids") or response.get("primary_keys") or fallback_ids
    else:
        count = fallback_count
        ids = fallback_ids
    return MutationResult(
        affected_count=int(count),
        primary_keys=[str(item) for item in ids],
    )
