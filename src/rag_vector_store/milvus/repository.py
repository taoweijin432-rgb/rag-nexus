"""正式 Contract Collection 的 Milvus Repository 实现。

Repository 是 Milvus Client 的唯一业务访问入口，负责连接生命周期、Collection
初始化/只读兼容性检查、CRUD、Query 和 Search 编排。Schema/Index 定义、兼容性
判断、Contract 行映射分别委托给 ``schemas``、``indexes``、``compatibility`` 和
``MilvusRowMapper``；本模块不负责 Contract 校验、Embedding 或 Task State。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from threading import Lock
from typing import Any

from expert_contracts.models import EmbeddedContractPreparedChunk

from ..config import MilvusSettings
from ..model_space import is_placeholder_model_version
from ..models import MutationResult, SearchHit
from .compatibility import (
    compare_contract_indexes as compare_contract_indexes_definition,
)
from .compatibility import (
    compare_model_space,
    compare_schema_specs,
)
from .contract import MilvusRowMapper
from .indexes import build_index_params, index_specs, vector_index_spec
from .schemas import (
    build_contract_schema,
    contract_milvus_field_specs,
)


class ContractMilvusVectorRepository:
    """封装 Milvus 连接、Schema 管理和 CRUD 写操作。

    SDK client 会在第一次操作时延迟创建。传入 ``client`` 适合测试，或适合由
    应用自行管理 PyMilvus client 生命周期的场景。所有正式写入都先确定向量维度，
    再通过 Schema/Index Compatibility Gate；已存在但不兼容的 Collection 会阻断，
    不自动 drop、alter 或 recreate。
    """

    def __init__(
        self,
        settings: MilvusSettings,
        vector_dimension: int | None = None,
        *,
        client: Any | None = None,
        allow_collection_creation: bool = True,
    ) -> None:
        if settings.schema_mode != "contract":
            raise ValueError(
                "ContractMilvusVectorRepository 要求 MILVUS_SCHEMA_MODE=contract"
            )
        self.settings = settings
        self._vector_dimension: int | None = None
        self._client = client
        self._initialized = False
        self._owns_client = client is None
        self._allow_collection_creation = allow_collection_creation
        self._contract_row_mapper = MilvusRowMapper()
        self._lock = Lock()

        if vector_dimension is not None:
            self._set_vector_dimension(vector_dimension)

    @property
    def client(self) -> Any:
        """暴露延迟创建的 SDK client，供高级集成场景使用。"""

        return self._get_client()

    def contract_schema_summary(
        self,
        vector_dimension: int | None = None,
    ) -> list[dict[str, object]]:
        """返回正式契约 Milvus Schema 摘要，不连接或修改 Milvus。"""

        dimension = self._resolve_vector_dimension(vector_dimension)
        return [spec.summary() for spec in contract_milvus_field_specs(dimension)]

    def compare_contract_schema(
        self,
        description: Mapping[str, Any],
        vector_dimension: int | None = None,
    ) -> dict[str, object]:
        """只读比较服务端 Collection 与正式契约 Schema。"""

        dimension = self._resolve_vector_dimension(vector_dimension)
        return compare_schema_specs(
            description,
            contract_milvus_field_specs(dimension),
        ).as_dict()

    def expected_index_summary(self) -> dict[str, object]:
        """返回当前配置要求的向量索引定义。"""

        return vector_index_spec(self.settings).summary(
            include_name=False
        )

    def expected_contract_indexes(self) -> list[dict[str, object]]:
        """返回正式契约 Collection 的向量和高频标量索引定义。"""

        return [
            spec.summary(include_name=True)
            for spec in index_specs(self.settings)
        ]

    def compare_contract_indexes(self, client: Any | None = None) -> dict[str, object]:
        """只读比较正式 Contract Collection 的向量和标量索引。

        该方法只调用 describe/list 类只读接口，不创建或修改索引；结果供初始化
        阶段的 zero-write gate 和验证脚本使用。
        """

        active_client = self._get_client() if client is None else client
        collection_name = self.settings.collection_name
        index_names = list(
            active_client.list_indexes(collection_name=collection_name)
        )
        descriptions: dict[str, Mapping[str, Any]] = {}
        for index_name in sorted(
            set(index_names)
            & {
                str(item["index_name"])
                for item in self.expected_contract_indexes()
            }
        ):
            description = active_client.describe_index(
                collection_name=collection_name,
                index_name=index_name,
            )
            descriptions[index_name] = description
        return compare_contract_indexes_definition(
            index_names,
            descriptions,
            index_specs(self.settings),
        ).as_dict()

    def initialize(self, vector_dimension: int) -> None:
        """建立连接，并创建或校验集合 Schema。

        Collection 已存在时只做 Schema 与 Index 兼容性检查；只有明确允许创建、
        且目标不存在时才调用 create_collection。任何不兼容都会在后续写入前抛出，
        从而保持 fail-closed。
        """

        self._set_vector_dimension(vector_dimension)
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            client = self._get_client()
            collection_name = self.settings.collection_name
            if client.has_collection(collection_name=collection_name):
                # 先校验物理契约，再允许任何 upsert/search 继续；不能用“维度相同”
                # 代替完整 Schema 和模型空间检查。
                description = client.describe_collection(collection_name=collection_name)
                self._validate_existing_collection(description)
                self._validate_existing_indexes(client)
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

    def upsert_batch(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> MutationResult:
        """按 Contract 主键 upsert 一批已经完成向量化的记录。

        调用顺序是输入类型/批次主键检查 → 维度锁定 → Collection gate → 行映射
        → SDK upsert。Repository 不自行生成向量或修复字段，Mapper 拒绝容量/主键
        不合法的数据。
        """

        items = list(records)
        if not items:
            return MutationResult(affected_count=0)
        if not all(isinstance(item, EmbeddedContractPreparedChunk) for item in items):
            raise TypeError(
                "ContractMilvusVectorRepository 只接受 EmbeddedContractPreparedChunk"
            )

        self._ensure_dimension_from_records(items)
        self._validate_batch_model_space(items)
        dimension = self._vector_dimension
        if dimension is None:  # 理论上不会发生，防止未来修改破坏维度约束
            raise ValueError("vector dimension is not configured")
        self.initialize(dimension)
        self._validate_existing_model_space(items[0])
        rows = self._to_rows(items)
        response = self._get_client().upsert(
            collection_name=self.settings.collection_name,
            data=rows,
            **self._timeout_kwargs(),
        )
        return _mutation_result(
            response,
            "upsert_count",
            expected_count=len(items),
            expected_ids=self._primary_keys(items),
        )

    def get_by_vector_doc_id(
        self,
        vector_doc_id: str,
        *,
        include_embedding: bool = True,
    ) -> dict[str, Any] | None:
        """按 Contract 主键回查单条 Entity，不执行向量检索。

        Metadata-only/audit-only 更新必须读取完整向量和 JSON 快照，调用方才能在
        不重新 Embedding 的前提下重建完整 Entity；查询为空时由上层决定是否修复。
        """

        if not isinstance(vector_doc_id, str) or not vector_doc_id.strip():
            raise ValueError("vector_doc_id 必须是非空字符串")

        dimension = self._resolve_vector_dimension(None)
        output_fields = [
            spec.name for spec in contract_milvus_field_specs(dimension)
        ]
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

    def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int = 3,
        chunk_type: str | None = None,
    ) -> list[SearchHit]:
        """执行 Contract Collection 基础向量检索，并转换为 ``SearchHit``。

        输入向量会先做有限数值检查并锁定维度，过滤条件只允许当前基础的
        ``chunk_type`` 等值表达式；原始 SDK 结果不会直接泄漏到业务层。
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit 必须是正整数")
        if chunk_type is not None and (
            not isinstance(chunk_type, str) or not chunk_type.strip()
        ):
            raise ValueError("chunk_type 必须是非空字符串或 None")

        vector = _normalize_search_embedding(embedding)
        self._set_vector_dimension(len(vector))
        self.initialize(len(vector))

        filter_expression = ""
        if chunk_type is not None:
            filter_expression = (
                "chunk_type == "
                f"'{_escape_milvus_string_literal(chunk_type)}'"
            )

        response = self._get_client().search(
            collection_name=self.settings.collection_name,
            data=[vector],
            anns_field="embedding",
            filter=filter_expression,
            limit=limit,
            output_fields=[
                "vector_doc_id",
                "expert_id",
                "chunk_type",
                "chunk_text",
            ],
            search_params=self._search_params(limit),
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        if not isinstance(response, list) or len(response) != 1:
            raise TypeError("Milvus search 返回结果必须包含一个 query 结果列表")
        hits = response[0]
        if not isinstance(hits, list):
            raise TypeError("Milvus search 的 query 结果必须是列表")
        return [_to_search_hit(hit) for hit in hits]

    def close(self) -> None:
        """关闭由当前 Repository 创建的 PyMilvus client。

        外部注入的 client 不由本对象关闭，避免破坏调用方共享的连接生命周期。
        """

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
                    "pymilvus is required to use ContractMilvusVectorRepository; "
                    "run `uv sync` first"
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

    def _ensure_dimension_from_records(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> None:
        self._set_vector_dimension(len(records[0].embedding))
        for record in records:
            if len(record.embedding) != self._vector_dimension:
                raise ValueError(
                    f"embedding dimension mismatch for chunk {record.vector_doc_id!r}: "
                    f"expected {self._vector_dimension}, got {len(record.embedding)}"
                )

    @staticmethod
    def _validate_batch_model_space(
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> None:
        """在任何 Collection 操作前拒绝批内混用模型空间。"""

        first = records[0]
        if is_placeholder_model_version(first.embedding_model_version):
            raise ValueError("正式 Contract 写入禁止使用 embedding_model_version=dev-unknown")
        difference = compare_model_space(
            [
                {
                    "embedding_model_alias": record.embedding_model_alias,
                    "embedding_model_version": record.embedding_model_version,
                    "embedding_dim": record.embedding_dim,
                }
                for record in records
            ],
            embedding_model_alias=first.embedding_model_alias,
            embedding_model_version=first.embedding_model_version,
            embedding_dimension=first.embedding_dim,
        )
        if not difference.compatible:
            raise ValueError(
                "同一 mutation batch 不能混用 Embedding model-space: "
                f"{json.dumps(difference.as_dict(), ensure_ascii=False, sort_keys=True)}"
            )

    def _validate_existing_model_space(
        self,
        record: EmbeddedContractPreparedChunk,
    ) -> None:
        """在 upsert 前分批遍历已有 Entity，确认 Collection 模型空间一致。"""

        iterator = self._get_client().query_iterator(
            collection_name=self.settings.collection_name,
            batch_size=1000,
            limit=-1,
            filter="embedding_dim >= 0",
            output_fields=[
                "embedding_model_alias",
                "embedding_model_version",
                "embedding_dim",
            ],
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        try:
            while True:
                rows = iterator.next()
                if not isinstance(rows, list) or not all(
                    isinstance(row, Mapping) for row in rows
                ):
                    raise TypeError("Milvus model-space iterator 返回了非法 Entity 批次")
                if not rows:
                    break
                difference = compare_model_space(
                    rows,
                    embedding_model_alias=record.embedding_model_alias,
                    embedding_model_version=record.embedding_model_version,
                    embedding_dimension=record.embedding_dim,
                )
                if not difference.compatible:
                    raise ValueError(
                        "Milvus Collection 已有 Entity 与当前 Embedding model-space 不一致: "
                        f"{json.dumps(difference.as_dict(), ensure_ascii=False, sort_keys=True)}"
                    )
        finally:
            iterator.close()

    def _build_schema(self, vector_dimension: int | None) -> Any:
        """构造正式 Contract Schema，不修改任何已有 Collection。"""

        return build_contract_schema(self.settings, vector_dimension)

    def _build_contract_schema(self, vector_dimension: int | None) -> Any:
        """保留内部构建入口，实际使用正式 Contract Schema。"""

        return build_contract_schema(self.settings, vector_dimension)

    def _build_index_params(self, client: Any) -> Any:
        return build_index_params(
            client,
            index_specs(self.settings),
        )

    def _validate_existing_collection(self, description: Mapping[str, Any]) -> None:
        """校验正式 Contract Collection，绝不自动迁移不兼容 Schema。

        这里是 Repository 内的最后一道写入前保护；差异会以安全 JSON 摘要抛出，
        不包含凭据或业务正文。
        """

        difference = self.compare_contract_schema(description)
        if not difference["compatible"]:
            raise ValueError(
                f"Milvus contract collection {self.settings.collection_name!r} "
                "与正式契约 schema 不兼容: "
                f"{json.dumps(difference, ensure_ascii=False, sort_keys=True)}"
            )

    def _validate_existing_indexes(self, client: Any) -> None:
        """校验正式契约 Collection 的向量与标量索引。"""

        difference = self.compare_contract_indexes(client)
        if not difference["compatible"]:
            raise ValueError(
                f"Milvus contract collection {self.settings.collection_name!r} "
                "的索引不兼容: "
                f"{json.dumps(difference, ensure_ascii=False, sort_keys=True)}"
            )

    def _to_rows(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> list[dict[str, Any]]:
        """在交给 SDK 前统一执行批次主键检查和 Contract 行映射。"""

        self._validate_batch_ids(records)
        return self._contract_row_mapper.to_rows(records)

    @staticmethod
    def _validate_batch_ids(
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> None:
        ids = [record.vector_doc_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("一个 mutation batch 不能包含重复的主键")

    def _primary_keys(
        self,
        records: Sequence[EmbeddedContractPreparedChunk],
    ) -> list[str]:
        return [record.vector_doc_id for record in records]

    def _timeout_kwargs(self) -> dict[str, Any]:
        return {} if self.settings.timeout is None else {"timeout": self.settings.timeout}

    def _search_params(self, limit: int) -> dict[str, object]:
        """构造当前开发索引可用的最小 search 参数。"""

        params: dict[str, object] = {
            "metric_type": self.settings.metric_type.upper(),
        }
        if self.settings.index_type.upper() == "HNSW":
            params["params"] = {"ef": max(64, limit)}
        return params


def _mutation_result(
    response: Any,
    count_key: str,
    *,
    expected_count: int,
    expected_ids: Sequence[str],
) -> MutationResult:
    """严格解析 SDK mutation 响应，不根据输入推定写入成功。"""

    if not isinstance(response, Mapping):
        raise TypeError("Milvus mutation 响应必须是 Mapping")

    count = response.get(count_key)
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"Milvus mutation 响应缺少合法的 {count_key}")
    if count != expected_count:
        raise ValueError(
            f"Milvus mutation 的 {count_key} 与请求数量不一致: "
            f"expected {expected_count}, got {count}"
        )

    if "ids" in response:
        ids = response["ids"]
    elif "primary_keys" in response:
        ids = response["primary_keys"]
    else:
        raise ValueError("Milvus mutation 响应缺少 ids 或 primary_keys")
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise TypeError("Milvus mutation 响应中的主键必须是序列")
    primary_keys = [str(item) for item in ids]
    if primary_keys != list(expected_ids):
        raise ValueError("Milvus mutation 响应的主键与请求主键不一致")

    return MutationResult(
        affected_count=count,
        primary_keys=primary_keys,
    )


def _normalize_search_embedding(embedding: Sequence[float]) -> list[float]:
    """校验并复制 search 输入向量，避免把可变输入直接传给 SDK。"""

    if isinstance(embedding, (str, bytes)):
        raise TypeError("embedding 必须是数值序列")

    vector: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("embedding 元素必须是数值")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding 元素必须是有限数值")
        vector.append(number)
    if not vector:
        raise ValueError("embedding 不能为空")
    return vector


def _escape_milvus_string_literal(value: str) -> str:
    """转义等值过滤中的字符串字面量。"""

    return value.replace("\\", "\\\\").replace("'", "\\'")


def _to_search_hit(raw_hit: object) -> SearchHit:
    """将 PyMilvus 的单条 search 结果转换为稳定的 SearchHit。"""

    if not isinstance(raw_hit, Mapping):
        raise TypeError("Milvus search 单条结果必须是映射结构")

    raw_entity = raw_hit.get("entity")
    entity: Mapping[str, Any] = raw_entity if isinstance(raw_entity, Mapping) else raw_hit
    payload = {
        "vector_doc_id": entity.get("vector_doc_id", raw_hit.get("id")),
        "expert_id": entity.get("expert_id"),
        "chunk_type": entity.get("chunk_type"),
        "chunk_text": entity.get("chunk_text"),
        "score": raw_hit.get("distance", raw_hit.get("score")),
    }
    return SearchHit.model_validate(payload)
