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
    field_map,
)
from .contract import MilvusRowMapper
from .indexes import build_index_params, index_specs, vector_index_spec
from .schemas import (
    build_contract_schema,
    contract_milvus_field_specs,
)


class ContractMilvusVectorRepository:
    """封装 Milvus 连接、Schema 管理和 CRUD 写操作。

    调用方：
        - ``ContractIngestionService.ingest_one`` / ``ContractBatchIngestionService.ingest_batch``
          （通过 ``ContractRepository`` Protocol）；
        - 验收/验证脚本（只读 gate、结构对比、PK 回查）。

    输入：
        ``EmbeddedContractPreparedChunk``（已经过 Contract 校验、Eligibility 与
        Embedding 的结果记录）。

    内部主要步骤（见 ``upsert_batch``）：
        1. Collection / Schema / Index compatibility gate；
        2. model-space / embedding dimension 检查；
        3. ``MilvusRowMapper`` 行转换；
        4. vector_doc_id / PK 一致性检查；
        5. SDK upsert；
        6. ``MutationResult``（affected_count / primary_keys）fail-closed 校验。

    职责边界：
        Repository **不负责** Contract Validation、Eligibility、调用 Embedding、
        Task State；它只消费已经向量化的正式记录。

    连接生命周期：
        SDK client 在第一次操作时延迟创建。传入 ``client`` 适合测试，或适合由
        应用自行管理 PyMilvus client 生命周期的场景。所有正式写入都先确定向量
        维度，再通过 Schema/Index Compatibility Gate；已存在但不兼容的 Collection
        会阻断，不自动 drop、alter 或 recreate。

    持久化 profile：
        V14（本类默认）使用 25 字段 Schema、``MilvusRowMapper`` 与
        ``index_specs``；V15 由 ``ContractMilvusV15VectorRepository`` 子类整体
        替换下面声明的 profile 属性，禁止在方法内部按版本分支。
    """

    # ---- 持久化 profile（默认 V14；V15 子类整体替换，禁止方法内按版本分支）----
    _schema_field_specs = staticmethod(contract_milvus_field_specs)
    _schema_builder = staticmethod(build_contract_schema)
    _index_specs_fn = staticmethod(index_specs)
    _row_mapper_cls = MilvusRowMapper
    # 全量扫描/模型空间一致性检查使用的 filter 表达式与输出字段。
    _full_scan_filter = "embedding_dim >= 0"
    _model_space_fields = (
        "embedding_model_alias",
        "embedding_model_version",
        "embedding_dim",
    )

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
        self._contract_row_mapper = self._row_mapper_cls()
        self._lock = Lock()
        # 模型空间一致性缓存：同一 Repository 生命周期内，同一个
        # (embedding_model_alias, embedding_model_version, embedding_dim) 只校验
        # 一次既有 Collection 数据；后续相同 model space 的 upsert 直接复用，跳过
        # 全表扫描。model space 变化时缓存失效并重新校验，保持 fail-closed。
        self._validated_model_space: dict[str, object] | None = None

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
        return [spec.summary() for spec in self._schema_field_specs(dimension)]

    def compare_contract_schema(
        self,
        description: Mapping[str, Any],
        vector_dimension: int | None = None,
    ) -> dict[str, object]:
        """只读比较服务端 Collection 与正式契约 Schema。"""

        dimension = self._resolve_vector_dimension(vector_dimension)
        return compare_schema_specs(
            description,
            self._schema_field_specs(dimension),
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
            for spec in self._index_specs_fn(self.settings)
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
            self._index_specs_fn(self.settings),
        ).as_dict()

    def assert_destructive_maintenance_allowed(
        self,
        *,
        expected_dimension: int | None = None,
    ) -> dict[str, object]:
        """Full Sync 删除 stale vectors 前的 Collection 安全 preflight（fail-closed）。

        该方法只对当前 Repository 已绑定的 ``settings.collection_name`` 做只读
        检查，不接收调用方传入的任意 Collection 名。普通 CRUD 不应、也无需
        调用它；只有 Full Sync 在准备物理删除 stale vectors 前必须显式通过，
        否则 fail-closed。

        检查项（任一项失败即抛异常，不执行任何删除）：
            1. 授权：当前 ``collection_name`` 必须在
               ``destructive_maintenance_collections``（默认空，即默认不允许任何
               destructive cleanup）；历史 / 受保护 Collection 因此天然被拒绝。
            2. 存在：当前 Collection 必须真实存在。
            3. Schema 兼容：与正式 Contract Collection 兼容。
            4. 主键字段必须为 ``vector_doc_id``。
            5. ``auto_id`` 必须为 False。
            6. ``embedding`` 向量字段必须存在。
            7. embedding 维度与正式模型空间（默认期望维度）一致。

        Args:
            expected_dimension: 期望的向量维度；省略时使用 Repository 已配置维度。

        Returns:
            dict[str, object]：只读安全摘要，供 Full Sync 报告使用。
        """

        dimension = self._resolve_vector_dimension(expected_dimension)
        collection_name = self.settings.collection_name

        if collection_name not in self.settings.destructive_maintenance_collections:
            raise RuntimeError(
                f"Collection {collection_name!r} 未获授权执行 destructive maintenance；"
                "请在环境变量 MILVUS_ALLOWED_DESTRUCTIVE_COLLECTIONS 中显式声明"
            )

        active_client = self._get_client()
        if not active_client.has_collection(collection_name=collection_name):
            raise RuntimeError(
                f"Collection {collection_name!r} 不存在，无法执行 destructive maintenance"
            )

        description = active_client.describe_collection(
            collection_name=collection_name
        )

        fields = description.get("fields", [])
        # 逐项显式校验结构性不变量，先于整体 Schema 对比给出精确的可定位错误。
        primary_fields = [
            field.get("name")
            for field in fields
            if isinstance(field, Mapping) and field.get("is_primary")
        ]
        if primary_fields != ["vector_doc_id"]:
            raise ValueError(
                f"Collection {collection_name!r} 主键字段必须为 vector_doc_id，"
                f"实际为 {primary_fields!r}"
            )
        if description.get("auto_id", False):
            raise ValueError(
                f"Collection {collection_name!r} 的 auto_id 必须为 False，"
                "禁止 destructive maintenance"
            )
        embedding_field = field_map(description).get("embedding")
        if embedding_field is None:
            raise ValueError(
                f"Collection {collection_name!r} 缺少向量字段 embedding"
            )
        params = embedding_field.get("params")
        embedding_dim = (
            params.get("dim") if isinstance(params, Mapping) else None
        )
        if embedding_dim != dimension:
            raise ValueError(
                f"Collection {collection_name!r} 的 embedding 维度 {embedding_dim!r} "
                f"与期望维度 {dimension} 不一致"
            )

        # 最后整体对比：任何其他与正式契约 Schema 的差异都 fail-closed。
        difference = self.compare_contract_schema(description, dimension)
        if not difference["compatible"]:
            raise ValueError(
                f"Milvus contract collection {collection_name!r} 与正式契约 schema "
                "不兼容，禁止 destructive maintenance: "
                f"{json.dumps(difference, ensure_ascii=False, sort_keys=True)}"
            )

        return {
            "collection_name": collection_name,
            "authorized": True,
            "schema_mode": self.settings.schema_mode,
            "collection_exists": True,
            "schema_compatible": True,
            "primary_key": "vector_doc_id",
            "auto_id": False,
            "embedding_field": "embedding",
            "embedding_dim": dimension,
        }

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

        调用链：
            upsert_batch(records)
            -> 类型检查 + 空批处理
            -> _ensure_dimension_from_records      # 从记录锁定向量维度
            -> _validate_batch_model_space         # 批内模型空间一致 + 非占位版本
            -> initialize(dimension)               # Collection/Index gate（已存在只读校验）
            -> _validate_existing_model_space      # 遍历已有 Entity 对比模型空间
            -> _to_rows(records)                   # MilvusRowMapper 行映射 + PK 去重
            -> SDK upsert
            -> _mutation_result                    # upsert_count/PK 严格解析
            -> MutationResult

        输入：
            ``EmbeddedContractPreparedChunk[]`` —— 已经过 Contract 校验、Eligibility
            与 Embedding 的正式记录。

        输出：
            ``MutationResult`` —— affected_count 与 primary_keys；两者必须与请求
            完全一致（fail-closed），否则抛异常，不推定写入成功。

        分支：
            - 空批：直接返回 ``MutationResult(affected_count=0)``；
            - 任一校验失败：抛异常，不执行部分写入。

        职责边界：
            Repository 不自行生成向量或修复字段；Mapper 拒绝容量/主键不合法的
            数据；Repository 不负责 Contract Validation / Eligibility / Embedding。
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

    def delete_by_vector_doc_ids(
        self,
        vector_doc_ids: Sequence[str],
    ) -> MutationResult:
        """按 vector_doc_id 集合删除 Milvus 实体。

        语义：
            - 按主键精确删除，幂等；不存在的 vector_doc_id 不视为错误（不参与计数）。
            - 空列表安全返回（不发起 SDK 调用）。
            - 本方法不生成或修改 vector_doc_id，只透传调用方提供的主键。
            - 返回 ``MutationResult``，其中 ``affected_count`` 为实际删除条数、
              ``primary_keys`` 为空（pymilvus delete 不返回已删主键列表）。

        Args:
            vector_doc_ids: 待删除的主键集合。

        Returns:
            MutationResult：``affected_count`` 为实际删除条数。
        """

        cleaned: list[str] = []
        seen: set[str] = set()
        for vector_doc_id in vector_doc_ids:
            if not isinstance(vector_doc_id, str) or not vector_doc_id.strip():
                raise ValueError("vector_doc_id 必须是非空字符串")
            normalized = vector_doc_id.strip()
            if normalized not in seen:
                seen.add(normalized)
                cleaned.append(normalized)

        if not cleaned:
            return MutationResult(affected_count=0)

        response = self._get_client().delete(
            collection_name=self.settings.collection_name,
            ids=cleaned,
            **self._timeout_kwargs(),
        )
        return MutationResult(affected_count=_delete_count(response))

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
            spec.name for spec in self._schema_field_specs(dimension)
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

    def query_by_expert_id(
        self,
        expert_id: int,
        *,
        output_fields: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        """按 ``expert_id`` 查询该专家的全部向量记录（默认不含 embedding）。

        语义：
            - 基于标量字段 ``expert_id`` 过滤（INT64 索引字段），分页遍历全部匹配行。
            - 默认只返回最小字段集 ``vector_doc_id`` / ``expert_id`` /
              ``content_hash`` / ``metadata_hash``，供上游做残留差集判断；
              不返回 embedding，避免大字段拉取。
            - 调用方可通过 ``output_fields`` 显式扩展返回字段。

        Args:
            expert_id: 专家 ID（INT64）。
            output_fields: 要返回的字段；默认最小字段集。

        Returns:
            list[dict[str, object]]：按查询顺序返回每行的字段字典。
        """

        if isinstance(expert_id, bool) or not isinstance(expert_id, int):
            raise ValueError(  # noqa: TRY004
                "expert_id 必须是整数"
            )
        if expert_id < 1:
            raise ValueError("expert_id 必须是正整数")

        fields = list(
            output_fields
            if output_fields is not None
            else ["vector_doc_id", "expert_id", "content_hash", "metadata_hash"]
        )
        if not fields or any(
            not isinstance(field, str) or not field.strip() for field in fields
        ):
            raise ValueError("output_fields 必须是非空字符串序列")

        iterator = self._get_client().query_iterator(
            collection_name=self.settings.collection_name,
            batch_size=1000,
            limit=-1,
            filter=f"expert_id == {int(expert_id)}",
            output_fields=fields,
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        rows: list[dict[str, object]] = []
        try:
            while True:
                batch = iterator.next()
                if not isinstance(batch, list) or not all(
                    isinstance(row, Mapping) for row in batch
                ):
                    raise TypeError("Milvus expert_id iterator 返回了非法 Entity 批次")
                if not batch:
                    break
                rows.extend(dict(row) for row in batch)
        finally:
            iterator.close()
        return rows

    def list_all_vector_states(
        self,
        *,
        output_fields: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        """列出当前 Collection 中全部向量的最小状态（默认不含 embedding）。

        语义：
            - 用于全量更新 / 残留对账：读取当前全部存在的 ``vector_doc_id``
              及其 ``content_hash`` / ``metadata_hash``，供差异判断与 stale 计算；
              不返回 embedding，避免大字段拉取。
            - 基于 ``embedding_dim >= 0`` 的全量过滤表达式，使用与内部
              model-space 一致性检查相同的分页遍历模式。
            - 全量结果出现重复 ``vector_doc_id`` 时 fail-closed：视为读取结果
              不可信，抛 ``RuntimeError``，不静默去重。
            - 调用方可通过 ``output_fields`` 显式扩展返回字段。

        Args:
            output_fields: 要返回的字段；默认最小字段集
              ``vector_doc_id`` / ``content_hash`` / ``metadata_hash``。

        Returns:
            list[dict[str, object]]：按查询顺序返回每行的字段字典。
        """

        fields = list(
            output_fields
            if output_fields is not None
            else ["vector_doc_id", "content_hash", "metadata_hash"]
        )
        if not fields or any(
            not isinstance(field, str) or not field.strip() for field in fields
        ):
            raise ValueError("output_fields 必须是非空字符串序列")

        iterator = self._get_client().query_iterator(
            collection_name=self.settings.collection_name,
            batch_size=1000,
            limit=-1,
            filter=self._full_scan_filter,
            output_fields=fields,
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        rows: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        check_duplicates = "vector_doc_id" in fields
        try:
            while True:
                batch = iterator.next()
                if not isinstance(batch, list) or not all(
                    isinstance(row, Mapping) for row in batch
                ):
                    raise TypeError("Milvus 全量状态 iterator 返回了非法 Entity 批次")
                if not batch:
                    break
                for row in batch:
                    projected = dict(row)
                    rows.append(projected)
                    if check_duplicates:
                        vector_doc_id = projected.get("vector_doc_id")
                        if isinstance(vector_doc_id, str):
                            if vector_doc_id in seen_ids:
                                raise RuntimeError(
                                    "Milvus 全量状态扫描返回了重复 vector_doc_id: "
                                    f"{vector_doc_id!r}"
                                )
                            seen_ids.add(vector_doc_id)
        finally:
            iterator.close()
        return rows

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
        """在 upsert 前分批遍历已有 Entity，确认 Collection 模型空间一致。

        缓存：同一 Repository 生命周期内，同一 model space
        (embedding_model_alias / embedding_model_version / embedding_dim) 只校验
        一次既有 Collection 数据；后续相同 model space 的 upsert 直接跳过全表
        扫描。model space 任一字段变化都会使缓存失效并重新校验，保持 fail-closed。
        扫描中发现不一致或 iterator 异常时不写缓存，下次同名 model space 仍会
        重新扫描。
        """

        requested = {
            "embedding_model_alias": record.embedding_model_alias,
            "embedding_model_version": record.embedding_model_version,
            "embedding_dim": record.embedding_dim,
        }
        if self._validated_model_space == requested:
            return

        iterator = self._get_client().query_iterator(
            collection_name=self.settings.collection_name,
            batch_size=1000,
            limit=-1,
            filter=self._full_scan_filter,
            output_fields=list(self._model_space_fields),
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
                    [self._model_space_from_row(row) for row in rows],
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
        # 扫描全部通过后才写入缓存，使后续相同 model space 跳过扫描。
        self._validated_model_space = requested

    def _model_space_from_row(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, object]:
        """从单行 Entity 提取模型空间字段（默认 V14 一级字段投影）。"""

        return {
            "embedding_model_alias": row.get("embedding_model_alias"),
            "embedding_model_version": row.get("embedding_model_version"),
            "embedding_dim": row.get("embedding_dim"),
        }

    def _build_schema(self, vector_dimension: int | None) -> Any:
        """构造 profile 指定的 Contract Schema，不修改任何已有 Collection。"""

        return self._schema_builder(self.settings, vector_dimension)

    def _build_contract_schema(self, vector_dimension: int | None) -> Any:
        """保留内部构建入口，实际使用 profile 指定的 Contract Schema。"""

        return self._schema_builder(self.settings, vector_dimension)

    def _build_index_params(self, client: Any) -> Any:
        return build_index_params(
            client,
            self._index_specs_fn(self.settings),
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


def _delete_count(response: Any) -> int:
    """解析 pymilvus delete 响应，返回实际删除条数。

    delete 用于删除不存在的向量：请求的 ID 可能部分或全部未命中，因此这里只
    读取实际 ``delete_count``，不要求与请求数量一致（存在视为成功）。响应格式
    兼容 pymilvus 2.3.2+（``{'delete_count': N}``）及更早版本。
    """

    if not isinstance(response, Mapping):
        raise TypeError("Milvus delete 响应必须是 Mapping")
    count = response.get("delete_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("Milvus delete 响应缺少合法的 delete_count")
    if count < 0:
        raise ValueError("Milvus delete_count 不能为负数")
    return count


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
