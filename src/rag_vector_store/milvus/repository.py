"""正式 Contract Collection 的通用 Milvus Repository 基类。

Repository 是 Milvus Client 的唯一业务访问入口，负责连接生命周期、Collection
初始化/只读兼容性检查、CRUD、Query 和 Search 编排。Schema/Index 定义、兼容性
判断、Contract 行映射分别委托给 ``schemas``、``indexes``、``compatibility`` 和
子类提供的 ``RowMapper``；本模块不负责 Contract 校验、Embedding 或 Task State。

本类是**通用基类**，只承载与具体物理 Schema 无关的 CRUD / Schema / Index /
Model Space / 删除保护编排逻辑。具体的持久化 profile（字段集合、Schema
builder、索引集合、RowMapper、全量扫描 filter、模型空间字段）由子类整体
提供，禁止在方法内部按版本分支。当前唯一正式子类是
``ContractMilvusV15VectorRepository``（V15 精简 4 字段 Schema）。

旧的 V14 25 字段物理形态（``MilvusRowMapper`` / ``contract_milvus_field_specs`` /
``index_specs`` 等）已退役；其代码与专属测试已删除，但真实 V14 Collection
仍受保护性检查保护，不被 drop、alter 或迁移。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

from .._utils import batched, dedupe_nonempty_strings
from ..config import MilvusSettings
from ..core.storage import VectorRepositoryProfile
from ..model_space import is_placeholder_model_version
from ..models import MutationResult
from .compatibility import (
    compare_contract_indexes as compare_contract_indexes_definition,
)
from .compatibility import (
    compare_model_space,
    compare_schema_specs,
    field_map,
)
from .indexes import build_index_params, vector_index_spec

_PRIMARY_KEY_QUERY_BATCH_SIZE = 200


class ContractMilvusRepositoryBase:
    """封装 Milvus 连接、Schema 管理和 CRUD 写操作的通用基类。

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
        3. RowMapper 行转换；
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
        本基类不携带任何物理 Schema 默认值；构造时必须注入完整且不可变的
        :class:`VectorRepositoryProfile`，禁止再通过子类类属性重复声明配置。
    """

    def __init__(
        self,
        settings: MilvusSettings,
        vector_dimension: int | None = None,
        *,
        client: Any | None = None,
        allow_collection_creation: bool = True,
        profile: VectorRepositoryProfile[Any],
    ) -> None:
        if settings.schema_mode != "contract":
            raise ValueError(
                "ContractMilvusRepositoryBase 要求 MILVUS_SCHEMA_MODE=contract"
            )
        self.settings = settings
        self._vector_dimension: int | None = None
        self._client = client
        self._initialized = False
        self._owns_client = client is None
        self._allow_collection_creation = allow_collection_creation
        if not isinstance(profile, VectorRepositoryProfile):
            raise TypeError("profile 必须是 VectorRepositoryProfile")
        self._profile = profile
        self._contract_row_mapper = self._row_mapper_cls()
        self._contract_adapter = self._contract_adapter_cls()
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

    @property
    def profile(self) -> VectorRepositoryProfile[Any]:
        """返回当前实例唯一、不可变的持久化 Profile。"""

        return self._profile

    @property
    def _schema_field_specs(self) -> Any:
        return self._profile._schema_field_specs

    @property
    def _schema_builder(self) -> Any:
        return self._profile._schema_builder

    @property
    def _index_specs_fn(self) -> Any:
        return self._profile._index_specs_fn

    @property
    def _row_mapper_cls(self) -> Any:
        return self._profile._row_mapper_cls

    @property
    def _full_scan_filter(self) -> str:
        return self._profile._full_scan_filter

    @property
    def _model_space_fields(self) -> tuple[str, ...]:
        return self._profile._model_space_fields

    @property
    def _contract_adapter_cls(self) -> Any:
        return self._profile._contract_adapter_cls

    @property
    def _owner_field(self) -> str:
        return self._profile._owner_field

    @property
    def _vector_doc_id_prefix(self) -> str:
        return self._profile._vector_doc_id_prefix

    @property
    def _owner_filter_fn(self) -> Any:
        return self._profile._owner_filter_fn

    @property
    def _owner_output_fields(self) -> tuple[str, ...]:
        return self._profile._owner_output_fields

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
        records: Sequence[Any],
    ) -> MutationResult:
        """按 Contract 主键 upsert 一批已经完成向量化的记录。

        调用链：
            upsert_batch(records)
            -> 类型检查 + 空批处理
            -> _ensure_dimension_from_records      # 从记录锁定向量维度
            -> _validate_batch_model_space         # 批内模型空间一致 + 非占位版本
            -> initialize(dimension)               # Collection/Index gate（已存在只读校验）
            -> _validate_existing_model_space      # 遍历已有 Entity 对比模型空间
            -> _to_rows(records)                   # RowMapper 行映射 + PK 去重
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
        if not all(self._contract_adapter.is_embedded(item) for item in items):
            raise TypeError("ContractMilvusRepositoryBase 只接受当前 Profile 的 Embedded Contract")

        self._validate_identity_profile(items)

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

        rows = self.get_many_by_vector_doc_ids(
            [vector_doc_id],
            include_embedding=include_embedding,
        )
        return rows.get(vector_doc_id.strip())

    def get_many_by_vector_doc_ids(
        self,
        vector_doc_ids: Sequence[str],
        *,
        include_embedding: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """按主键批量回查 Entity，并严格校验返回身份。"""

        cleaned = dedupe_nonempty_strings(
            vector_doc_ids,
            prefix=self._vector_doc_id_prefix,
        )
        if not cleaned:
            return {}

        dimension = self._resolve_vector_dimension(None)
        output_fields = [
            spec.name for spec in self._schema_field_specs(dimension)
        ]
        if not include_embedding:
            output_fields.remove("embedding")

        requested = set(cleaned)
        found: dict[str, dict[str, Any]] = {}
        for batch in batched(cleaned, size=_PRIMARY_KEY_QUERY_BATCH_SIZE):
            rows = self._get_client().query(
                collection_name=self.settings.collection_name,
                ids=list(batch),
                output_fields=output_fields,
                consistency_level="Strong",
                **self._timeout_kwargs(),
            )
            if not isinstance(rows, list) or not all(
                isinstance(row, Mapping) for row in rows
            ):
                raise TypeError("Milvus query 返回了非法 Entity 批次")
            for raw in rows:
                row = dict(raw)
                vector_doc_id = row.get("vector_doc_id")
                if not isinstance(vector_doc_id, str):
                    raise TypeError("Milvus query 返回的 Entity 缺少合法 vector_doc_id")
                if vector_doc_id not in requested:
                    raise RuntimeError(
                        f"Milvus query 返回了未请求的 vector_doc_id: {vector_doc_id!r}"
                    )
                if vector_doc_id in found:
                    raise RuntimeError(
                        f"同一个 vector_doc_id 回查到多条 Entity: {vector_doc_id!r}"
                    )
                found[vector_doc_id] = row
        return found

    def query_by_owner_id(
        self,
        owner_id: Any,
        *,
        output_fields: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        """按当前 Profile 的 owner 字段查询全部向量记录。"""

        fields = list(
            output_fields
            if output_fields is not None
            else self._owner_output_fields
        )
        if not fields or any(
            not isinstance(field, str) or not field.strip() for field in fields
        ):
            raise ValueError("output_fields 必须是非空字符串序列")
        iterator = self._get_client().query_iterator(
            collection_name=self.settings.collection_name,
            batch_size=1000,
            limit=-1,
            filter=self._owner_filter_fn(owner_id),
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
                    raise TypeError("Milvus owner iterator 返回了非法 Entity 批次")
                if not batch:
                    break
                for row in batch:
                    projected = dict(row)
                    vector_doc_id = projected.get("vector_doc_id")
                    if vector_doc_id is not None and (
                        not isinstance(vector_doc_id, str)
                        or not vector_doc_id.startswith(self._vector_doc_id_prefix)
                    ):
                        raise RuntimeError(
                            "Milvus owner 查询返回了不属于当前 Profile 的 "
                            f"vector_doc_id: {vector_doc_id!r}"
                        )
                    rows.append(projected)
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

    def list_all_vector_activity_states(self) -> list[dict[str, object]]:
        """列出全部向量的轻量状态并附带经过校验的 ``is_active``（活动状态扫描）。

        与 ``list_all_vector_states`` 的区别：返回行额外包含严格布尔
        ``is_active``，供 Full Reconciliation 以单一扫描同时获得 active /
        inactive 分区（inactive 只作统计审计，不参与差异分类）。

        默认实现不可用：活动状态字段的位置由具体 Schema Profile 决定
        （V14 为一级标量列，已退役；V15 位于 flat metadata JSON），
        因此基类不提供通用实现，子类必须覆盖。
        """

        raise NotImplementedError(
            "当前 Repository Profile 不支持活动状态扫描；"
            "list_all_vector_activity_states 仅由 V15 Repository 实现"
        )

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
                    "pymilvus is required to use ContractMilvusRepositoryBase; "
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
        records: Sequence[Any],
    ) -> None:
        self._set_vector_dimension(len(self._contract_adapter.embedding(records[0])))
        for record in records:
            dimension = len(self._contract_adapter.embedding(record))
            if dimension != self._vector_dimension:
                raise ValueError(
                    f"embedding dimension mismatch for chunk "
                    f"{self._contract_adapter.vector_doc_id(record)!r}: "
                    f"expected {self._vector_dimension}, got {dimension}"
                )

    def _validate_batch_model_space(
        self,
        records: Sequence[Any],
    ) -> None:
        """在任何 Collection 操作前拒绝批内混用模型空间。"""

        model_spaces = [self._contract_adapter.model_space(record) for record in records]
        first_model_space = model_spaces[0]
        if is_placeholder_model_version(first_model_space["embedding_model_version"]):
            raise ValueError("正式 Contract 写入禁止使用 embedding_model_version=dev-unknown")
        difference = compare_model_space(
            model_spaces,
            embedding_model_alias=first_model_space["embedding_model_alias"],
            embedding_model_version=first_model_space["embedding_model_version"],
            embedding_dimension=first_model_space["embedding_dim"],
        )
        if not difference.compatible:
            raise ValueError(
                "同一 mutation batch 不能混用 Embedding model-space: "
                f"{json.dumps(difference.as_dict(), ensure_ascii=False, sort_keys=True)}"
            )

    def _validate_existing_model_space(
        self,
        record: Any,
    ) -> None:
        """在 upsert 前分批遍历已有 Entity，确认 Collection 模型空间一致。

        缓存：同一 Repository 生命周期内，同一 model space
        (embedding_model_alias / embedding_model_version / embedding_dim) 只校验
        一次既有 Collection 数据；后续相同 model space 的 upsert 直接跳过全表
        扫描。model space 任一字段变化都会使缓存失效并重新校验，保持 fail-closed。
        扫描中发现不一致或 iterator 异常时不写缓存，下次同名 model space 仍会
        重新扫描。
        """

        requested = self._contract_adapter.model_space(record)
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
                    embedding_model_alias=requested["embedding_model_alias"],
                    embedding_model_version=requested["embedding_model_version"],
                    embedding_dimension=requested["embedding_dim"],
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
        """从单行 Entity 提取模型空间字段。

        基类提供默认的一级字段投影实现；子类可覆盖为从 flat metadata JSON 读取
        （如 V15 的模型空间字段位于 metadata）。
        """

        return {
            "embedding_model_alias": row.get("embedding_model_alias"),
            "embedding_model_version": row.get("embedding_model_version"),
            "embedding_dim": row.get("embedding_dim"),
        }

    def _build_schema(self, vector_dimension: int | None) -> Any:
        """构造 profile 指定的 Contract Schema，不修改任何已有 Collection。"""

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
        records: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """在交给 SDK 前统一执行批次主键检查和 Contract 行映射。"""

        self._validate_batch_ids(records)
        return self._contract_row_mapper.to_rows(records)

    def _validate_batch_ids(
        self,
        records: Sequence[Any],
    ) -> None:
        ids = [self._contract_adapter.vector_doc_id(record) for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("一个 mutation batch 不能包含重复的主键")

    def _primary_keys(
        self,
        records: Sequence[Any],
    ) -> list[str]:
        return [self._contract_adapter.vector_doc_id(record) for record in records]

    def _validate_identity_profile(self, records: Sequence[Any]) -> None:
        """验证 owner、Metadata 和主键前缀属于同一个 Repository Profile。"""

        owner_field = self._owner_field
        prefix = self._vector_doc_id_prefix
        if self._contract_adapter.owner_field() != owner_field:
            raise ValueError("Contract Adapter 与 Repository Profile 的 owner 字段不一致")
        if self._contract_adapter.id_prefix() != prefix:
            raise ValueError("Contract Adapter 与 Repository Profile 的 vector_doc_id 前缀不一致")
        for record in records:
            vector_doc_id = self._contract_adapter.vector_doc_id(record)
            if not vector_doc_id.startswith(prefix):
                raise ValueError(
                    f"vector_doc_id {vector_doc_id!r} 不符合当前 Profile 前缀 {prefix!r}"
                )
            owner_id = self._contract_adapter.owner_id(record)
            metadata = self._contract_adapter.metadata(record)
            if metadata.get(owner_field) != owner_id:
                raise ValueError(
                    f"Embedded Contract 的 owner 字段 {owner_field!r} 与 owner_id 不一致"
                )

    def _timeout_kwargs(self) -> dict[str, Any]:
        return {} if self.settings.timeout is None else {"timeout": self.settings.timeout}


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
