"""V15 精简 Contract Collection 的 Milvus Repository 实现。

V15 是当前唯一正式物理 profile，直接继承通用基类 ``ContractMilvusRepositoryBase``，
整体替换其持久化 profile 类属性（不复制任何 CRUD 逻辑，也不在方法内按版本分支）：
    * Schema：``v15_milvus_field_specs``（4 字段）+ ``build_v15_schema``；
    * 行映射：``MilvusV15RowMapper``（4 字段 + flat metadata JSON）；
    * 索引：``index_specs_v15``（仅 embedding HNSW/COSINE）；
    * 全量扫描 filter：基于一级 PK ``vector_doc_id`` 的恒真合法范围
      （``vector_doc_id != ""``），Full Reconciliation 状态扫描不依赖
      任何不存在的 V15 scalar；
    * 模型空间一致性检查从 metadata JSON 读取（V15 的
      ``embedding_model_alias`` / ``embedding_model_version`` /
      ``embedding_dim`` 位于其中）。

旧的 V14 25 字段物理形态已退役；通用 CRUD / Query / Safety 编排逻辑全部
继承自基类，本类只替换持久化 profile 与 V15 专属的 Query/状态提取行为。

V15 物理 Schema（4 字段，dynamic=False）：

    vector_doc_id  VARCHAR PK (auto_id=False)
    chunk_text     VARCHAR(8192)
    embedding      FLOAT_VECTOR(1024)
    metadata       JSON（单层 flat，其余全部 Contract 信息，
                   含 expert_id / content_hash / metadata_hash）

所有在线写入与查询统一通过 ``vector_doc_id`` 定位：
    - upsert：直接使用传入 Contract 的 ``vector_doc_id``（不生成 / 不重算）；
    - update_metadata：按 ``vector_doc_id`` 读旧 Entity → 保留原 embedding →
      重建 chunk_text/metadata → 同 PK upsert（不重新 Embedding）；
    - disable / get：按 PK ``vector_doc_id``；Repository 不提供物理删除。

正式调用统一使用 ``query_by_owner_id``，由 Profile 决定 flat metadata JSON 中的 owner 字段，
且不会把 owner 重新提升为一级字段。
``list_all_vector_states`` 物理读取 ``["vector_doc_id", "metadata"]``，再从
metadata 提取 ``content_hash`` / ``metadata_hash`` —— 这是纯 4 字段设计明确
接受的 tradeoff，不为了优化而重新增加 hash scalar。

职责边界与基类完全一致：不负责 Contract Validation、Eligibility、
Embedding、Task State；不创建真实 Collection（是否创建由调用方结合
``allow_collection_creation`` 决定）；不修改 .env；不迁移真实数据。

检索能力不在当前 Repository 公共接口内；未来需要按多领域查询契约另行设计。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import MilvusSettings
from ..core.hash_state import read_hash_field
from ..core.storage import VectorRepositoryProfile
from ..domains import ExpertContractAdapter, ResumeContractAdapter
from .compatibility import compare_model_space
from .contract_v15 import (
    MilvusV15RowMapper,
    ResumeMilvusV15RowMapper,
)
from .indexes import index_specs_v15
from .repository import ContractMilvusRepositoryBase
from .schemas.contract_v15 import build_v15_schema, v15_milvus_field_specs

__all__ = [
    "EXPERT_V15_PROFILE",
    "RESUME_V15_PROFILE",
    "ContractMilvusV15VectorRepository",
]

EXPERT_V15_PROFILE = VectorRepositoryProfile(
    schema_field_specs=v15_milvus_field_specs,
    schema_builder=build_v15_schema,
    index_specs_fn=index_specs_v15,
    row_mapper_cls=MilvusV15RowMapper,
    full_scan_filter='vector_doc_id != ""',
    model_space_fields=("metadata",),
    contract_adapter_cls=ExpertContractAdapter,
    owner_field="expert_id",
    vector_doc_id_prefix="expert:",
    owner_filter_fn=lambda owner_id: f'metadata["expert_id"] == {int(owner_id)}',
    owner_output_fields=("vector_doc_id",),
)

RESUME_V15_PROFILE = VectorRepositoryProfile(
    schema_field_specs=v15_milvus_field_specs,
    schema_builder=build_v15_schema,
    index_specs_fn=index_specs_v15,
    row_mapper_cls=ResumeMilvusV15RowMapper,
    full_scan_filter='vector_doc_id != ""',
    model_space_fields=("metadata",),
    contract_adapter_cls=ResumeContractAdapter,
    owner_field="resume_id",
    vector_doc_id_prefix="resume:",
    owner_filter_fn=lambda owner_id: f'metadata["resume_id"] == {int(owner_id)}',
    owner_output_fields=("vector_doc_id",),
)

# 限量采样行数：V15 Collection 规模增长后，upsert 前的既有 model-space
# 一致性检查不再全表扫描，改为采样前若干行校验。当前正式数据向量空间一致，
# 采样校验既避免隧道链路下的全表扫描超时，又保留 fail-closed 的方向性
# （采样到不一致即拒绝）。
_MODEL_SPACE_SAMPLE_LIMIT = 500

# V15 不存在的一级字段（已进入 flat metadata JSON）：禁止作为一级 output field。
_V15_FORBIDDEN_TOP_LEVEL_FIELDS = ("expert_id", "content_hash", "metadata_hash")


class ContractMilvusV15VectorRepository(ContractMilvusRepositoryBase):
    """V15 精简 Contract Collection 的 Repository 类型。

    构造参数、initialize / upsert_batch / get_by_vector_doc_id /
    query_by_owner_id / list_all_vector_states /
    assert_destructive_maintenance_allowed / close 的语义与基类一致；差异只体现
    在持久化 profile 上（见模块 docstring）。

    Expert 是默认 Profile；Resume Factory 必须显式注入 ``RESUME_V15_PROFILE``。
    """

    def __init__(
        self,
        settings: MilvusSettings,
        vector_dimension: int | None = None,
        *,
        client: Any | None = None,
        allow_collection_creation: bool = False,
        profile: VectorRepositoryProfile[Any] = EXPERT_V15_PROFILE,
    ) -> None:
        super().__init__(
            settings,
            vector_dimension,
            client=client,
            allow_collection_creation=allow_collection_creation,
            profile=profile,
        )

    def _model_space_from_row(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, object]:
        """从 V15 行的 flat metadata JSON 提取模型空间字段。

        与基类的区别：V15 的 ``embedding_model_alias`` /
        ``embedding_model_version`` / ``embedding_dim`` 位于 metadata JSON 内，
        而不是一级字段；提取失败（缺少合法 metadata 对象）一律 fail-closed。
        """

        flat = row.get("metadata")
        if not isinstance(flat, Mapping):
            raise TypeError("Milvus V15 行缺少合法 metadata JSON 对象")
        if self.profile.owner_field == "resume_id":
            if "resume_id" not in flat:
                raise ValueError(
                    "Resume Collection 既有 Entity 缺少 metadata['resume_id']，"
                    "拒绝与新 Resume 数据混写"
                )
            if "expert_id" in flat:
                raise ValueError(
                    "Resume Collection 既有 Entity 不得包含 metadata['expert_id']"
                )
        return {
            "embedding_model_alias": flat.get("embedding_model_alias"),
            "embedding_model_version": flat.get("embedding_model_version"),
            "embedding_dim": flat.get("embedding_dim"),
        }

    def _validate_existing_model_space(
        self,
        record: Any,
    ) -> None:
        """在 upsert 前采样校验既有 Entity 的模型空间一致性（V15 版本）。

        与基类的区别：V15 不再对整个 Collection 做全表
        ``query_iterator(limit=-1)`` 扫描——该扫描会随 Collection 行数线性
        增长，在 SSH 隧道 + 远端 Milvus 链路下达到数十秒级超时，导致 upsert
        在写入前卡死（已写入约 7 万行时复现）。改为限量采样前
        ``_MODEL_SPACE_SAMPLE_LIMIT`` 行校验 model-space 一致性。

        语义变化（已与业务确认）：
            - 由“验证全部历史行”放宽为“限量采样验证”；
            - 当前正式数据向量空间一致，采样校验足以阻止异空间混写；
            - 采样到任一不一致行仍 fail-closed，拒绝本次 upsert。

        缓存语义与基类一致：同一 Repository 生命周期内、同一 model space
        采样通过后才写缓存；采样中发现不一致或查询异常时不写缓存，下次
        同名 model space 仍会重新采样。model space 任一字段变化会使缓存失效。
        """

        requested = {
            "embedding_model_alias": record.embedding_model_alias,
            "embedding_model_version": record.embedding_model_version,
            "embedding_dim": record.embedding_dim,
        }
        if self._validated_model_space == requested:
            return

        client = self._get_client()
        stats = client.get_collection_stats(
            collection_name=self.settings.collection_name,
        )
        row_count_raw = stats.get("row_count") if isinstance(stats, Mapping) else None
        try:
            row_count = int(row_count_raw)
        except (TypeError, ValueError) as error:
            raise TypeError("Milvus Collection stats 缺少合法 row_count") from error
        if row_count < 0:
            raise ValueError("Milvus Collection stats 的 row_count 不能为负数")
        if row_count == 0:
            self._validated_model_space = requested
            return
        if row_count <= _MODEL_SPACE_SAMPLE_LIMIT:
            sample_filter = self.profile.full_scan_filter
        else:
            # 目标候选数约为上限的两倍，再由 limit 截断到500条；最低比例避免
            # 超大 Collection 因浮点下溢形成空采样。
            factor = min(
                1.0,
                max(0.000001, (_MODEL_SPACE_SAMPLE_LIMIT * 2) / row_count),
            )
            sample_filter = f"RANDOM_SAMPLE({factor:.12g})"

        rows = client.query(
            collection_name=self.settings.collection_name,
            filter=sample_filter,
            output_fields=["vector_doc_id", *self.profile.model_space_fields],
            limit=_MODEL_SPACE_SAMPLE_LIMIT,
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        if not isinstance(rows, list) or not all(
            isinstance(row, Mapping) for row in rows
        ):
            raise TypeError("Milvus model-space 采样查询返回了非法 Entity 批次")
        if rows:
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
        # 采样校验通过后才写入缓存，使后续相同 model space 跳过采样查询。
        self._validated_model_space = requested

    def query_by_owner_id(
        self,
        owner_id: int,
        *,
        output_fields: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        """按当前 V15 Profile 的 owner JSON 字段查询向量记录。"""

        if isinstance(owner_id, bool) or not isinstance(owner_id, int):
            raise TypeError("owner_id 必须是整数")
        if owner_id < 1:
            raise ValueError("owner_id 必须是正整数")

        fields = list(
            output_fields if output_fields is not None else ["vector_doc_id"]
        )
        if not fields or any(
            not isinstance(field, str) or not field.strip() for field in fields
        ):
            raise ValueError("output_fields 必须是非空字符串序列")
        forbidden_fields = set(_V15_FORBIDDEN_TOP_LEVEL_FIELDS)
        forbidden_fields.add(self.profile.owner_field)
        for forbidden in forbidden_fields:
            if forbidden in fields:
                raise ValueError(
                    f"V15 不存在一级字段 {forbidden!r}，请通过 'metadata' 读取"
                )
        return super().query_by_owner_id(owner_id, output_fields=fields)

    def list_all_vector_states(
        self,
        *,
        output_fields: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        """列出当前 Collection 中全部向量的最小状态（V15 版本）。

        公共返回语义与基类保持一致：
            每行包含 ``vector_doc_id`` / ``content_hash`` / ``metadata_hash``
            （供 Full Reconciliation 差异判断与 stale 计算）。

        但 V15 物理读取为 ``["vector_doc_id", "metadata"]``，再从 metadata JSON
        提取两个 hash——这是纯 4 字段设计明确接受的 tradeoff（V15 不再有一级
        hash scalar），不为了优化而重新增加 hash scalar。

        fail-closed：
            - 显式请求不存在的 V15 一级字段（expert_id/content_hash/
              metadata_hash）→ ``ValueError``；
            - metadata 缺 ``content_hash`` / ``metadata_hash``、hash 类型错误 →
              ``RuntimeError``；
            - 全量结果出现重复 ``vector_doc_id`` → ``RuntimeError``（不静默去重）；
            - iterator 返回非法批次 → ``TypeError``。
        """

        fields = list(
            output_fields
            if output_fields is not None
            else ["vector_doc_id", "metadata"]
        )
        if not fields or any(
            not isinstance(field, str) or not field.strip() for field in fields
        ):
            raise ValueError("output_fields 必须是非空字符串序列")
        for forbidden in _V15_FORBIDDEN_TOP_LEVEL_FIELDS:
            if forbidden in fields:
                raise ValueError(
                    f"V15 不存在一级字段 {forbidden!r}，请通过 'metadata' 读取"
                )

        iterator = self._get_client().query_iterator(
            collection_name=self.settings.collection_name,
            batch_size=1000,
            limit=-1,
            filter=self.profile.full_scan_filter,
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
                    if "metadata" in fields:
                        projected = self._extract_light_state(projected)
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

    def _extract_light_state(
        self,
        projected: dict[str, object],
    ) -> dict[str, object]:
        """从 V15 行的 flat metadata 提取轻量状态模型（fail-closed）。

        读取 ``metadata["content_hash"]`` / ``metadata["metadata_hash"]``，
        从返回行中移除原始 metadata key，替换为两个 hash —— 保持公共返回语义
        （vector_doc_id / content_hash / metadata_hash）。
        """

        flat = projected.pop("metadata")
        if not isinstance(flat, Mapping):
            raise TypeError("Milvus V15 全量状态行缺少合法 metadata JSON 对象")
        try:
            content_hash = read_hash_field(
                {"metadata": flat}, "content_hash", required=True
            )
            metadata_hash = read_hash_field(
                {"metadata": flat}, "metadata_hash", required=True
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Milvus V15 全量状态 metadata Hash 不合法: {exc}"
            ) from exc
        # required=True 保证这里不会是 None；显式断言让类型检查和后续模型都保持
        # 严格字符串语义。
        assert content_hash is not None
        assert metadata_hash is not None
        projected["content_hash"] = content_hash
        projected["metadata_hash"] = metadata_hash
        return projected

    def list_all_vector_activity_states(self) -> list[dict[str, object]]:
        """列出全部向量的轻量状态并附带经过校验的 ``is_active``（V15 版本）。

        与 ``list_all_vector_states`` 的区别：返回行额外包含严格布尔
        ``is_active``（读自 flat metadata），供 Full Reconciliation 以单一扫描
        同时获得 active / inactive 分区；inactive 只作统计审计，不参与差异分类。
        物理读取与全量状态扫描一致（Strong 一致性 + 全扫 filter）。

        fail-closed：
            - metadata 缺 ``is_active`` 或非严格 bool → ``RuntimeError``
              （不默认为 active，避免掩盖数据缺陷）；
            - 其余语义（非法批次 / 重复主键 / 非法 metadata）与
              ``list_all_vector_states`` 一致。
        """

        iterator = self._get_client().query_iterator(
            collection_name=self.settings.collection_name,
            batch_size=1000,
            limit=-1,
            filter=self.profile.full_scan_filter,
            output_fields=["vector_doc_id", "metadata"],
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        rows: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        try:
            while True:
                batch = iterator.next()
                if not isinstance(batch, list) or not all(
                    isinstance(row, Mapping) for row in batch
                ):
                    raise TypeError("Milvus 活动状态 iterator 返回了非法 Entity 批次")
                if not batch:
                    break
                for row in batch:
                    raw = dict(row)
                    flat = raw.get("metadata")
                    if not isinstance(flat, Mapping):
                        raise TypeError(
                            "Milvus V15 活动状态行缺少合法 metadata JSON 对象"
                        )
                    # 先在原始 flat metadata 上提取 is_active，再复用轻量状态提取。
                    is_active = flat.get("is_active")
                    if not isinstance(is_active, bool):
                        # 与 _extract_light_state 的 hash 校验保持同类语义：
                        # 数据形态缺陷而非调用方类型错误，保留 RuntimeError。
                        raise RuntimeError(  # noqa: TRY004
                            "Milvus V15 活动状态 metadata 缺少合法 is_active 布尔值"
                        )
                    projected = self._extract_light_state(raw)
                    projected["is_active"] = is_active
                    rows.append(projected)
                    vector_doc_id = projected.get("vector_doc_id")
                    if isinstance(vector_doc_id, str):
                        if vector_doc_id in seen_ids:
                            raise RuntimeError(
                                "Milvus 活动状态扫描返回了重复 vector_doc_id: "
                                f"{vector_doc_id!r}"
                            )
                        seen_ids.add(vector_doc_id)
        finally:
            iterator.close()
        return rows
