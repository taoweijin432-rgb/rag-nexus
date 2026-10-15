"""metadata JSON 模型空间检查；V15/V16 共用，不含版本专属查询。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .compatibility import compare_model_space
from .repository import ContractMilvusRepositoryBase

# 保持已有采样契约；不是全量历史模型空间一致性的证明。
_MODEL_SPACE_SAMPLE_LIMIT = 500


class MetadataModelSpaceRepository(ContractMilvusRepositoryBase):
    """复用 metadata 模型空间提取、限量采样及实例缓存，不携带默认 Profile。"""

    def _model_space_from_row(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, object]:
        """读取 flat metadata 模型空间；Resume 仍按构造 Profile 拒绝身份混写。"""

        flat = row.get("metadata")
        if not isinstance(flat, Mapping):
            raise TypeError("Milvus 行缺少合法 metadata JSON 对象")
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
        """保持既有最多 500 行采样契约，不把采样当作全表一致性证明。

        同一实例、同一模型空间只有采样通过才缓存；异常或不匹配不缓存。
        模型空间变化重新采样，空集合沿用原有缓存语义。
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
            **self._timeout_kwargs(),
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
