"""单人物的读取、标准化、Chunk 构建、校验和向量同步。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from expert_contracts import ExpertChunkEmbeddingInput

from ..contract_bridge import chunk_records_to_contracts
from ..dataset import EXPERT_SPEC
from ..models import ChunkRecord
from ..normalize import normalize_aggregate
from ..ports import ExpertBatchSyncInput
from ..validators import validate_chunks
from ..vector_sync import ExpertVectorSyncResult
from .state import _ExpertProgress


class _PersonProcessingMixin:
    """封装所有构建入口共用的单人物领域加工链路。

    使用方需要提供 source、dataset_spec、factory、vector_sync 和 settings 属性，
    并实现 _load_person_aggregate。单人物入口直接同步，批量入口先生成 Contract，
    再进行跨人物批量同步。
    """

    def _process_active_expert(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_id: int,
        progress: _ExpertProgress,
    ) -> str:
        """执行一位 active 专家的完整领域链路并返回业务终态。

        * Source 读取主档、工作、项目、教育四表一致性快照；
        * Normalize 在内存中生成 Profile/Employment/Project Stage；
        * ChunkFactory 生成三类 Chunk；
        * Validator 在向量同步前执行完整契约检查；
        * 有 Chunk 时，通过内存 Contract 同步完整专家快照；
        * 无 Chunk 时，通过空快照删除该专家的旧向量。

        返回 ``success`` 或 ``rejected``。异常原样交给入口处理：单专家入口结束整个
        批次，批量入口只结束当前专家。``progress`` 保存异常发生前已完成阶段的数量。

        Args:
            expert_id: 待读取和同步的人物业务 ID。
            data_version: 标准化及 Chunk Contract 使用的数据版本。
            batch_id: 记录到 Chunk 追踪字段中的构建批次主键。
            progress: 当前人物的可变进度对象，用于保留已完成阶段计数。

        Returns:
            success 表示已生成并同步 Chunk；rejected 表示无内容且已同步空快照。
        """

        aggregate = self._load_person_aggregate(expert_id)
        progress.extracted_count = (
            1 + len(aggregate.employments) + len(aggregate.projects) + len(aggregate.educations)
        )

        profile, employments, projects = normalize_aggregate(
            aggregate,
            data_version=data_version,
            build_batch_id=batch_id,
            dataset_spec=self.dataset_spec,
        )
        chunks = self.factory.build_all(profile, employments, projects)
        if not chunks:
            # 无可生成内容属于 rejected 业务结果，逻辑清除旧向量。
            progress.vector_sync = self._sync_expert_vector_snapshot(expert_id, [])
            return "rejected"

        if self.dataset_spec is EXPERT_SPEC:
            validate_chunks(chunks)
        else:
            validate_chunks(chunks, dataset_spec=self.dataset_spec)
        # 校验完成后记录 Chunk 数量；向量同步失败时外层会把本专家标记为 failed。
        progress.chunk_count = len(chunks)
        progress.vector_sync = self._sync_expert_vector_snapshot(expert_id, chunks)
        return "success"

    def _sync_expert_vector_snapshot(
        self,
        expert_id: int,
        chunks: list[ChunkRecord],
    ) -> ExpertVectorSyncResult:
        """把 Chunk 转为公共 Contract，并同步一位专家的完整向量快照。

        ``chunks`` 为空时也会调用下游，用于删除该专家已存在但本次不再产出的向量。

        Args:
            expert_id: 下游向量记录归属的人物业务 ID。
            chunks: 当前人物完整且已校验的 Chunk；空列表表示当前无有效 Chunk。

        Returns:
            下游对新增、更新、不变和删除向量的可靠计数。
        """

        updated_at = datetime.now(ZoneInfo(self.settings.timezone))
        contracts = chunk_records_to_contracts(
            chunks,
            updated_at=updated_at,
            timezone_name=self.settings.timezone,
            dataset_spec=self.dataset_spec,
        )
        return self.vector_sync.sync_expert(expert_id, contracts)

    def _build_expert_contracts_for_batch(
        self,
        expert_id: int,
        *,
        data_version: str,
        batch_id: int,
        progress: _ExpertProgress,
    ) -> tuple[str, list[ExpertChunkEmbeddingInput]]:
        """执行一位 active 专家的读、生成与校验，但不触发向量同步。

        返回 ``(业务终态, contracts)``：有 Chunk 返回 ("success", contracts)，
        无可生成内容返回 ("rejected", [])。异常原样向外抛，由批量循环隔离为
        failed。``progress`` 记录已完成的阶段数量。本方法与 :meth:`_process_active_expert`
        共用读、生成、校验逻辑，但不调用下游，以便批量循环合并 Embedding。

        Args:
            expert_id: 当前页中待加工的人物业务 ID。
            data_version: 标准化及 Chunk Contract 使用的数据版本。
            batch_id: 记录到 Chunk 追踪字段中的构建批次主键。
            progress: 当前人物的可变进度对象。

        Returns:
            二元组，第一项是 success 或 rejected，第二项是待批量同步的 Contract。
        """

        aggregate = self._load_person_aggregate(expert_id)
        progress.extracted_count = (
            1 + len(aggregate.employments) + len(aggregate.projects) + len(aggregate.educations)
        )

        profile, employments, projects = normalize_aggregate(
            aggregate,
            data_version=data_version,
            build_batch_id=batch_id,
            dataset_spec=self.dataset_spec,
        )
        chunks = self.factory.build_all(profile, employments, projects)
        if not chunks:
            # 无可生成内容属于 rejected 业务结果，逻辑清除旧向量。
            return "rejected", self._contracts_for_expert([])

        if self.dataset_spec is EXPERT_SPEC:
            validate_chunks(chunks)
        else:
            validate_chunks(chunks, dataset_spec=self.dataset_spec)
        progress.chunk_count = len(chunks)
        return "success", self._contracts_for_expert(chunks)

    def _contracts_for_expert(
        self,
        chunks: list[ChunkRecord],
    ) -> list[ExpertChunkEmbeddingInput]:
        """把 Chunk 转为公共 Contract，向量同步前的纯内存转换。

        Args:
            chunks: 一位人物当前有效的 Chunk 列表；允许为空。

        Returns:
            带统一更新时间、时区和数据集标识的 Contract 列表。
        """

        updated_at = datetime.now(ZoneInfo(self.settings.timezone))
        return chunk_records_to_contracts(
            chunks,
            updated_at=updated_at,
            timezone_name=self.settings.timezone,
            dataset_spec=self.dataset_spec,
        )

    def _sync_experts_batch(
        self,
        inputs: Sequence[ExpertBatchSyncInput],
    ) -> dict[int, ExpertVectorSyncResult]:
        """批量同步多位专家，合并 Embedding 与 Milvus 写入。

        空输入返回空 dict。返回值按 expert_id 索引，供批量循环回填 per-expert 结果。

        Args:
            inputs: 当前页内可同步人物的完整 Contract 快照。

        Returns:
            以人物业务 ID 为键的向量同步结果；空输入返回空字典。
        """

        if not inputs:
            return {}
        results = self.vector_sync.sync_expert_batch(inputs)
        return {r.expert_id: r for r in results}
