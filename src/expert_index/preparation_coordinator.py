"""P4.8 外层 Preparation Coordinator：P3 任务租约与阶段协调。

把"已由 P3 领取且持有有效 attempt token 的任务"经 P3 Repository 续租和阶段推进，
驱动到 P4 Dispatcher 的纯事实准备入口，最终停在 ``validated``，产出内存
:class:`PreparedChunkSet`。

职责边界（严格）：

- 真正的 lease 有效性由 Index MySQL ``CURRENT_TIMESTAMP(6)`` + ``attempt_token`` +
  rowcount fencing 权威判定；Coordinator **不解读** ``task.lease_until`` 的时区或
  与本机时间比较，不给 naive DATETIME 猜测或强行附加时区。
- 复用现有 ``renew_source_sync_task_lease`` 与 ``advance_source_sync_task_stage`` 两个
  Repository 方法；不新增第二套 lease API，不修改 MySQL schema。
- 首次执行和 Artifact 前恢复都重读 MySQL 当前事实并重建内存结果；阶段只补齐当前阶段
  之后的相邻节点，绝不回退。``contract_ready``/``handed_off``/``completed`` 属于 P5，
  P4 不推进，也不调用 ``mark_source_sync_task_success``。
- 每次 renew 和 stage 更新都携带 ``task_id`` + 当前 ``attempt_token``；任一冲突
  （lease 已变、token 失配、阶段错位）立即传播异常，**不再读取事实、不再构建、
  不再推进、不返回部分结果**。
- Source/Normalize/Builder/Validation 失败不推进 ``chunk_built`` 或 ``validated``；
  异常 fail-closed 向上传播。
- 不调用 Consumer、不写 Artifact、不推进 Checkpoint、不访问 Milvus/Embedding，
  不调用 ``contract_bridge``/``vector_sync``，不修改 ``scanner``/``repositories``/
  ``source.py``/``scope_handlers.py``/``rag_vector_store``。

P5.3B 的入口契约：只接收 ``processing_stage == 'validated'`` 的任务，handed_off
仍属于 P5，P4 Coordinator 不负责。
"""

from __future__ import annotations

from typing import Protocol

from .chunk_builder import ChunkFactory
from .models import PreparedChunkSet, SourceSyncTask
from .preparation_dispatcher import build_prepared_chunk_set, read_scoped_facts
from .source import MySQLSourceAdapter

__all__ = ["PreparationCoordinator", "TaskRepositoryPort"]

#: 续租默认 lease 秒数。与 P3 扫描/claim 默认对齐，仅作为传给 Repository 的建议值；
#: 真正的 lease 有效性仍由 Repository 以 MySQL CURRENT_TIMESTAMP(6) 判定。
DEFAULT_LEASE_SECONDS = 120

_PREPARATION_STAGES = (
    "candidate",
    "source_read",
    "normalized",
    "chunk_built",
    "validated",
)
_PREPARATION_STAGE_INDEX = {
    stage: index for index, stage in enumerate(_PREPARATION_STAGES)
}


class TaskRepositoryPort(Protocol):
    """P4 Coordinator 依赖的 P3 Repository 最小契约。

    只复用现有 ``renew_source_sync_task_lease`` 与 ``advance_source_sync_task_stage``
    两个方法；二者内部以 Index MySQL ``CURRENT_TIMESTAMP(6)`` + ``attempt_token`` +
    rowcount fencing 权威判定 lease 与阶段，Coordinator 不重复实现。
    """

    def renew_source_sync_task_lease(
        self,
        task_id: int,
        attempt_token: str,
        *,
        lease_seconds: int,
    ) -> SourceSyncTask:
        """续租当前有效 attempt 的 lease，返回数据库最新任务。"""

    def advance_source_sync_task_stage(
        self,
        task_id: int,
        attempt_token: str,
        *,
        next_stage: str,
    ) -> SourceSyncTask:
        """把有效 processing 任务幂等推进到紧邻的下一个标准阶段。"""


class PreparationCoordinator:
    """P4 外层协调器：续租 + 单调阶段推进 + 纯事实准备。

    注入依赖：
    :param task_repository: P3 Repository（或最小 Protocol 实现），提供续租与阶段推进。
    :param source: P4.3 定向只读事实适配器。
    :param factory: ChunkFactory，注入 ``task_id/batch_id/data_version`` 并提供策略版本。
    :param lease_seconds: 续租秒数；作为传给 Repository 的建议值，真正有效性仍由
        Index MySQL ``CURRENT_TIMESTAMP(6)`` + ``attempt_token`` + rowcount 判定。
    """

    def __init__(
        self,
        task_repository: TaskRepositoryPort,
        source: MySQLSourceAdapter,
        factory: ChunkFactory,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> None:
        self._repo = task_repository
        self._source = source
        self._factory = factory
        if not isinstance(lease_seconds, int) or lease_seconds <= 0:
            raise ValueError("lease_seconds 必须为正整数")
        self._lease_seconds = lease_seconds

    def prepare(
        self,
        task: SourceSyncTask,
        attempt_token: str,
    ) -> PreparedChunkSet:
        """驱动已领取任务到 ``validated``，返回内存 :class:`PreparedChunkSet`。

        :param task: P3 已领取任务模型（``sync_status == 'processing'``）。
        :param attempt_token: 调用方持有的本次认领令牌，必须与 ``task.attempt_token`` 一致。

        执行序（首次或 Artifact 前恢复均从当前事实重建）：
        1. 内存层校验 task 是 SourceSyncTask、状态 processing、token 匹配。
        2. 续租（Repository 以 MySQL CURRENT_TIMESTAMP(6) + token + rowcount 权威判定）。
        3. 读取当前事实（纯读取，不构建）；读取异常立即停止，不推进 source_read。
        4. 当前仍为 ``candidate`` 时推进 ``source_read``；恢复任务不回退阶段。
        5. 续租。
        6. 调用 ``build_prepared_chunk_set`` 完成 Normalize/Build/Validate；任一失败
           异常立即向上传播，**阶段最多停留在 source_read**（不提前写 normalized）。
        7. 续租。
        8. 从当前阶段起依次补齐 ``normalized`` → ``chunk_built`` → ``validated``。
        9. 返回 :class:`PreparedChunkSet`。

        注：``build_prepared_chunk_set`` 当前尚未拆成三个独立步骤，本次在它完整成功
        后依次补齐三个已完成阶段；不在执行前提前写 normalized。
        """

        # 1) 内存层一致性校验：真正的 lease 有效性由后续续租在 DB 权威判定。
        if not isinstance(task, SourceSyncTask):
            raise TypeError("task 必须是 SourceSyncTask")
        if task.sync_status != "processing":
            raise ValueError(
                f"任务状态非 processing，实际为 {task.sync_status!r}，无法协调"
            )
        if not isinstance(attempt_token, str) or not attempt_token.strip():
            raise ValueError("attempt_token 必须是非空字符串")
        if task.attempt_token is None or task.attempt_token != attempt_token:
            raise ValueError("attempt_token 与任务当前令牌不一致，调用方不再持有有效 lease")
        if task.processing_stage not in _PREPARATION_STAGE_INDEX:
            raise ValueError(
                f"P4 无法从 {task.processing_stage!r} 阶段准备当前事实"
            )

        task_id = task.task_id
        token = attempt_token
        current_stage = task.processing_stage

        # 2) 续租：以 Index MySQL CURRENT_TIMESTAMP(6) 为权威。返回的最新任务作为
        #    后续阶段推进的身份上下文；不解读其 lease_until 的时区或与本机时间比较。
        task = self._repo.renew_source_sync_task_lease(
            task_id, token, lease_seconds=self._lease_seconds,
        )
        # 续租返回的任务仍是 processing，token 仍是当前 token；防御性校验防止 DB 异常。
        if task.sync_status != "processing" or task.attempt_token != token:
            raise RuntimeError("续租后任务状态或令牌异常")

        # 3) 读取当前事实（纯读取，不构建）。任何读取/归属/时序异常立即向上传播，
        #    不推进 source_read，不构建 Chunk。
        facts = read_scoped_facts(task, self._source)

        # 4) 首次任务推进 source_read；恢复任务保留已完成阶段，只重读当前事实。
        if current_stage == "candidate":
            self._repo.advance_source_sync_task_stage(
                task_id, token, next_stage="source_read",
            )
            current_stage = "source_read"

        # 5) 续租：以 Index MySQL CURRENT_TIMESTAMP(6) 为权威，为后续构建与阶段
        #    推进续住 lease。
        self._repo.renew_source_sync_task_lease(
            task_id, token, lease_seconds=self._lease_seconds,
        )

        # 6) Normalize/Build/Validate 由 scope_handlers 在 build_prepared_chunk_set 内
        #    一次性完成。任一失败立即向上传播；此时阶段最多停留在 source_read，
        #    不提前写 normalized、不推进 chunk_built 或 validated。
        prepared = build_prepared_chunk_set(task, facts, self._factory)

        # 7) 构建成功后续租，准备连续推进三个已完成阶段。
        self._repo.renew_source_sync_task_lease(
            task_id, token, lease_seconds=self._lease_seconds,
        )

        # 8) 只补齐当前阶段之后的节点。构建成功前不前移；恢复时也不回退或重复写
        #    较早阶段，确保 Repository 的相邻推进约束始终成立。
        current_index = _PREPARATION_STAGE_INDEX[current_stage]
        for next_stage in ("normalized", "chunk_built", "validated"):
            if current_index < _PREPARATION_STAGE_INDEX[next_stage]:
                self._repo.advance_source_sync_task_stage(
                    task_id, token, next_stage=next_stage,
                )
                current_stage = next_stage
                current_index = _PREPARATION_STAGE_INDEX[current_stage]

        return prepared
