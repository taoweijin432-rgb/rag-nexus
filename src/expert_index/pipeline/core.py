"""专家索引构建流程的依赖装配与统一入口类。"""

from __future__ import annotations

from ..chunk_builder import ChunkFactory
from ..config import Settings
from ..dataset import EXPERT_SPEC, DatasetSpec
from ..ports import ExpertVectorSyncPort
from ..repositories import (
    MySQLBatchItemRepository,
    MySQLBatchRepository,
    MySQLBuildLockRepository,
)
from ..source import MySQLSourceAdapter
from .batch import _BatchBuildMixin
from .person import _PersonProcessingMixin
from .retry import _RetryBuildMixin
from .single import _SingleBuildMixin
from .source_access import _SourceAccessMixin


class ExpertIndexPipeline(
    _SingleBuildMixin,
    _BatchBuildMixin,
    _RetryBuildMixin,
    _PersonProcessingMixin,
    _SourceAccessMixin,
):
    """装配单人物、全量、续跑和失败重试构建能力。

    Args:
        settings: 数据源、状态库、时区和 Chunk 构建规则配置。
        source: 人物来源读取实现；省略时创建 MySQLSourceAdapter。
        build_lock_repository: single/full 构建互斥锁实现；省略时连接状态库。
        batch_repository: 批次主表访问实现；省略时连接状态库。
        batch_item_repository: 批次人物明细访问实现；省略时连接状态库。
        vector_sync: 必传的向量同步端口，负责单人物及跨人物批量同步。
        dataset_spec: 数据集表名、字段和 Contract 标识；默认使用 EXPERT_SPEC。

    构造函数只完成依赖校验和装配，不读取人物数据，也不启动构建任务。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        source: object | None = None,
        build_lock_repository: MySQLBuildLockRepository | None = None,
        batch_repository: MySQLBatchRepository | None = None,
        batch_item_repository: MySQLBatchItemRepository | None = None,
        vector_sync: ExpertVectorSyncPort,
        dataset_spec: DatasetSpec = EXPERT_SPEC,
    ) -> None:
        # 默认 Repository 会真实写入状态库；完整注入 Fake 的测试路径不建立数据库连接。
        if any(
            repository is None
            for repository in (
                build_lock_repository,
                batch_repository,
                batch_item_repository,
            )
        ):
            settings.validate_index_write_boundary()
        self.settings = settings
        self.source = source or MySQLSourceAdapter(settings.source_db)
        self.dataset_spec = dataset_spec
        self.build_lock_repository = build_lock_repository or MySQLBuildLockRepository(
            settings.index_db
        )
        self.batch_repository = batch_repository or MySQLBatchRepository(settings.index_db)
        self.batch_item_repository = batch_item_repository or MySQLBatchItemRepository(
            settings.index_db
        )
        self.vector_sync = vector_sync
        self.factory = ChunkFactory(settings, dataset_spec=dataset_spec)
