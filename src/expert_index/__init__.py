"""专家索引 Chunk 加工包。

包的公开入口刻意保持很小：业务调用方通过 :class:`ExpertIndexPipeline`
运行单专家或全量编排；清洗器、Source Adapter 和 Repository 仍保持独立，方便单测
和替换。命令行分别通过 ``build`` 和 ``build-batch`` 启动两类任务。
"""

from .pipeline import BatchBuildResult, BuildResult, ExpertIndexPipeline

__all__ = ["BatchBuildResult", "BuildResult", "ExpertIndexPipeline"]
