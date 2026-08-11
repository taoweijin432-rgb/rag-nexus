"""专家索引 Chunk 加工包。

包的公开入口刻意保持很小：业务调用方通过 :class:`ExpertIndexPipeline`
运行完整事务；清洗器、Source Adapter 和 Repository 仍保持独立，方便单测和替换。
"""

from .pipeline import BuildResult, ExpertIndexPipeline

__all__ = ["BuildResult", "ExpertIndexPipeline"]

