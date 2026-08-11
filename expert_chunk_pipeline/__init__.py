"""专家数据清洗与向量化前处理包。

公开 API 保持精简：调用方通常只需要使用 ``build_expert_chunk`` 或
``write_chunks``。数据库读取、规则配置和文件编排分别位于独立模块中。
"""

from .chunk import build_expert_chunk

__all__ = ["build_expert_chunk"]
