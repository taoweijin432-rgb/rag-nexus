"""Pipeline Runner 模块。

Task2 Step 2：任务执行编排层。

Runner 只负责编排，不实现业务逻辑：
- 不实现 Contract 校验、Eligibility、Embedding（复用 ContractBatchIngestionService）
- 不实现 Milvus 写入（复用 ContractRepository）
- 不实现 MySQL 读取（复用 MySqlExpertChunkReader）
- 不实现 Task State 持久化（复用 TaskStateStore）
"""

from .pipeline_runner import (
    PipelineRunnerError,
    VectorizationPipelineRunner,
)

__all__ = [
    "PipelineRunnerError",
    "VectorizationPipelineRunner",
]