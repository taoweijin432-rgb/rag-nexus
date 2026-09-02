"""Resume 数据处理的独立上游边界。"""

from .checkpoint_initializer import (
    ResumeCheckpointInitializationResult,
    initialize_resume_checkpoints,
)
from .config import PhysicalDatabaseTarget, ResumeSettings
from .dataset import RESUME_SPEC
from .incremental_runtime import ResumeIncrementalRuntime, build_resume_incremental_runtime
from .pipeline import (
    ResumeBatchBuildResult,
    ResumeBuildResult,
    ResumeIndexPipeline,
    ResumeVectorSyncSummary,
)
from .preflight import ResumeSourcePreflightResult, run_resume_source_preflight
from .runtime import ResumeRuntime, build_resume_runtime
from .schema_initializer import (
    DEFAULT_RESUME_INDEX_DDL,
    MySQLResumeSchemaInitializer,
    ResumeSchemaInitializationResult,
)
from .source import MySQLResumeSourceAdapter, ResumeCandidate, ResumeNotFoundError
from .source_schema import (
    RESUME_SOURCE_TABLES,
    ResumeSourceSchemaError,
    ResumeSourceSchemaReport,
    inspect_resume_source_schema,
    require_resume_source_schema,
)

__all__ = [
    "DEFAULT_RESUME_INDEX_DDL",
    "RESUME_SOURCE_TABLES",
    "RESUME_SPEC",
    "MySQLResumeSchemaInitializer",
    "MySQLResumeSourceAdapter",
    "PhysicalDatabaseTarget",
    "ResumeBatchBuildResult",
    "ResumeBuildResult",
    "ResumeCandidate",
    "ResumeCheckpointInitializationResult",
    "ResumeIncrementalRuntime",
    "ResumeIndexPipeline",
    "ResumeNotFoundError",
    "ResumeRuntime",
    "ResumeSchemaInitializationResult",
    "ResumeSettings",
    "ResumeSourcePreflightResult",
    "ResumeSourceSchemaError",
    "ResumeSourceSchemaReport",
    "ResumeVectorSyncSummary",
    "build_resume_incremental_runtime",
    "build_resume_runtime",
    "initialize_resume_checkpoints",
    "inspect_resume_source_schema",
    "require_resume_source_schema",
    "run_resume_source_preflight",
]
