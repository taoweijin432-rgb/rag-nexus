"""Resume 数据处理的独立上游边界。"""

from .config import PhysicalDatabaseTarget, ResumeSettings
from .preflight import ResumeSourcePreflightResult, run_resume_source_preflight
from .schema_initializer import (
    DEFAULT_RESUME_INDEX_DDL,
    MySQLResumeSchemaInitializer,
    ResumeSchemaInitializationResult,
)
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
    "MySQLResumeSchemaInitializer",
    "PhysicalDatabaseTarget",
    "ResumeSchemaInitializationResult",
    "ResumeSettings",
    "ResumeSourcePreflightResult",
    "ResumeSourceSchemaError",
    "ResumeSourceSchemaReport",
    "inspect_resume_source_schema",
    "require_resume_source_schema",
    "run_resume_source_preflight",
]
