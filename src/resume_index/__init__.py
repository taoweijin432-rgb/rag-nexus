"""Resume 数据处理的独立上游边界。"""

from .config import PhysicalDatabaseTarget, ResumeSettings
from .dataset import RESUME_SPEC
from .preflight import ResumeSourcePreflightResult, run_resume_source_preflight
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
    "ResumeCandidate",
    "ResumeNotFoundError",
    "ResumeSchemaInitializationResult",
    "ResumeSettings",
    "ResumeSourcePreflightResult",
    "ResumeSourceSchemaError",
    "ResumeSourceSchemaReport",
    "inspect_resume_source_schema",
    "require_resume_source_schema",
    "run_resume_source_preflight",
]
