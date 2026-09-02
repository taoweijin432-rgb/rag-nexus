"""Resume 数据处理的独立上游边界。"""

from .config import PhysicalDatabaseTarget, ResumeSettings
from .preflight import ResumeSourcePreflightResult, run_resume_source_preflight
from .source_schema import (
    RESUME_SOURCE_TABLES,
    ResumeSourceSchemaError,
    ResumeSourceSchemaReport,
    inspect_resume_source_schema,
    require_resume_source_schema,
)

__all__ = [
    "RESUME_SOURCE_TABLES",
    "PhysicalDatabaseTarget",
    "ResumeSettings",
    "ResumeSourcePreflightResult",
    "ResumeSourceSchemaError",
    "ResumeSourceSchemaReport",
    "inspect_resume_source_schema",
    "require_resume_source_schema",
    "run_resume_source_preflight",
]
