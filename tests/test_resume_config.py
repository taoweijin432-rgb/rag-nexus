"""Resume 专用配置和 Expert 隔离门禁测试。"""

from __future__ import annotations

import pytest

from expert_index.config import DatabaseBoundaryError, DatabaseRole
from resume_index.config import ResumeSettings


def _environment() -> dict[str, str]:
    """构造显式隔离的 Resume/Expert 目标，不连接任何外部服务。"""

    return {
        "RESUME_SOURCE_DB_HOST": "source.mysql.local",
        "RESUME_SOURCE_DB_PORT": "3306",
        "RESUME_SOURCE_DB_NAME": "shared_source",
        "RESUME_SOURCE_DB_USER": "resume_reader",
        "RESUME_SOURCE_DB_PASSWORD": "resume-source-secret",
        "RESUME_INDEX_DB_HOST": "127.0.0.1",
        "RESUME_INDEX_DB_PORT": "3307",
        "RESUME_INDEX_DB_NAME": "resume_index",
        "RESUME_INDEX_DB_USER": "resume_writer",
        "RESUME_INDEX_DB_PASSWORD": "resume-index-secret",
        "RESUME_MILVUS_COLLECTION_NAME": "resume_candidates_v15_contract",
        # 以下 Expert 变量只提供隔离参照；Resume 不读取 Expert 凭据。
        "INDEX_DB_HOST": "127.0.0.1",
        "INDEX_DB_PORT": "3306",
        "INDEX_DB_NAME": "expert_index",
        "MILVUS_COLLECTION_NAME": "expert_candidates_v15_contract",
        # 即使通用变量存在，Resume 也不得把它们当作自己的回退配置。
        "SOURCE_DB_HOST": "expert-source.mysql.local",
        "SOURCE_DB_PORT": "3306",
        "SOURCE_DB_NAME": "shared_source",
        "SOURCE_DB_USER": "expert_reader",
        "SOURCE_DB_PASSWORD": "expert-source-secret",
    }


def test_from_env_builds_explicit_resume_roles_and_targets() -> None:
    settings = ResumeSettings.from_env(_environment())

    assert settings.source_db.role is DatabaseRole.SOURCE
    assert settings.source_db.host == "source.mysql.local"
    assert settings.source_db.database == "shared_source"
    assert settings.index_db.role is DatabaseRole.INDEX
    assert settings.index_db.database == "resume_index"
    assert settings.milvus_collection_name == "resume_candidates_v15_contract"


@pytest.mark.parametrize(
    "missing_name",
    [
        "RESUME_SOURCE_DB_HOST",
        "RESUME_SOURCE_DB_PORT",
        "RESUME_SOURCE_DB_NAME",
        "RESUME_SOURCE_DB_USER",
        "RESUME_SOURCE_DB_PASSWORD",
        "RESUME_INDEX_DB_HOST",
        "RESUME_INDEX_DB_PORT",
        "RESUME_INDEX_DB_NAME",
        "RESUME_INDEX_DB_USER",
        "RESUME_INDEX_DB_PASSWORD",
        "RESUME_MILVUS_COLLECTION_NAME",
    ],
)
def test_resume_variables_never_fall_back_to_expert_values(missing_name: str) -> None:
    environment = _environment()
    environment.pop(missing_name)

    with pytest.raises(RuntimeError, match=missing_name):
        ResumeSettings.from_env(environment)


@pytest.mark.parametrize(
    "missing_name",
    [
        "SOURCE_DB_HOST",
        "SOURCE_DB_PORT",
        "SOURCE_DB_NAME",
        "INDEX_DB_HOST",
        "INDEX_DB_PORT",
        "INDEX_DB_NAME",
    ],
)
def test_missing_expert_database_reference_fails_closed(missing_name: str) -> None:
    environment = _environment()
    environment.pop(missing_name)

    with pytest.raises(RuntimeError, match=missing_name):
        ResumeSettings.from_env(environment)


def test_resume_source_may_share_expert_read_only_source_schema() -> None:
    environment = _environment()
    environment["RESUME_SOURCE_DB_HOST"] = environment["SOURCE_DB_HOST"]
    environment["RESUME_SOURCE_DB_NAME"] = environment["SOURCE_DB_NAME"]

    settings = ResumeSettings.from_env(environment)

    assert settings.source_db.database == "shared_source"


def test_resume_index_rejects_resume_source_schema() -> None:
    environment = _environment()
    environment["RESUME_INDEX_DB_HOST"] = environment["RESUME_SOURCE_DB_HOST"]
    environment["RESUME_INDEX_DB_PORT"] = environment["RESUME_SOURCE_DB_PORT"]
    environment["RESUME_INDEX_DB_NAME"] = environment["RESUME_SOURCE_DB_NAME"]

    with pytest.raises(DatabaseBoundaryError, match="Resume Source"):
        ResumeSettings.from_env(environment)


def test_resume_index_rejects_expert_index_schema_with_localhost_alias() -> None:
    environment = _environment()
    environment["RESUME_INDEX_DB_HOST"] = "localhost"
    environment["RESUME_INDEX_DB_PORT"] = environment["INDEX_DB_PORT"]
    environment["RESUME_INDEX_DB_NAME"] = environment["INDEX_DB_NAME"].upper()

    with pytest.raises(DatabaseBoundaryError, match="Expert Index"):
        ResumeSettings.from_env(environment)


def test_resume_index_rejects_expert_source_schema() -> None:
    environment = _environment()
    environment["RESUME_INDEX_DB_HOST"] = environment["SOURCE_DB_HOST"]
    environment["RESUME_INDEX_DB_PORT"] = environment["SOURCE_DB_PORT"]
    environment["RESUME_INDEX_DB_NAME"] = environment["SOURCE_DB_NAME"]

    with pytest.raises(DatabaseBoundaryError, match="Expert Source"):
        ResumeSettings.from_env(environment)


def test_resume_collection_rejects_expert_collection_case_insensitively() -> None:
    environment = _environment()
    environment["RESUME_MILVUS_COLLECTION_NAME"] = environment["MILVUS_COLLECTION_NAME"].upper()

    with pytest.raises(DatabaseBoundaryError, match="Expert Collection"):
        ResumeSettings.from_env(environment)


def test_blank_expert_collection_uses_formal_expert_default_for_comparison() -> None:
    environment = _environment()
    environment["MILVUS_COLLECTION_NAME"] = ""
    environment["RESUME_MILVUS_COLLECTION_NAME"] = "expert_candidates_v15_contract"

    with pytest.raises(DatabaseBoundaryError, match="Expert Collection"):
        ResumeSettings.from_env(environment)
