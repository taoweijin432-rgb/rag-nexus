"""sql/014 扩展 error_stage 取值集的轻量静态测试。

不连接 MySQL、不执行任何 DDL，仅读取迁移文本断言范围与取值一致性。

断言：
1. SQL 只作用于 expert_index_sync_task；
2. 原有 9 个 error_stage 全部保留；
3. 4 个 P5 阶段（artifact/handoff/scope_consumer/receipt）全部存在；
4. 无 DROP TABLE、DELETE、TRUNCATE；
5. 不包含字段操作、索引操作或其他约束名，避免迁移范围扩大。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SQL_PATH = (
    Path(__file__).resolve().parents[1]
    / "sql"
    / "014_expand_expert_sync_task_error_stages.sql"
)

_ORIGINAL_STAGES = (
    "source",
    "parse",
    "pii",
    "chunk_builder",
    "generation",
    "validation",
    "embedding",
    "milvus",
    "checkpoint",
)
_P5_STAGES = ("artifact", "handoff", "scope_consumer", "receipt")
_CONSTRAINT_NAME = "chk_expert_sync_task_error_stage"


@pytest.fixture(scope="module")
def sql_text() -> str:
    return _SQL_PATH.read_text(encoding="utf-8")


def _value_in_sql(sql: str, value: str) -> bool:
    """迁移文本用双单引号转义字符串字面量（''value''）。"""
    return f"''{value}''" in sql


class TestExpandErrorStagesMigration:
    def test_targets_only_expert_index_sync_task(self, sql_text: str) -> None:
        alter_targets = set(re.findall(r"ALTER\s+TABLE\s+(\w+)", sql_text, re.IGNORECASE))
        assert alter_targets == {"expert_index_sync_task"}, (
            f"014 只应作用于 expert_index_sync_task，实际 ALTER TABLE 目标: {alter_targets}"
        )
        assert "expert_index_sync_task" in sql_text

    @pytest.mark.parametrize("stage", _ORIGINAL_STAGES)
    def test_original_error_stages_preserved(self, sql_text: str, stage: str) -> None:
        assert _value_in_sql(sql_text, stage), f"原有 error_stage '{stage}' 必须保留"

    @pytest.mark.parametrize("stage", _P5_STAGES)
    def test_p5_error_stages_present(self, sql_text: str, stage: str) -> None:
        assert _value_in_sql(sql_text, stage), f"P5 error_stage '{stage}' 必须存在"

    def test_no_destructive_statements(self, sql_text: str) -> None:
        upper = sql_text.upper()
        assert "DROP TABLE" not in upper, "禁止 DROP TABLE"
        assert "TRUNCATE" not in upper, "禁止 TRUNCATE"
        assert re.search(r"\bDELETE\b", sql_text) is None, "禁止 DELETE"

    def test_only_modifies_target_constraint(self, sql_text: str) -> None:
        add_constraints = re.findall(
            r"ADD\s+CONSTRAINT\s+(\w+)", sql_text, re.IGNORECASE
        )
        assert add_constraints == [_CONSTRAINT_NAME], (
            f"014 只应新增/重建 { _CONSTRAINT_NAME}，实际 ADD CONSTRAINT: {add_constraints}"
        )
        assert "DROP CHECK" in sql_text.upper()
        # 不应出现任何列或索引操作，避免范围扩大。
        assert re.search(r"ADD\s+COLUMN", sql_text, re.IGNORECASE) is None, "禁止新增列"
        assert re.search(r"MODIFY\s+COLUMN", sql_text, re.IGNORECASE) is None, "禁止修改列"
        assert re.search(r"DROP\s+COLUMN", sql_text, re.IGNORECASE) is None, "禁止删除列"
        assert re.search(r"\bADD\s+KEY\b", sql_text, re.IGNORECASE) is None, "禁止新增索引"
        assert re.search(r"\bADD\s+INDEX\b", sql_text, re.IGNORECASE) is None, "禁止新增索引"
        assert re.search(r"\bADD\s+UNIQUE\b", sql_text, re.IGNORECASE) is None, "禁止新增唯一键"

    def test_missing_constraint_is_recreated_on_rerun(self, sql_text: str) -> None:
        """DROP 成功而 ADD 中断后，下一次必须按约束缺失路径重新 ADD。"""

        assert "@expert_index_p5_constraint_exists" in sql_text
        assert "@expert_index_p5_constraint_current" in sql_text
        assert "@expert_index_p5_need_drop" in sql_text
        assert "@expert_index_p5_need_add" in sql_text
        assert "@expert_index_p5_constraint_current = 0" in sql_text

    def test_values_match_model_error_stages(self, sql_text: str) -> None:
        # 与 expert_index/models.py 的 SOURCE_SYNC_ERROR_STAGES 保持一致。
        from expert_index.models import SOURCE_SYNC_ERROR_STAGES

        expected = set(_ORIGINAL_STAGES) | set(_P5_STAGES)
        assert expected == SOURCE_SYNC_ERROR_STAGES, (
            "迁移取值集应与 SOURCE_SYNC_ERROR_STAGES 完全一致"
        )
        for value in expected:
            assert _value_in_sql(sql_text, value), f"迁移缺少 error_stage: {value}"
