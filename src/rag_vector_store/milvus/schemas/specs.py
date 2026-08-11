"""Milvus Schema 字段的通用不可变定义。

``MilvusFieldSpec`` 是本地 Contract Schema、建表和兼容性比较共享的声明对象。
它不依赖 Repository，也不调用 PyMilvus，避免字段定义在多个执行路径中漂移。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MilvusFieldSpec:
    """一个 Collection 字段的稳定定义。

    ``max_length`` 按 Milvus VARCHAR bytes 约束，``dim`` 仅对向量字段有意义；
    ``summary`` 供只读 Schema 检查和安全报告使用。
    """

    name: str
    datatype: str
    nullable: bool = False
    primary_key: bool = False
    max_length: int | None = None
    dim: int | None = None

    def summary(self) -> dict[str, object]:
        """返回用于创建前展示和兼容性比较的字段摘要。"""

        return {
            "field": self.name,
            "datatype": self.datatype,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
            "max_length": self.max_length,
            "dim": self.dim,
        }
