"""领域 Contract Adapter 的最小协议。

通用向量化内核只通过本协议读取 Contract 的必要信息。这里故意不导入任何业务
Contract，避免未来新增领域时把某个领域的 Pydantic 类型带入核心层。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

InputT = TypeVar("InputT")
PreparedT = TypeVar("PreparedT")
EmbeddedT = TypeVar("EmbeddedT")


@runtime_checkable
class VectorContractAdapter(Protocol[InputT, PreparedT, EmbeddedT]):
    """把一个领域 Contract 接入向量化内核所需的协议。"""

    def is_input(self, value: object) -> bool:
        """判断值是否属于该领域的原始 Contract。"""

        ...

    def validate(self, value: InputT) -> PreparedT:
        """完成领域 Contract 校验并返回可消费视图。"""

        ...

    def vector_doc_id(self, value: object) -> str:
        """读取领域 Contract 的稳定向量主键。"""

        ...

    def chunk_text(self, value: PreparedT) -> str:
        """读取唯一允许发送给 Embedding 服务的正文。"""

        ...

    def skip_reason(self, value: PreparedT) -> str | None:
        """返回资格跳过原因；None 表示允许向量化。"""

        ...

    def attach_embedding(
        self,
        value: PreparedT,
        embedding: Sequence[float],
        *,
        embedding_model_alias: str,
        embedding_model_version: str,
        embedded_at: str,
    ) -> EmbeddedT:
        """把向量和模型空间信息回填为领域 Embedded Contract。"""

        ...

    def is_embedded(self, value: object) -> bool:
        """判断值是否属于该领域的 Embedded Contract。"""

        ...

    def is_prepared(self, value: object) -> bool:
        """判断值是否属于该领域的 Prepared Contract。"""

        ...

    def model_space(self, value: EmbeddedT) -> dict[str, object]:
        """读取模型空间字段，供领域 Repository 做一致性校验。"""

        ...

    def embedding(self, value: EmbeddedT) -> Sequence[float]:
        """读取已生成向量，供 Repository 锁定维度和写入前校验。"""

        ...

    def embedded_at(self, value: EmbeddedT) -> str:
        """读取原始向量生成时间，供 metadata-only 更新保持不变。"""

        ...

    def metadata(self, value: EmbeddedT) -> dict[str, object]:
        """返回可存储的业务 Metadata 映射。"""

        ...

    def with_active(self, value: EmbeddedT, is_active: bool) -> EmbeddedT:
        """只改变生命周期激活标记，保留原向量及其它字段。"""

        ...

    def content_hash(self, value: object) -> str:
        """读取正文 Hash。"""

        ...

    def metadata_hash(self, value: object) -> str:
        """读取 Metadata Hash。"""

        ...

    def source_hash(self, value: object) -> str | None:
        """读取来源 Hash；历史记录可能没有该字段。"""

        ...

    def text_and_identity_equal(self, left: object, right: object) -> bool:
        """判断 metadata-only 更新所需的正文和身份是否完全一致。"""

        ...

    def owner_id(self, value: object) -> Any:
        """返回领域 owner ID，用于隔离查询。"""

        ...

    def owner_field(self) -> str:
        """返回 flat Metadata 中的 owner 字段名。"""

        ...

    def id_prefix(self) -> str:
        """返回 vector_doc_id 的领域前缀。"""

        ...


class VectorConsumerContractAdapter(Protocol):
    """增量 Consumer 读取 Contract 差分字段所需的最小协议。"""

    def vector_doc_id(self, value: object) -> str:
        ...

    def content_hash(self, value: object) -> str:
        ...

    def metadata_hash(self, value: object) -> str:
        ...

    def source_hash(self, value: object) -> str | None:
        ...

    def skip_reason(self, value: object) -> str | None:
        ...


@runtime_checkable
class VectorStorageAdapter(VectorContractAdapter[InputT, PreparedT, EmbeddedT], Protocol):
    """包含 V15 物理行投影能力的领域 Adapter 扩展协议。"""

    def storage_projection(
        self,
        value: EmbeddedT,
    ) -> dict[str, Mapping[str, object]]:
        """返回 V15 flat metadata 所需的分区投影。"""

        ...

    def restore_embedded(
        self,
        row: Mapping[str, object],
    ) -> EmbeddedT:
        """从 V15 物理行还原领域 Embedded Contract。"""

        ...


__all__ = [
    "VectorConsumerContractAdapter",
    "VectorContractAdapter",
    "VectorStorageAdapter",
]
