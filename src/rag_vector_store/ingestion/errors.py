"""VectorIngestionService 的公共异常层级。

调用方只需捕获 ``VectorIngestionError`` 或其子类，无需捕获 pymilvus / HTTP / SDK
底层异常。Facade 在边界统一把底层异常转换为本模块的公共异常，并只携带安全摘要，
不泄漏凭据、连接参数、chunk_text 或 embedding。

异常层级保持精简，不做过度细分。
"""

from __future__ import annotations

_MAX_MESSAGE_LENGTH = 200


def _normalize_message(message: object) -> str:
    """把消息规整为单行并截断，避免大对象（chunk_text / embedding）或换行泄漏。"""

    text = " ".join(str(message).split())
    if len(text) > _MAX_MESSAGE_LENGTH:
        return text[:_MAX_MESSAGE_LENGTH] + "..."
    return text


class VectorIngestionError(Exception):
    """rag_vector_store 向量能力公共异常基类。

    构造时自动把消息规整为单行并截断到安全长度。子类只扩展语义，不更改抛出方式。
    """

    def __init__(self, message: object = "") -> None:
        super().__init__(_normalize_message(message))


class VectorContractError(VectorIngestionError):
    """输入 Contract 不合法（校验失败）。"""


class VectorEmbeddingError(VectorIngestionError):
    """Embedding 调用 / 结果校验失败。"""


class VectorMilvusError(VectorIngestionError):
    """Milvus 读写失败。"""


class VectorInconsistentStateError(VectorIngestionError):
    """metadata-only 安全门检测到不一致（Entity 缺失 / model_space / embedding_dim）。"""


__all__ = [
    "VectorContractError",
    "VectorEmbeddingError",
    "VectorInconsistentStateError",
    "VectorIngestionError",
    "VectorMilvusError",
]