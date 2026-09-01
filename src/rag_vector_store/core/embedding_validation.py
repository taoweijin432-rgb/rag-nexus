"""领域无关的 Embedding 响应校验工具。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def validate_embedding_vector(
    raw_embedding: Sequence[float],
    *,
    expected_dimension: int | None = None,
    label: str = "Embedding",
) -> list[float]:
    """校验并规范化单条向量。

    Provider 解析层可以不传 ``expected_dimension``，只负责校验一条响应；
    Contract Worker 传入正式维度，执行最终的领域无关维度安全门。
    """

    if isinstance(raw_embedding, (str, bytes)) or not isinstance(
        raw_embedding, Sequence
    ):
        raise TypeError(f"{label} 返回的 embedding 必须是数值序列")
    if not raw_embedding:
        raise ValueError(f"{label} 返回了空 embedding")

    vector: list[float] = []
    for value in raw_embedding:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{label} 返回的 embedding 元素必须是数值")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} 返回的 embedding 只能包含有限数值")
        vector.append(number)

    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError(
            f"{label} dimension mismatch: expected {expected_dimension}, "
            f"got {len(vector)}"
        )
    return vector


__all__ = ["validate_embedding_vector"]
