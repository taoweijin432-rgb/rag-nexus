"""入库服务使用的 embedding 模型接口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingModel(Protocol):
    """适配任意文本 embedding 服务所需的最小接口。

    实现可以包装本地模型、托管 API 或团队内部服务。入库模块不会感知模型厂商。
    """

    @property
    def dimension(self) -> int:
        """每条输入文本对应的浮点数个数。"""

        ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """按输入顺序为每条文本返回一个向量。"""

        ...
