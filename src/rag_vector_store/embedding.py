"""入库链路使用的 Embedding 最小协议。

本模块位于 Contract 校验与具体厂商适配器之间，只约定“接收一批文本并按顺序
返回向量”的接口。它不负责 HTTP、鉴权、重试、模型版本治理，也不负责决定哪些
Chunk 有资格向量化；这些职责分别属于 ``embedding_adapters`` 和 Contract
ingestion 编排层。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingModel(Protocol):
    """适配任意文本 Embedding 服务所需的最小接口。

    设计意图：
        ``ContractEmbeddingWorker`` 依赖的是这个抽象 ``EmbeddingModel``，而不是
        具体厂商 SDK；这样阿里云百炼、本地模型或团队内部服务可以在不修改
        Validator / Worker / Ingestion / Repository 的前提下替换。

    实现：
        - 正式实现为 ``embedding_adapters/company_cloud.py`` 的
          ``CompanyCloudEmbeddingModel``（阿里云百炼适配器）。

    调用方：
        ``ContractEmbeddingWorker.embed`` —— 只传入已通过 Contract/Eligibility
        的 ``chunk_text``，不会把 Metadata、Source 或 Versions 拼接进模型输入。

    接口：
        - ``embed(texts)``：接收 ``Sequence[str]``，返回与输入**顺序一一对应**
          的向量序列。
        - ``dimension``：每条输入文本对应的浮点数个数。

    失败契约：
        具体适配器负责把网络或厂商错误转成安全异常；Contract Worker 再统一
        校验向量的有限性和维度（fail-closed）。
    """

    @property
    def dimension(self) -> int:
        """每条输入文本对应的浮点数个数。"""

        ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """按输入顺序为每条文本返回一个向量。

        输入为空时的行为由具体适配器决定，但正式 Worker 不会在没有 eligible
        记录时调用该方法；返回数量或顺序不满足协议时，调用方必须拒绝该批次。
        """

        ...
