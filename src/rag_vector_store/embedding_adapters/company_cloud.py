"""阿里云百炼 Embedding API 适配器。

本模块位于稳定的 ``EmbeddingModel`` Protocol 与阿里云百炼 HTTP 服务之间，只负责
把环境配置转换为请求、解析响应并做网络/向量防御性校验。Contract 校验、幂等、
Task State 和 Milvus 写入不在这里处理。类名保留 ``CompanyCloud`` 以兼容既有导出。

当前适配器按照阿里云百炼约定的 HTTP JSON 协议调用服务：

请求：
    POST {EMBEDDING_API_URL}
    Authorization: Bearer {EMBEDDING_API_KEY}
    {"model": "text-embedding-v4", "input": ["文本一", "文本二"]}

响应：
    {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

API 地址、模型名和密钥全部从环境变量读取。具体公司 API 如果使用不同的
请求或响应字段，只需要替换本适配器，不需要修改 EmbeddingModel、Contract
ingestion 或 Milvus Repository。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..embedding import EmbeddingModel
from ..model_space import (
    FORMAL_EMBEDDING_DIMENSION,
    FORMAL_EMBEDDING_MODEL,
    FORMAL_EMBEDDING_MODEL_ALIAS,
    FORMAL_EMBEDDING_MODEL_VERSION,
    build_model_space_id,
    is_placeholder_model_version,
)


class CompanyCloudEmbeddingError(RuntimeError):
    """阿里云百炼 Embedding 调用或响应校验失败。"""


DEFAULT_EMBEDDING_BATCH_SIZE = 10


@dataclass(frozen=True)
class CompanyCloudEmbeddingSettings:
    """阿里云百炼 Embedding API 的非敏感配置和密钥配置。

    谁使用：
        ``CompanyCloudEmbeddingModel.from_env`` 构造模型时读取；
        测试可显式构造本对象注入。

    来源：
        全部来自环境变量（``EMBEDDING_API_URL`` / ``EMBEDDING_API_KEY`` 必填，
        其余可选并默认正式常量）。密钥字段 ``api_key`` 设置 ``repr=False``，
        避免日志/异常中泄露。
    """

    api_url: str
    api_key: str = field(repr=False)
    model: str = FORMAL_EMBEDDING_MODEL
    model_alias: str = FORMAL_EMBEDDING_MODEL_ALIAS
    # 这是项目侧模型空间版本，不冒充供应商内部 revision。
    model_version: str | None = FORMAL_EMBEDDING_MODEL_VERSION
    expected_dimension: int | None = FORMAL_EMBEDDING_DIMENSION
    timeout: float = 60.0

    def __post_init__(self) -> None:
        """校验配置，避免把无效配置带到网络请求阶段。"""

        if not isinstance(self.api_url, str) or not self.api_url.strip():
            raise ValueError("EMBEDDING_API_URL 不能为空")
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("EMBEDDING_API_KEY 不能为空")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("EMBEDDING_MODEL 不能为空")
        if not isinstance(self.model_alias, str) or not self.model_alias.strip():
            raise ValueError("EMBEDDING_MODEL_ALIAS 不能为空")
        if self.model_version is not None and not self.model_version.strip():
            raise ValueError("EMBEDDING_MODEL_VERSION 不能为空字符串")
        if is_placeholder_model_version(self.model_version):
            raise ValueError("正式 Embedding 配置禁止使用 dev-unknown")
        if self.expected_dimension is not None and (
            isinstance(self.expected_dimension, bool)
            or not isinstance(self.expected_dimension, int)
            or self.expected_dimension < 1
        ):
            raise ValueError("EMBEDDING_DIMENSION 必须是正整数")
        if isinstance(self.timeout, bool) or self.timeout <= 0:
            raise ValueError("EMBEDDING_TIMEOUT 必须是正数")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> CompanyCloudEmbeddingSettings:
        """从环境变量创建云端 Embedding 配置。"""

        env = os.environ if environ is None else environ
        timeout_text = _optional(env, "EMBEDDING_TIMEOUT")
        dimension_text = _optional(env, "EMBEDDING_DIMENSION")
        return cls(
            api_url=_required(env, "EMBEDDING_API_URL"),
            api_key=_required(env, "EMBEDDING_API_KEY"),
            model=_get(env, "EMBEDDING_MODEL", FORMAL_EMBEDDING_MODEL),
            model_alias=_get(
                env,
                "EMBEDDING_MODEL_ALIAS",
                FORMAL_EMBEDDING_MODEL_ALIAS,
            ),
            model_version=(
                _optional(env, "EMBEDDING_MODEL_VERSION")
                or FORMAL_EMBEDDING_MODEL_VERSION
            ),
            expected_dimension=(
                FORMAL_EMBEDDING_DIMENSION
                if dimension_text is None
                else _parse_dimension(dimension_text)
            ),
            timeout=60.0 if timeout_text is None else _parse_timeout(timeout_text),
        )


class CompanyCloudEmbeddingModel(EmbeddingModel):
    """通过阿里云百炼兼容 HTTP API 批量生成 Embedding。

    调用方：
        ``ContractEmbeddingWorker``（通过 ``EmbeddingModel`` Protocol 抽象）。

    输入：
        一批 ``chunk_text``（经 Contract/Eligibility 过滤后的正式文本）。

    内部流程：
        - 按 Provider 限制微批（每次最多 ``DEFAULT_EMBEDDING_BATCH_SIZE`` = 10 条）；
        - 每批调用 ``_embed_batch()``（一次真正的 Provider 请求边界）；
        - 调用阿里云百炼 ``text-embedding-v4``；
        - 保持输入输出顺序；
        - 校验响应结构 / index / 数值有限性 / 维度。

    输出：
        与 ``texts`` 一一对应的 embedding 列表。

    设计边界：
        - 10 条微批限制只属于本 Adapter；Worker / Ingestion 不应感知。
        - 本适配器不做重试或文本清洗，调用方可以据异常决定是否重试任务；
        - Contract 校验、幂等、Task State 和 Milvus 写入不在这里处理。
    """

    def __init__(
        self,
        settings: CompanyCloudEmbeddingSettings,
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.settings = settings
        self._opener = opener
        self._dimension = settings.expected_dimension

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> CompanyCloudEmbeddingModel:
        """从环境变量创建阿里云百炼 Embedding 模型。"""

        return cls(CompanyCloudEmbeddingSettings.from_env(environ), opener=opener)

    @property
    def dimension(self) -> int:
        """返回已配置并会由真实响应再次校验的向量维度。"""

        if self._dimension is None:
            raise CompanyCloudEmbeddingError("尚未获得 Embedding dimension")
        return self._dimension

    @property
    def model_alias(self) -> str:
        """返回业务侧使用的模型别名。"""

        return self.settings.model_alias

    @property
    def model_version(self) -> str | None:
        """返回项目侧模型空间版本，不声称是供应商内部 revision。"""

        return self.settings.model_version

    @property
    def model_space_id(self) -> str:
        """返回当前模型、维度和版本组成的稳定模型空间标识。"""

        if self.model_version is None:
            raise CompanyCloudEmbeddingError("Embedding model version 未配置")
        try:
            return build_model_space_id(
                model_alias=self.model_alias,
                dimension=self.dimension,
                model_version=self.model_version,
            )
        except ValueError as exc:
            raise CompanyCloudEmbeddingError("Embedding model-space 配置无效") from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """调用云端 API，并返回与输入顺序一致的向量列表。

        每次百炼请求最多携带十条文本。API 若返回 ``index``，解析器会在片内
        复原顺序；所有片都成功后才返回合并结果，避免调用方拿到部分向量。
        """

        input_texts = list(texts)
        _validate_input_texts(input_texts)
        if not input_texts:
            # 空输入直接返回空列表：正式 Worker 不会在无 eligible 时调用本方法。
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(input_texts), DEFAULT_EMBEDDING_BATCH_SIZE):
            batch = input_texts[start : start + DEFAULT_EMBEDDING_BATCH_SIZE]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def _embed_batch(self, input_texts: Sequence[str]) -> list[list[float]]:
        """请求并校验一个不超过厂商限制的文本片。

        这是一次真正 Provider 请求边界：构造 HTTP 请求 → 发送 → 解析 
        ``data[].embedding`` → 校验数量/顺序/有限性/维度 → 锁定运行时维度。
        任一环节失败都会让本片失败（fail-closed），不会返回部分结果。
        """

        request = Request(
            self.settings.api_url,
            data=json.dumps(
                {
                    "model": self.settings.model,
                    "input": input_texts,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        # 请求体只包含 Worker 传入的 chunk_text；认证信息放在进程内存的 Header，
        # 不进入报告、异常或返回对象。
        response_body = self._request(request)
        embeddings = _parse_embeddings(response_body)
        _validate_embeddings(input_texts, embeddings)

        # 首次响应确定运行时维度，后续响应必须保持一致；Contract Worker 还会
        # 按正式预期维度进行最后一道校验。
        actual_dimension = len(embeddings[0])
        if self._dimension is not None and self._dimension != actual_dimension:
            raise CompanyCloudEmbeddingError(
                "同一个 Embedding 模型返回了不一致的 dimension"
            )
        self._dimension = actual_dimension
        return embeddings

    def _request(self, request: Request) -> bytes:
        """发送 HTTP 请求，并将底层网络异常转换为安全的业务异常。"""

        try:
            with self._opener(request, timeout=self.settings.timeout) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise CompanyCloudEmbeddingError("Embedding API 认证失败") from exc
            raise CompanyCloudEmbeddingError(
                f"Embedding API 返回 HTTP {exc.code}"
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise CompanyCloudEmbeddingError("Embedding API 连接或请求超时") from exc


def _parse_embeddings(response_body: bytes) -> list[list[float]]:
    """解析约定的 ``data[].embedding`` 响应结构。

    外部 API 响应不视为可信输入：缺字段、空向量、非数值、非有限值和重复 index
    都会阻止该批次继续，避免部分结果按错误顺序写入。
    """

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompanyCloudEmbeddingError("Embedding API 返回的 JSON 无法解析") from exc

    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise CompanyCloudEmbeddingError(
            "Embedding API 响应缺少合法的 data 列表"
        )

    data = payload["data"]
    embeddings: list[list[float]] = []
    indexed_embeddings: list[tuple[int, list[float]]] = []
    has_index = []

    for item in data:
        if not isinstance(item, Mapping) or not isinstance(item.get("embedding"), list):
            raise CompanyCloudEmbeddingError(
                "Embedding API 响应中的 embedding 必须是列表"
            )
        raw_embedding = item["embedding"]
        if not raw_embedding:
            raise CompanyCloudEmbeddingError("Embedding API 返回了空 embedding")

        vector: list[float] = []
        for value in raw_embedding:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise CompanyCloudEmbeddingError(
                    "Embedding API 返回的 embedding 元素必须是数值"
                )
            number = float(value)
            if not math.isfinite(number):
                raise CompanyCloudEmbeddingError(
                    "Embedding API 返回的 embedding 只能包含有限数值"
                )
            vector.append(number)

        raw_index = item.get("index")
        has_index.append(raw_index is not None)
        if raw_index is not None:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise CompanyCloudEmbeddingError(
                    "Embedding API 响应中的 index 必须是整数"
                )
            indexed_embeddings.append((raw_index, vector))
        embeddings.append(vector)

    if any(has_index) and not all(has_index):
        raise CompanyCloudEmbeddingError(
            "Embedding API 响应中的 index 字段不完整"
        )
    if all(has_index) and indexed_embeddings:
        indexes = [index for index, _ in indexed_embeddings]
        if len(indexes) != len(set(indexes)):
            raise CompanyCloudEmbeddingError(
                "Embedding API 响应中的 index 存在重复值"
            )
        expected_indexes = set(range(len(data)))
        if set(indexes) != expected_indexes:
            raise CompanyCloudEmbeddingError(
                "Embedding API 响应中的 index 必须覆盖当前片的连续范围"
            )
        embeddings_by_index = {
            index: vector for index, vector in indexed_embeddings
        }
        embeddings = [embeddings_by_index[index] for index in range(len(data))]

    return embeddings


def _validate_input_texts(texts: Sequence[str]) -> None:
    """校验发送给云端 API 的文本输入。"""

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding 输入文本必须是非空字符串")


def _validate_embeddings(texts: Sequence[str], embeddings: Sequence[Sequence[float]]) -> None:
    """校验返回数量和同一批次内的向量维度。"""

    if len(texts) != len(embeddings):
        raise CompanyCloudEmbeddingError(
            "Embedding API 返回数量与输入文本数量不一致"
        )
    if not embeddings:
        raise CompanyCloudEmbeddingError("Embedding API 未返回向量")

    dimension = len(embeddings[0])
    if dimension < 1:
        raise CompanyCloudEmbeddingError("Embedding API 返回了空 embedding")
    if any(len(embedding) != dimension for embedding in embeddings[1:]):
        raise CompanyCloudEmbeddingError(
            "同一个 Embedding batch 中的向量 dimension 不一致"
        )


def _required(environ: Mapping[str, str], name: str) -> str:
    """读取必填环境变量，但不在异常中暴露变量值。"""

    value = environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"缺少环境变量 {name}")
    return value.strip()


def _optional(environ: Mapping[str, str], name: str) -> str | None:
    """读取可选环境变量。"""

    value = environ.get(name)
    return None if value is None or not value.strip() else value.strip()


def _get(environ: Mapping[str, str], name: str, default: str) -> str:
    """读取带默认值的环境变量。"""

    return _optional(environ, name) or default


def _parse_timeout(value: str) -> float:
    """解析超时时间，并拒绝非正数。"""

    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError("EMBEDDING_TIMEOUT 必须是数字") from exc
    if timeout <= 0:
        raise ValueError("EMBEDDING_TIMEOUT 必须是正数")
    return timeout


def _parse_dimension(value: str) -> int:
    """解析预期向量维度，供首次调用前的 ingestion 初始化使用。"""

    try:
        dimension = int(value)
    except ValueError as exc:
        raise ValueError("EMBEDDING_DIMENSION 必须是正整数") from exc
    if dimension < 1:
        raise ValueError("EMBEDDING_DIMENSION 必须是正整数")
    return dimension
