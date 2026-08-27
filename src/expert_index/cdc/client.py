"""真实 Canal SDK 的最小传输适配。

本模块不把第三方 SDK 类型泄漏到领域层；调用方只需提供 SDK Client 和已验证的
``CanalBatchDecoder``。SDK 作为可选依赖懒加载，未安装时在构造阶段明确报错。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .parser import CanalBatchDecoder, CanalClientAdapter, CanalConnectorPort
from .protocol import CANAL_DESTINATION, CANAL_SUBSCRIPTION, CanalClientPort


class CanalSdkUnavailableError(RuntimeError):
    """表示真实 Canal Python SDK 未安装或无法导入。"""


class CanalPythonConnector(CanalConnectorPort):
    """将常见 Canal Python Client API 适配为项目传输端口。

    ``canal-python`` 不同版本的连接方法签名略有差异，因此仅在连接阶段按公开
    参数名选择兼容调用；消息解码仍由显式注入的 decoder 负责。
    """

    def __init__(
        self,
        host: str,
        port: int,
        destination: str = CANAL_DESTINATION,
        username: str = "",
        password: str = "",
        *,
        sdk_client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        """初始化连接参数；不在构造阶段连接 Canal。"""

        if not isinstance(host, str) or not host.strip():
            raise ValueError("host 必须是非空字符串")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port 必须是 1 到 65535 的整数")
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError("destination 必须是非空字符串")
        self.host = host
        self.port = port
        self.destination = destination
        self.username = username
        self.password = password
        if sdk_client is not None and client_factory is not None:
            raise ValueError("sdk_client 与 client_factory 不能同时提供")
        if sdk_client is not None:
            self._client = sdk_client
        else:
            factory = client_factory or _load_default_client_factory()
            self._client = _instantiate_sdk_client(
                factory,
                host=self.host,
                port=self.port,
                destination=self.destination,
                username=self.username,
                password=self.password,
            )

    def connect(self) -> None:
        """使用当前 Canal endpoint 建立 SDK 连接。"""

        method = getattr(self._client, "connect", None)
        if not callable(method):
            raise CanalSdkUnavailableError("Canal SDK Client 缺少 connect 方法")
        # canal-python 通常在构造函数接收 endpoint，connect() 无参；保留参数化
        # 连接回退以兼容自定义 Client，但不把连接参数写入领域模型。
        try:
            method()
        except TypeError:
            try:
                method(
                    host=self.host,
                    port=self.port,
                    destination=self.destination,
                    username=self.username,
                    password=self.password,
                )
            except TypeError:
                method(self.host, self.port)

    def disconnect(self) -> None:
        """断开 SDK 连接。"""

        method = getattr(self._client, "disconnect", None)
        if callable(method):
            method()

    def subscribe(self, subscription: str) -> None:
        """注册表过滤表达式，不自动确认任何 batch。"""

        method = getattr(self._client, "subscribe", None)
        if not callable(method):
            raise CanalSdkUnavailableError("Canal SDK Client 缺少 subscribe 方法")
        try:
            method(subscription)
        except TypeError:
            try:
                method(client_filter=subscription)
            except TypeError:
                method(filter=subscription)

    def get_without_ack(self, batch_size: int) -> object:
        """读取一个未确认 batch，兼容 SDK 的 get/get_without_ack 命名。"""

        method = getattr(self._client, "get_without_ack", None)
        if not callable(method):
            method = getattr(self._client, "get", None)
        if not callable(method):
            raise CanalSdkUnavailableError("Canal SDK Client 缺少 get 方法")
        try:
            return method(batch_size)
        except TypeError:
            return method(batch_size=batch_size)

    def rollback(self, batch_id: int) -> None:
        """回滚指定 batch，保留 Canal 重投能力。"""

        method = getattr(self._client, "rollback", None)
        if not callable(method):
            raise CanalSdkUnavailableError("Canal SDK Client 缺少 rollback 方法")
        method(batch_id)

    def ack(self, batch_id: int) -> None:
        """确认已完成 batch；调用时机由上层业务编排决定。"""

        method = getattr(self._client, "ack", None)
        if not callable(method):
            raise CanalSdkUnavailableError("Canal SDK Client 缺少 ack 方法")
        method(batch_id)


def build_real_canal_client(
    *,
    host: str,
    port: int,
    destination: str = CANAL_DESTINATION,
    subscription: str = CANAL_SUBSCRIPTION,
    username: str = "",
    password: str = "",
    decoder: CanalBatchDecoder,
    sdk_client: Any | None = None,
) -> CanalClientPort:
    """构造领域层真实 Canal Client 端口。

    decoder 必须把 SDK Message 转换为无 SDK 的 :class:`CanalBatch`；这样可以根据
    实际安装的 SDK 版本实现解码，而不会在业务层猜测 RowData 结构。
    """

    if not callable(decoder):
        raise TypeError("decoder 必须是可调用对象")
    connector = CanalPythonConnector(
        host,
        port,
        destination,
        username,
        password,
        sdk_client=sdk_client,
    )
    return CanalClientAdapter(
        connector,
        decoder,
        destination=destination,
        subscription=subscription,
    )


def _load_default_client_factory() -> Callable[..., Any]:
    """懒加载 canal-python Client，避免普通 Poll 依赖该可选包。"""

    try:
        from canal.client import Client  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CanalSdkUnavailableError(
            "未安装 Canal Python SDK，请安装 canal-python 或显式注入 sdk_client"
        ) from exc
    return Client


def _instantiate_sdk_client(
    factory: Callable[..., Any],
    *,
    host: str,
    port: int,
    destination: str,
    username: str,
    password: str,
) -> Any:
    """兼容 canal-python Client 的构造参数差异。"""

    attempts = (
        {
            "host": host,
            "port": port,
            "destination": destination,
            "username": username,
            "password": password,
        },
        {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
        },
        {"host": host, "port": port},
    )
    for kwargs in attempts:
        try:
            return factory(**kwargs)
        except TypeError:
            continue
    try:
        return factory(host, port)
    except TypeError:
        return factory()


__all__ = [
    "CanalPythonConnector",
    "CanalSdkUnavailableError",
    "build_real_canal_client",
]
