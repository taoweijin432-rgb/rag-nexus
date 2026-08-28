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
        client_id: str = "1001",
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
        if (
            not isinstance(client_id, str)
            or not client_id.isdecimal()
            or not 1 <= int(client_id) <= 32767
        ):
            raise ValueError("client_id 必须是 1 到 32767 的十进制整数字符串")
        self.host = host
        self.port = port
        self.destination = destination
        self.client_id = client_id
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
        # canal-python 0.4 的 endpoint 在 connect(host, port) 接收；先传入显式
        # endpoint，失败后才兼容无参或关键字连接实现，避免误连本机默认端口。
        try:
            method(host=self.host, port=self.port)
        except TypeError:
            try:
                method()
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
        # canal-python 0.4/Canal 1.1.8 需要先注册 ClientIdentity，再执行
        # rollback(0) 清理该 destination 的未确认 batch；该操作不会确认位点。
        # 对暴露原始 Connector 的版本使用兼容实现，避开旧 SDK 的 ACK 枚举错误。
        if _supports_raw_canal_protocol(self._client):
            _subscribe_v04(self._client, subscription, self.destination, self.client_id)
            return
        try:
            # canal-python 0.4 使用 bytes 参数，并在 subscribe 中保存 destination
            # 供后续 get/ack/rollback 使用；不能把订阅表达式误传为 client_id。
            method(
                client_id=self.client_id.encode("utf-8"),
                destination=self.destination.encode("utf-8"),
                filter=subscription.encode("utf-8"),
            )
        except TypeError:
            try:
                method(client_filter=subscription)
            except TypeError:
                try:
                    method(filter=subscription)
                except TypeError:
                    method(subscription)

    def get_without_ack(self, batch_size: int) -> object:
        """读取一个未确认 batch，兼容 SDK 的 get/get_without_ack 命名。"""

        # canal-python 0.4 的 ACK 分支调用了不存在的 ``PacketType.Ack``，导致
        # 服务器返回空批次或错误 ACK 时直接抛出 AttributeError。对该版本暴露的
        # 原始 Connector 走兼容实现，仍复用 SDK 的 TCP 连接和 protobuf 定义。
        if _supports_raw_canal_protocol(self._client):
            return _get_without_ack_v04(self._client, batch_size, self.client_id)
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
    client_id: str = "1001",
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
        client_id=client_id,
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


def _supports_raw_canal_protocol(client: Any) -> bool:
    """判断 SDK 是否暴露 canal-python 0.4 的原始 Connector 能力。"""

    connector = getattr(client, "connector", None)
    return callable(getattr(connector, "write_with_header", None)) and callable(
        getattr(connector, "read_next_packet", None)
    )


def _get_without_ack_v04(client: Any, batch_size: int, client_id: str) -> object:
    """绕过 canal-python 0.4 的错误 ACK 枚举，读取一个未确认 batch。"""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    CanalProtocol_pb2 = _load_canal_protocol()

    get = CanalProtocol_pb2.Get()
    get.client_id = getattr(client, "client_id", client_id.encode("utf-8"))
    get.destination = getattr(client, "destination", b"")
    get.auto_ack = False
    get.fetch_size = batch_size
    get.timeout = -1
    # Canal 1.1.8 Java Client 的 getWithoutAck 使用毫秒枚举值 2。
    get.unit = 2

    packet = CanalProtocol_pb2.Packet()
    packet.type = _packet_type(CanalProtocol_pb2, "GET")
    packet.body = get.SerializeToString()
    connector = client.connector
    connector.write_with_header(packet.SerializeToString())

    response = CanalProtocol_pb2.Packet()
    response.MergeFromString(connector.read_next_packet())
    response_type = int(response.type)
    if response_type == _packet_type(CanalProtocol_pb2, "MESSAGES"):
        messages = CanalProtocol_pb2.Messages()
        messages.MergeFromString(response.body)
        message: dict[str, object] = {"id": 0, "entries": []}
        if messages.batch_id > 0:
            message["id"] = messages.batch_id
            entries = message["entries"]
            assert isinstance(entries, list)
            for item in messages.messages:
                entries.append(_decode_entry_protocol(item))
        return message
    if response_type == _packet_type(CanalProtocol_pb2, "ACK"):
        ack = CanalProtocol_pb2.Ack()
        ack.MergeFromString(response.body)
        if ack.error_code > 0:
            raise RuntimeError(
                "Canal GET 返回错误 ACK: "
                f"error_code={ack.error_code}, error_message={ack.error_message}"
            )
        return {"id": 0, "entries": []}
    raise RuntimeError(f"Canal GET 返回未知 packet type: {response_type}")


def _subscribe_v04(client: Any, subscription: str, destination: str, client_id: str) -> None:
    """按 Canal 1.1.8 会话顺序订阅成功后执行 rollback(0)。"""

    CanalProtocol_pb2 = _load_canal_protocol()
    client_id = client_id.encode("utf-8")
    destination_bytes = destination.encode("utf-8")
    client.client_id = client_id
    client.destination = destination_bytes

    sub = CanalProtocol_pb2.Sub()
    sub.destination = destination_bytes
    sub.client_id = client_id
    sub.filter = subscription.encode("utf-8")

    packet = CanalProtocol_pb2.Packet()
    packet.type = _packet_type(CanalProtocol_pb2, "SUBSCRIPTION")
    packet.body = sub.SerializeToString()
    connector = client.connector
    connector.write_with_header(packet.SerializeToString())

    response = CanalProtocol_pb2.Packet()
    response.MergeFromString(connector.read_next_packet())
    if int(response.type) != _packet_type(CanalProtocol_pb2, "ACK"):
        raise RuntimeError(f"Canal SUBSCRIBE 返回未知 packet type: {int(response.type)}")
    ack = CanalProtocol_pb2.Ack()
    ack.MergeFromString(response.body)
    if ack.error_code > 0:
        raise RuntimeError(
            "Canal SUBSCRIBE 返回错误 ACK: "
            f"error_code={ack.error_code}, error_message={ack.error_message}"
        )

    rollback = getattr(client, "rollback", None)
    if not callable(rollback):
        raise CanalSdkUnavailableError("Canal SDK Client 缺少 rollback 方法")
    rollback(0)


def _decode_entry_protocol(payload: bytes) -> object:
    """将 GET 响应中的 Entry bytes 转成 SDK Entry 对象。"""

    EntryProtocol_pb2 = _load_entry_protocol()
    entry = EntryProtocol_pb2.Entry()
    entry.MergeFromString(payload)
    return entry


def _packet_type(protocol: Any, name: str) -> int:
    """兼容旧 protobuf 的模块常量和 EnumTypeWrapper 两种枚举访问方式。"""

    value = getattr(protocol, name, None)
    if value is None:
        value = getattr(protocol.PacketType, name)
    return int(value)


def _load_canal_protocol() -> Any:
    """加载 Canal 控制协议 protobuf，统一转换导入失败信息。"""

    try:
        from canal.protocol import CanalProtocol_pb2
    except (ImportError, TypeError, ValueError, AttributeError) as exc:
        raise CanalSdkUnavailableError("无法加载 Canal Protocol protobuf 定义") from exc
    return CanalProtocol_pb2


def _load_entry_protocol() -> Any:
    """加载 Canal Entry protobuf，统一转换导入失败信息。"""

    try:
        from canal.protocol import EntryProtocol_pb2
    except (ImportError, TypeError, ValueError, AttributeError) as exc:
        raise CanalSdkUnavailableError("无法加载 Canal Entry protobuf 定义") from exc
    return EntryProtocol_pb2


__all__ = [
    "CanalPythonConnector",
    "CanalSdkUnavailableError",
    "build_real_canal_client",
]
