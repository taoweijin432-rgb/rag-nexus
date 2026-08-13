from __future__ import annotations

import json
from collections.abc import Callable
from urllib.error import HTTPError, URLError

import pytest

from rag_vector_store import (
    FORMAL_EMBEDDING_DIMENSION,
    FORMAL_EMBEDDING_MODEL,
    FORMAL_EMBEDDING_MODEL_ALIAS,
    FORMAL_EMBEDDING_MODEL_SPACE_ID,
    FORMAL_EMBEDDING_MODEL_VERSION,
    CompanyCloudEmbeddingError,
    CompanyCloudEmbeddingModel,
    CompanyCloudEmbeddingSettings,
)


class FakeResponse:
    """模拟 urllib 响应对象。"""

    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> object:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def make_opener(payload: object) -> Callable[..., FakeResponse]:
    """创建一个返回固定响应的假 HTTP opener。"""

    def opener(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(payload)

    return opener


def make_recording_opener(payload: object, requests: list[object]) -> Callable[..., FakeResponse]:
    """创建一个记录请求对象但不访问网络的假 opener。"""

    def opener(request: object, **kwargs: object) -> FakeResponse:
        requests.append(request)
        return FakeResponse(payload)

    return opener


def settings(*, expected_dimension: int | None = None) -> CompanyCloudEmbeddingSettings:
    """创建不包含真实密钥的测试配置。"""

    return CompanyCloudEmbeddingSettings(
        api_url="https://embedding.example.test/v1/embeddings",
        api_key="unit-test-key",
        expected_dimension=expected_dimension,
    )


def test_formal_embedding_defaults_identify_the_project_model_space() -> None:
    configured = CompanyCloudEmbeddingSettings(
        api_url="https://embedding.example.test/v1/embeddings",
        api_key="unit-test-key",
    )

    assert configured.model == FORMAL_EMBEDDING_MODEL
    assert configured.model_alias == FORMAL_EMBEDDING_MODEL_ALIAS
    assert configured.model_version == FORMAL_EMBEDDING_MODEL_VERSION
    assert configured.expected_dimension == FORMAL_EMBEDDING_DIMENSION

    model = CompanyCloudEmbeddingModel(configured)
    assert model.model_space_id == FORMAL_EMBEDDING_MODEL_SPACE_ID


def test_formal_embedding_rejects_placeholder_model_version() -> None:
    with pytest.raises(ValueError, match="dev-unknown"):
        CompanyCloudEmbeddingSettings(
            api_url="https://embedding.example.test/v1/embeddings",
            api_key="unit-test-key",
            model_version="dev-unknown",
        )


def test_company_cloud_embedding_supports_batch_and_discovers_dimension() -> None:
    model = CompanyCloudEmbeddingModel(
        settings(),
        opener=make_opener(
            {
                "data": [
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                ]
            }
        ),
    )

    embeddings = model.embed(["第一段测试文本", "第二段测试文本"])

    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert model.dimension == 3
    assert model.model_alias == FORMAL_EMBEDDING_MODEL_ALIAS


def test_company_cloud_embedding_splits_large_input_and_preserves_order() -> None:
    requests: list[object] = []

    def opener(request: object, **kwargs: object) -> FakeResponse:
        requests.append(request)
        payload = json.loads(request.data.decode("utf-8"))
        data = [
            {
                "index": index,
                "embedding": [float(text.rsplit("-", 1)[1])],
            }
            for index, text in reversed(list(enumerate(payload["input"])))
        ]
        return FakeResponse({"data": data})

    model = CompanyCloudEmbeddingModel(
        CompanyCloudEmbeddingSettings(
            api_url="https://embedding.example.test/v1/embeddings",
            api_key="unit-test-key",
            expected_dimension=1,
        ),
        opener=opener,
    )

    embeddings = model.embed([f"chunk-{index}" for index in range(25)])

    assert [len(json.loads(request.data.decode("utf-8"))["input"]) for request in requests] == [
        10,
        10,
        5,
    ]
    assert embeddings == [[float(index)] for index in range(25)]


def test_company_cloud_embedding_fails_whole_call_when_a_slice_fails() -> None:
    requests: list[object] = []

    def opener(request: object, **kwargs: object) -> FakeResponse:
        requests.append(request)
        if len(requests) == 2:
            raise URLError("temporary network failure")
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "data": [
                    {"index": index, "embedding": [0.1]}
                    for index in range(len(payload["input"]))
                ]
            }
        )

    model = CompanyCloudEmbeddingModel(
        CompanyCloudEmbeddingSettings(
            api_url="https://embedding.example.test/v1/embeddings",
            api_key="unit-test-key",
            expected_dimension=1,
        ),
        opener=opener,
    )

    with pytest.raises(CompanyCloudEmbeddingError, match="连接或请求超时"):
        model.embed([f"chunk-{index}" for index in range(25)])

    assert len(requests) == 2


def test_company_cloud_embedding_builds_the_agreed_http_request() -> None:
    requests: list[object] = []
    model = CompanyCloudEmbeddingModel(
        settings(),
        opener=make_recording_opener(
            {"data": [{"embedding": [0.1, 0.2]}]},
            requests,
        ),
    )

    model.embed(["测试文本"])

    request = requests[0]
    assert request.get_method() == "POST"
    assert request.full_url == settings().api_url
    assert request.get_header("Authorization") == "Bearer unit-test-key"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "text-embedding-v4",
        "input": ["测试文本"],
    }


def test_company_cloud_embedding_rejects_http_error() -> None:
    def opener(*args: object, **kwargs: object) -> object:
        raise HTTPError(
            "https://embedding.example.test/v1/embeddings",
            401,
            "Unauthorized",
            hdrs=None,
            fp=None,
        )

    model = CompanyCloudEmbeddingModel(settings(), opener=opener)

    with pytest.raises(CompanyCloudEmbeddingError, match="认证失败"):
        model.embed(["测试文本"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": []}, "返回数量与输入文本数量不一致"),
        ({"data": [{"embedding": []}]}, "空 embedding"),
        ({"data": [{"embedding": [0.1, 0.2]}]}, "返回数量与输入文本数量不一致"),
        (
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3]}]},
            "dimension 不一致",
        ),
        ({"data": [{"embedding": ["not-a-number"]}]}, "元素必须是数值"),
        ({"unexpected": []}, "缺少合法的 data 列表"),
    ],
)
def test_company_cloud_embedding_validates_response(
    payload: object,
    message: str,
) -> None:
    model = CompanyCloudEmbeddingModel(settings(), opener=make_opener(payload))

    with pytest.raises(CompanyCloudEmbeddingError, match=message):
        model.embed(["测试文本", "第二段文本"])


def test_company_cloud_embedding_reads_configuration_from_environment() -> None:
    model = CompanyCloudEmbeddingModel.from_env(
        {
            "EMBEDDING_API_URL": "https://embedding.example.test/v1/embeddings",
            "EMBEDDING_API_KEY": "unit-test-key",
            "EMBEDDING_MODEL": "text-embedding-v4",
            "EMBEDDING_MODEL_ALIAS": "company-text-v4",
            "EMBEDDING_MODEL_VERSION": "revision-a",
            "EMBEDDING_DIMENSION": "1",
            "EMBEDDING_TIMEOUT": "12.5",
        },
        opener=make_opener({"data": [{"embedding": [0.1]}]}),
    )

    assert model.settings.model == "text-embedding-v4"
    assert model.model_alias == "company-text-v4"
    assert model.model_version == "revision-a"
    assert model.dimension == 1
    assert model.settings.timeout == 12.5
    assert model.embed(["测试文本"]) == [[0.1]]


def test_company_cloud_embedding_validates_configured_dimension() -> None:
    model = CompanyCloudEmbeddingModel(
        CompanyCloudEmbeddingSettings(
            api_url="https://embedding.example.test/v1/embeddings",
            api_key="unit-test-key",
            expected_dimension=3,
        ),
        opener=make_opener({"data": [{"embedding": [0.1, 0.2]}]}),
    )

    assert model.dimension == 3
    with pytest.raises(CompanyCloudEmbeddingError, match="不一致的 dimension"):
        model.embed(["测试文本"])


def test_company_cloud_embedding_does_not_expose_api_key_in_settings_repr() -> None:
    configured = settings()

    assert "unit-test-key" not in repr(configured)
