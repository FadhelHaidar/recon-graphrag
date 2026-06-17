import sys
from types import SimpleNamespace

import pytest

from recon_graphrag.embeddings.factory import create_embedder
from recon_graphrag.embeddings.factory import _openai_embedding


class _FakeOpenAIClient:
    last_kwargs = None

    def __init__(self, **kwargs):
        self.__class__.last_kwargs = kwargs
        self.embeddings = SimpleNamespace(create=lambda **_: None)


class _FakeAsyncOpenAIClient(_FakeOpenAIClient):
    pass


def test_azure_embedder_binds_model_as_deployment(monkeypatch):
    fake_openai = SimpleNamespace(
        OpenAI=_FakeOpenAIClient,
        AsyncOpenAI=_FakeAsyncOpenAIClient,
        AzureOpenAI=_FakeOpenAIClient,
        AsyncAzureOpenAI=_FakeAsyncOpenAIClient,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    embedder = create_embedder(
        "azure_openai",
        model="my-embedding-deployment",
        api_key="key",
        api_version="2025-03-01-preview",
        azure_endpoint="https://example.openai.azure.com/",
    )

    assert embedder.model == "my-embedding-deployment"
    assert _FakeOpenAIClient.last_kwargs["azure_deployment"] == (
        "my-embedding-deployment"
    )


def test_azure_embedder_preserves_explicit_deployment(monkeypatch):
    fake_openai = SimpleNamespace(
        OpenAI=_FakeOpenAIClient,
        AsyncOpenAI=_FakeAsyncOpenAIClient,
        AzureOpenAI=_FakeOpenAIClient,
        AsyncAzureOpenAI=_FakeAsyncOpenAIClient,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    create_embedder(
        "azure_openai",
        model="display-model",
        azure_deployment="my-embedding-deployment",
        api_key="key",
        api_version="2025-03-01-preview",
        azure_endpoint="https://example.openai.azure.com/",
    )

    assert _FakeOpenAIClient.last_kwargs["azure_deployment"] == (
        "my-embedding-deployment"
    )


def test_openai_embedding_response_without_data_has_clear_error():
    with pytest.raises(ValueError, match="did not include data"):
        _openai_embedding(SimpleNamespace(data=None))


def test_openai_embedding_response_with_provider_error_has_clear_error():
    response = SimpleNamespace(
        model_dump=lambda: {
            "data": None,
            "error": {
                "message": "HTTP 429: Model busy, retry later",
                "code": 429,
            },
        }
    )

    with pytest.raises(RuntimeError, match="provider returned error"):
        _openai_embedding(response)
