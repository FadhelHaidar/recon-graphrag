import sys
from types import SimpleNamespace

import pytest

from recon_graphrag.llm.factory import create_llm
from recon_graphrag.llm.factory import _openai_chat_response_to_llm_response


class _FakeOpenAIClient:
    last_kwargs = None

    def __init__(self, **kwargs):
        self.__class__.last_kwargs = kwargs
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="ok"))
                    ],
                    usage=None,
                )
            )
        )


class _FakeAsyncOpenAIClient(_FakeOpenAIClient):
    pass


def test_azure_llm_binds_model_name_as_deployment(monkeypatch):
    fake_openai = SimpleNamespace(
        OpenAI=_FakeOpenAIClient,
        AsyncOpenAI=_FakeAsyncOpenAIClient,
        AzureOpenAI=_FakeOpenAIClient,
        AsyncAzureOpenAI=_FakeAsyncOpenAIClient,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = create_llm(
        "azure_openai",
        model_name="my-chat-deployment",
        api_key="key",
        api_version="2025-03-01-preview",
        azure_endpoint="https://example.openai.azure.com/",
    )

    assert llm.model_name == "my-chat-deployment"
    assert _FakeOpenAIClient.last_kwargs["azure_deployment"] == "my-chat-deployment"


def test_azure_llm_preserves_explicit_deployment(monkeypatch):
    fake_openai = SimpleNamespace(
        OpenAI=_FakeOpenAIClient,
        AsyncOpenAI=_FakeAsyncOpenAIClient,
        AzureOpenAI=_FakeOpenAIClient,
        AsyncAzureOpenAI=_FakeAsyncOpenAIClient,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    create_llm(
        "azure_openai",
        model_name="display-model",
        azure_deployment="my-chat-deployment",
        api_key="key",
        api_version="2025-03-01-preview",
        azure_endpoint="https://example.openai.azure.com/",
    )

    assert _FakeOpenAIClient.last_kwargs["azure_deployment"] == "my-chat-deployment"


def test_openai_chat_response_without_choices_has_clear_error():
    with pytest.raises(ValueError, match="did not include choices"):
        _openai_chat_response_to_llm_response(SimpleNamespace(choices=None))


def test_openai_chat_response_with_provider_error_has_clear_error():
    response = SimpleNamespace(
        model_dump=lambda: {
            "choices": None,
            "error": {
                "message": "HTTP 429: Model busy, retry later",
                "code": 429,
            },
        }
    )

    with pytest.raises(RuntimeError, match="provider returned error"):
        _openai_chat_response_to_llm_response(response)
