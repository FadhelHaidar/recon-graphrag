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


def test_create_llm_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm("unknown_provider")


@pytest.mark.parametrize("reason", [None, "stop", "length", "content_filter"])
def test_openai_chat_response_preserves_finish_reason_and_usage(reason):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason=reason)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    result = _openai_chat_response_to_llm_response(response)
    assert result.content == "ok"
    assert result.finish_reason == reason
    assert result.usage is not None
    assert result.usage.total_tokens == 3


def test_openai_chat_refusal_is_fixed_status_not_raw_text():
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=None, refusal="sensitive refusal text"), finish_reason="stop",
    )], usage=None)
    result = _openai_chat_response_to_llm_response(response)
    assert result.content == ""
    assert result.finish_reason == "refusal"
    assert "sensitive" not in repr(result)


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("reason", [None, "stop", "length"])
def test_ollama_chat_response_preserves_done_reason(as_dict, reason):
    from recon_graphrag.llm.factory import _ollama_chat_response_to_llm_response

    response = {"message": {"content": "ok"}, "done_reason": reason,
                "prompt_eval_count": 1, "eval_count": 2}
    result = _ollama_chat_response_to_llm_response(response if as_dict else SimpleNamespace(**response))
    assert result.finish_reason == reason
    assert result.content == "ok"
    assert result.usage is not None
    assert result.usage.total_tokens == 3


@pytest.mark.parametrize("reason", [None, "end_turn", "max_tokens", "refusal"])
def test_anthropic_response_preserves_stop_reason(reason):
    from recon_graphrag.llm.factory import _anthropic_response_to_llm_response

    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], stop_reason=reason,
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    result = _anthropic_response_to_llm_response(response)
    assert result.finish_reason == reason
    assert result.content == "ok"
    assert result.usage is not None
    assert result.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_openai_adapter_and_usage_wrapper_forward_schema_and_completion(monkeypatch):
    from recon_graphrag.communities.reports import COMMUNITY_REPORT_RESPONSE_FORMAT
    from recon_graphrag.llm.factory import OpenAIChatLLM
    from recon_graphrag.observability import track_usage

    calls = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="length")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    def create(**kwargs):
        calls.append(kwargs)
        return response

    async def acreate(**kwargs):
        return create(**kwargs)

    adapter = object.__new__(OpenAIChatLLM)
    adapter.model_params = {"max_tokens": 100}
    adapter.model_name = "test-model"
    monkeypatch.setattr(adapter, "client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), raising=False)
    monkeypatch.setattr(adapter, "async_client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=acreate))), raising=False)
    llm = track_usage(adapter)
    assert getattr(llm, "supports_structured_output") is True
    sync_result = llm.invoke("prompt", response_format=COMMUNITY_REPORT_RESPONSE_FORMAT)
    async_result = await llm.ainvoke("prompt", response_format=COMMUNITY_REPORT_RESPONSE_FORMAT)
    for result in (sync_result, async_result):
        assert result.finish_reason == "length"
        assert result.usage is not None
        assert result.usage.total_tokens == 3
    assert calls == [{
        "messages": [{"role": "user", "content": "prompt"}], "model": "test-model",
        "max_tokens": 100, "response_format": COMMUNITY_REPORT_RESPONSE_FORMAT,
    }] * 2
