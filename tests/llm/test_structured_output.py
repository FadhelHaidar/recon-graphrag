"""Native structured-output wire parameters and failure handling."""

import copy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from recon_graphrag.llm.factory import create_llm
from recon_graphrag.retrieval.search_drift import _drift_response_format


@pytest.fixture
def native_llm(monkeypatch):
    def build(provider, response=None, error=None, model_params=None):
        if response is None:
            response = SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="{}", refusal=None), finish_reason="stop",
                )],
                message=SimpleNamespace(content="{}"),
                content=[SimpleNamespace(type="text", text="{}")],
                usage=None, stop_reason="end_turn", done_reason="stop",
            )
        sync = Mock(return_value=response, side_effect=error)
        asynchronous = AsyncMock(return_value=response, side_effect=error)

        def client(call):
            return SimpleNamespace(
                chat=(call if provider == "ollama" else SimpleNamespace(
                    completions=SimpleNamespace(create=call))),
                messages=SimpleNamespace(create=call),
            )

        sdk = SimpleNamespace(
            OpenAI=lambda **kw: client(sync), AsyncOpenAI=lambda **kw: client(asynchronous),
            AzureOpenAI=lambda **kw: client(sync), AsyncAzureOpenAI=lambda **kw: client(asynchronous),
            Client=lambda **kw: client(sync), AsyncClient=lambda **kw: client(asynchronous),
            Anthropic=lambda **kw: client(sync), AsyncAnthropic=lambda **kw: client(asynchronous),
        )
        module = provider if provider in {"ollama", "anthropic"} else "openai"
        monkeypatch.setitem(sys.modules, module, sdk)
        return create_llm(provider, model_name="model", model_params=model_params), sync, asynchronous
    return build


@pytest.mark.parametrize("provider", ["openai", "azure_openai", "openrouter", "ollama", "anthropic"])
@pytest.mark.asyncio
async def test_native_schema_sync_async_forwarding(native_llm, provider):
    defaults = {"temperature": 0, "extra_body": {
        "unrelated": True, "provider": {"order": ["preferred"], "require_parameters": False},
        "output_config": {"effort": "low"},
    }}
    original = copy.deepcopy(defaults)
    llm, sync, asynchronous = native_llm(provider, model_params=defaults)
    response_format = _drift_response_format(primer=True)
    assert llm.supports_structured_output is True
    llm.invoke("prompt", response_format=response_format)
    await llm.ainvoke("prompt", response_format=response_format)
    assert sync.call_args.kwargs == asynchronous.call_args.kwargs
    params = sync.call_args.kwargs
    assert params["temperature"] == 0
    assert params["messages"] == [{"role": "user", "content": "prompt"}]
    assert params["model"] == "model"
    if provider == "ollama":
        assert "response_format" not in params
        assert params["format"] == response_format["json_schema"]["schema"]
    elif provider == "anthropic":
        assert "response_format" not in params
        assert params["extra_body"]["output_config"] == {
            "effort": "low", "format": {
                "type": "json_schema", "schema": response_format["json_schema"]["schema"],
            },
        }
        assert params["max_tokens"] == 4096
    else:
        assert params["response_format"] == response_format
    assert params["extra_body"]["unrelated"] is True
    expected_provider = {**original["extra_body"]["provider"]}
    if provider == "openrouter":
        expected_provider["require_parameters"] = True
    assert params["extra_body"]["provider"] == expected_provider
    assert defaults == original  # per-call routing/schema must not mutate defaults

    llm.invoke("prose")
    await llm.ainvoke("prose")
    assert sync.call_args.kwargs == asynchronous.call_args.kwargs
    assert "response_format" not in sync.call_args.kwargs
    assert "format" not in sync.call_args.kwargs
    assert sync.call_args.kwargs["extra_body"] == original["extra_body"]


@pytest.mark.parametrize("provider,field,value", [
    ("openai", "refusal", "Cannot comply"),
    ("openai", "finish_reason", "length"),
    ("azure_openai", "finish_reason", "content_filter"),
    ("openrouter", "finish_reason", "length"),
    ("anthropic", "stop_reason", "refusal"),
    ("anthropic", "stop_reason", "max_tokens"),
    ("ollama", "done_reason", "length"),
])
@pytest.mark.asyncio
async def test_structured_refusal_and_truncation(native_llm, provider, field, value):
    message = SimpleNamespace(content="partial", refusal=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    response = SimpleNamespace(
        choices=[choice], message=message, content=[], usage=None,
        stop_reason="end_turn", done_reason="stop",
    )
    setattr(message if field == "refusal" else choice if field == "finish_reason" else response, field, value)
    llm, sync, asynchronous = native_llm(provider, response=response)
    for call in (llm.invoke, llm.ainvoke):
        with pytest.raises(ValueError, match=value):
            if call == llm.invoke:
                call("p", response_format=_drift_response_format(primer=False))
            else:
                await call("p", response_format=_drift_response_format(primer=False))
    assert sync.call_count == asynchronous.call_count == 1
    # Unstructured behavior is unchanged.
    llm.invoke("prose")
    await llm.ainvoke("prose")


@pytest.mark.parametrize("provider", ["openai", "azure_openai", "openrouter", "ollama", "anthropic"])
@pytest.mark.asyncio
async def test_schema_rejection_preserves_error_without_downgrade(native_llm, provider):
    error = RuntimeError("provider rejected schema")
    llm, sync, asynchronous = native_llm(provider, error=error)
    with pytest.raises(RuntimeError) as caught:
        llm.invoke("p", response_format=_drift_response_format(primer=False))
    assert caught.value is error
    with pytest.raises(RuntimeError) as caught:
        await llm.ainvoke("p", response_format=_drift_response_format(primer=False))
    assert caught.value is error
    assert sync.call_count == asynchronous.call_count == 1


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
@pytest.mark.asyncio
async def test_anthropic_effort_only_does_not_enable_structured_response_checks(native_llm, stop_reason):
    response = SimpleNamespace(content=[], usage=None, stop_reason=stop_reason)
    llm, sync, asynchronous = native_llm("anthropic", response=response, model_params={
        "extra_body": {"output_config": {"effort": "low"}},
    })
    assert llm.invoke("prose").content == ""
    assert (await llm.ainvoke("prose")).content == ""
    assert sync.call_args.kwargs == asynchronous.call_args.kwargs
    assert sync.call_args.kwargs["extra_body"] == {"output_config": {"effort": "low"}}


@pytest.mark.asyncio
async def test_openai_custom_vllm_base_url_forwards_schema_without_router_preferences(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}", refusal=None), finish_reason="stop")],
        usage=None,
    )
    sync = Mock(return_value=response)
    asynchronous = AsyncMock(return_value=response)
    sync_constructor = Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=sync)),
    ))
    async_constructor = Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=asynchronous)),
    ))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=sync_constructor, AsyncOpenAI=async_constructor,
        AzureOpenAI=sync_constructor, AsyncAzureOpenAI=async_constructor,
    ))
    llm = create_llm("openai", model_name="qwen3.8-27b", base_url="http://vllm.test/v1", api_key="test")
    assert sync_constructor.call_args.kwargs["base_url"] == "http://vllm.test/v1"
    assert async_constructor.call_args == sync_constructor.call_args
    response_format = _drift_response_format(primer=False, mixed_action=True)
    llm.invoke("p", response_format=response_format)
    await llm.ainvoke("p", response_format=response_format)
    assert sync.call_args.kwargs == asynchronous.call_args.kwargs == {
        "messages": [{"role": "user", "content": "p"}],
        "model": "qwen3.8-27b", "response_format": response_format,
    }


@pytest.mark.asyncio
async def test_anthropic_schema_preserves_top_level_and_extra_body_output_config(native_llm):
    llm, sync, asynchronous = native_llm("anthropic", model_params={
        "output_config": {"effort": "low", "format": {"type": "old"}},
        "extra_body": {"output_config": {"unrelated": True}},
    })
    response_format = _drift_response_format(primer=False)
    llm.invoke("p", response_format=response_format)
    await llm.ainvoke("p", response_format=response_format)
    assert sync.call_args.kwargs == asynchronous.call_args.kwargs
    params = sync.call_args.kwargs
    assert "output_config" not in params  # works with old SDK signatures
    assert params["extra_body"]["output_config"] == {
        "effort": "low", "unrelated": True,
        "format": {"type": "json_schema", "schema": response_format["json_schema"]["schema"]},
    }
