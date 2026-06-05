"""LLM factory functions.

Convenience module to create LLM instances without depending on
``neo4j-graphrag`` provider wrappers. The public provider keys and common
constructor kwargs intentionally mirror the previous factory.
"""

from __future__ import annotations

from typing import Any, Optional

from recon_graphrag.llm.base import BaseLLM, LLMResponse, LLMUsage

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_llm(provider: str, **kwargs: Any) -> BaseLLM:
    """Create an LLM instance for a supported provider.

    Args:
        provider: One of "openai", "azure_openai", "ollama", "openrouter".
        **kwargs: Passed to the underlying provider adapter.
    """
    providers = {
        "openai": _create_openai_llm,
        "azure_openai": _create_azure_openai_llm,
        "ollama": _create_ollama_llm,
        "openrouter": _create_openrouter_llm,
    }
    if provider not in providers:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: {', '.join(providers.keys())}"
        )
    return providers[provider](**kwargs)


class OpenAIChatLLM:
    """OpenAI-compatible chat-completions adapter."""

    supports_structured_output = True

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        azure: bool = False,
        **kwargs: Any,
    ):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "Could not import openai Python client. "
                "Install it with `pip install openai`."
            ) from exc

        self.model_name = model_name
        self.model_params = model_params or {}
        self.openai = openai
        client_cls = openai.AzureOpenAI if azure else openai.OpenAI
        async_client_cls = openai.AsyncAzureOpenAI if azure else openai.AsyncOpenAI
        self.client = client_cls(**kwargs)
        self.async_client = async_client_cls(**kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        params = {**self.model_params, **kwargs}
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            **params,
        )
        return _openai_chat_response_to_llm_response(response)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        params = {**self.model_params, **kwargs}
        response = await self.async_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            **params,
        )
        return _openai_chat_response_to_llm_response(response)

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        async_close = getattr(self.async_client, "close", None)
        if callable(async_close):
            await async_close()


class OllamaLLM:
    """Ollama chat adapter."""

    supports_structured_output = False

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "Could not import ollama Python client. "
                "Install it with `pip install ollama`."
            ) from exc

        self.model_name = model_name
        self.model_params = model_params or {}
        self.client = ollama.Client(**kwargs)
        self.async_client = ollama.AsyncClient(**kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **{**self.model_params, **kwargs},
        )
        return _ollama_chat_response_to_llm_response(response)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        response = await self.async_client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **{**self.model_params, **kwargs},
        )
        return _ollama_chat_response_to_llm_response(response)


def _create_openai_llm(**kwargs: Any) -> BaseLLM:
    return OpenAIChatLLM(**kwargs)


def _create_azure_openai_llm(**kwargs: Any) -> BaseLLM:
    return OpenAIChatLLM(azure=True, **kwargs)


def _create_ollama_llm(**kwargs: Any) -> BaseLLM:
    return OllamaLLM(**kwargs)


def _create_openrouter_llm(**kwargs: Any) -> BaseLLM:
    kwargs.setdefault("base_url", _OPENROUTER_BASE_URL)
    return OpenAIChatLLM(**kwargs)


def _openai_chat_response_to_llm_response(response: Any) -> LLMResponse:
    content = response.choices[0].message.content or ""
    usage = None
    if getattr(response, "usage", None):
        usage = LLMUsage(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    return LLMResponse(content=content, usage=usage)


def _ollama_chat_response_to_llm_response(response: Any) -> LLMResponse:
    message = getattr(response, "message", None)
    if isinstance(response, dict):
        message = response.get("message", message)
    content = ""
    if isinstance(message, dict):
        content = message.get("content") or ""
    elif message is not None:
        content = getattr(message, "content", "") or ""

    request_tokens = _response_value(response, "prompt_eval_count")
    response_tokens = _response_value(response, "eval_count")
    total_tokens = None
    if request_tokens is not None and response_tokens is not None:
        total_tokens = request_tokens + response_tokens
    usage = (
        LLMUsage(
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
        )
        if request_tokens is not None or response_tokens is not None
        else None
    )
    return LLMResponse(content=content, usage=usage)


def _response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)
