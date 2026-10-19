"""LLM factory functions.

Convenience module to create LLM instances without depending on
``neo4j-graphrag`` provider wrappers.
"""

from __future__ import annotations

from typing import Any, Optional

from recon_graphrag.llm.base import BaseLLM, LLMResponse, LLMUsage
from recon_graphrag.providers._compat import (
    OpenAICompatibleProviderError,
    _response_error,
    _safe_response_summary,
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_llm(provider: str, **kwargs: Any) -> BaseLLM:
    """Create an LLM instance for a supported provider.

    Args:
        provider: One of "openai", "azure_openai", "ollama", "openrouter",
            "anthropic".
        **kwargs: Passed to the underlying provider adapter.
    """
    providers = {
        "openai": OpenAIChatLLM,
        "azure_openai": _create_azure_openai_llm,
        "ollama": OllamaLLM,
        "openrouter": _create_openrouter_llm,
        "anthropic": AnthropicLLM,
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
    _require_structured_parameters = False

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
        self.azure = azure
        self.azure_endpoint = kwargs.get("azure_endpoint")
        self.openai = openai
        client_cls = openai.AzureOpenAI if azure else openai.OpenAI
        async_client_cls = openai.AsyncAzureOpenAI if azure else openai.AsyncOpenAI
        self.client = client_cls(**kwargs)
        self.async_client = async_client_cls(**kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        params = self._request_params(kwargs)
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                **params,
            )
        except Exception as exc:
            raise self._provider_error(exc) from exc
        return _openai_chat_response_to_llm_response(
            response, structured=_is_json_schema(params),
        )

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        params = self._request_params(kwargs)
        try:
            response = await self.async_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                **params,
            )
        except Exception as exc:
            raise self._provider_error(exc) from exc
        return _openai_chat_response_to_llm_response(
            response, structured=_is_json_schema(params),
        )

    def _request_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        params = {**self.model_params, **kwargs}
        if self._require_structured_parameters and _is_json_schema(params):
            extra_body = dict(params.get("extra_body") or {})
            extra_body["provider"] = {
                **extra_body.get("provider", {}), "require_parameters": True,
            }
            params["extra_body"] = extra_body
        return params

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        async_close = getattr(self.async_client, "close", None)
        if callable(async_close):
            await async_close()

    def _provider_error(self, exc: Exception) -> Exception:
        if not self.azure or not _is_azure_deployment_not_found(exc):
            return exc

        endpoint = self.azure_endpoint or "<missing azure_endpoint>"
        return RuntimeError(
            "Azure OpenAI deployment was not found. "
            f"Requested deployment/model_name='{self.model_name}' on endpoint "
            f"'{endpoint}'. For Azure OpenAI, model_name must be the Azure "
            "deployment name, not just the base model name. Check "
            "AZURE_OPENAI_LLM_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, and "
            "AZURE_OPENAI_API_VERSION."
        )


class OllamaLLM:
    """Ollama chat adapter."""

    supports_structured_output = True

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
        params = _native_schema_params({**self.model_params, **kwargs}, "ollama")
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        return _ollama_chat_response_to_llm_response(
            response, structured=isinstance(params.get("format"), dict),
        )

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        params = _native_schema_params({**self.model_params, **kwargs}, "ollama")
        response = await self.async_client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        return _ollama_chat_response_to_llm_response(
            response, structured=isinstance(params.get("format"), dict),
        )


def _create_azure_openai_llm(**kwargs: Any) -> BaseLLM:
    deployment_name = kwargs.get("azure_deployment") or kwargs.get("model_name")
    if deployment_name:
        kwargs.setdefault("azure_deployment", deployment_name)
    return OpenAIChatLLM(azure=True, **kwargs)


def _create_openrouter_llm(**kwargs: Any) -> BaseLLM:
    kwargs.setdefault("base_url", _OPENROUTER_BASE_URL)
    llm = OpenAIChatLLM(**kwargs)
    llm._require_structured_parameters = True
    return llm


class AnthropicLLM:
    """Anthropic Claude chat adapter."""

    supports_structured_output = True

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Could not import anthropic Python client. "
                "Install it with `pip install anthropic`."
            ) from exc

        self.model_name = model_name
        self.model_params = model_params or {}
        self.client = anthropic.Anthropic(**kwargs)
        self.async_client = anthropic.AsyncAnthropic(**kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        params = {**self.model_params, **kwargs}
        structured = _is_json_schema(params)
        params = _native_schema_params(params, "anthropic")
        max_tokens = params.pop("max_tokens", 4096)
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        return _anthropic_response_to_llm_response(response, structured=structured)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        params = {**self.model_params, **kwargs}
        structured = _is_json_schema(params)
        params = _native_schema_params(params, "anthropic")
        max_tokens = params.pop("max_tokens", 4096)
        response = await self.async_client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        return _anthropic_response_to_llm_response(response, structured=structured)

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        async_close = getattr(self.async_client, "close", None)
        if callable(async_close):
            await async_close()


def _openai_chat_response_to_llm_response(
    response: Any, *, structured: bool = False,
) -> LLMResponse:
    error = _response_error(response)
    if error is not None:
        raise OpenAICompatibleProviderError.from_error(
            "chat",
            error,
            response,
        )

    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(
            "OpenAI-compatible chat response did not include choices. "
            f"Response: {_safe_response_summary(response)}"
        )

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError(
            "OpenAI-compatible chat choice did not include a message. "
            f"Response: {_safe_response_summary(response)}"
        )

    if structured:
        refusal = getattr(message, "refusal", None)
        reason = getattr(choices[0], "finish_reason", None)
        if refusal or reason in {"length", "content_filter"}:
            raise ValueError(f"OpenAI structured response failed: {refusal or reason}")
    content = getattr(message, "content", None) or ""
    usage = None
    if getattr(response, "usage", None):
        usage = LLMUsage(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    return LLMResponse(content=content, usage=usage)



def _is_azure_deployment_not_found(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code", code)

    message = str(exc)
    return (
        status_code == 404
        and (
            code == "DeploymentNotFound"
            or "DeploymentNotFound" in message
            or "deployment for this resource does not exist" in message
        )
    )


def _ollama_chat_response_to_llm_response(
    response: Any, *, structured: bool = False,
) -> LLMResponse:
    if structured and _response_value(response, "done_reason") == "length":
        raise ValueError("Ollama structured response truncated: length")
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


def _anthropic_response_to_llm_response(
    response: Any, *, structured: bool = False,
) -> LLMResponse:
    reason = getattr(response, "stop_reason", None)
    if structured and reason in {"refusal", "max_tokens"}:
        raise ValueError(f"Anthropic structured response failed: {reason}")
    content = ""
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            content += block.text

    usage = None
    if getattr(response, "usage", None):
        usage = LLMUsage(
            request_tokens=response.usage.input_tokens,
            response_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
    return LLMResponse(content=content, usage=usage)


def _response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def _is_json_schema(params: dict[str, Any]) -> bool:
    response_format = params.get("response_format")
    return isinstance(response_format, dict) and response_format.get("type") == "json_schema"


def _native_schema_params(params: dict[str, Any], provider: str) -> dict[str, Any]:
    if _is_json_schema(params):
        schema = params.pop("response_format")["json_schema"]["schema"]
        if provider == "ollama":
            params["format"] = schema
        else:
            # extra_body supports native output_config even on older Anthropic SDKs.
            extra_body = dict(params.get("extra_body") or {})
            extra_body["output_config"] = {
                **params.pop("output_config", {}),
                **extra_body.get("output_config", {}),
                "format": {"type": "json_schema", "schema": schema},
            }
            params["extra_body"] = extra_body
    return params
