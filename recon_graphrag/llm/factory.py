"""LLM factory functions.

Convenience module to create LLM instances without needing to know the exact
import paths. Supports OpenAI, Azure OpenAI, Ollama, and OpenRouter.

Users can also pass their own LLMInterface implementations directly —
the factory is just convenience.
"""

from __future__ import annotations

from typing import Any

from recon_graphrag.llm.base import BaseLLM

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_llm(provider: str, **kwargs: Any) -> BaseLLM:
    """Create an LLM instance for a supported provider.

    Args:
        provider: One of "openai", "azure_openai", "ollama", "openrouter".
        **kwargs: Passed to the underlying LLM constructor.

    Returns:
        A BaseLLM (LLMInterface) instance.

    Example:
        llm = create_llm("openai", model_name="gpt-4o", api_key="sk-...")

        llm = create_llm("azure_openai", model_name="my-deployment",
                          api_key="...", api_version="2025-03-01-preview",
                          azure_endpoint="https://my-resource.openai.azure.com/")

        llm = create_llm("ollama", model_name="llama3")

        llm = create_llm("openrouter", model_name="anthropic/claude-sonnet",
                          api_key="sk-or-...")

        # Custom base_url for any OpenAI-compatible API
        llm = create_llm("openai", model_name="custom-model",
                          base_url="http://localhost:8000/v1", api_key="dummy")
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


# --- LLM constructors ---


def _create_openai_llm(**kwargs: Any) -> BaseLLM:
    from neo4j_graphrag.llm import OpenAILLM

    return OpenAILLM(**kwargs)


def _create_azure_openai_llm(**kwargs: Any) -> BaseLLM:
    from neo4j_graphrag.llm import AzureOpenAILLM

    return AzureOpenAILLM(**kwargs)


def _create_ollama_llm(**kwargs: Any) -> BaseLLM:
    from neo4j_graphrag.llm import OllamaLLM

    return OllamaLLM(**kwargs)


def _create_openrouter_llm(**kwargs: Any) -> BaseLLM:
    """Create OpenRouter LLM (OpenAI-compatible API).

    Sets base_url to OpenRouter by default, but allows override
    via explicit base_url kwarg.
    """
    from neo4j_graphrag.llm import OpenAILLM

    kwargs.setdefault("base_url", _OPENROUTER_BASE_URL)
    return OpenAILLM(**kwargs)
