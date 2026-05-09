"""Embedder factory functions.

Convenience module to create Embedder instances without needing to know the
exact import paths. Supports OpenAI, Azure OpenAI, Ollama, sentence-transformer,
and OpenRouter.

Users can also pass their own Embedder implementations directly —
the factory is just convenience.
"""

from __future__ import annotations

from typing import Any

from recon_graphrag.embeddings.base import BaseEmbedder, ModelParamsEmbedder

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_embedder(provider: str, *, model_params: dict[str, Any] | None = None, **kwargs: Any) -> BaseEmbedder:
    """Create an Embedder instance for a supported provider.

    Args:
        provider: One of "openai", "azure_openai", "ollama",
            "sentence-transformer", "openrouter".
        model_params: Extra kwargs to inject into every embed call.
            These are parameters that the underlying SDK expects at call time
            (e.g. ``dimensions``, ``encoding_format``, ``extra_body``) rather
            than at construction time. If provided, the embedder is
            automatically wrapped with :class:`ModelParamsEmbedder`.
        **kwargs: Passed to the underlying Embedder constructor.

    Returns:
        A BaseEmbedder instance, wrapped with ModelParamsEmbedder if
        model_params is provided.

    Example:
        embedder = create_embedder("openai", model="text-embedding-3-small",
                                    api_key="sk-...")

        embedder = create_embedder("openrouter",
                                    model="qwen/qwen3-embedding-8b",
                                    api_key="sk-or-...",
                                    model_params={"dimensions": 1536,
                                                  "encoding_format": "float"})

        embedder = create_embedder("sentence-transformer", model="all-MiniLM-L6-v2")
    """
    providers = {
        "openai": _create_openai_embedder,
        "azure_openai": _create_azure_openai_embedder,
        "ollama": _create_ollama_embedder,
        "sentence-transformer": _create_sentence_transformer_embedder,
        "openrouter": _create_openrouter_embedder,
    }
    if provider not in providers:
        raise ValueError(
            f"Unknown embedder provider: '{provider}'. "
            f"Supported: {', '.join(providers.keys())}"
        )
    embedder = providers[provider](**kwargs)
    if model_params:
        embedder = ModelParamsEmbedder(embedder, model_params)
    return embedder


# --- Embedder constructors ---


def _create_openai_embedder(**kwargs: Any) -> BaseEmbedder:
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    return OpenAIEmbeddings(**kwargs)


def _create_azure_openai_embedder(**kwargs: Any) -> BaseEmbedder:
    from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(**kwargs)


def _create_ollama_embedder(**kwargs: Any) -> BaseEmbedder:
    from neo4j_graphrag.embeddings import OllamaEmbeddings

    return OllamaEmbeddings(**kwargs)


def _create_sentence_transformer_embedder(**kwargs: Any) -> BaseEmbedder:
    from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(**kwargs)


def _create_openrouter_embedder(**kwargs: Any) -> BaseEmbedder:
    """Create OpenRouter embedder (OpenAI-compatible API).

    Sets base_url to OpenRouter by default, but allows override
    via explicit base_url kwarg.

    Defaults encoding_format="float" since OpenRouter defaults to base64,
    which causes "No embedding data received" errors.
    """
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    kwargs.setdefault("base_url", _OPENROUTER_BASE_URL)
    return OpenAIEmbeddings(**kwargs)
