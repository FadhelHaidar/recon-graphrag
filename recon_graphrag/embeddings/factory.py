"""Embedder factory functions.

Creates local provider adapters without relying on ``neo4j-graphrag``
embedding wrappers.
"""

from __future__ import annotations

import asyncio
from typing import Any

from recon_graphrag.embeddings.base import BaseEmbedder, ModelParamsEmbedder
from recon_graphrag.providers._compat import (
    OpenAICompatibleProviderError,
    _error_value,
    _response_error,
    _response_payload,
    _safe_response_summary,
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_embedder(
    provider: str,
    *,
    model_params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseEmbedder:
    """Create an embedder instance for a supported provider."""
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


class OpenAIEmbeddings:
    """OpenAI-compatible embeddings adapter."""

    def __init__(self, model: str = "text-embedding-ada-002", azure: bool = False, **kwargs: Any):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "Could not import openai Python client. "
                "Install it with `pip install openai`."
            ) from exc

        self.model = model
        client_cls = openai.AzureOpenAI if azure else openai.OpenAI
        async_client_cls = openai.AsyncAzureOpenAI if azure else openai.AsyncOpenAI
        self.client = client_cls(**kwargs)
        self.async_client = async_client_cls(**kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.model, **kwargs)
        return _openai_embedding(response)

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        response = await self.async_client.embeddings.create(
            input=text,
            model=self.model,
            **kwargs,
        )
        return _openai_embedding(response)


class OllamaEmbeddings:
    """Ollama embeddings adapter."""

    def __init__(self, model: str, **kwargs: Any):
        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "Could not import ollama Python client. "
                "Install it with `pip install ollama`."
            ) from exc

        self.model = model
        self.client = ollama.Client(**kwargs)
        self.async_client = ollama.AsyncClient(**kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        response = self.client.embed(model=self.model, input=text, **kwargs)
        return _ollama_embedding(response)

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        response = await self.async_client.embed(model=self.model, input=text, **kwargs)
        return _ollama_embedding(response)


class SentenceTransformerEmbeddings:
    """SentenceTransformers embeddings adapter."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", *args: Any, **kwargs: Any):
        try:
            import numpy as np
            import sentence_transformers
            import torch
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self.np = np
        self.torch = torch
        self.model = sentence_transformers.SentenceTransformer(model, *args, **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        result = self.model.encode([text], **kwargs)
        if isinstance(result, (self.torch.Tensor, self.np.ndarray)):
            return result.flatten().tolist()
        if isinstance(result, list):
            values = []
            for item in result:
                if isinstance(item, self.torch.Tensor):
                    values.extend(item.flatten().tolist())
                elif isinstance(item, self.np.ndarray):
                    values.extend(item.flatten().tolist())
                elif isinstance(item, list):
                    values.extend(item)
            if values:
                return values
        raise ValueError("Unexpected return type from SentenceTransformer.encode")

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text, **kwargs)


def _create_openai_embedder(**kwargs: Any) -> BaseEmbedder:
    return OpenAIEmbeddings(**kwargs)


def _create_azure_openai_embedder(**kwargs: Any) -> BaseEmbedder:
    deployment_name = kwargs.get("azure_deployment") or kwargs.get("model")
    if deployment_name:
        kwargs.setdefault("azure_deployment", deployment_name)
    return OpenAIEmbeddings(azure=True, **kwargs)


def _create_ollama_embedder(**kwargs: Any) -> BaseEmbedder:
    return OllamaEmbeddings(**kwargs)


def _create_sentence_transformer_embedder(**kwargs: Any) -> BaseEmbedder:
    return SentenceTransformerEmbeddings(**kwargs)


def _create_openrouter_embedder(**kwargs: Any) -> BaseEmbedder:
    kwargs.setdefault("base_url", _OPENROUTER_BASE_URL)
    return OpenAIEmbeddings(**kwargs)


def _ollama_embedding(response: Any) -> list[float]:
    embeddings = getattr(response, "embeddings", None)
    if isinstance(response, dict):
        embeddings = response.get("embeddings", embeddings)
    if not embeddings:
        raise ValueError("Failed to retrieve embeddings.")
    embedding = embeddings[0]
    if not isinstance(embedding, list):
        embedding = list(embedding)
    return embedding


def _openai_embedding(response: Any) -> list[float]:
    error = _response_error(response)
    if error is not None:
        raise OpenAICompatibleProviderError.from_error(
            "embedding",
            error,
            response,
        )

    data = getattr(response, "data", None)
    if not data:
        raise ValueError(
            "OpenAI-compatible embedding response did not include data. "
            f"Response: {_safe_response_summary(response)}"
        )

    embedding = getattr(data[0], "embedding", None)
    if embedding is None:
        raise ValueError(
            "OpenAI-compatible embedding response item did not include an embedding. "
            f"Response: {_safe_response_summary(response)}"
        )
    return list(embedding)

