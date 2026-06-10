"""Local embedding interface and dimension detection."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol for embedding providers accepted by Recon-GraphRAG."""

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed query text."""
        ...

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Asynchronously embed query text."""
        ...


class ModelParamsEmbedder:
    """Wrap any embedder and inject params into every embed call."""

    def __init__(self, embedder: BaseEmbedder, model_params: dict[str, Any]):
        self._embedder = embedder
        self._model_params = model_params

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        merged = {**self._model_params, **kwargs}
        return self._embedder.embed_query(text, **merged)

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        merged = {**self._model_params, **kwargs}
        async_embed = getattr(self._embedder, "async_embed_query", None)
        if callable(async_embed):
            return await async_embed(text, **merged)
        return await asyncio.to_thread(self._embedder.embed_query, text, **merged)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embedder, name)


def detect_embedding_dim(embedder: BaseEmbedder) -> Optional[int]:
    """Auto-detect embedding dimension for local sentence-transformer models."""
    inner = getattr(embedder, "_embedder", embedder)
    model = getattr(inner, "model", None)
    if model is not None:
        get_dim = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(get_dim):
            return get_dim()
    return None


__all__ = ["BaseEmbedder", "ModelParamsEmbedder", "detect_embedding_dim"]
