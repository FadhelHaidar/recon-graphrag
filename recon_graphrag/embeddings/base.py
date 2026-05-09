"""Embedder base interface and dimension detection.

Re-exports neo4j_graphrag's Embedder so users can depend on
recon_graphrag.embeddings.BaseEmbedder without importing internal
neo4j-graphrag paths.
"""

from __future__ import annotations

from typing import Any, Optional

from neo4j_graphrag.embeddings import Embedder as BaseEmbedder

__all__ = ["BaseEmbedder", "ModelParamsEmbedder", "detect_embedding_dim"]


class ModelParamsEmbedder(BaseEmbedder):
    """Wraps any embedder to inject extra params into every embed call.

    The upstream neo4j_graphrag embedders accept per-call kwargs in
    ``embed_query(text, **kwargs)`` that get forwarded to the underlying
    API (e.g. ``dimensions``, ``encoding_format``, ``extra_body``).
    But there's no way to set these at construction time — they must be
    passed on each call.

    This wrapper stores ``model_params`` and merges them into every
    ``embed_query`` / ``async_embed_query`` call automatically.

    Args:
        embedder: The underlying embedder instance.
        model_params: Dict of extra kwargs to pass to every embed call.
    """

    def __init__(self, embedder: BaseEmbedder, model_params: dict[str, Any]):
        self._embedder = embedder
        self._model_params = model_params

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        merged = {**self._model_params, **kwargs}
        return self._embedder.embed_query(text, **merged)

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        merged = {**self._model_params, **kwargs}
        return self._embedder.embed_query(text, **merged)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embedder, name)


def detect_embedding_dim(embedder: BaseEmbedder) -> Optional[int]:
    """Auto-detect embedding dimension for local models.

    For sentence-transformers models, reads the dimension directly from
    the model object (no API call needed).

    Returns None for API-based embedders (OpenAI, OpenRouter, etc.) since
    detecting their dimension would require an API call. Callers should
    provide the dimension explicitly for those providers.

    Args:
        embedder: An embedder instance to inspect.

    Returns:
        The embedding dimension, or None if it cannot be detected without
        an API call.
    """
    inner = getattr(embedder, "_embedder", embedder)
    model = getattr(inner, "model", None)
    if model is not None:
        get_dim = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(get_dim):
            return get_dim()
    return None
