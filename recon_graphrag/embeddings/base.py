"""Local embedding interface and dimension detection."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import math
from collections.abc import Awaitable, Callable
from typing import Any, Optional, Protocol, cast, runtime_checkable


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
            call = cast(Callable[..., Awaitable[list[float]]], async_embed)
            return await call(text, **merged)
        return await asyncio.to_thread(self._embedder.embed_query, text, **merged)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embedder, name)


class TruncatingEmbedder:
    """Truncate embedding vectors to ``target_dim`` and normalize their length.

    Prefix truncation can preserve useful rankings for Matryoshka-style models.
    Other models can lose retrieval quality. Use the same wrapper for graph
    building and queries so both operations produce compatible vectors.
    """

    def __init__(self, embedder: BaseEmbedder, target_dim: int):
        if target_dim <= 0:
            raise ValueError(f"target_dim must be a positive integer, got {target_dim}")
        self._embedder = embedder
        self.target_embedding_dim = target_dim

    def _truncate(self, vector: list[float]) -> list[float]:
        native = len(vector)
        if native < self.target_embedding_dim:
            raise ValueError(
                f"Embedder produced {native} dims; cannot truncate to "
                f"{self.target_embedding_dim}"
            )
        cut = list(vector[: self.target_embedding_dim])
        norm = math.sqrt(math.fsum(x * x for x in cut))
        if norm:
            cut = [x / norm for x in cut]
        return cut

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self._truncate(self._embedder.embed_query(text, **kwargs))

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        vector = await self._embedder.async_embed_query(text, **kwargs)
        return self._truncate(vector)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embedder, name)


def detect_embedding_dim(embedder: BaseEmbedder) -> Optional[int]:
    """Return the output dimension, or None when detection fails.

    Read TruncatingEmbedder's target dimension or a sentence-transformer model's
    encoder dimension when available. Otherwise, make one asynchronous embedding
    request in a worker thread. The probe can incur provider costs.
    Cache the probe result on the underlying instance when possible.
    An unavailable async method or a failed probe returns None.
    """
    dim = getattr(embedder, "target_embedding_dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim

    inner = getattr(embedder, "_embedder", embedder)
    cached = getattr(inner, "_recon_native_dim", None)
    if cached is not None:
        return cast(int | None, cached) or None

    model = getattr(inner, "model", None)
    if model is not None:
        get_dim = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(get_dim):
            call = cast(Callable[[], int | None], get_dim)
            dim = call()
            if dim:
                return dim

    async_embed = getattr(embedder, "async_embed_query", None)
    if not callable(async_embed):
        return None

    probe_call = cast(Callable[[str], Awaitable[list[float]]], async_embed)

    def _probe() -> int | None:
        return len(asyncio.run(probe_call("dimension probe")))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            dim = executor.submit(_probe).result(timeout=60)
    except Exception:
        dim = None
    with contextlib.suppress(Exception):
        cast(Any, inner)._recon_native_dim = dim
    return dim


__all__ = [
    "BaseEmbedder",
    "ModelParamsEmbedder",
    "TruncatingEmbedder",
    "detect_embedding_dim",
]
