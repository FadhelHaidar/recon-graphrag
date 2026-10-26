"""Vector embedding generation for source chunks.

Chunk vectors let local/DRIFT retrieval rank candidate source chunks by
similarity to the query instead of falling back to chunk-id order. Vectors are
stored L2-normalized, so ranking is a plain dot product and works identically on
Neo4j and Memgraph without a vector index or backend-specific function.
"""

from __future__ import annotations

import asyncio
import logging
import math

from tqdm.asyncio import tqdm_asyncio

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.entity_text import embedder_fingerprint
from recon_graphrag.graphdb.base import GraphStore

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 10


def _normalize(vector: list[float]) -> list[float]:
    """Scale to unit length so a dot product equals cosine similarity."""
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [float(value) / norm for value in vector]


class ChunkEmbedder:
    """Generate and store vector embeddings for source chunks."""

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: BaseEmbedder,
        graph_name: str = "entity-graph",
        concurrency: int = _DEFAULT_CONCURRENCY,
    ):
        self.graph_store = graph_store
        self.embedder = embedder
        self.graph_name = graph_name
        self.concurrency = concurrency
        self.model_fingerprint = embedder_fingerprint(embedder)

    async def embed_chunks(self, batch_size: int = 500):
        """Generate embeddings for chunks that have none for the current model.

        Chunk text is immutable once written, so only a missing vector or a
        changed embedding model triggers work. Embeds concurrently up to
        ``self.concurrency`` at a time, then batch-upserts all vectors at once.
        """
        total = 0
        semaphore = asyncio.Semaphore(self.concurrency)
        last_batch_ids: frozenset | None = None
        model_fingerprint = self.model_fingerprint

        while True:
            chunks = await asyncio.to_thread(
                self.graph_store.get_unembedded_chunks,
                graph_name=self.graph_name,
                limit=batch_size,
                model_fingerprint=model_fingerprint,
            )
            if not chunks:
                break

            # No-forward-progress guard: persistent embed failures, or an upsert
            # that does not clear the missing-vector condition, would otherwise
            # loop here forever.
            batch_ids = frozenset(chunk["id"] for chunk in chunks)
            if batch_ids == last_batch_ids:
                logger.warning(
                    "stopping: %s chunks could not be embedded (no progress); "
                    "leaving them unembedded",
                    len(chunks),
                )
                break
            last_batch_ids = batch_ids

            async def _embed(chunk: dict):
                async with semaphore:
                    try:
                        embedding = await self.embedder.async_embed_query(chunk["text"])
                        return chunk["id"], _normalize(embedding)
                    except Exception as embed_error:
                        return embed_error

            results = await tqdm_asyncio.gather(
                *[_embed(chunk) for chunk in chunks],
                desc="Embedding chunks",
                disable=None,
            )

            ids, embeddings = [], []
            for chunk, result in zip(chunks, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning(
                        "error embedding chunk id=%s type=%s",
                        chunk.get("chunk_id", chunk["id"]),
                        type(result).__name__,
                    )
                    logger.debug(
                        "error embedding chunk id=%s: %s",
                        chunk.get("chunk_id", chunk["id"]),
                        result,
                    )
                elif result is not None:
                    ids.append(result[0])
                    embeddings.append(result[1])

            if ids:
                await asyncio.to_thread(
                    self.graph_store.upsert_vectors,
                    ids,
                    "embedding",
                    embeddings,
                    model_fingerprint=model_fingerprint,
                )
                total += len(ids)

        if total == 0:
            logger.info("all chunks already have embeddings")
        else:
            logger.info("embedded %s chunks", total)
