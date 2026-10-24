"""Vector embedding generation for entities.

Generates embeddings for entity descriptions, then upserts them into
the graph for semantic retrieval.
"""

from __future__ import annotations

import asyncio
import logging

from tqdm.asyncio import tqdm_asyncio

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.entity_text import (
    embedder_fingerprint,
    entity_embedding_fingerprint,
    entity_embedding_text,
    value_to_text,
)
from recon_graphrag.graphdb.base import GraphStore

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 10


class EntityEmbedder:
    """Generate and store vector embeddings for entities."""

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

    async def embed_entities(self, batch_size: int = 500):
        """Generate embeddings for entities without embeddings.

        Loops until every entity has an embedding whose fingerprints match the
        text and model in use, so changed entities are re-embedded.
        Embeds entities concurrently up to ``self.concurrency`` at a time,
        then batch-upserts all embeddings at once.
        """
        total = 0
        semaphore = asyncio.Semaphore(self.concurrency)
        last_batch_ids: frozenset | None = None
        model_fingerprint = self.model_fingerprint

        while True:
            entities = await asyncio.to_thread(
                self.graph_store.get_unembedded_entities,
                graph_name=self.graph_name,
                limit=batch_size,
                model_fingerprint=model_fingerprint,
            )
            if not entities:
                break

            # No-forward-progress guard: if the store hands back the same batch
            # it just did, embedding it again cannot change the result — persistent
            # embed failures (or an upsert that doesn't clear `embedding IS NULL`)
            # would otherwise loop here forever.
            batch_ids = frozenset(e["id"] for e in entities)
            if batch_ids == last_batch_ids:
                logger.warning(
                    "stopping: %s entities could not be embedded (no progress); "
                    "leaving them unembedded",
                    len(entities),
                )
                break
            last_batch_ids = batch_ids

            # Returns the exception on failure so tqdm_asyncio.gather (no
            # return_exceptions=) still lets the caller sort ok/failed results.
            async def _embed(entity: dict):
                async with semaphore:
                    try:
                        text = entity_embedding_text(
                            labels=entity.get("labels"),
                            name=entity.get("name", ""),
                            description=entity.get("description", ""),
                        )
                        embedding = await self.embedder.async_embed_query(text)
                        return entity["id"], embedding, text
                    except Exception as embed_error:
                        return embed_error

            results = await tqdm_asyncio.gather(
                *[_embed(e) for e in entities],
                desc="Embedding entities",
                disable=None,
            )

            ids, embeddings, fingerprints = [], [], []
            for entity, result in zip(entities, results, strict=True):
                if isinstance(result, Exception):
                    name = value_to_text(
                        entity.get("name", entity.get("description", ""))
                    )
                    logger.warning("error embedding entity '%s': %s", name, result)
                elif result is not None:
                    ids.append(result[0])
                    embeddings.append(result[1])
                    fingerprints.append(
                        entity_embedding_fingerprint(
                            labels=entity.get("labels"),
                            name=entity.get("name", ""),
                            description=entity.get("description", ""),
                        )
                    )

            if ids:
                await asyncio.to_thread(
                    self.graph_store.upsert_vectors,
                    ids,
                    "embedding",
                    embeddings,
                    content_fingerprints=fingerprints,
                    model_fingerprint=model_fingerprint,
                )
                total += len(ids)

        if total == 0:
            logger.info("all entities already have embeddings")
        else:
            logger.info("embedded %s entities", total)
