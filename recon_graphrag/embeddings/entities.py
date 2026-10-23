"""Vector embedding generation for entities.

Generates embeddings for entity descriptions, then upserts them into
the graph for semantic retrieval.
"""

from __future__ import annotations

import asyncio
import logging

from tqdm.asyncio import tqdm_asyncio

from recon_graphrag.embeddings.base import BaseEmbedder
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

    async def embed_entities(self, batch_size: int = 500):
        """Generate embeddings for entities without embeddings.

        Loops until all unembedded entities are processed.
        Embeds entities concurrently up to ``self.concurrency`` at a time,
        then batch-upserts all embeddings at once.
        """
        total = 0
        semaphore = asyncio.Semaphore(self.concurrency)
        last_batch_ids: frozenset | None = None

        while True:
            entities = self.graph_store.get_unembedded_entities(
                graph_name=self.graph_name, limit=batch_size
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
                        text = self._entity_to_text(entity)
                        embedding = await self.embedder.async_embed_query(text)
                        return entity["id"], embedding
                    except Exception as e:
                        return e

            results = await tqdm_asyncio.gather(
                *[_embed(e) for e in entities],
                desc="Embedding entities",
                disable=None,
            )

            ids, embeddings = [], []
            for entity, result in zip(entities, results):
                if isinstance(result, Exception):
                    name = self._value_to_text(
                        entity.get("name", entity.get("description", ""))
                    )
                    logger.warning("error embedding entity '%s': %s", name, result)
                elif result is not None:
                    ids.append(result[0])
                    embeddings.append(result[1])

            if ids:
                self.graph_store.upsert_vectors(ids, "embedding", embeddings)
                total += len(ids)

        if total == 0:
            logger.info("all entities already have embeddings")
        else:
            logger.info("embedded %s entities", total)

    @staticmethod
    def _entity_to_text(entity: dict) -> str:
        labels = entity.get("labels", [])
        label = [lbl for lbl in labels if lbl != "__Entity__"]
        label = label[0] if label else "Entity"
        name = EntityEmbedder._value_to_text(entity.get("name", ""))
        desc = EntityEmbedder._value_to_text(entity.get("description", ""))
        parts = [f"{label}: {name}"]
        if desc:
            parts.append(desc)
        return " - ".join(parts)

    @staticmethod
    def _value_to_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return ", ".join(
                f"{key}: {EntityEmbedder._value_to_text(item)}"
                for key, item in value.items()
                if item is not None
            )
        if isinstance(value, (list, tuple, set)):
            return ", ".join(
                text
                for item in value
                if (text := EntityEmbedder._value_to_text(item))
            )
        return str(value)
