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
    entity_embedding_text,
    text_fingerprint,
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
        """Generate missing entity embeddings and refresh stale embeddings.

        Attempt each pending entity at most once per pass.
        Exclude attempted IDs so failures cannot block later batches.
        Keep failed entities pending for the next pass.
        Embed up to ``self.concurrency`` entities concurrently.
        Write each batch of successful vectors together.
        """
        total = 0
        semaphore = asyncio.Semaphore(self.concurrency)
        attempted_ids: set[str | int] = set()
        model_fingerprint = self.model_fingerprint

        while True:
            entities = await asyncio.to_thread(
                self.graph_store.get_unembedded_entities,
                graph_name=self.graph_name,
                limit=batch_size,
                model_fingerprint=model_fingerprint,
                exclude_ids=list(attempted_ids),
            )
            if not entities:
                break

            # Keep termination protection even if an adapter ignores exclusions.
            entities = [entity for entity in entities if entity["id"] not in attempted_ids]
            if not entities:
                logger.warning(
                    "stopping: %s entities could not be embedded (no progress); "
                    "leaving them unembedded",
                    len(attempted_ids),
                )
                break
            attempted_ids.update(entity["id"] for entity in entities)

            # Returns the exception on failure so tqdm_asyncio.gather (no
            # return_exceptions=) still lets the caller sort ok/failed results.
            async def _embed(entity: dict):
                async with semaphore:
                    try:
                        text = entity_embedding_text(
                            labels=entity.get("labels"),
                            name=entity.get("name", ""),
                            description=entity.get("description", ""),
                            entity_type=entity.get("type"),
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
                    # Entity ids are uuid5 URNs, so the warning stays traceable
                    # without putting names or provider payloads in logs.
                    logger.warning(
                        "error embedding entity id=%s type=%s",
                        entity["id"],
                        type(result).__name__,
                    )
                    logger.debug(
                        "error embedding entity id=%s name=%s: %s",
                        entity["id"],
                        value_to_text(entity.get("name", entity.get("description", ""))),
                        result,
                    )
                elif result is not None:
                    ids.append(result[0])
                    embeddings.append(result[1])
                    # Fingerprint the exact text sent to the provider; the
                    # entity row can change while the await is pending.
                    fingerprints.append(text_fingerprint(result[2]))

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

        if not attempted_ids:
            logger.info("all entities already have embeddings")
        else:
            logger.info("embedded %s of %s attempted entities; failed items remain pending", total, len(attempted_ids))
