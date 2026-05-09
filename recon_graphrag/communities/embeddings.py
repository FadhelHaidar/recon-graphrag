"""Vector embedding generation for communities and entities.

Generates embeddings for community summaries and entity descriptions,
then upserts them into the graph for semantic retrieval.
"""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.embeddings import Embedder

from recon_graphrag.graph.base import GraphStore


class CommunityEmbedder:
    """Generate and store vector embeddings for communities and entities."""

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: Embedder,
    ):
        self.graph_store = graph_store
        self.embedder = embedder

    async def embed_communities(self, level: int = 0):
        """Generate embeddings for all community summaries at a given level.

        Reads Community nodes with summaries but without embeddings,
        generates embeddings via the embedder, and batch-upserts them.
        """
        communities = self._get_communities_without_embeddings(level)
        if not communities:
            print(f"All communities at level {level} already have embeddings.")
            return

        ids, embeddings = [], []
        for comm in communities:
            try:
                embedding = await self.embedder.async_embed_query(comm["summary"])
                ids.append(comm["id"])
                embeddings.append(embedding)
                print(f"  Embedded community {comm['community_id']}")
            except Exception as e:
                print(f"  Error embedding community {comm['community_id']}: {e}")

        if ids:
            self.graph_store.upsert_vectors(ids, "embedding", embeddings)

    async def embed_entities(self):
        """Generate embeddings for entities without embeddings.

        Concatenates label + name + description as the text to embed,
        then batch-upserts all embeddings at once.
        """
        entities = self._get_entities_without_embeddings()
        if not entities:
            print("All entities already have embeddings.")
            return

        ids, embeddings = [], []
        for entity in entities:
            text = self._entity_to_text(entity)
            try:
                embedding = await self.embedder.async_embed_query(text)
                ids.append(entity["id"])
                embeddings.append(embedding)
            except Exception as e:
                name = entity.get("name", entity.get("description", ""))
                print(f"  Error embedding entity '{name}': {e}")

        if ids:
            self.graph_store.upsert_vectors(ids, "embedding", embeddings)

    def _get_communities_without_embeddings(self, level: int) -> list[dict]:
        query = """
        MATCH (c:Community {level: $level})
        WHERE c.summary IS NOT NULL AND c.embedding IS NULL
        RETURN elementId(c) AS id, c.id AS community_id, c.summary AS summary
        """
        return self.graph_store.execute_query(query, {"level": level})

    def _get_entities_without_embeddings(self) -> list[dict]:
        query = """
        MATCH (e:__Entity__)
        WHERE e.embedding IS NULL
        RETURN elementId(e) AS id, labels(e) AS labels,
               e.name AS name,
               CASE WHEN e.description IS NOT NULL THEN e.description ELSE '' END AS description
        LIMIT 500
        """
        return self.graph_store.execute_query(query)

    @staticmethod
    def _entity_to_text(entity: dict) -> str:
        labels = entity.get("labels", [])
        label = [lbl for lbl in labels if lbl != "__Entity__"]
        label = label[0] if label else "Entity"
        name = entity.get("name", "") or ""
        desc = entity.get("description", "") or ""
        parts = [f"{label}: {name}"]
        if desc:
            parts.append(desc)
        return " - ".join(parts)
