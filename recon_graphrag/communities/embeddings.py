"""Vector embedding generation for communities and entities.

Generates embeddings for community summaries and entity descriptions,
then upserts them into the graph for semantic retrieval.
"""

from __future__ import annotations

from recon_graphrag.communities.detection import DEFAULT_GRAPH_NAME
from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graph.base import GraphStore


class CommunityEmbedder:
    """Generate and store vector embeddings for communities and entities."""

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: BaseEmbedder,
        graph_name: str = DEFAULT_GRAPH_NAME,
    ):
        self.graph_store = graph_store
        self.embedder = embedder
        self.graph_name = graph_name

    async def embed_communities(self, level: int = 0):
        """Generate embeddings for all community summaries at a given level.

        Reads Community nodes with summaries but without embeddings,
        generates embeddings via the embedder, and batch-upserts them.
        """
        communities = self._get_communities_without_embeddings(level)
        if not communities:
            print(f"  All communities at level {level} already have embeddings.")
            return

        ids, embeddings = [], []
        for comm in communities:
            try:
                text = self._community_to_text(comm)
                embedding = await self.embedder.async_embed_query(text)
                ids.append(comm["id"])
                embeddings.append(embedding)
                print(f"  Embedded community {comm['community_id']} level {comm['level']}")
            except Exception as e:
                print(f"  Error embedding community {comm['community_id']}: {e}")

        if ids:
            self.graph_store.upsert_vectors(ids, "embedding", embeddings)

    async def embed_entities(self, batch_size: int = 500):
        """Generate embeddings for entities without embeddings.

        Loops until all unembedded entities are processed.
        Concatenates label + name + description as the text to embed,
        then batch-upserts all embeddings at once.
        """
        total = 0

        while True:
            entities = self._get_entities_without_embeddings(limit=batch_size)
            if not entities:
                break

            ids, embeddings = [], []

            for entity in entities:
                try:
                    text = self._entity_to_text(entity)
                    embedding = await self.embedder.async_embed_query(text)
                    ids.append(entity["id"])
                    embeddings.append(embedding)
                except Exception as e:
                    name = self._value_to_text(
                        entity.get("name", entity.get("description", ""))
                    )
                    print(f"  Error embedding entity '{name}': {e}")

            if ids:
                self.graph_store.upsert_vectors(ids, "embedding", embeddings)
                total += len(ids)

        if total == 0:
            print("  All entities already have embeddings.")
        else:
            print(f"  Embedded {total} entities.")

    def _get_communities_without_embeddings(self, level: int) -> list[dict]:
        query = """
        MATCH (c:Community {graph_name: $graph_name, level: $level})
        WHERE c.summary IS NOT NULL AND c.embedding IS NULL
        RETURN elementId(c) AS id,
               c.id AS community_id,
               c.level AS level,
               c.summary AS summary
        """
        return self.graph_store.execute_query(
            query,
            {"graph_name": self.graph_name, "level": level},
        )

    def _get_entities_without_embeddings(self, limit: int = 500) -> list[dict]:
        query = """
        MATCH (e:__Entity__)
        WHERE e.embedding IS NULL
        RETURN elementId(e) AS id, labels(e) AS labels,
               e.name AS name,
               CASE WHEN e.description IS NOT NULL THEN e.description ELSE '' END AS description
        LIMIT $limit
        """
        return self.graph_store.execute_query(query, {"limit": limit})

    @staticmethod
    def _community_to_text(community: dict) -> str:
        return (
            f"Community level: {community['level']}\n"
            f"Community id: {community['community_id']}\n\n"
            f"Summary:\n{community['summary']}"
        )

    @staticmethod
    def _entity_to_text(entity: dict) -> str:
        labels = entity.get("labels", [])
        label = [lbl for lbl in labels if lbl != "__Entity__"]
        label = label[0] if label else "Entity"
        name = CommunityEmbedder._value_to_text(entity.get("name", ""))
        desc = CommunityEmbedder._value_to_text(entity.get("description", ""))
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
                f"{key}: {CommunityEmbedder._value_to_text(item)}"
                for key, item in value.items()
                if item is not None
            )
        if isinstance(value, (list, tuple, set)):
            return ", ".join(
                text
                for item in value
                if (text := CommunityEmbedder._value_to_text(item))
            )
        return str(value)
