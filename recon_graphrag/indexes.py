"""Vector and fulltext index creation + entity resolution.

Creates all required Neo4j indexes for semantic retrieval and runs
entity resolution to merge duplicate entities.
"""

from __future__ import annotations

from typing import Optional

from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)

from recon_graphrag.graph_store import GraphStore
from recon_graphrag.types import IndexConfig


class IndexManager:
    """Create indexes and resolve duplicate entities in the graph."""

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: Optional[Embedder] = None,
        embedding_dim: int = 1536,
        index_config: Optional[IndexConfig] = None,
    ):
        self.graph_store = graph_store
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.config = index_config or IndexConfig()

    def create_indexes(self):
        """Create all required vector and fulltext indexes."""
        self.graph_store.create_vector_index(
            name=self.config.chunk_vector_index,
            label=self.config.chunk_label,
            embedding_property="embedding",
            dimensions=self.embedding_dim,
        )
        self.graph_store.create_vector_index(
            name=self.config.entity_vector_index,
            label=self.config.entity_label,
            embedding_property="embedding",
            dimensions=self.embedding_dim,
        )
        self.graph_store.create_vector_index(
            name=self.config.community_vector_index,
            label=self.config.community_label,
            embedding_property="embedding",
            dimensions=self.embedding_dim,
        )
        self.graph_store.create_fulltext_index(
            name=self.config.entity_fulltext_index,
            label=self.config.entity_label,
            node_properties=["name"],
        )

    async def resolve_entities(self):
        """Run entity resolution to merge duplicate __Entity__ nodes.

        Uses SinglePropertyExactMatchResolver which merges entities with
        the same label and matching 'name' property. Requires APOC plugin.
        """
        resolver = SinglePropertyExactMatchResolver(
            driver=self.graph_store.driver,
            resolve_property="name",
            neo4j_database=getattr(self.graph_store, "_database", None),
        )
        result = await resolver.run()
        return result

    def verify(self):
        """Print current graph schema info: indexes, node counts, relationship counts."""
        print("\n=== Indexes ===")
        indexes = self.graph_store.execute_query("SHOW INDEXES")
        for idx in indexes:
            print(
                f"  {idx.get('name', '?')}: {idx.get('type', '?')} "
                f"on {idx.get('labelsOrTypes', '?')}.{idx.get('properties', '?')}"
            )

        print("\n=== Node Counts ===")
        counts = self.graph_store.execute_query(
            "MATCH (n) RETURN labels(n) AS labels, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in counts:
            print(f"  {rec['labels']}: {rec['count']}")

        print("\n=== Relationship Counts ===")
        rels = self.graph_store.execute_query(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in rels:
            print(f"  {rec['type']}: {rec['count']}")
