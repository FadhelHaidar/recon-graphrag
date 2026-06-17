"""Vector and fulltext index creation for Memgraph.

Memgraph uses native vector and text indexes:

- Vector: CREATE VECTOR INDEX name ON :Label(prop) WITH CONFIG {...}
- Text:   CREATE TEXT INDEX name ON :Label(prop1, prop2)

These are not MAGE modules; they are built into the Memgraph engine.
"""

from __future__ import annotations

from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.memgraph.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)
from recon_graphrag.models.types import IndexConfig


class IndexManager:
    """Create indexes for the Memgraph backend."""

    def __init__(
        self,
        graph_store: GraphStore,
        embedder: Optional[BaseEmbedder] = None,
        embedding_dim: Optional[int] = None,
        index_config: Optional[IndexConfig] = None,
    ):
        self.graph_store = graph_store
        self.embedder = embedder
        self.config = index_config or IndexConfig()

        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        elif embedder is not None:
            from recon_graphrag.embeddings.base import detect_embedding_dim

            self.embedding_dim = detect_embedding_dim(embedder) or 1536
        else:
            self.embedding_dim = 1536

    def create_indexes(self):
        """Create all required vector and text indexes."""
        self._drop_indexes()
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
        self._create_constraints()

    def _drop_indexes(self):
        """Drop existing indexes so they can be recreated with updated settings."""
        for name in [
            self.config.chunk_vector_index,
            self.config.entity_vector_index,
            self.config.community_vector_index,
            self.config.entity_fulltext_index,
        ]:
            try:
                self.graph_store.execute_query(
                    f"DROP VECTOR INDEX {escape_cypher_identifier(name)}"
                )
            except Exception:
                pass
            try:
                self.graph_store.execute_query(
                    f"DROP TEXT INDEX {escape_cypher_identifier(name)}"
                )
            except Exception:
                pass

    def _create_constraints(self):
        """Create constraints required by the community hierarchy.

        Memgraph supports only single-property unique constraints, so we
        enforce the composite semantic (graph_name, level, id) via a `uid`
        property computed as "{graph_name}:{level}:{id}".
        """
        try:
            self.graph_store.execute_query(
                """
                CREATE CONSTRAINT ON (c:Community)
                ASSERT c.uid IS UNIQUE
                """
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                return
            print(f"  Warning: community uniqueness constraint failed: {e}")

    def verify(self):
        """Print current graph schema info: indexes, node counts, relationship counts."""
        print("\n=== Indexes ===")
        try:
            indexes = self.graph_store.execute_query("SHOW INDEX INFO")
            for idx in indexes:
                print(f"  {idx}")
        except Exception as exc:
            print(f"  Could not list indexes: {exc}")

        print("\n=== Node Counts ===")
        counts = self.graph_store.execute_query(
            "MATCH (n) RETURN labels(n) AS labels, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in counts:
            print(f"  {rec['labels']}: {rec['count']}")

        print("\n=== Relationship Counts ===")
        rels = self.graph_store.execute_query(
            "MATCH ()-[r]-() RETURN type(r) AS type, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in rels:
            print(f"  {rec['type']}: {rec['count']}")
