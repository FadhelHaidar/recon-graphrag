"""Vector and fulltext index creation for FalkorDB."""

from __future__ import annotations

from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.falkordb.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)
from recon_graphrag.models.types import IndexConfig


class IndexManager:
    """Create indexes for the FalkorDB backend."""

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
        """Create all required vector and fulltext indexes."""
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

    def _drop_indexes(self):
        """Drop existing indexes so they can be recreated with updated settings."""
        # FalkorDB vector indexes are dropped by label/property, not by name.
        vector_indexes = [
            (self.config.chunk_label, "embedding"),
            (self.config.entity_label, "embedding"),
            (self.config.community_label, "embedding"),
        ]
        for label, prop in vector_indexes:
            try:
                self.graph_store.execute_query(
                    f"DROP VECTOR INDEX FOR (n:{escape_cypher_identifier(label)}) ON (n.{escape_cypher_identifier(prop)})"
                )
            except Exception:
                pass
        # Fulltext indexes are dropped by label.
        try:
            self.graph_store.execute_query(
                f"CALL db.idx.fulltext.drop({cypher_string_literal(self.config.entity_label)})"
            )
        except Exception:
            pass

    def verify(self):
        """Print current graph schema info: indexes, node counts, relationship counts."""
        print("\n=== Indexes ===")
        try:
            indexes = self.graph_store.execute_query("CALL db.indexes()")
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
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        for rec in rels:
            print(f"  {rec['type']}: {rec['count']}")
