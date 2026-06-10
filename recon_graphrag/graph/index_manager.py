"""Vector and fulltext index creation + entity resolution.

Creates all required Neo4j indexes for semantic retrieval and runs
entity resolution to merge duplicate entities.
"""

from __future__ import annotations

from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graph.base import GraphStore
from recon_graphrag.graph.cypher import escape_cypher_identifier
from recon_graphrag.models.types import IndexConfig


class ExactMatchEntityResolver:
    """Merge duplicate entity nodes using APOC when available."""

    def __init__(self, graph_store: GraphStore, resolve_property: str = "name"):
        self.graph_store = graph_store
        self.resolve_property = resolve_property

    async def run(self) -> dict:
        try:
            self.graph_store.execute_query("RETURN apoc.version() AS version")
        except Exception as exc:
            return {
                "skipped": True,
                "reason": f"APOC is unavailable: {exc}",
                "merged_groups": 0,
            }

        prop = escape_cypher_identifier(self.resolve_property)
        result = self.graph_store.execute_query(
            f"""
            MATCH (e:__Entity__)
            WHERE e.{prop} IS NOT NULL
            WITH e,
                 coalesce(e.graph_name, '') AS graph_name,
                 e.{prop} AS resolve_value,
                 [label IN labels(e) WHERE label <> '__Entity__'] AS domain_labels
            UNWIND CASE
                WHEN size(domain_labels) = 0 THEN ['__Entity__']
                ELSE domain_labels
            END AS domain_label
            WITH graph_name, domain_label, resolve_value, collect(DISTINCT e) AS nodes
            WHERE size(nodes) > 1
            CALL apoc.refactor.mergeNodes(
                nodes,
                {{properties: 'combine', mergeRels: true}}
            ) YIELD node
            RETURN count(node) AS merged_groups
            """
        )
        merged_groups = result[0].get("merged_groups", 0) if result else 0
        return {"skipped": False, "merged_groups": merged_groups}


class IndexManager:
    """Create indexes and resolve duplicate entities in the graph."""

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
        """Create all required vector, fulltext, and uniqueness indexes."""
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
                    f"DROP INDEX {escape_cypher_identifier(name)} IF EXISTS"
                )
            except Exception:
                pass

    def _create_constraints(self):
        """Create constraints required by the community hierarchy."""
        try:
            self.graph_store.execute_query(
                """
                CREATE CONSTRAINT community_unique IF NOT EXISTS
                FOR (c:Community)
                REQUIRE (c.graph_name, c.level, c.id) IS UNIQUE
                """
            )
        except Exception as e:
            print(f"  Warning: community uniqueness constraint failed: {e}")

    async def resolve_entities(self):
        """Run entity resolution to merge duplicate __Entity__ nodes.

        Merges entities with the same graph name, domain label, and name.
        If APOC is unavailable, resolution is skipped and graph building can
        continue.
        """
        resolver = ExactMatchEntityResolver(self.graph_store, resolve_property="name")
        return await resolver.run()

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
