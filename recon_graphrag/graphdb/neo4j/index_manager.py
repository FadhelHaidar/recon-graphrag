"""Vector and fulltext index creation + entity resolution.

Creates all required Neo4j indexes for semantic retrieval and runs
entity resolution to merge duplicate entities.
"""

from __future__ import annotations

from typing import Optional

from recon_graphrag.embeddings.base import BaseEmbedder
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.neo4j.cypher import escape_cypher_identifier
from recon_graphrag.graphdb.neo4j.entity_resolution import _Neo4jEntityResolver
from recon_graphrag.models.types import IndexConfig


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

    async def resolve_entities(
        self,
        graph_name: str = "entity-graph",
        strategy: str = "normalized",
        resolve_property: str = "name",
        dry_run: bool = False,
        merge_threshold: float = 95.0,
        review_threshold: float = 85.0,
        max_candidates_per_entity: int = 20,
        aliases: Optional[dict] = None,
        embedder=None,
        llm=None,
        llm_guidance: Optional[str] = None,
        allow_ai_auto_merge: bool = False,
        context_properties: Optional[dict[str, list[str]] | list[str]] = None,
        conflict_properties: Optional[dict[str, list[str]] | list[str]] = None,
        context_mode: str = "safe_defaults",
    ) -> dict:
        """Run entity resolution to merge duplicate __Entity__ nodes.

        If APOC is unavailable, resolution is skipped and graph building can
        continue.
        """
        resolver = _Neo4jEntityResolver(self.graph_store)
        return await resolver.resolve(
            graph_name=graph_name,
            strategy=strategy,
            resolve_property=resolve_property,
            dry_run=dry_run,
            merge_threshold=merge_threshold,
            review_threshold=review_threshold,
            max_candidates_per_entity=max_candidates_per_entity,
            aliases=aliases,
            embedder=embedder or self.embedder,
            llm=llm,
            llm_guidance=llm_guidance,
            allow_ai_auto_merge=allow_ai_auto_merge,
            context_properties=context_properties,
            conflict_properties=conflict_properties,
            context_mode=context_mode,
        )

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
