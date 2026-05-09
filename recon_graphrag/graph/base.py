"""Graph database abstraction layer.

Defines the GraphStore protocol that all backends must implement,
plus Neo4jGraphStore as the default Neo4j backend.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import neo4j
from neo4j_graphrag.indexes import (
    create_fulltext_index,
    create_vector_index,
    upsert_vectors,
)


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for graph database operations.

    All SDK components depend on this protocol, not on any specific
    database driver. Implement this to add a new backend.
    """

    def execute_query(
        self, query: str, parameters: Optional[dict] = None
    ) -> list[dict]:
        """Execute a query and return results as list of dicts."""
        ...

    def create_vector_index(
        self,
        name: str,
        label: str,
        embedding_property: str,
        dimensions: int,
        similarity_fn: str = "cosine",
    ) -> None:
        """Create a vector index."""
        ...

    def create_fulltext_index(
        self,
        name: str,
        label: str,
        node_properties: list[str],
    ) -> None:
        """Create a fulltext index."""
        ...

    def upsert_vectors(
        self,
        node_ids: list[str],
        embedding_property: str,
        vectors: list[list[float]],
    ) -> None:
        """Batch upsert vector embeddings onto nodes."""
        ...

    @property
    def driver(self) -> neo4j.Driver:
        """Return the underlying neo4j.Driver (for neo4j-graphrag compatibility).

        This is a temporary bridge — neo4j-graphrag's SimpleKGPipeline and
        HybridCypherRetriever require a raw driver. Future versions of
        neo4j-graphrag may accept a protocol instead, at which point this
        property can be removed.
        """
        ...
