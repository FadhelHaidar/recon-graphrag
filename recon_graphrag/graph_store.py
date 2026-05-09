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


class Neo4jGraphStore:
    """Neo4j backend: wraps neo4j.Driver + neo4j-graphrag index helpers."""

    def __init__(
        self,
        driver: neo4j.Driver,
        database: Optional[str] = None,
    ):
        self._driver = driver
        self._database = database

    @property
    def driver(self) -> neo4j.Driver:
        return self._driver

    def execute_query(
        self, query: str, parameters: Optional[dict] = None
    ) -> list[dict]:
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def create_vector_index(
        self,
        name: str,
        label: str,
        embedding_property: str,
        dimensions: int,
        similarity_fn: str = "cosine",
    ) -> None:
        create_vector_index(
            self._driver,
            name=name,
            label=label,
            embedding_property=embedding_property,
            dimensions=dimensions,
            similarity_fn=similarity_fn,
            fail_if_exists=False,
            neo4j_database=self._database,
        )

    def create_fulltext_index(
        self,
        name: str,
        label: str,
        node_properties: list[str],
    ) -> None:
        create_fulltext_index(
            self._driver,
            name=name,
            label=label,
            node_properties=node_properties,
            fail_if_exists=False,
            neo4j_database=self._database,
        )

    def upsert_vectors(
        self,
        node_ids: list[str],
        embedding_property: str,
        vectors: list[list[float]],
    ) -> None:
        upsert_vectors(
            self._driver,
            node_ids,
            embedding_property,
            vectors,
            neo4j_database=self._database,
        )
