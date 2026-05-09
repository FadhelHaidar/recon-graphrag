"""Neo4j graph store backend.

Wraps neo4j.Driver + neo4j-graphrag index helpers.
"""

from __future__ import annotations

from typing import Optional

import neo4j
from neo4j_graphrag.indexes import (
    create_fulltext_index,
    create_vector_index,
    upsert_vectors,
)

from recon_graphrag.graph.base import GraphStore


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
        with self._driver.session(
            database=self._database,
            notifications_disabled_categories=["DEPRECATION"],
        ) as session:
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
