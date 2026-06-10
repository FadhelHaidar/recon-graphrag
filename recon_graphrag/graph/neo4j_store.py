"""Neo4j graph store backend."""

from __future__ import annotations

from typing import Optional

import neo4j

from recon_graphrag.graph.base import GraphStore
from recon_graphrag.graph.cypher import (
    cypher_string_literal,
    escape_cypher_identifier,
)


class Neo4jGraphStore:
    """Neo4j backend backed by the official Neo4j driver."""

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
            notifications_disabled_categories=["DEPRECATION", "UNRECOGNIZED", "HINT"],
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
        query = f"""
        CREATE VECTOR INDEX {escape_cypher_identifier(name)} IF NOT EXISTS
        FOR (n:{escape_cypher_identifier(label)})
        ON (n.{escape_cypher_identifier(embedding_property)})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {int(dimensions)},
            `vector.similarity_function`: {cypher_string_literal(similarity_fn)}
          }}
        }}
        """
        self.execute_query(query)

    def create_fulltext_index(
        self,
        name: str,
        label: str,
        node_properties: list[str],
    ) -> None:
        properties = ", ".join(
            f"n.{escape_cypher_identifier(prop)}" for prop in node_properties
        )
        query = f"""
        CREATE FULLTEXT INDEX {escape_cypher_identifier(name)} IF NOT EXISTS
        FOR (n:{escape_cypher_identifier(label)})
        ON EACH [{properties}]
        """
        self.execute_query(query)

    def upsert_vectors(
        self,
        node_ids: list[str],
        embedding_property: str,
        vectors: list[list[float]],
    ) -> None:
        """Batch upsert vector embeddings onto nodes matched by elementId().

        This intentionally follows the current embedding call path, which reads
        elementId() values immediately before writing embeddings back.
        """
        if len(node_ids) != len(vectors):
            raise ValueError("node_ids and vectors must have the same length")
        if not node_ids:
            return

        rows = [
            {"id": node_id, "vector": vector}
            for node_id, vector in zip(node_ids, vectors)
        ]
        self.execute_query(
            """
            UNWIND $rows AS row
            MATCH (n)
            WHERE elementId(n) = row.id
            CALL db.create.setNodeVectorProperty(n, $embedding_property, row.vector)
            RETURN count(n) AS updated_count
            """,
            {"rows": rows, "embedding_property": embedding_property},
        )
