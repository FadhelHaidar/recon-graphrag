"""Shared graph store helpers.

Concrete backends keep their own packages and dialect-specific operations.
This base only contains backend-neutral Cypher-shaped helpers whose return
shape is part of the shared GraphStore contract.
"""

from __future__ import annotations


class BaseGraphStore:
    """Mixin for graph-store methods that are identical across backends."""

    def execute_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        raise NotImplementedError

    def get_entity_count(self) -> int:
        return self._count("MATCH (e:__Entity__) RETURN count(e) AS cnt")

    def get_chunk_count(self) -> int:
        return self._count("MATCH (c:Chunk) RETURN count(c) AS cnt")

    def get_evidence_link_count(self) -> int:
        return self._count(
            "MATCH (:Chunk)-[r:FROM_CHUNK]->(:__Entity__) RETURN count(r) AS cnt"
        )

    def get_relationship_count(self) -> int:
        return self._count(
            "MATCH (:__Entity__)-[r]-(:__Entity__) RETURN count(r) AS cnt"
        )

    def validate_graph_build(self) -> dict:
        counts = {
            "entity_count": self.get_entity_count(),
            "chunk_count": self.get_chunk_count(),
            "evidence_link_count": self.get_evidence_link_count(),
            "entity_relationship_count": self.get_relationship_count(),
        }

        for key, query in self._extra_validation_count_queries().items():
            counts[key] = self._count(query)
        return counts

    def _extra_validation_count_queries(self) -> dict[str, str]:
        return {}

    def _count(self, query: str) -> int:
        result = self.execute_query(query)
        return result[0]["cnt"] if result else 0
