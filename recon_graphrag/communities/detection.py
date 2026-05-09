"""GDS Leiden community detection for the entity knowledge graph.

Groups related entity nodes based on connectivity, forming a hierarchical
community structure stored as Community nodes with IN_COMMUNITY relationships.
"""

from __future__ import annotations

from typing import Optional

from recon_graphrag.graph.base import GraphStore


DEFAULT_GRAPH_NAME = "entity-graph"


class CommunityDetector:
    """Run Leiden community detection via Neo4j GDS on the entity graph."""

    def __init__(
        self,
        graph_store: GraphStore,
        relationship_types: Optional[list[str]] = None,
        max_levels: int = 3,
        gamma: float = 1.0,
        theta: float = 0.01,
        tolerance: float = 1e-7,
        graph_name: str = DEFAULT_GRAPH_NAME,
    ):
        self.graph_store = graph_store
        self.relationship_types = relationship_types or []
        self.max_levels = max_levels
        self.gamma = gamma
        self.theta = theta
        self.tolerance = tolerance
        self.graph_name = graph_name

    def detect(self) -> list[dict]:
        """Run Leiden and create Community nodes.

        Steps:
        1. Clean up existing communities
        2. Drop existing projection if it exists
        3. Project the entity graph (undirected)
        4. Run gds.leiden.stream with hierarchical levels
        5. Create Community nodes + IN_COMMUNITY relationships
        6. Build PARENT_COMMUNITY hierarchy between levels
        7. Clean up the GDS projection

        Returns list of {community_id, level, entity_count}.
        """
        self._cleanup_communities()
        self._drop_projection()
        self._project_graph()
        try:
            leiden_results = self._run_leiden()
            self._write_communities(leiden_results)
            self._write_intermediate_communities(leiden_results)
        finally:
            self._drop_projection()

        return self._get_community_stats()

    def _cleanup_communities(self):
        self.graph_store.execute_query("MATCH (c:Community) DETACH DELETE c")

    def _project_graph(self):
        count = self.graph_store.execute_query(
            "MATCH (e:__Entity__) RETURN count(e) AS cnt"
        )
        if not count or count[0]["cnt"] == 0:
            raise RuntimeError(
                "No __Entity__ nodes found. Run ingestion before community detection."
            )

        # Only project relationship types that exist in the graph
        existing = self.graph_store.execute_query(
            "MATCH ()-[r]->() RETURN DISTINCT type(r) AS t"
        )
        existing_types = {r["t"] for r in existing}
        valid_types = [rt for rt in self.relationship_types if rt in existing_types]

        if not valid_types:
            raise RuntimeError(
                f"None of the requested relationship types {self.relationship_types} "
                f"exist in the graph. Found: {sorted(existing_types)}"
            )

        rel_config = ", ".join(
            f"{rel}: {{orientation: 'UNDIRECTED'}}"
            for rel in valid_types
        )
        query = f"""
        CALL gds.graph.project(
            $graph_name,
            '__Entity__',
            {{{rel_config}}}
        )
        YIELD graphName, nodeCount, relationshipCount
        """
        self.graph_store.execute_query(query, {"graph_name": self.graph_name})

    def _drop_projection(self):
        self.graph_store.execute_query(
            "CALL gds.graph.drop($graph_name, false)",
            {"graph_name": self.graph_name},
        )

    def _run_leiden(self) -> list[dict]:
        query = """
        CALL gds.leiden.stream($graph_name, {
            maxLevels: $max_levels,
            gamma: $gamma,
            theta: $theta,
            tolerance: $tolerance,
            includeIntermediateCommunities: true
        })
        YIELD nodeId, communityId, intermediateCommunityIds
        RETURN gds.util.asNode(nodeId) AS entity,
               communityId,
               intermediateCommunityIds
        """
        return self.graph_store.execute_query(query, {
            "graph_name": self.graph_name,
            "max_levels": self.max_levels,
            "gamma": self.gamma,
            "theta": self.theta,
            "tolerance": self.tolerance,
        })

    def _write_communities(self, leiden_results: list[dict]):
        data = [
            {
                "entity_name": str(
                    rec["entity"].get("name", "")
                    or rec["entity"].get("description", "")
                ),
                "community_id": str(rec["communityId"]),
            }
            for rec in leiden_results
        ]

        query = """
        UNWIND $data AS row
        MERGE (c:Community {id: row.community_id, level: 0})
        ON CREATE SET c.created = timestamp()
        WITH c, row
        MATCH (e:__Entity__)
        WHERE e.name = row.entity_name OR e.description = row.entity_name
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """
        self.graph_store.execute_query(query, {"data": data})

    def _write_intermediate_communities(self, leiden_results: list[dict]):
        for level_idx in range(self.max_levels - 1):
            level = level_idx + 1
            data = []
            for rec in leiden_results:
                ids = rec.get("intermediateCommunityIds", [])
                if level_idx < len(ids):
                    data.append({
                        "leaf_community_id": str(rec["communityId"]),
                        "parent_community_id": str(ids[level_idx]),
                    })
            if not data:
                break

            query = """
            UNWIND $data AS row
            MERGE (c:Community {id: row.parent_community_id, level: $level})
            ON CREATE SET c.created = timestamp()
            WITH c, row
            MATCH (leaf:Community {id: row.leaf_community_id, level: $leaf_level})
            MERGE (leaf)-[:PARENT_COMMUNITY]->(c)
            """
            self.graph_store.execute_query(query, {
                "data": data, "level": level, "leaf_level": level_idx,
            })

    def _get_community_stats(self) -> list[dict]:
        query = """
        MATCH (c:Community)
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        WITH c, count(e) AS entity_count
        RETURN c.id AS community_id, c.level AS level, entity_count
        ORDER BY c.level, entity_count DESC
        """
        return self.graph_store.execute_query(query)
