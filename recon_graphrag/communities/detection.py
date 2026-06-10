"""GDS Leiden community detection for the entity knowledge graph.

Groups related entity nodes based on connectivity, forming a hierarchical
community structure stored as Community nodes with IN_COMMUNITY and
PARENT_COMMUNITY relationships.
"""

from __future__ import annotations

from typing import Any, Optional

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
        tolerance: float = 1e-4,
        graph_name: str = DEFAULT_GRAPH_NAME,
        relationship_weight_property: Optional[str] = None,
        random_seed: Optional[int] = 42,
        entity_label: str = "__Entity__",
        community_label: str = "Community",
    ):
        self.graph_store = graph_store
        self.relationship_types = relationship_types
        self.max_levels = max_levels
        self.gamma = gamma
        self.theta = theta
        self.tolerance = tolerance
        self.graph_name = graph_name
        self.relationship_weight_property = relationship_weight_property
        self.random_seed = random_seed
        self.entity_label = entity_label
        self.community_label = community_label

        if self.max_levels < 1:
            raise ValueError("max_levels must be >= 1")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if self.theta <= 0:
            raise ValueError("theta must be > 0")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be > 0")

    def detect(self) -> list[dict[str, Any]]:
        """Run Leiden and create hierarchical Community nodes.

        Steps:
        1. Clean up existing communities for this graph_name
        2. Drop existing projection if it exists
        3. Project the entity graph (undirected)
        4. Run gds.leiden.stream with hierarchical levels
        5. Create Community nodes for every Leiden level
        6. Link entities to their community at every level
        7. Build PARENT_COMMUNITY hierarchy between adjacent levels
        8. Clean up the GDS projection

        Returns list of {community_id, level, entity_count, child_community_count}.
        """
        print(
            f"Starting community detection: graph_name={self.graph_name} "
            f"max_levels={self.max_levels} gamma={self.gamma} theta={self.theta} tolerance={self.tolerance}"
        )

        print(f"Cleaning existing communities: graph_name={self.graph_name}")
        self._cleanup_communities()

        print(f"Dropping existing GDS projection: graph_name={self.graph_name}")
        self._drop_projection()

        print(f"Projecting entity graph: graph_name={self.graph_name} relationship_types={self.relationship_types or 'AUTO'}")
        self._project_graph()
        try:
            print(f"Running Leiden community detection: graph_name={self.graph_name} max_levels={self.max_levels}")
            leiden_results = self._run_leiden()
            print(f"Leiden community detection returned assignments: count={len(leiden_results)}")
            self._write_community_hierarchy(leiden_results)
        finally:
            print(f"Dropping GDS projection: graph_name={self.graph_name}")
            self._drop_projection()

        stats = self._get_community_stats()
        levels = sorted({s["level"] for s in stats})
        print(f"Community detection complete: communities={len(stats)} levels={levels}")
        return stats

    @staticmethod
    def _escape_cypher_identifier(identifier: str) -> str:
        """Escape labels, relationship types, and property names for Cypher."""
        return "`" + identifier.replace("`", "``") + "`"

    @staticmethod
    def _cypher_string_literal(value: str) -> str:
        """Safely create a Cypher string literal."""
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    def _cleanup_communities(self):
        community_label = self._escape_cypher_identifier(self.community_label)
        query = f"""
        MATCH (c:{community_label} {{graph_name: $graph_name}})
        DETACH DELETE c
        """
        self.graph_store.execute_query(query, {"graph_name": self.graph_name})

    def _project_graph(self):
        entity_label = self._escape_cypher_identifier(self.entity_label)
        count = self.graph_store.execute_query(
            f"""
            MATCH (e:{entity_label} {{graph_name: $graph_name}})
            RETURN count(e) AS cnt
            """,
            {"graph_name": self.graph_name},
        )
        if not count or count[0]["cnt"] == 0:
            print(
                f"No entity nodes found for community detection: graph_name={self.graph_name} "
                f"entity_label={self.entity_label}"
            )
            raise RuntimeError(
                f"No {self.entity_label} nodes found. "
                "Run ingestion before community detection."
            )

        valid_types = self._get_valid_relationship_types()
        print(f"Valid relationship types for community detection: relationship_types={valid_types}")

        query = f"""
        MATCH (source:{entity_label} {{graph_name: $graph_name}})-[r]-(target:{entity_label} {{graph_name: $graph_name}})
        WHERE r.graph_name = $graph_name
          AND type(r) IN $relationship_types
        WITH gds.graph.project(
            $graph_name,
            source,
            target,
            {{relationshipType: type(r)}},
            {{undirectedRelationshipTypes: $relationship_types}}
        ) AS g
        RETURN g.graphName AS graphName,
               g.nodeCount AS nodeCount,
               g.relationshipCount AS relationshipCount
        """
        result = self.graph_store.execute_query(
            query,
            {
                "graph_name": self.graph_name,
                "relationship_types": valid_types,
            },
        )

        if result:
            print(
                f"Projected entity graph: graph_name={result[0].get('graphName')} "
                f"nodes={result[0].get('nodeCount', 0)} relationships={result[0].get('relationshipCount', 0)}"
            )
        else:
            print(f"GDS projection returned no result: graph_name={self.graph_name}")

        if not result or result[0]["relationshipCount"] == 0:
            print(f"GDS projection has zero relationships: graph_name={self.graph_name}")
            raise RuntimeError(
                "GDS projection was created with zero relationships. "
                "Community detection needs entity-to-entity relationships."
            )

    def _get_valid_relationship_types(self) -> list[str]:
        entity_label = self._escape_cypher_identifier(self.entity_label)
        existing = self.graph_store.execute_query(
            f"""
            MATCH (:{entity_label} {{graph_name: $graph_name}})-[r]-(:{entity_label} {{graph_name: $graph_name}})
            WHERE r.graph_name = $graph_name
            RETURN DISTINCT type(r) AS t
            """,
            {"graph_name": self.graph_name},
        )
        existing_types = {r["t"] for r in existing}

        if self.relationship_types is None:
            excluded_types = {"IN_COMMUNITY", "PARENT_COMMUNITY"}
            valid_types = sorted(existing_types - excluded_types)
        else:
            valid_types = [rt for rt in self.relationship_types if rt in existing_types]

        if not valid_types:
            requested = self.relationship_types if self.relationship_types is not None else "AUTO"
            print(
                f"No valid relationship types found: requested={requested} existing={sorted(existing_types)}"
            )
            raise RuntimeError(
                f"No valid entity-to-entity relationship types found. "
                f"Requested: {requested}. "
                f"Existing entity-to-entity relationship types: {sorted(existing_types)}"
            )

        return valid_types

    def _build_relationship_projection(self, relationship_types: list[str]) -> str:
        parts = []
        for rel_type in relationship_types:
            escaped_rel_type = self._escape_cypher_identifier(rel_type)
            if self.relationship_weight_property:
                escaped_weight = self._escape_cypher_identifier(
                    self.relationship_weight_property
                )
                weight_literal = self._cypher_string_literal(
                    self.relationship_weight_property
                )
                parts.append(
                    f"""
                    {escaped_rel_type}: {{
                        orientation: 'UNDIRECTED',
                        properties: {{
                            {escaped_weight}: {{
                                property: {weight_literal},
                                defaultValue: 1.0
                            }}
                        }}
                    }}
                    """
                )
            else:
                parts.append(f"{escaped_rel_type}: {{orientation: 'UNDIRECTED'}}")
        return ", ".join(parts)

    def _drop_projection(self):
        self.graph_store.execute_query(
            "CALL gds.graph.drop($graph_name, false)",
            {"graph_name": self.graph_name},
        )

    def _run_leiden(self) -> list[dict[str, Any]]:
        config_lines = [
            "maxLevels: $max_levels",
            "gamma: $gamma",
            "theta: $theta",
            "tolerance: $tolerance",
            "includeIntermediateCommunities: true",
        ]
        params: dict[str, Any] = {
            "graph_name": self.graph_name,
            "max_levels": self.max_levels,
            "gamma": self.gamma,
            "theta": self.theta,
            "tolerance": self.tolerance,
        }

        if self.relationship_weight_property:
            config_lines.append(
                "relationshipWeightProperty: $relationship_weight_property"
            )
            params["relationship_weight_property"] = self.relationship_weight_property

        if self.random_seed is not None:
            config_lines.append("randomSeed: $random_seed")
            params["random_seed"] = self.random_seed

        config = ",\n".join(config_lines)
        query = f"""
        CALL gds.leiden.stream($graph_name, {{
            {config}
        }})
        YIELD nodeId, communityId, intermediateCommunityIds
        WITH gds.util.asNode(nodeId) AS entity,
             communityId,
             intermediateCommunityIds
        RETURN elementId(entity) AS entity_element_id,
               communityId,
               intermediateCommunityIds
        """
        return self.graph_store.execute_query(query, params)

    def _normalize_community_path(self, rec: dict[str, Any]) -> list[str]:
        """Return community IDs from finest level to coarsest level."""
        ids = rec.get("intermediateCommunityIds")
        if ids:
            path = [str(x) for x in ids]
        else:
            path = [str(rec["communityId"])]

        final_id = str(rec["communityId"])
        if path[-1] != final_id:
            path.append(final_id)

        compact_path = []
        for community_id in path:
            if not compact_path or compact_path[-1] != community_id:
                compact_path.append(community_id)
        return compact_path

    def _write_community_hierarchy(self, leiden_results: list[dict[str, Any]]):
        membership_rows = []
        parent_edges = set()

        for rec in leiden_results:
            path = self._normalize_community_path(rec)
            entity_element_id = rec["entity_element_id"]

            for level, community_id in enumerate(path):
                membership_rows.append(
                    {
                        "entity_element_id": entity_element_id,
                        "community_id": community_id,
                        "level": level,
                    }
                )

            for level in range(len(path) - 1):
                parent_edges.add((path[level], level, path[level + 1], level + 1))

        if not membership_rows:
            print(
                f"Leiden returned no community assignments: graph_name={self.graph_name}"
            )
            raise RuntimeError("Leiden returned no community assignments.")

        print(f"Writing community hierarchy: memberships={len(membership_rows)} parent_edges={len(parent_edges)}")
        self._write_entity_memberships(membership_rows)
        self._write_parent_community_edges(parent_edges)

    def _write_entity_memberships(self, rows: list[dict[str, Any]]):
        entity_label = self._escape_cypher_identifier(self.entity_label)
        community_label = self._escape_cypher_identifier(self.community_label)
        query = f"""
        UNWIND $rows AS row
        MATCH (e:{entity_label})
        WHERE elementId(e) = row.entity_element_id
        MERGE (c:{community_label} {{
            graph_name: $graph_name,
            level: row.level,
            id: row.community_id
        }})
        ON CREATE SET c.created = timestamp()
        SET c.updated = timestamp()
        MERGE (e)-[rel:IN_COMMUNITY]->(c)
        ON CREATE SET rel.created = timestamp()
        """
        self.graph_store.execute_query(
            query, {"rows": rows, "graph_name": self.graph_name}
        )

    def _write_parent_community_edges(self, parent_edges: set[tuple[str, int, str, int]]):
        if not parent_edges:
            return

        community_label = self._escape_cypher_identifier(self.community_label)
        rows = [
            {
                "child_id": child_id,
                "child_level": child_level,
                "parent_id": parent_id,
                "parent_level": parent_level,
            }
            for child_id, child_level, parent_id, parent_level in parent_edges
        ]

        query = f"""
        UNWIND $rows AS row
        MATCH (child:{community_label} {{
            graph_name: $graph_name,
            id: row.child_id,
            level: row.child_level
        }})
        MATCH (parent:{community_label} {{
            graph_name: $graph_name,
            id: row.parent_id,
            level: row.parent_level
        }})
        MERGE (child)-[rel:PARENT_COMMUNITY]->(parent)
        ON CREATE SET rel.created = timestamp()
        """
        self.graph_store.execute_query(
            query, {"rows": rows, "graph_name": self.graph_name}
        )

    def _get_community_stats(self) -> list[dict[str, Any]]:
        entity_label = self._escape_cypher_identifier(self.entity_label)
        community_label = self._escape_cypher_identifier(self.community_label)
        query = f"""
        MATCH (c:{community_label} {{graph_name: $graph_name}})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:{entity_label})
        OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:{community_label})
        RETURN c.id AS community_id,
               c.level AS level,
               count(DISTINCT e) AS entity_count,
               count(DISTINCT child) AS child_community_count
        ORDER BY c.level, entity_count DESC
        """
        return self.graph_store.execute_query(query, {"graph_name": self.graph_name})
