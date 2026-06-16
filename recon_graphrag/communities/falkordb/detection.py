"""Community detection for the FalkorDB backend.

FalkorDB does not ship Neo4j GDS, so this module implements label propagation
in Python over the entity subgraph exported from FalkorDB. The resulting
communities are stored as single-level (:Community) nodes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.falkordb.cypher import escape_cypher_identifier

DEFAULT_GRAPH_NAME = "entity-graph"


class CommunityDetector:
    """Run label-propagation community detection on the entity graph."""

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

    def detect(self) -> list[dict[str, Any]]:
        """Run label propagation and create Community nodes.

        Steps:
        1. Clean up existing communities for this graph_name.
        2. Export entity nodes and relationships.
        3. Run label propagation in Python.
        4. Create Community nodes and IN_COMMUNITY relationships.

        Returns list of {community_id, level, entity_count, child_community_count}.
        """
        self._cleanup_communities()
        nodes, edges = self._export_subgraph()

        if not nodes:
            raise RuntimeError(
                f"No {self.entity_label} nodes found. "
                "Run ingestion before community detection."
            )
        if not edges:
            raise RuntimeError(
                "Community detection needs entity-to-entity relationships."
            )

        communities = self._label_propagation(nodes, edges)
        self._write_communities(communities)
        return self._get_community_stats()

    def _cleanup_communities(self):
        community_label = escape_cypher_identifier(self.community_label)
        query = f"""
        MATCH (c:{community_label} {{graph_name: $graph_name}})
        DETACH DELETE c
        """
        self.graph_store.execute_query(query, {"graph_name": self.graph_name})

    def _export_subgraph(self) -> tuple[set[str], list[tuple[str, str, str, Any]]]:
        """Export entity IDs and relationships for the current graph."""
        entity_label = escape_cypher_identifier(self.entity_label)

        node_rows = self.graph_store.execute_query(
            f"""
            MATCH (e:{entity_label} {{graph_name: $graph_name}})
            RETURN id(e) AS id
            """,
            {"graph_name": self.graph_name},
        )
        nodes = {row["id"] for row in node_rows}

        valid_types = self._get_valid_relationship_types()

        rel_rows = self.graph_store.execute_query(
            f"""
            MATCH (source:{entity_label} {{graph_name: $graph_name}})-[r]-(target:{entity_label} {{graph_name: $graph_name}})
            WHERE r.graph_name = $graph_name
              AND type(r) IN $relationship_types
            RETURN id(source) AS source_id, id(target) AS target_id, type(r) AS rel_type, properties(r) AS rel_props
            """,
            {
                "graph_name": self.graph_name,
                "relationship_types": valid_types,
            },
        )
        edges = [
            (
                row["source_id"],
                row["target_id"],
                row["rel_type"],
                row.get("rel_props", {}) or {},
            )
            for row in rel_rows
        ]
        return nodes, edges

    def _get_valid_relationship_types(self) -> list[str]:
        entity_label = escape_cypher_identifier(self.entity_label)
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
            raise RuntimeError(
                f"No valid entity-to-entity relationship types found. "
                f"Requested: {requested}. "
                f"Existing entity-to-entity relationship types: {sorted(existing_types)}"
            )

        return valid_types

    def _label_propagation(
        self, nodes: set[str], edges: list[tuple[str, str, str, Any]]
    ) -> dict[str, list[str]]:
        """Run asynchronous label propagation and return community_id -> node_ids."""
        # Build adjacency list (undirected)
        adjacency: dict[str, list[str]] = {node: [] for node in nodes}
        for source, target, _rel_type, _props in edges:
            if source in adjacency and target in adjacency:
                adjacency[source].append(target)
                adjacency[target].append(source)

        # Initialize each node with its own label
        labels = {node: node for node in nodes}

        # Deterministic iteration order
        node_list = sorted(nodes)
        if self.random_seed is not None:
            import random

            rng = random.Random(self.random_seed)
            order = node_list.copy()
            rng.shuffle(order)
        else:
            order = node_list

        max_iterations = 10
        for _ in range(max_iterations):
            changed = False
            for node in order:
                neighbors = adjacency.get(node, [])
                if not neighbors:
                    continue
                neighbor_labels = [labels[n] for n in neighbors]
                # Pick the most frequent label; ties broken by label value for determinism
                counts = Counter(neighbor_labels)
                most_common = max(
                    counts.items(), key=lambda item: (item[1], item[0])
                )[0]
                if labels[node] != most_common:
                    labels[node] = most_common
                    changed = True
            if not changed:
                break

        # Invert labels into communities; use string IDs for consistency with Neo4j.
        communities: dict[str, list] = {}
        for node, label in labels.items():
            communities.setdefault(str(label), []).append(node)
        return communities

    def _write_communities(self, communities: dict[str, list[str]]):
        if not communities:
            raise RuntimeError("Label propagation returned no community assignments.")

        entity_label = escape_cypher_identifier(self.entity_label)
        community_label = escape_cypher_identifier(self.community_label)

        # Create Community nodes
        community_rows = [
            {
                "community_id": community_id,
                "level": 0,
                "graph_name": self.graph_name,
            }
            for community_id in communities.keys()
        ]
        self.graph_store.execute_query(
            f"""
            UNWIND $rows AS row
            MERGE (c:{community_label} {{
                graph_name: row.graph_name,
                level: row.level,
                id: row.community_id
            }})
            ON CREATE SET c.created = timestamp()
            SET c.updated = timestamp()
            """,
            {"rows": community_rows},
        )

        # Create IN_COMMUNITY relationships
        membership_rows = [
            {
                "entity_id": node_id,
                "community_id": community_id,
                "level": 0,
            }
            for community_id, node_ids in communities.items()
            for node_id in node_ids
        ]
        self.graph_store.execute_query(
            f"""
            UNWIND $rows AS row
            MATCH (e:{entity_label})
            WHERE id(e) = row.entity_id
            MATCH (c:{community_label} {{
                graph_name: $graph_name,
                level: row.level,
                id: row.community_id
            }})
            MERGE (e)-[rel:IN_COMMUNITY]->(c)
            ON CREATE SET rel.created = timestamp()
            """,
            {"rows": membership_rows, "graph_name": self.graph_name},
        )

    def _get_community_stats(self) -> list[dict[str, Any]]:
        entity_label = escape_cypher_identifier(self.entity_label)
        community_label = escape_cypher_identifier(self.community_label)
        query = f"""
        MATCH (c:{community_label} {{graph_name: $graph_name}})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:{entity_label})
        RETURN c.id AS community_id,
               c.level AS level,
               count(DISTINCT e) AS entity_count,
               0 AS child_community_count
        ORDER BY c.level, entity_count DESC
        """
        return self.graph_store.execute_query(query, {"graph_name": self.graph_name})
