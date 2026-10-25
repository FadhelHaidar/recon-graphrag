"""Community detection for the Memgraph backend using MAGE Leiden.

MAGE provides `leiden_community_detection.get_subgraph()`, which runs server-side
and returns community memberships per node. This module maps them to the same
Community/IN_COMMUNITY/PARENT_COMMUNITY structure used by the Neo4j backend.

Note: in practice MAGE returns a single community level per node (verified
against `memgraph/memgraph-mage` across several `gamma` values), so hierarchies
and `PARENT_COMMUNITY` edges only appear on Memgraph if a future MAGE release
reports intermediate levels. `max_levels` is therefore a Neo4j-only dimension.

Note: MAGE's Leiden implementation is non-deterministic and does not expose a
random seed parameter. `random_seed` is accepted for API symmetry with the Neo4j
backend but is not forwarded to the algorithm.
"""

from __future__ import annotations

from typing import Any, Optional

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.cypher import (
    escape_cypher_identifier,
)
from recon_graphrag.graphdb.store_base import QueryExecutor, replacement_transaction

DEFAULT_GRAPH_NAME = "entity-graph"


class CommunityDetector:
    """Run MAGE Leiden community detection on the entity graph in Memgraph.

    random_seed is accepted for API symmetry with the Neo4j backend but is not
    used by MAGE, whose Leiden implementation does not expose a seed parameter.
    """

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
        1. Run MAGE leiden_community_detection.get_subgraph() on the entity
           subgraph. `tolerance` is mapped to MAGE's `resolution_parameter`
           because MAGE does not expose a separate tolerance parameter.
        2. Normalize the returned community paths and build every community,
           membership, and parent row.
        3. In one transaction: replace existing communities for this graph_name,
           create Community nodes and IN_COMMUNITY relationships, then build the
           PARENT_COMMUNITY hierarchy between adjacent levels.

        Existing communities survive any failure before the replacement
        transaction commits.

        Returns list of {community_id, level, entity_count, child_community_count}.
        """
        leiden_results = self._run_leiden()
        if not leiden_results:
            raise RuntimeError("Leiden returned no community assignments.")

        paths = self._normalize_paths(leiden_results)
        if not paths:
            raise RuntimeError("Leiden returned no community assignments.")

        self._write_community_hierarchy(paths)
        return self._get_community_stats()

    def _cleanup_communities(self, executor=None):
        community_label = escape_cypher_identifier(self.community_label)
        query = f"""
        MATCH (c:{community_label} {{graph_name: $graph_name}})
        DETACH DELETE c
        """
        (executor or self.graph_store).execute_query(
            query, {"graph_name": self.graph_name}
        )

    def _get_valid_relationship_types(self) -> list[str]:
        entity_label = escape_cypher_identifier(self.entity_label)
        count = self.graph_store.execute_query(
            f"""
            MATCH (e:{entity_label} {{graph_name: $graph_name}})
            RETURN count(e) AS cnt
            """,
            {"graph_name": self.graph_name},
        )
        if not count or count[0]["cnt"] == 0:
            raise RuntimeError(
                f"No {self.entity_label} nodes found. "
                "Run ingestion before community detection."
            )

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

        rel_count = self.graph_store.execute_query(
            f"""
            MATCH (:{entity_label} {{graph_name: $graph_name}})-[r]-(:{entity_label} {{graph_name: $graph_name}})
            WHERE r.graph_name = $graph_name
              AND type(r) IN $relationship_types
            RETURN count(r) AS cnt
            """,
            {"graph_name": self.graph_name, "relationship_types": valid_types},
        )
        if not rel_count or rel_count[0]["cnt"] == 0:
            raise RuntimeError(
                "Memgraph community projection has zero relationships. "
                "Community detection needs entity-to-entity relationships."
            )

        return valid_types

    def _run_leiden(self) -> list[dict[str, Any]]:
        valid_types = self._get_valid_relationship_types()

        # MAGE leiden_community_detection.get_subgraph scopes the algorithm to the
        # selected entity subgraph. The procedure accepts optional arguments
        # positionally after the two required subgraph arguments. The order is:
        # weight, gamma, theta, resolution_parameter, max_iterations.
        #
        # gamma is the resolution/granularity knob (higher -> more, smaller
        # communities); self.gamma feeds it, matching the Neo4j backend's gamma
        # so both resolve at the same granularity. MAGE's `resolution_parameter`
        # is misleadingly named: it's the minimum modularity change when merging
        # (a convergence threshold), so self.tolerance correctly maps there.
        #
        # There is no random seed parameter exposed by MAGE; self.random_seed is
        # kept for API symmetry with the Neo4j backend but is not forwarded.
        weight_property = self.relationship_weight_property or "weight"
        params: dict[str, Any] = {
            "graph_name": self.graph_name,
            "weight_property": weight_property,
            "gamma": self.gamma,
            "theta": self.theta,
            "tolerance": self.tolerance,
            "relationship_types": valid_types,
        }

        query = f"""
        MATCH (source:{escape_cypher_identifier(self.entity_label)} {{graph_name: $graph_name}})-[r]-(target:{escape_cypher_identifier(self.entity_label)} {{graph_name: $graph_name}})
        WHERE r.graph_name = $graph_name
          AND type(r) IN $relationship_types
        WITH COLLECT(DISTINCT source) + COLLECT(DISTINCT target) AS nodes, COLLECT(DISTINCT r) AS relationships
        CALL leiden_community_detection.get_subgraph(
            nodes,
            relationships,
            $weight_property,
            $gamma,
            $theta,
            $tolerance
        )
        YIELD node, community_id, communities
        RETURN id(node) AS entity_id, community_id, communities
        ORDER BY entity_id
        """
        return self.graph_store.execute_query(query, params)

    @staticmethod
    def _normalize_paths(leiden_results: list[dict[str, Any]]) -> dict[int, list[int]]:
        """Return entity_id -> community path from finest to coarsest level.

        MAGE returns communities in coarsest→finest order. We reverse to
        finest→coarsest so that _write_community_hierarchy can apply the
        same reversal logic as the Neo4j backend.
        """
        paths: dict[int, list[int]] = {}
        for row in leiden_results:
            entity_id = row.get("entity_id")
            final_id = row.get("community_id")
            hierarchy = row.get("communities") or []

            if entity_id is None or final_id is None:
                continue

            path = [int(x) for x in hierarchy]
            final_id_int = int(final_id)
            if not path or path[-1] != final_id_int:
                path.append(final_id_int)

            # Deduplicate consecutive identical IDs.
            compact_path = []
            for cid in path:
                if not compact_path or compact_path[-1] != cid:
                    compact_path.append(cid)

            # Reverse: MAGE gives coarsest→finest, we want finest→coarsest.
            compact_path.reverse()

            paths[entity_id] = compact_path
        return paths

    def _write_community_hierarchy(self, paths: dict[int, list[int]]):
        community_rows, membership_rows, parent_rows = self._build_hierarchy_rows(paths)

        community_label = escape_cypher_identifier(self.community_label)
        entity_label = escape_cypher_identifier(self.entity_label)

        executor: QueryExecutor
        with replacement_transaction(self.graph_store) as executor:
            self._cleanup_communities(executor)
            executor.execute_query(
                f"""
                UNWIND $rows AS row
                MERGE (c:{community_label} {{
                    graph_name: row.graph_name,
                    level: row.level,
                    id: row.community_id
                }})
                ON CREATE SET c.created = timestamp(), c.uid = row.uid
                SET c.updated = timestamp(), c.uid = row.uid
                """,
                {"rows": community_rows},
            )
            executor.execute_query(
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
            if parent_rows:
                executor.execute_query(
                    f"""
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
                    """,
                    {"rows": parent_rows, "graph_name": self.graph_name},
                )

    def _build_hierarchy_rows(
        self, paths: dict[int, list[int]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Build and validate every row before any write happens."""
        if not paths:
            raise RuntimeError("No community paths to write.")

        # Limit to max_levels (finest first in path).
        max_depth = max(len(p) for p in paths.values())
        levels_to_write = min(max_depth, self.max_levels)

        # Level assignment is reversed: path[0] (finest) → highest level,
        # path[-1] (coarsest) → level 0.  This matches Microsoft GraphRAG
        # semantics where level 0 = coarsest.
        max_level = levels_to_write - 1

        # Create Community nodes for each path index.
        community_rows = []
        for index in range(levels_to_write):
            level = max_level - index
            level_communities = set()
            for path in paths.values():
                if index < len(path):
                    level_communities.add(path[index])
            for community_id in sorted(level_communities):
                community_rows.append({
                    "community_id": str(community_id),
                    "level": level,
                    "graph_name": self.graph_name,
                    "uid": f"{self.graph_name}:{level}:{community_id}",
                })

        # Sort for deterministic MERGE order.
        community_rows.sort(key=lambda r: (r["level"], r["community_id"]))

        # Create IN_COMMUNITY relationships.
        membership_rows = []
        for entity_id, path in sorted(paths.items()):
            for index in range(min(len(path), levels_to_write)):
                level = max_level - index
                membership_rows.append({
                    "entity_id": entity_id,
                    "community_id": str(path[index]),
                    "level": level,
                })

        # Create PARENT_COMMUNITY edges between adjacent levels.
        # After reversal: path[index] (finer) → level=max_level-index,
        # path[index+1] (coarser) → level=max_level-(index+1).
        # PARENT_COMMUNITY points from child (finer) to parent (coarser).
        parent_edges: set[tuple[str, int, str, int]] = set()
        for path in paths.values():
            for index in range(min(len(path) - 1, levels_to_write - 1)):
                child_id = str(path[index])
                child_level = max_level - index
                parent_id = str(path[index + 1])
                parent_level = max_level - (index + 1)
                parent_edges.add((child_id, child_level, parent_id, parent_level))

        if parent_edges:
            parent_rows = [
                {
                    "child_id": child_id,
                    "child_level": child_level,
                    "parent_id": parent_id,
                    "parent_level": parent_level,
                }
                for child_id, child_level, parent_id, parent_level in sorted(parent_edges)
            ]
        else:
            parent_rows = []

        for entity_id, path in paths.items():
            if not path or any(community_id is None for community_id in path):
                raise RuntimeError(
                    "Leiden returned a malformed community path for entity "
                    f"{entity_id!r}; existing communities were left untouched."
                )

        return community_rows, membership_rows, parent_rows

    def _get_community_stats(self) -> list[dict[str, Any]]:
        entity_label = escape_cypher_identifier(self.entity_label)
        community_label = escape_cypher_identifier(self.community_label)
        query = f"""
        MATCH (c:{community_label} {{graph_name: $graph_name}})
        OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:{entity_label})
        WITH c, count(DISTINCT e) AS entity_count
        OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:{community_label})
        WITH c, entity_count, count(DISTINCT child) AS child_community_count
        RETURN c.id AS community_id,
               c.level AS level,
               entity_count,
               child_community_count
        ORDER BY c.level, entity_count DESC, community_id
        """
        return self.graph_store.execute_query(query, {"graph_name": self.graph_name})
