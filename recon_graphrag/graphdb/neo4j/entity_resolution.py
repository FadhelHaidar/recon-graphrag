"""Neo4j-specific entity resolution implementation."""

from __future__ import annotations

from recon_graphrag.graphdb.entity_resolution import (
    BaseEntityResolver,
    _EntityRecord,
    _first_property_value,
    _normalize_name,
)
from recon_graphrag.graphdb.neo4j.cypher import escape_cypher_identifier


class _Neo4jEntityResolver(BaseEntityResolver):
    def _preflight(self, *, dry_run: bool) -> dict | None:
        if dry_run:
            return None
        try:
            self.graph_store.execute_query("RETURN apoc.version() AS version")
        except Exception as exc:
            return {
                "skipped": True,
                "reason": f"APOC is unavailable: {exc}",
                "merged_groups": 0,
                "merged_nodes": 0,
                "candidate_groups": 0,
                "review_groups": [],
                "signals": {"apoc": "unavailable"},
            }
        return None

    def _load_entities(
        self, graph_name: str, resolve_property: str
    ) -> list[_EntityRecord]:
        prop = escape_cypher_identifier(resolve_property)
        rows = self.graph_store.execute_query(
            f"""
            MATCH (e:__Entity__)
            WHERE e.graph_name = $graph_name
              AND e.{prop} IS NOT NULL
            RETURN
              elementId(e) AS node_id,
              e.id AS entity_id,
              e.graph_name AS graph_name,
              e.{prop} AS resolve_value,
              labels(e) AS labels,
              properties(e) AS properties
            """,
            {"graph_name": graph_name},
        )

        entities = []
        for row in rows:
            labels = row.get("labels", [])
            domain_labels = [label for label in labels if label != "__Entity__"]
            domain_label = domain_labels[0] if domain_labels else "__Entity__"
            resolve_value = row.get("resolve_value", "")
            entities.append(
                _EntityRecord(
                    node_id=row.get("node_id", ""),
                    entity_id=row.get("entity_id", ""),
                    graph_name=row.get("graph_name", ""),
                    domain_label=domain_label,
                    resolve_value=resolve_value,
                    normalized_value=_normalize_name(resolve_value),
                    properties=row.get("properties", {}),
                )
            )
        return entities

    def _merge_groups(
        self, groups: list[list[_EntityRecord]], resolve_property: str
    ) -> int:
        merged_nodes = 0
        for group in groups:
            sorted_group = sorted(
                group,
                key=lambda e: len(e.resolve_value) if e.resolve_value else 0,
                reverse=True,
            )
            node_ids = [e.node_id for e in sorted_group]
            canonical_entity_id = sorted_group[0].entity_id if sorted_group else ""
            canonical_key = _first_property_value(
                sorted_group[0].properties.get("canonical_key")
                if sorted_group
                else None
            )
            canonical_readable_id = _first_property_value(
                sorted_group[0].properties.get("human_readable_id")
                if sorted_group
                else None
            )
            canonical_name = sorted_group[0].resolve_value if sorted_group else ""
            aliases = list(
                {
                    e.resolve_value
                    for e in group
                    if e.resolve_value and e.resolve_value != canonical_name
                }
            )

            result = self.graph_store.execute_query(
                f"""
                MATCH (n)
                WHERE elementId(n) IN $node_ids
                WITH collect(n) AS nodes
                CALL apoc.refactor.mergeNodes(
                    nodes,
                    {{properties: 'combine', mergeRels: true}}
                ) YIELD node
                SET node.{escape_cypher_identifier(resolve_property)} = $canonical_name,
                    node.id = $canonical_entity_id,
                    node.canonical_key = coalesce($canonical_key, node.canonical_key),
                    node.human_readable_id = coalesce($canonical_readable_id, node.human_readable_id)
                RETURN elementId(node) AS merged_id
                """,
                {
                    "node_ids": node_ids,
                    "canonical_entity_id": canonical_entity_id,
                    "canonical_key": canonical_key,
                    "canonical_readable_id": canonical_readable_id,
                    "canonical_name": canonical_name,
                },
            )
            if result:
                merged_nodes += len(node_ids)
                if aliases:
                    merged_id = result[0].get("merged_id")
                    if merged_id:
                        try:
                            self.graph_store.execute_query(
                                """
                                MATCH (n)
                                WHERE elementId(n) = $node_id
                                SET n.aliases = coalesce(n.aliases, []) + $aliases
                                RETURN n
                                """,
                                {"node_id": merged_id, "aliases": aliases},
                            )
                        except Exception:
                            pass
        return merged_nodes
