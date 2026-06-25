"""Memgraph-specific entity resolution implementation."""

from __future__ import annotations

from recon_graphrag.graphdb.entity_resolution import (
    BaseEntityResolver,
    _EntityRecord,
    _first_property_value,
    _normalize_name,
)
from recon_graphrag.graphdb.memgraph.cypher import escape_cypher_identifier


class _MemgraphEntityResolver(BaseEntityResolver):
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
              id(e) AS node_id,
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
                    node_id=row.get("node_id", 0),
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

            canonical_id = node_ids[0]
            other_ids = node_ids[1:]
            if not other_ids:
                continue

            combined_props = self._combine_properties(node_ids)
            combined_props["id"] = canonical_entity_id
            if canonical_key:
                combined_props["canonical_key"] = canonical_key
            if canonical_readable_id:
                combined_props["human_readable_id"] = canonical_readable_id
            combined_props[resolve_property] = canonical_name
            if aliases:
                existing_aliases = combined_props.get("aliases", [])
                if not isinstance(existing_aliases, list):
                    existing_aliases = [existing_aliases]
                combined_props["aliases"] = list(
                    dict.fromkeys(existing_aliases + aliases)
                )

            self.graph_store.execute_query(
                """
                MATCH (n)
                WHERE id(n) = $canonical_id
                SET n += $props
                RETURN id(n) AS merged_id
                """,
                {"canonical_id": canonical_id, "props": combined_props},
            )

            self._rewire_relationships(canonical_id, other_ids)

            self.graph_store.execute_query(
                """
                MATCH (n)
                WHERE id(n) IN $other_ids
                DELETE n
                """,
                {"other_ids": other_ids},
            )

            merged_nodes += len(node_ids)

        return merged_nodes

    def _combine_properties(self, node_ids: list[int]) -> dict:
        rows = self.graph_store.execute_query(
            """
            MATCH (n)
            WHERE id(n) IN $node_ids
            RETURN id(n) AS node_id, properties(n) AS props
            """,
            {"node_ids": node_ids},
        )
        combined: dict = {}
        for row in rows:
            props = row.get("props", {}) or {}
            for key, value in props.items():
                if key not in combined:
                    combined[key] = value
                elif isinstance(combined[key], list) and isinstance(value, list):
                    for v in value:
                        if v not in combined[key]:
                            combined[key].append(v)
                elif isinstance(combined[key], list):
                    if value not in combined[key]:
                        combined[key].append(value)
                elif isinstance(value, list):
                    combined[key] = [combined[key]] + [
                        item for item in value if item != combined[key]
                    ]
                elif combined[key] != value:
                    combined[key] = [combined[key], value]
        return combined

    def _rewire_relationships(
        self, canonical_id: int, other_ids: list[int]
    ) -> None:
        merged_ids = [canonical_id] + other_ids
        type_rows = self.graph_store.execute_query(
            """
            MATCH (n)-[r]-(other)
            WHERE id(n) IN $other_ids
            RETURN DISTINCT type(r) AS rel_type, id(n) AS node_id
            """,
            {"other_ids": other_ids},
        )

        types_by_node: dict = {}
        for row in type_rows:
            rel_type = row.get("rel_type")
            node_id = row.get("node_id")
            if rel_type and node_id is not None:
                types_by_node.setdefault(node_id, []).append(rel_type)

        for node_id, rel_types in types_by_node.items():
            for rel_type in set(rel_types):
                escaped_rel = escape_cypher_identifier(rel_type)
                self.graph_store.execute_query(
                    f"""
                    MATCH (source)-[r:{escaped_rel}]->(target)
                    WHERE id(source) = $node_id
                    WITH r, target, properties(r) AS rel_props
                    MATCH (canonical)
                    WHERE id(canonical) = $canonical_id
                    WITH r, canonical, target, rel_props
                    WHERE NOT id(target) IN $merged_ids
                    MERGE (canonical)-[new_r:{escaped_rel}]->(target)
                    SET new_r += rel_props
                    DELETE r
                    """,
                    {
                        "node_id": node_id,
                        "canonical_id": canonical_id,
                        "merged_ids": merged_ids,
                    },
                )
                self.graph_store.execute_query(
                    f"""
                    MATCH (source)-[r:{escaped_rel}]->(target)
                    WHERE id(source) = $node_id
                      AND id(target) IN $merged_ids
                    DELETE r
                    """,
                    {"node_id": node_id, "merged_ids": merged_ids},
                )
                self.graph_store.execute_query(
                    f"""
                    MATCH (source)-[r:{escaped_rel}]->(target)
                    WHERE id(target) = $node_id
                    WITH r, source, properties(r) AS rel_props
                    MATCH (canonical)
                    WHERE id(canonical) = $canonical_id
                    WITH r, source, canonical, rel_props
                    WHERE NOT id(source) IN $merged_ids
                    MERGE (source)-[new_r:{escaped_rel}]->(canonical)
                    SET new_r += rel_props
                    DELETE r
                    """,
                    {
                        "node_id": node_id,
                        "canonical_id": canonical_id,
                        "merged_ids": merged_ids,
                    },
                )
                self.graph_store.execute_query(
                    f"""
                    MATCH (source)-[r:{escaped_rel}]->(target)
                    WHERE id(target) = $node_id
                      AND id(source) IN $merged_ids
                    DELETE r
                    """,
                    {"node_id": node_id, "merged_ids": merged_ids},
                )
