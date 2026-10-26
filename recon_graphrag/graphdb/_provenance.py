"""Property-compatible per-document contribution records shared by writers/resolvers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from recon_graphrag.graphdb.base import GraphStore
    from recon_graphrag.graphdb.entity_resolution import _EntityRecord


def contribution_entries(value: Any) -> list[list[Any]]:
    """Read JSON pairs, including singleton lists collapsed by APOC combine."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)):
        raise ValueError("Invalid contribution provenance; re-ingest the affected documents.")
    entries = []
    for item in items:
        try:
            pair = json.loads(item) if isinstance(item, str) else item
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid contribution provenance; re-ingest the affected documents.") from exc
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not isinstance(pair[0], str):
            raise ValueError("Invalid contribution provenance; re-ingest the affected documents.")
        if list(pair) not in entries:
            entries.append(list(pair))
    return entries


def encode_contributions(entries: list[list[Any]]) -> list[str]:
    """Graph properties cannot hold nested lists or maps."""
    return [json.dumps(entry, ensure_ascii=False) for entry in entries]


def property_list(value: Any) -> list:
    return value if isinstance(value, list) else ([] if value is None else [value])


def build_uid(graph_name: str, node_id: str) -> str:
    # Length-prefix keeps graph names containing colons unambiguous.
    return f"{len(graph_name)}:{graph_name}:{node_id}"


def raw_descriptions(properties: dict) -> list[str]:
    values = properties.get("descriptions")
    if values is None:
        values = properties.get("description")
    return [str(text) for text in property_list(values) if str(text).strip()]


def reject_mixed_ownership(records: list[dict]) -> None:
    """Do not let merge deduplication hide an unowned legacy contribution."""
    for key, value_key in (("description_sources", "descriptions"), ("strength_sources", "strength")):
        tracked = False
        unknown = False
        for props in records:
            entries = contribution_entries(props.get(key))
            tracked = tracked or bool(entries)
            if value_key == "descriptions":
                values = raw_descriptions(props)
                unknown = unknown or any(text and text not in [value for _, value in entries] for text in values)
            else:
                unknown = unknown or (props.get(value_key) is not None and not entries)
        if tracked and unknown:
            raise ValueError(
                f"Cannot merge mixed tracked/legacy {value_key} ownership; "
                "re-ingest the affected documents before resolution."
            )


def preflight_merge_ownership(
    graph_store: GraphStore, groups: list[list[_EntityRecord]], node_id_function: str
) -> None:
    """Check only nodes and coincident incident domain edges in these groups."""
    merged_node_ids = {
        entity.node_id: group[0].node_id for group in groups for entity in group
    }
    for group in groups:
        reject_mixed_ownership([entity.properties for entity in group])
        node_ids = [entity.node_id for entity in group]
        rows = graph_store.execute_query(
            f"""
            MATCH (s:__Entity__ {{graph_name: $graph_name}})-[r]->(t:__Entity__ {{graph_name: $graph_name}})
            WHERE {node_id_function}(s) IN $node_ids OR {node_id_function}(t) IN $node_ids
            RETURN {node_id_function}(s) AS source_node_id,
                   {node_id_function}(t) AS target_node_id,
                   type(r) AS rel_type, properties(r) AS rel_props
            """,
            {"graph_name": group[0].graph_name, "node_ids": node_ids},
        )
        edges: dict[tuple, list[dict]] = {}
        for row in rows:
            source = row["source_node_id"]
            target = row["target_node_id"]
            key = (merged_node_ids.get(source, source),
                   merged_node_ids.get(target, target), row["rel_type"])
            edges.setdefault(key, []).append(row["rel_props"])
        for records in edges.values():
            reject_mixed_ownership(records)
