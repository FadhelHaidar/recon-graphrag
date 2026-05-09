"""GraphSchema re-exports and builder helpers.

Users define their own domain schema using neo4j-graphrag's GraphSchema
primitives, then pass it to the pipeline.
"""

from __future__ import annotations

from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)

__all__ = ["GraphSchema", "NodeType", "PropertyType", "RelationshipType"]


def build_schema(
    node_types: list[dict],
    relationship_types: list[dict],
    patterns: list[tuple[str, str, str]],
) -> GraphSchema:
    """Build a GraphSchema from dicts and pattern tuples.

    Args:
        node_types: List of {"label": str, "description": str, "properties": list[str]}.
            Properties can be simple names or {"name": str, "type": str} dicts.
        relationship_types: List of {"label": str, "description": str}.
        patterns: List of (source_label, relation_label, target_label) tuples.

    Returns:
        A GraphSchema instance ready for the pipeline.

    Example:
        schema = build_schema(
            node_types=[
                {"label": "Company", "description": "A company", "properties": ["name"]},
                {"label": "Product", "description": "A product", "properties": ["name", "brand"]},
            ],
            relationship_types=[
                {"label": "SUPPLIES", "description": "Company supplies product"},
            ],
            patterns=[
                ("Company", "SUPPLIES", "Product"),
            ],
        )
    """
    nodes = []
    for nt in node_types:
        props = []
        for p in nt.get("properties", []):
            if isinstance(p, str):
                props.append(PropertyType(name=p, type="STRING"))
            elif isinstance(p, dict):
                props.append(PropertyType(name=p["name"], type=p.get("type", "STRING")))
        nodes.append(NodeType(
            label=nt["label"],
            description=nt.get("description", ""),
            properties=props,
        ))

    rels = [
        RelationshipType(label=rt["label"], description=rt.get("description", ""))
        for rt in relationship_types
    ]

    return GraphSchema(node_types=nodes, relationship_types=rels, patterns=patterns)
