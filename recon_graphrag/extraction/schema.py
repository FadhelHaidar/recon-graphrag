"""GraphSchema definition and builder helpers.

Users define their own domain schema using internal GraphSchema
primitives, then pass it to the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


PropertyDataType = Literal[
    "STRING",
    "INTEGER",
    "FLOAT",
    "BOOLEAN",
    "DATE",
    "DATETIME",
    "LIST",
]


@dataclass(frozen=True)
class PropertyType:
    name: str
    type: PropertyDataType = "STRING"
    description: str = ""
    required: bool = False


@dataclass(frozen=True)
class NodeType:
    label: str
    description: str = ""
    properties: list[PropertyType] = field(default_factory=list)
    identity_property: str = "name"

    @property
    def property_names(self) -> set[str]:
        return {prop.name for prop in self.properties}


@dataclass(frozen=True)
class RelationshipType:
    label: str
    description: str = ""
    properties: list[PropertyType] = field(default_factory=list)

    @property
    def property_names(self) -> set[str]:
        return {prop.name for prop in self.properties}


@dataclass(frozen=True)
class GraphSchema:
    node_types: list[NodeType]
    relationship_types: list[RelationshipType]
    patterns: list[tuple[str, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_validated", True)
        self.validate()

    def node_labels(self) -> set[str]:
        return {node.label for node in self.node_types}

    def relationship_labels(self) -> set[str]:
        return {rel.label for rel in self.relationship_types}

    def pattern_set(self) -> set[tuple[str, str, str]]:
        return set(self.patterns)

    def get_node_type(self, label: str) -> Optional[NodeType]:
        return next((node for node in self.node_types if node.label == label), None)

    def get_relationship_type(self, label: str) -> Optional[RelationshipType]:
        return next(
            (rel for rel in self.relationship_types if rel.label == label),
            None,
        )

    def is_valid_pattern(
        self,
        source_label: str,
        relationship_label: str,
        target_label: str,
    ) -> bool:
        if not self.patterns:
            return relationship_label in self.relationship_labels()

        return (source_label, relationship_label, target_label) in self.pattern_set()

    def validate(self) -> None:
        node_labels = self.node_labels()
        relationship_labels = self.relationship_labels()

        duplicate_nodes = _find_duplicates([node.label for node in self.node_types])
        duplicate_rels = _find_duplicates(
            [rel.label for rel in self.relationship_types]
        )

        if duplicate_nodes:
            raise ValueError(f"Duplicate node labels: {sorted(duplicate_nodes)}")

        if duplicate_rels:
            raise ValueError(
                f"Duplicate relationship labels: {sorted(duplicate_rels)}"
            )

        for source, rel, target in self.patterns:
            if source not in node_labels:
                raise ValueError(f"Pattern uses unknown source node label: {source}")
            if target not in node_labels:
                raise ValueError(f"Pattern uses unknown target node label: {target}")
            if rel not in relationship_labels:
                raise ValueError(f"Pattern uses unknown relationship label: {rel}")


def _find_duplicates(values: list[str]) -> set[str]:
    seen = set()
    duplicates = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    return duplicates


__all__ = [
    "GraphSchema",
    "NodeType",
    "PropertyType",
    "RelationshipType",
    "build_schema",
]


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
        for prop in nt.get("properties", []):
            if isinstance(prop, str):
                props.append(PropertyType(name=prop, type="STRING"))
            elif isinstance(prop, dict):
                props.append(
                    PropertyType(
                        name=prop["name"],
                        type=prop.get("type", "STRING"),
                        description=prop.get("description", ""),
                        required=prop.get("required", False),
                    )
                )

        nodes.append(
            NodeType(
                label=nt["label"],
                description=nt.get("description", ""),
                properties=props,
                identity_property=nt.get("identity_property", "name"),
            )
        )

    rels = []
    for rt in relationship_types:
        props = []
        for prop in rt.get("properties", []):
            if isinstance(prop, str):
                props.append(PropertyType(name=prop, type="STRING"))
            elif isinstance(prop, dict):
                props.append(
                    PropertyType(
                        name=prop["name"],
                        type=prop.get("type", "STRING"),
                        description=prop.get("description", ""),
                        required=prop.get("required", False),
                    )
                )

        rels.append(
            RelationshipType(
                label=rt["label"],
                description=rt.get("description", ""),
                properties=props,
            )
        )

    schema = GraphSchema(
        node_types=nodes,
        relationship_types=rels,
        patterns=patterns,
    )
    schema.validate()
    return schema
