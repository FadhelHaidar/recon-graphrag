"""Schema validation for extracted graph data."""

from typing import Any

from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.types import (
    ExtractedNode,
    ExtractedRelationship,
    GraphExtraction,
)


class SchemaValidator:
    def validate(
        self,
        extraction: GraphExtraction,
        schema: GraphSchema,
    ) -> GraphExtraction:
        schema.validate()

        allowed_node_labels = schema.node_labels()
        allowed_rel_labels = schema.relationship_labels()

        valid_nodes = []
        node_by_id = {}

        for node in extraction.nodes:
            if node.label not in allowed_node_labels:
                continue

            node_type = schema.get_node_type(node.label)
            properties = self._filter_properties(node.properties, node_type)

            identity_value = (
                properties.get(node_type.identity_property)
                if node_type
                else None
            ) or properties.get("name") or properties.get("title") or node.id

            properties["name"] = str(identity_value)
            properties.setdefault("description", "")

            clean_node = ExtractedNode(
                id=node.id,
                label=node.label,
                properties=properties,
            )
            valid_nodes.append(clean_node)
            node_by_id[clean_node.id] = clean_node

        valid_relationships = []

        for rel in extraction.relationships:
            if rel.type not in allowed_rel_labels:
                continue

            source = node_by_id.get(rel.source_id)
            target = node_by_id.get(rel.target_id)
            if not source or not target:
                continue

            if not schema.is_valid_pattern(source.label, rel.type, target.label):
                continue

            rel_type = schema.get_relationship_type(rel.type)
            properties = self._filter_properties(rel.properties, rel_type)
            properties.setdefault("description", "")
            properties.setdefault("weight", 1.0)

            valid_relationships.append(
                ExtractedRelationship(
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    type=rel.type,
                    properties=properties,
                )
            )

        return GraphExtraction(
            nodes=valid_nodes,
            relationships=valid_relationships,
        )

    def _filter_properties(self, properties, schema_type):
        allowed = schema_type.property_names if schema_type else set()
        property_types = {
            prop.name: prop.type
            for prop in (schema_type.properties if schema_type else [])
        }
        base_allowed = {"name", "description", "weight"}

        return {
            key: self._normalize_property_value(
                value,
                property_types.get(key, self._base_property_type(key)),
            )
            for key, value in properties.items()
            if key in allowed or key in base_allowed
        }

    def _base_property_type(self, key: str) -> str:
        if key in {"name", "description"}:
            return "STRING"
        if key == "weight":
            return "FLOAT"
        return "STRING"

    def _normalize_property_value(self, value: Any, property_type: str) -> Any:
        if property_type == "STRING":
            return self._stringify_value(value)
        if property_type == "FLOAT":
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
        if property_type == "INTEGER":
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if property_type == "BOOLEAN":
            return bool(value)
        if property_type == "LIST":
            return value if isinstance(value, list) else [value]
        return value

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return ", ".join(
                f"{key}: {self._stringify_value(item)}"
                for key, item in value.items()
                if item is not None
            )
        if isinstance(value, (list, tuple, set)):
            return ", ".join(
                text
                for item in value
                if (text := self._stringify_value(item))
            )
        return str(value)
