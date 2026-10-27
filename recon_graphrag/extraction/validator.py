"""Schema validation for extracted graph data."""

import math
from datetime import date, datetime
from typing import Any

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    RelationshipType,
)
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

        for index, node in enumerate(extraction.nodes):
            if node.label not in allowed_node_labels:
                continue

            node_type = schema.get_node_type(node.label)
            assert node_type is not None
            path = f"nodes[{index}].properties"
            properties = self._filter_properties(node.properties, node_type, path)
            # Check supplied values before standard STRING normalization.
            self._validate_properties(properties, node_type, path, required=False)

            identity_value = (
                properties.get(node_type.identity_property)
                if node_type
                else None
            ) or properties.get("name") or properties.get("title") or node.id

            declared_types = {prop.name: prop.type for prop in node_type.properties}
            # Explicit schema overrides retain their supplied type and required
            # semantics; only the historical STRING fields receive defaults.
            if declared_types.get("name", "STRING") == "STRING":
                properties["name"] = str(identity_value)
            if declared_types.get("description", "STRING") == "STRING":
                properties.setdefault("description", "")
            self._validate_properties(properties, node_type, path)

            clean_node = ExtractedNode(
                id=node.id,
                label=node.label,
                properties=properties,
                canonical_key_trusted=node.canonical_key_trusted,
            )
            valid_nodes.append(clean_node)
            node_by_id[clean_node.id] = clean_node

        valid_relationships = []

        for index, rel in enumerate(extraction.relationships):
            if rel.type not in allowed_rel_labels:
                continue

            source = node_by_id.get(rel.source_id)
            target = node_by_id.get(rel.target_id)
            if not source or not target:
                continue

            if not schema.is_valid_pattern(source.label, rel.type, target.label):
                continue

            rel_type = schema.get_relationship_type(rel.type)
            assert rel_type is not None
            path = f"relationships[{index}].properties"
            properties = self._filter_properties(rel.properties, rel_type, path)
            declared_types = {prop.name: prop.type for prop in rel_type.properties}
            if declared_types.get("description", "STRING") == "STRING":
                properties.setdefault("description", "")
            properties.setdefault("weight", 1 if declared_types.get("weight") == "INTEGER" else 1.0)
            self._validate_properties(properties, rel_type, path)
            if not self._is_finite_number(properties["weight"]):
                raise ValueError(f"{path}.weight: expected a finite number")

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

    def _filter_properties(self, properties, schema_type, path):
        if not isinstance(properties, dict):
            raise ValueError(f"{path}: expected an object")
        allowed = schema_type.property_names if schema_type else set()
        base_allowed = {"name", "description", "weight"}

        return {
            key: value
            for key, value in properties.items()
            if key in allowed or key in base_allowed
        }

    def _validate_properties(
        self,
        properties: dict[str, Any],
        schema_type: NodeType | RelationshipType,
        path: str,
        *,
        required: bool = True,
    ) -> None:
        declared = {prop.name: prop for prop in schema_type.properties}
        if required:
            for prop in declared.values():
                if prop.required and properties.get(prop.name) is None:
                    raise ValueError(f"{path}.{prop.name}: required property is missing or null")

        for key, value in properties.items():
            if value is None:
                continue
            prop = declared.get(key)
            ptype = prop.type if prop else (
                "STRING" if key in {"name", "description"} else "FLOAT"
            )
            if not self._matches_type(value, ptype):
                raise ValueError(f"{path}.{key}: expected {ptype}")

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    def _matches_type(self, value: Any, ptype: str) -> bool:
        # LLM JSON is already typed: row-specific stringification, CSV splitting,
        # truthy aliases, and lossy integer conversions do not apply here.
        if ptype == "STRING":
            return isinstance(value, str)
        if ptype == "INTEGER":
            return type(value) is int
        if ptype == "FLOAT":
            return self._is_finite_number(value)
        if ptype == "BOOLEAN":
            return type(value) is bool
        if ptype in {"DATE", "DATETIME"}:
            if not isinstance(value, str):
                return False
            try:
                if ptype == "DATE":
                    date.fromisoformat(value)
                else:
                    datetime.fromisoformat(value)
            except ValueError:
                return False
            return True
        if ptype == "LIST":
            return isinstance(value, list) and all(
                isinstance(item, str)
                or type(item) is bool
                or self._is_finite_number(item)
                for item in value
            )
        return False
