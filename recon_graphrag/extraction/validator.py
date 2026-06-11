"""Schema validation for extracted graph data."""

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
        base_allowed = {"name", "description", "weight"}

        return {
            key: value
            for key, value in properties.items()
            if key in allowed or key in base_allowed
        }
