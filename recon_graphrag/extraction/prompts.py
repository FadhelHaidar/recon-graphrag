"""Schema-aware prompt generation for LLM graph extraction."""

from recon_graphrag.extraction.schema import GraphSchema


class SchemaPromptBuilder:
    def build_prompt(self, text: str, schema: GraphSchema) -> str:
        schema.validate()

        node_section = self._format_nodes(schema)
        relationship_section = self._format_relationships(schema)
        pattern_section = self._format_patterns(schema)

        return f"""
You are extracting a knowledge graph from text.

Allowed node types:
{node_section}

Allowed relationship types:
{relationship_section}

Allowed relationship patterns:
{pattern_section}

Rules:
1. Extract only facts explicitly supported by the text.
2. Use only the allowed node labels.
3. Use only the allowed relationship types.
4. Use only the allowed relationship patterns.
5. Every relationship source_id and target_id must refer to a node in "nodes".
6. Every node must have a stable "id".
7. Prefer IDs in this format: "<label>:<normalized-name>".
8. Return valid JSON only. Do not include markdown.
9. If there are no valid nodes or relationships, return empty arrays.

JSON format:
{{
  "nodes": [
    {{
      "id": "person:example",
      "label": "Person",
      "properties": {{
        "name": "Example"
      }}
    }}
  ],
  "relationships": [
    {{
      "source_id": "person:example",
      "target_id": "movie:example",
      "type": "ACTED_IN",
      "properties": {{
        "description": "The text states that Example acted in Example."
      }}
    }}
  ]
}}

Text:
{text}
""".strip()

    def _format_nodes(self, schema: GraphSchema) -> str:
        lines = []
        for node in schema.node_types:
            props = ", ".join(
                f"{prop.name}: {prop.type}" for prop in node.properties
            )
            lines.append(
                f"- {node.label}: {node.description}\n"
                f"  Identity property: {node.identity_property}\n"
                f"  Properties: {props or 'none'}"
            )
        return "\n".join(lines)

    def _format_relationships(self, schema: GraphSchema) -> str:
        return "\n".join(
            f"- {rel.label}: {rel.description}"
            for rel in schema.relationship_types
        )

    def _format_patterns(self, schema: GraphSchema) -> str:
        return "\n".join(
            f"- {source} -[{rel}]-> {target}"
            for source, rel, target in schema.patterns
        )
