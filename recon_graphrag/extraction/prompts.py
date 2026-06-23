"""Schema-aware prompt generation for LLM graph extraction."""

from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.types import GraphExtraction


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
10. Every relationship should include a numeric "weight" property. Use 1.0
    for a normal explicit relationship, higher values for unusually strong
    or repeatedly supported relationships, and lower positive values for weak
    but explicit relationships.

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
        "description": "The text states that Example acted in Example.",
        "weight": 1.0
      }}
    }}
  ]
}}

Text:
{text}
""".strip()

    def build_assessment_prompt(
        self, text: str, schema: GraphSchema, current: GraphExtraction
    ) -> str:
        """Build a prompt asking the LLM if it missed any entities."""
        schema.validate()

        node_section = self._format_nodes(schema)
        relationship_section = self._format_relationships(schema)
        existing_section = self._format_existing(current)

        return f"""
You previously extracted a knowledge graph from text. Review whether you
missed any important entities or relationships.

Allowed node types:
{node_section}

Allowed relationship types:
{relationship_section}

Already extracted:
{existing_section}

Rules:
1. Answer only "yes" or "no".
2. Say "yes" only if there are important entities or relationships clearly
   supported by the text that are NOT in the already-extracted list.
3. Do not suggest duplicates or minor variations of existing items.

Text:
{text}

Did you miss any entities or relationships? Answer only "yes" or "no".
""".strip()

    def build_continuation_prompt(
        self, text: str, schema: GraphSchema, current: GraphExtraction
    ) -> str:
        """Build a prompt asking the LLM to extract only missed items."""
        schema.validate()

        node_section = self._format_nodes(schema)
        relationship_section = self._format_relationships(schema)
        pattern_section = self._format_patterns(schema)
        existing_section = self._format_existing(current)

        return f"""
You previously extracted a knowledge graph from text, but missed some items.
Extract ONLY the missing entities and relationships.

Allowed node types:
{node_section}

Allowed relationship types:
{relationship_section}

Allowed relationship patterns:
{pattern_section}

Already extracted (do NOT duplicate these):
{existing_section}

Rules:
1. Extract only NEW items not in the already-extracted list.
2. Use the same ID format: "<label>:<normalized-name>".
3. Every relationship endpoint must refer to a node (new or already extracted).
4. Return valid JSON only. Do not include markdown.
5. If there are no missing items, return {{"nodes": [], "relationships": []}}.

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
        "description": "The text states that Example acted in Example.",
        "weight": 1.0
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

    @staticmethod
    def _format_existing(extraction: GraphExtraction) -> str:
        """Format already-extracted items as compact reference text."""
        lines = []
        if extraction.nodes:
            lines.append("Entities:")
            for node in extraction.nodes:
                desc = node.properties.get("description", "")
                name = node.properties.get("name", node.id)
                lines.append(f"  - {node.id} ({node.label}): {name}")
                if desc:
                    lines.append(f"    Description: {desc}")
        if extraction.relationships:
            lines.append("Relationships:")
            for rel in extraction.relationships:
                lines.append(f"  - {rel.source_id} -[{rel.type}]-> {rel.target_id}")
        return "\n".join(lines) if lines else "(none)"
