"""Schema-aware prompt generation for LLM graph extraction."""

from recon_graphrag.extraction.schema import GraphSchema
from recon_graphrag.extraction.types import GraphExtraction
from recon_graphrag.utils.templates import format_template, has_placeholder


class SchemaPromptBuilder:
    """Build prompts for graph extraction, gleaning, and description summarization.

    Each public ``build_*`` method can be customized by passing a string
    template to the constructor. When a template is provided, it is used as
    the user-facing instruction/context and the backend appends the standard
    structural sections (schema, rules, JSON format, observations, etc.).

    Supported optional placeholders:

    - ``extraction_prompt``: ``{text}``
    - ``assessment_prompt``: ``{text}``, ``{existing}``
    - ``continuation_prompt``: ``{text}``, ``{existing}``
    - ``claim_prompt``: ``{text}``, ``{entity_ids}``
    - ``entity_summary_prompt``: ``{descriptions}``, ``{entity_name}``, ``{entity_type}``
    - ``relationship_summary_prompt``: ``{descriptions}``, ``{source}``, ``{target}``, ``{rel_type}``

    To include a literal brace in a template, escape it by doubling it
    (e.g. ``{{not a placeholder}}``).
    """

    def __init__(
        self,
        extraction_prompt: str | None = None,
        assessment_prompt: str | None = None,
        continuation_prompt: str | None = None,
        claim_prompt: str | None = None,
        entity_summary_prompt: str | None = None,
        relationship_summary_prompt: str | None = None,
    ):
        self.extraction_prompt = extraction_prompt
        self.assessment_prompt = assessment_prompt
        self.continuation_prompt = continuation_prompt
        self.claim_prompt = claim_prompt
        self.entity_summary_prompt = entity_summary_prompt
        self.relationship_summary_prompt = relationship_summary_prompt

    def build_entity_summary_prompt(
        self,
        descriptions: list[str],
        entity_name: str,
        entity_type: str,
        properties: dict | None = None,
    ) -> str:
        observations = "\n".join(
            f"- {description}" for description in descriptions if description.strip()
        )
        attributes = ""
        if properties:
            attribute_lines = "\n".join(
                f"- {key}: {value}" for key, value in sorted(properties.items())
            )
            attributes = f"\nKnown attributes:\n{attribute_lines}\n"

        base = (
            self.entity_summary_prompt
            or """\
Summarize the following observations about a single entity into one concise description.
Use only the provided observations and attributes. Return plain text only, with no JSON or markdown."""
        )

        has_descriptions = has_placeholder(base, "descriptions")
        has_name = has_placeholder(base, "entity_name")
        has_type = has_placeholder(base, "entity_type")
        base = format_template(
            base,
            descriptions=observations,
            entity_name=entity_name,
            entity_type=entity_type,
        )

        parts = [base]
        if not has_name:
            parts.append(f"Entity name: {entity_name}")
        if not has_type:
            parts.append(f"Entity type: {entity_type}")
        if attributes:
            parts.append(attributes.strip())
        if not has_descriptions:
            parts.append(f"Observations:\n{observations}")

        return "\n\n".join(parts)

    def build_relationship_summary_prompt(
        self, descriptions: list[str], source: str, target: str, rel_type: str
    ) -> str:
        observations = "\n".join(
            f"- {description}" for description in descriptions if description.strip()
        )

        base = (
            self.relationship_summary_prompt
            or """\
Summarize the following observations about a single relationship into one concise description.
Use only the provided observations. Return plain text only, with no JSON or markdown."""
        )

        has_descriptions = has_placeholder(base, "descriptions")
        has_source = has_placeholder(base, "source")
        has_target = has_placeholder(base, "target")
        has_rel_type = has_placeholder(base, "rel_type")
        base = format_template(
            base,
            descriptions=observations,
            source=source,
            target=target,
            rel_type=rel_type,
        )

        parts = [base]
        if not (has_source and has_target and has_rel_type):
            parts.append(f"Relationship: {source} -[{rel_type}]-> {target}")
        if not has_descriptions:
            parts.append(f"Observations:\n{observations}")

        return "\n\n".join(parts)

    def build_prompt(self, text: str, schema: GraphSchema) -> str:
        schema.validate()

        node_section = self._format_nodes(schema)
        relationship_section = self._format_relationships(schema)
        pattern_section = self._format_patterns(schema)

        base = (
            self.extraction_prompt
            or "You are extracting a knowledge graph from text."
        )
        has_text = has_placeholder(base, "text")
        base = format_template(base, text=text)

        parts = [
            base,
            f"Allowed node types:\n{node_section}",
            f"Allowed relationship types:\n{relationship_section}",
            f"Allowed relationship patterns:\n{pattern_section}",
            """Rules:
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
    but explicit relationships.""",
            """JSON format:
{
  "nodes": [
    {
      "id": "person:example",
      "label": "Person",
      "properties": {
        "name": "Example"
      }
    }
  ],
  "relationships": [
    {
      "source_id": "person:example",
      "target_id": "movie:example",
      "type": "ACTED_IN",
      "properties": {
        "description": "The text states that Example acted in Example.",
        "weight": 1.0
      }
    }
  ]
}""",
        ]
        if not has_text:
            parts.append(f"Text:\n{text}")

        return "\n\n".join(parts)

    def build_assessment_prompt(
        self, text: str, schema: GraphSchema, current: GraphExtraction
    ) -> str:
        """Build a prompt asking the LLM if it missed any entities."""
        schema.validate()

        node_section = self._format_nodes(schema)
        relationship_section = self._format_relationships(schema)
        existing_section = self._format_existing(current)

        base = (
            self.assessment_prompt
            or """\
You previously extracted a knowledge graph from text. Review whether you
missed any important entities or relationships."""
        )
        has_text = has_placeholder(base, "text")
        has_existing = has_placeholder(base, "existing")
        base = format_template(base, text=text, existing=existing_section)

        parts = [
            base,
            f"Allowed node types:\n{node_section}",
            f"Allowed relationship types:\n{relationship_section}",
        ]
        if not has_existing:
            parts.append(f"Already extracted:\n{existing_section}")
        parts.append("""Rules:
1. Answer only "yes" or "no".
2. Say "yes" only if there are important entities or relationships clearly
   supported by the text that are NOT in the already-extracted list.
3. Do not suggest duplicates or minor variations of existing items.""")
        if not has_text:
            parts.append(f"Text:\n{text}")
        parts.append(
            'Did you miss any entities or relationships? Answer only "yes" or "no".'
        )

        return "\n\n".join(parts)

    def build_continuation_prompt(
        self, text: str, schema: GraphSchema, current: GraphExtraction
    ) -> str:
        """Build a prompt asking the LLM to extract only missed items."""
        schema.validate()

        node_section = self._format_nodes(schema)
        relationship_section = self._format_relationships(schema)
        pattern_section = self._format_patterns(schema)
        existing_section = self._format_existing(current)

        base = (
            self.continuation_prompt
            or """\
You previously extracted a knowledge graph from text, but missed some items.
Extract ONLY the missing entities and relationships."""
        )
        has_text = has_placeholder(base, "text")
        has_existing = has_placeholder(base, "existing")
        base = format_template(base, text=text, existing=existing_section)

        parts = [
            base,
            f"Allowed node types:\n{node_section}",
            f"Allowed relationship types:\n{relationship_section}",
            f"Allowed relationship patterns:\n{pattern_section}",
        ]
        if not has_existing:
            parts.append(f"Already extracted (do NOT duplicate these):\n{existing_section}")
        parts.append("""Rules:
1. Extract only NEW items not in the already-extracted list.
2. Use the same ID format: "<label>:<normalized-name>".
3. Every relationship endpoint must refer to a node (new or already extracted).
4. Return valid JSON only. Do not include markdown.
5. If there are no missing items, return {"nodes": [], "relationships": []}.""")
        parts.append("""JSON format:
{
  "nodes": [
    {
      "id": "person:example",
      "label": "Person",
      "properties": {
        "name": "Example"
      }
    }
  ],
  "relationships": [
    {
      "source_id": "person:example",
      "target_id": "movie:example",
      "type": "ACTED_IN",
      "properties": {
        "description": "The text states that Example acted in Example.",
        "weight": 1.0
      }
    }
  ]
}""")
        if not has_text:
            parts.append(f"Text:\n{text}")

        return "\n\n".join(parts)

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

    def build_claim_prompt(
        self,
        text: str,
        entity_ids: list[str],
    ) -> str:
        """Build a prompt to extract claims/covariates about entities.

        Args:
            text: The source text to extract claims from.
            entity_ids: IDs of entities already extracted from this text.
                Claims must reference one of these IDs.

        Returns:
            Prompt string for claim extraction.
        """
        entity_list = "\n".join(f"  - {eid}" for eid in entity_ids)

        base = (
            self.claim_prompt
            or "You are extracting claims, assertions, and covariates about entities from text."
        )
        has_text = has_placeholder(base, "text")
        has_entity_ids = has_placeholder(base, "entity_ids")
        # Inline substitution gets a comma-separated list; the appended
        # fallback section keeps the block format.
        base = format_template(base, text=text, entity_ids=", ".join(entity_ids))

        parts = [base]
        if not has_entity_ids:
            parts.append(f"Known entities (use these exact IDs as subject_entity_id):\n{entity_list}")
        parts.append("""Rules:
1. Extract only claims explicitly supported by the text.
2. Each claim must have a subject_entity_id matching one of the known entities.
3. claim_type should be a short label for the kind of claim (e.g. "role",
   "status", "opinion", "action", "attribute", "event").
4. description must be a nonempty string supported by the source. Do not invent placeholder claims.
5. status is "active" by default; use "resolved", "expired", or "rejected"
   only if the text explicitly indicates a state change.
6. object_entity_id may reference another known entity when the claim is about
   a relationship or assertion involving another entity.
7. source_text should be a short source excerpt supporting the claim.
8. Use null for unavailable dates, object IDs, excerpts, and text_unit_id.
9. Return valid JSON only: an object with a claims array. Do not include markdown.
10. If there are no valid claims, return {"claims": []}.""")
        parts.append("""JSON format:
{"claims": [
  {
    "subject_entity_id": "person:example",
    "claim_type": "role",
    "description": "Example held the position of CEO.",
    "status": "active",
    "start_date": null,
    "end_date": null,
    "object_entity_id": null,
    "source_text": "Example was appointed CEO.",
    "text_unit_id": null
  }
]}""")
        if not has_text:
            parts.append(f"Text:\n{text}")

        return "\n\n".join(parts)
