"""Synthetic E2E test corpus, schema, prompts, and structural assertion helpers.

Uses a compact security/incident corpus with schema-friendly facts that are
domain-neutral enough for structural assertions without relying on exact LLM
output wording.
"""

from __future__ import annotations

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from tests.integration.support import cleanup_graph, single_count

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

SYNTHETIC_PAGES = [
    {
        "text": """
        Acme Security operates the Sentinel monitoring system.
        Alice Rivera is the incident commander for Acme Security.
        Sentinel detected a suspicious login incident on 2026-04-12.
        The incident affected the Payments API.
        """,
        "metadata": {"record_id": "synthetic-page-001", "collection": "e2e-synthetic"},
    },
    {
        "text": """
        Bob Chen investigated the suspicious login incident.
        The root cause was a weak password on a service account.
        Acme Security mitigated the incident by rotating credentials
        and enabling multi-factor authentication.
        """,
        "metadata": {"record_id": "synthetic-page-002", "collection": "e2e-synthetic"},
    },
    {
        "text": """
        The Payments API depends on the Identity Gateway.
        Sentinel monitors the Identity Gateway and sends alerts to Acme Security.
        Alice Rivera approved a follow-up task to review service account policies.
        """,
        "metadata": {"record_id": "synthetic-page-003", "collection": "e2e-synthetic"},
    },
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SYNTHETIC_SCHEMA = GraphSchema(
    node_types=[
        NodeType(
            label="Person",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="Organization",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="System",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="Incident",
            properties=[PropertyType(name="name", type="STRING")],
        ),
        NodeType(
            label="Mitigation",
            properties=[PropertyType(name="name", type="STRING")],
        ),
    ],
    relationship_types=[
        RelationshipType(label="OPERATES"),
        RelationshipType(label="DETECTED"),
        RelationshipType(label="AFFECTED"),
        RelationshipType(label="INVESTIGATED"),
        RelationshipType(label="MITIGATED_BY"),
        RelationshipType(label="DEPENDS_ON"),
        RelationshipType(label="MONITORS"),
        RelationshipType(label="APPROVED"),
    ],
    patterns=[
        ("Organization", "OPERATES", "System"),
        ("System", "DETECTED", "Incident"),
        ("Incident", "AFFECTED", "System"),
        ("Person", "INVESTIGATED", "Incident"),
        ("Incident", "MITIGATED_BY", "Mitigation"),
        ("System", "DEPENDS_ON", "System"),
        ("System", "MONITORS", "System"),
        ("Person", "APPROVED", "Mitigation"),
    ],
)

# Relationship types used for community detection projection.
SYNTHETIC_COMMUNITY_RELATIONSHIP_TYPES = [
    "OPERATES",
    "DETECTED",
    "AFFECTED",
    "INVESTIGATED",
    "MITIGATED_BY",
    "DEPENDS_ON",
    "MONITORS",
    "APPROVED",
]

# Neutral prompts that work with any domain.
SYNTHETIC_COMMUNITY_SUMMARY_PROMPT = """Summarize the following findings.

{context}

Provide a concise 2-4 paragraph summary identifying the key entities,
their relationships, and important patterns.

Summary:"""

SYNTHETIC_LOCAL_ANSWER_PROMPT = """Based on the findings below, answer the query.

Query: {query}

Findings:
{context}

Provide a detailed answer. If the context does not contain enough information, say so.

Answer:"""

SYNTHETIC_GLOBAL_MAP_PROMPT = """Based on this report segment, answer the question.

Question: {query}

Report Segment:
{batch_text}

Provide a partial answer. Return valid JSON only.

JSON format:
{{
  "answer": "Your partial answer...",
  "helpfulness": 50,
  "report_ids": [],
  "references": []
}}
"""

SYNTHETIC_GLOBAL_REDUCE_PROMPT = """Synthesize these perspectives into a final answer.

Question: {query}

Perspectives:
{partial_text}

Final Answer:"""

SYNTHETIC_DRIFT_ANSWER_PROMPT = """Answer the query using the specific findings, broader context, and related information.

Query: {query}

=== Specific Findings ===
{entity_context}

=== Broader Context ===
{community_context}

=== Related Information ===
{bridging_context}

Answer:"""

# ---------------------------------------------------------------------------
# Structural assertion helpers
# ---------------------------------------------------------------------------


def assert_extraction_invariants(
    graph_document,
    schema: GraphSchema,
    *,
    min_chunks: int = 1,
    min_entities: int = 2,
    min_relationships: int = 1,
) -> None:
    """Assert structural invariants on an extracted GraphDocument.

    Validates that:
    - chunk, entity, and relationship counts meet minimums
    - every entity type is in the schema labels
    - every relationship type is in the schema relationship labels
    - every relationship endpoint references an extracted entity id
    """
    assert graph_document is not None, "graph_document must not be None"
    assert len(graph_document.chunks) >= min_chunks, (
        f"Expected >= {min_chunks} chunks, got {len(graph_document.chunks)}"
    )
    assert len(graph_document.entities) >= min_entities, (
        f"Expected >= {min_entities} entities, got {len(graph_document.entities)}"
    )
    assert len(graph_document.relationships) >= min_relationships, (
        f"Expected >= {min_relationships} relationships, got {len(graph_document.relationships)}"
    )

    entity_ids = {e.id for e in graph_document.entities}
    schema_node_labels = schema.node_labels()
    schema_rel_labels = schema.relationship_labels()

    for entity in graph_document.entities:
        assert entity.type in schema_node_labels, (
            f"Entity type {entity.type!r} not in schema node labels: {schema_node_labels}"
        )

    for rel in graph_document.relationships:
        assert rel.type in schema_rel_labels, (
            f"Relationship type {rel.type!r} not in schema relationship labels: {schema_rel_labels}"
        )
        assert rel.source_id in entity_ids, (
            f"Relationship source {rel.source_id!r} not in extracted entity ids"
        )
        assert rel.target_id in entity_ids, (
            f"Relationship target {rel.target_id!r} not in extracted entity ids"
        )


def assert_graph_persisted(store, graph_name: str) -> None:
    """Assert that Document, Chunk, and Entity counts are positive for the scoped graph."""
    for label in ("Document", "Chunk", "__Entity__"):
        count = single_count(
            store,
            f"MATCH (n:{label} {{graph_name: $graph_name}}) RETURN count(n) AS count",
            graph_name,
        )
        assert count > 0, f"Expected positive {label} count, got {count}"


def assert_build_validation_positive(validation: dict) -> None:
    """Assert that build validation has positive counts for key metrics."""
    for key in ("chunk_count", "entity_count", "evidence_link_count", "entity_relationship_count"):
        assert validation.get(key, 0) > 0, (
            f"Expected {key} > 0, got {validation.get(key)}"
        )


def cleanup_synthetic(store, graph_name: str) -> None:
    """Remove all nodes scoped to a synthetic test graph_name."""
    cleanup_graph(store, graph_name)
