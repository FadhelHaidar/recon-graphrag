"""Corpus, schema, prompts, and inspection helpers for the stage-by-stage run.

Three short incident pages carry deliberate duplicate surface forms so the
resolution stage has one unambiguous expected outcome. The resolution rows are
seeded directly from structured rows (no LLM), so merge decisions stay
deterministic even though the extraction, summarization, community, and
retrieval stages use real providers.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from recon_graphrag.extraction.structured import ColumnEntity, RowMapping

STAGE_SCHEMA = GraphSchema(
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
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="detected_on", type="DATE"),
                PropertyType(name="root_cause", type="STRING"),
            ],
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
        RelationshipType(label="APPROVED"),
        RelationshipType(label="DEPENDS_ON"),
    ],
    patterns=[
        ("Organization", "OPERATES", "System"),
        ("System", "DETECTED", "Incident"),
        ("Incident", "AFFECTED", "System"),
        ("Person", "INVESTIGATED", "Incident"),
        ("Incident", "MITIGATED_BY", "Mitigation"),
        ("Person", "APPROVED", "Mitigation"),
        ("System", "DEPENDS_ON", "System"),
    ],
)

# "alice chen" and "ALICE CHEN" are deliberate duplicates of "Alice Chen";
# "A. Chen" is deliberately NOT a normalized duplicate (it normalizes to
# "achen"), so the resolution stage shows both a merge and a non-merge.
STAGE_PAGES = [
    {
        "text": """
        Northwind Health operates the Atlas Records system.
        Alice Chen is the incident commander at Northwind Health.
        Atlas Records detected a ransomware incident on 2026-05-03.
        The ransomware incident affected the Billing API.
        """,
        "metadata": {"record_id": "stage-page-001", "collection": "stage-scenario"},
    },
    {
        "text": """
        Bob Ortiz investigated the ransomware incident.
        alice chen approved the credential rotation that mitigated the incident.
        The root cause was a leaked service-account key.
        """,
        "metadata": {"record_id": "stage-page-002", "collection": "stage-scenario"},
    },
    {
        "text": """
        Northwind Health mitigated the ransomware incident by rolling out
        multi-factor authentication. ALICE CHEN signed off on the mitigation.
        The Billing API depends on the Ledger Database.
        """,
        "metadata": {"record_id": "stage-page-003", "collection": "stage-scenario"},
    },
]

STAGE_COMMUNITY_RELATIONSHIP_TYPES = [
    "OPERATES",
    "DETECTED",
    "AFFECTED",
    "INVESTIGATED",
    "MITIGATED_BY",
    "APPROVED",
    "DEPENDS_ON",
]

STAGE_LOCAL_ANSWER_PROMPT = """Based on the findings below, answer the query.

Query: {query}

Findings:
{context}

Answer:"""

# Six rows, three of which are normalized duplicates. Seeded without text
# columns, so no LLM is involved and the merge outcome is exact.
STAGE_RESOLUTION_ROWS = [
    {"record_id": "row-1", "person_name": "Alice Chen"},
    {"record_id": "row-2", "person_name": "alice chen"},
    {"record_id": "row-3", "person_name": "ALICE  CHEN"},
    {"record_id": "row-4", "person_name": "A. Chen"},
    {"record_id": "row-5", "person_name": "Bob Ortiz"},
    {"record_id": "row-6", "person_name": " bob ortiz "},
]

STAGE_RESOLUTION_MAPPING = RowMapping(
    entities=[ColumnEntity(label="Person", identity_column="person_name")],
    record_id_column="record_id",
)


def normalized_name(value: str) -> str:
    """Mirror the resolver's documented normalization for readable assertions."""
    text = unicodedata.normalize("NFC", value or "").casefold()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split()).replace(" ", "")


def entity_rows(store: Any, graph_name: str) -> list[dict]:
    """Return every entity node with the state each stage cares about."""
    return store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name})
        RETURN e.id AS id,
               e.name AS name,
               labels(e) AS labels,
               e.description AS description,
               e.descriptions AS descriptions,
               e.embedding IS NOT NULL AS has_vector
        ORDER BY e.id
        """,
        {"graph_name": graph_name},
    )


def relationship_rows(store: Any, graph_name: str) -> list[dict]:
    """Return entity-to-entity relationships (chunk provenance excluded)."""
    return store.execute_query(
        """
        MATCH (a:__Entity__ {graph_name: $graph_name})-[r]->(b:__Entity__ {graph_name: $graph_name})
        RETURN type(r) AS type, a.name AS source, b.name AS target
        ORDER BY type, source, target
        """,
        {"graph_name": graph_name},
    )


def chunk_rows(store: Any, graph_name: str) -> list[dict]:
    """Return stored chunks and whether each one carries a vector."""
    return store.execute_query(
        """
        MATCH (c:Chunk {graph_name: $graph_name})
        RETURN c.id AS id, c.text AS text, c.embedding IS NOT NULL AS has_vector
        ORDER BY c.id
        """,
        {"graph_name": graph_name},
    )


def community_members(store: Any, graph_name: str) -> list[dict]:
    """Return (community, level, member entity name) triples."""
    return store.execute_query(
        """
        MATCH (c:Community {graph_name: $graph_name})<-[:IN_COMMUNITY]-(e:__Entity__)
        RETURN c.id AS community_id, c.level AS level, e.name AS entity
        ORDER BY c.level, c.id, e.name
        """,
        {"graph_name": graph_name},
    )


def community_reports(store: Any, graph_name: str) -> list[dict]:
    """Return stored community reports and whether each has a vector."""
    return store.execute_query(
        """
        MATCH (c:Community {graph_name: $graph_name})
        WHERE coalesce(c.report_text, '') <> ''
        RETURN c.id AS id,
               c.level AS level,
               c.report_text AS report_text,
               c.report_embedding IS NOT NULL AS has_vector
        ORDER BY c.id
        """,
        {"graph_name": graph_name},
    )


def chunk_ids(store: Any, graph_name: str) -> set[str]:
    """Return the ids of chunks stored for one graph."""
    return {row["id"] for row in chunk_rows(store, graph_name)}
