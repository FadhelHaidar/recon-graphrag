"""Versioned artifact types for the GraphRAG pipeline.

These types represent provenance-carrying domain objects that flow through
extraction, aggregation, community reporting, and retrieval. They are
database-neutral and use dataclasses consistent with ``extraction/types.py``.

Version constants are module-level so a bump is deliberate and covered by tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

ARTIFACT_SCHEMA_VERSION = "1.0"
ARTIFACT_PROMPT_VERSION = "0.0"  # updated when prompts change


# ---------------------------------------------------------------------------
# Provenance types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceReference:
    """Internal provenance linking an observation back to its source chunk.

    ``document_name`` is resolved from metadata keys such as ``title``,
    ``source``, or ``filename``. Page values are present only for page-based
    ingestion; plain text returns ``None``.
    """

    document_id: str
    chunk_id: str
    document_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True)
class Citation:
    """User-facing citation returned in search results.

    ``excerpt`` is an optional bounded snippet from the cited chunk.
    """

    document_id: str
    chunk_id: str
    document_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    excerpt: str | None = None


@dataclass
class DocumentSource:
    """Citations grouped by document for user-facing display."""

    document_id: str
    document_name: str | None
    chunk_list: list[Citation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation types
# ---------------------------------------------------------------------------


@dataclass
class DescriptionObservation:
    """A single description observation for an entity."""

    entity_id: str
    description: str
    source: SourceReference
    created_at: str | None = None


@dataclass
class RelationshipObservation:
    """A single observation about a relationship."""

    source_id: str
    target_id: str
    relationship_type: str
    description: str
    source: SourceReference
    weight: float = 1.0
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Claim types
# ---------------------------------------------------------------------------


@dataclass
class ClaimRecord:
    """A claim or covariate extracted from text (Phase 3 scope)."""

    id: str
    entity_id: str
    claim_type: str
    description: str
    source: SourceReference
    status: str = "active"
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Community report types
# ---------------------------------------------------------------------------


@dataclass
class CommunityFinding:
    """A single finding within a community report."""

    id: str
    description: str
    rank: float = 0.0


@dataclass
class ArtifactVersion:
    """Version metadata attached to derived artifacts."""

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    prompt_version: str = ARTIFACT_PROMPT_VERSION
    input_fingerprint: str | None = None
    created_at: str | None = None


@dataclass
class CommunityReport:
    """Structured community report (Phase 5 scope)."""

    id: str
    community_id: str
    level: int
    findings: list[CommunityFinding] = field(default_factory=list)
    summary: str = ""
    version: ArtifactVersion = field(default_factory=ArtifactVersion)


@dataclass
class PartialAnswer:
    """A partial answer from one community during global search map phase."""

    community_id: str
    level: int
    answer: str
    token_usage: int | None = None


# ---------------------------------------------------------------------------
# Canonical rendering
# ---------------------------------------------------------------------------


def entity_ref(entity_id: str) -> str:
    """Stable reference rendering for an entity."""
    return f"[Entity:{entity_id}]"


def source_ref(source: SourceReference) -> str:
    """Stable reference rendering for a source."""
    return f"[Source:{source.document_id}:{source.chunk_id}]"


def report_to_json(report: CommunityReport) -> str:
    """Serialize a community report to sorted-key JSON."""

    def _finding_dict(f: CommunityFinding) -> dict[str, Any]:
        return {"id": f.id, "description": f.description, "rank": f.rank}

    def _version_dict(v: ArtifactVersion) -> dict[str, Any]:
        return {
            "schema_version": v.schema_version,
            "prompt_version": v.prompt_version,
            "input_fingerprint": v.input_fingerprint,
            "created_at": v.created_at,
        }

    data = {
        "id": report.id,
        "community_id": report.community_id,
        "level": report.level,
        "findings": [_finding_dict(f) for f in report.findings],
        "summary": report.summary,
        "version": _version_dict(report.version),
    }
    return json.dumps(data, indent=2, sort_keys=True)


def report_to_text(report: CommunityReport) -> str:
    """Render a community report as plain text for embeddings.

    Findings are listed first (sorted by rank descending, then id), followed
    by the summary. This produces a stable text representation suitable for
    embedding models.
    """
    sorted_findings = sorted(report.findings, key=lambda f: (-f.rank, f.id))
    lines = [f"Community {report.community_id} (level {report.level})"]
    if sorted_findings:
        lines.append("")
        for f in sorted_findings:
            lines.append(f"- {f.description}")
    if report.summary:
        lines.append("")
        lines.append(report.summary)
    return "\n".join(lines)


def citation_excerpt(text: str, max_chars: int = 200) -> str:
    """Produce a bounded excerpt from chunk text for citations."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_PROMPT_VERSION",
    "SourceReference",
    "Citation",
    "DocumentSource",
    "DescriptionObservation",
    "RelationshipObservation",
    "ClaimRecord",
    "CommunityFinding",
    "ArtifactVersion",
    "CommunityReport",
    "PartialAnswer",
    "entity_ref",
    "source_ref",
    "report_to_json",
    "report_to_text",
    "citation_excerpt",
]
