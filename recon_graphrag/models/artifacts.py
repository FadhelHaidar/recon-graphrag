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
SCHEMA_VERSION = "2.0"


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

    ``metadata`` contains source metadata copied from the cited document and
    chunk. It supports row/list-item use cases where arbitrary keys identify the
    source record.
    """

    document_id: str
    chunk_id: str
    document_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    excerpt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    graph_name: str = "entity-graph"
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Community report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FindingReference:
    """A typed reference from a finding to evidence in the community context.

    ``target_id`` is the entity ID, relationship key (``source:type:target``),
    or claim ID. ``target_type`` distinguishes what kind of evidence it is.
    """

    target_id: str
    target_type: str  # "entity", "relationship", "claim"


@dataclass
class CommunityFinding:
    """A single finding within a community report."""

    id: str
    description: str
    references: list[FindingReference] = field(default_factory=list)
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
    title: str = ""
    rating: float | None = None
    rating_explanation: str | None = None
    version: ArtifactVersion = field(default_factory=ArtifactVersion)



# ---------------------------------------------------------------------------
# Canonical rendering
# ---------------------------------------------------------------------------


def source_ref(source: SourceReference) -> str:
    """Stable reference rendering for a source."""
    return f"[Source:{source.document_id}:{source.chunk_id}]"


def report_to_json(report: CommunityReport) -> str:
    """Serialize a community report to sorted-key JSON."""

    def _ref_dict(r: FindingReference) -> dict[str, Any]:
        return {"target_id": r.target_id, "target_type": r.target_type}

    def _finding_dict(f: CommunityFinding) -> dict[str, Any]:
        return {
            "id": f.id,
            "description": f.description,
            "rank": f.rank,
            "references": [_ref_dict(r) for r in f.references],
        }

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
        "title": report.title,
        "findings": [_finding_dict(f) for f in report.findings],
        "summary": report.summary,
        "rating": report.rating,
        "rating_explanation": report.rating_explanation,
        "version": _version_dict(report.version),
    }
    return json.dumps(data, indent=2, sort_keys=True)


def report_to_text(report: CommunityReport) -> str:
    """Render a community report as plain text for storage and retrieval.

    Title, summary, rating (if present), and findings in stable order.
    This produces a stable text representation for global and DRIFT search.
    """
    sorted_findings = sorted(report.findings, key=lambda f: (-f.rank, f.id))
    lines = []
    if report.title:
        lines.append(report.title)
    lines.append(f"Community {report.community_id} (level {report.level})")
    if report.rating is not None:
        lines.append(f"Rating: {report.rating}")
        if report.rating_explanation:
            lines.append(report.rating_explanation)
    if sorted_findings:
        lines.append("")
        for f in sorted_findings:
            ref_text = ", ".join(
                f"{r.target_type}:{r.target_id}" for r in f.references
            )
            suffix = f" [refs: {ref_text}]" if ref_text else ""
            lines.append(f"- {f.description}{suffix}")
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
    "SCHEMA_VERSION",
    "SourceReference",
    "Citation",
    "DocumentSource",
    "DescriptionObservation",
    "ClaimRecord",
    "FindingReference",
    "CommunityFinding",
    "ArtifactVersion",
    "CommunityReport",
    "source_ref",
    "report_to_json",
    "report_to_text",
    "citation_excerpt",
]
