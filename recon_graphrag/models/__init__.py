"""Shared data models."""

from recon_graphrag.models.artifacts import (
    ARTIFACT_PROMPT_VERSION,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactVersion,
    Citation,
    ClaimRecord,
    CommunityFinding,
    CommunityReport,
    DescriptionObservation,
    DocumentSource,
    SourceReference,
    citation_excerpt,
    report_to_json,
    report_to_text,
    source_ref,
)
from recon_graphrag.models.types import IndexConfig, SearchResult

__all__ = [
    # Version constants
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_PROMPT_VERSION",
    # Provenance types
    "SourceReference",
    "Citation",
    "DocumentSource",
    # Observation types
    "DescriptionObservation",
    # Claim types
    "ClaimRecord",
    # Community report types
    "CommunityFinding",
    "ArtifactVersion",
    "CommunityReport",
    # Canonical rendering
    "source_ref",
    "report_to_json",
    "report_to_text",
    "citation_excerpt",
    # Shared types
    "SearchResult",
    "IndexConfig",
]
