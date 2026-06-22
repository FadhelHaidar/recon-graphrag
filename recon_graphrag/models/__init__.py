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
    PartialAnswer,
    RelationshipObservation,
    SourceReference,
    citation_excerpt,
    entity_ref,
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
    "RelationshipObservation",
    # Claim types
    "ClaimRecord",
    # Community report types
    "CommunityFinding",
    "ArtifactVersion",
    "CommunityReport",
    "PartialAnswer",
    # Canonical rendering
    "entity_ref",
    "source_ref",
    "report_to_json",
    "report_to_text",
    "citation_excerpt",
    # Shared types
    "SearchResult",
    "IndexConfig",
]
