"""Database-neutral graph data types.

Defines neutral graph objects independent from Neo4j, Postgres, or any other
database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recon_graphrag.models.artifacts import ClaimRecord


@dataclass
class ExtractedNode:
    id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelationship:
    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphExtraction:
    nodes: list[ExtractedNode] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)


@dataclass
class DocumentRecord:
    id: str
    text_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    graph_name: str = "entity-graph"


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    graph_name: str = "entity-graph"


@dataclass
class EntityRecord:
    id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    graph_name: str = "entity-graph"


@dataclass
class RelationshipRecord:
    id: str
    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    graph_name: str = "entity-graph"
    observation_count: int = 1
    strength: float | None = None


@dataclass
class EvidenceLink:
    chunk_id: str
    entity_id: str
    graph_name: str = "entity-graph"


@dataclass
class GraphDocument:
    document: DocumentRecord
    chunks: list[ChunkRecord]
    entities: list[EntityRecord]
    relationships: list[RelationshipRecord]
    evidence_links: list[EvidenceLink]
    claims: list[ClaimRecord] = field(default_factory=list)
