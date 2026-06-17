"""JSON artifacts for database-neutral graph documents."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)


def graph_document_to_dict(graph_document: GraphDocument) -> dict[str, Any]:
    """Convert a GraphDocument into a JSON-serializable dictionary."""
    return asdict(graph_document)


def graph_document_from_dict(payload: dict[str, Any]) -> GraphDocument:
    """Load a GraphDocument from a dictionary produced by graph_document_to_dict."""
    return GraphDocument(
        document=DocumentRecord(**payload["document"]),
        chunks=[ChunkRecord(**row) for row in payload.get("chunks", [])],
        entities=[EntityRecord(**row) for row in payload.get("entities", [])],
        relationships=[
            RelationshipRecord(**row) for row in payload.get("relationships", [])
        ],
        evidence_links=[
            EvidenceLink(**row) for row in payload.get("evidence_links", [])
        ],
    )


def save_graph_document_json(graph_document: GraphDocument, path: str | Path) -> None:
    """Write a GraphDocument JSON artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(graph_document_to_dict(graph_document), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_graph_document_json(path: str | Path) -> GraphDocument:
    """Read a GraphDocument JSON artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return graph_document_from_dict(payload)
