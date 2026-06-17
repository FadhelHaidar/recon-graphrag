"""Tests for neutral GraphDocument JSON artifacts."""

from __future__ import annotations

from recon_graphrag.extraction.artifacts import (
    graph_document_from_dict,
    graph_document_to_dict,
    load_graph_document_json,
    save_graph_document_json,
)
from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)


def _sample_graph_document() -> GraphDocument:
    return GraphDocument(
        document=DocumentRecord(
            id="doc-1",
            text_hash="hash",
            metadata={"source": "unit", "nested": {"a": 1}},
            graph_name="entity-graph",
        ),
        chunks=[
            ChunkRecord(
                id="chunk-1",
                document_id="doc-1",
                text="Alice directed Inception.",
                index=0,
                metadata={"page_start": 1, "tags": ["movie"]},
            )
        ],
        entities=[
            EntityRecord(
                id="alice",
                type="Person",
                properties={"name": "Alice", "aliases": ["A."]},
            ),
            EntityRecord(
                id="inception",
                type="Movie",
                properties={"name": "Inception", "year": 2010},
            ),
        ],
        relationships=[
            RelationshipRecord(
                id="alice:DIRECTED:inception",
                source_id="alice",
                target_id="inception",
                type="DIRECTED",
                properties={"source_chunk_ids": ["chunk-1"], "weight": 1.0},
            )
        ],
        evidence_links=[
            EvidenceLink(chunk_id="chunk-1", entity_id="alice"),
            EvidenceLink(chunk_id="chunk-1", entity_id="inception"),
        ],
    )


def test_graph_document_dict_round_trip():
    graph_document = _sample_graph_document()

    payload = graph_document_to_dict(graph_document)
    loaded = graph_document_from_dict(payload)

    assert loaded == graph_document
    assert loaded.relationships[0].properties["source_chunk_ids"] == ["chunk-1"]


def test_graph_document_json_round_trip(tmp_path):
    graph_document = _sample_graph_document()
    path = tmp_path / "movie_graph.json"

    save_graph_document_json(graph_document, path)
    loaded = load_graph_document_json(path)

    assert loaded == graph_document
