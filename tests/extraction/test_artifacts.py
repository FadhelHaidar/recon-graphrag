"""Tests for neutral GraphDocument JSON artifacts."""

from __future__ import annotations

from typing import cast

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
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.models.artifacts import (
    ClaimRecord,
    DescriptionObservation,
    SourceReference,
)
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore


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


def test_graph_document_dict_round_trip_preserves_description_observations():
    graph_document = _sample_graph_document()
    source = SourceReference("doc-1", "chunk-1", "Movie notes", page_start=1, char_end=25)
    graph_document.entities[0].description_observations = [
        DescriptionObservation("alice", "Director", source)
    ]

    payload = graph_document_to_dict(graph_document)
    loaded = graph_document_from_dict(payload)

    assert loaded == graph_document


def test_legacy_artifact_without_claims_or_observations():
    payload = graph_document_to_dict(_sample_graph_document())
    del payload["claims"]
    for entity in payload["entities"]:
        del entity["description_observations"]

    loaded = graph_document_from_dict(payload)

    assert loaded.claims == []
    assert all(entity.description_observations == [] for entity in loaded.entities)


def test_graph_document_json_round_trip_and_replay_preserves_claim(tmp_path):
    graph_document = _sample_graph_document()
    source = SourceReference("doc-1", "chunk-1", "Movie notes", page_start=1, char_end=25)
    graph_document.claims = [ClaimRecord(
        "claim:1", "alice", "credit", "Alice directed Inception.", source,
        object_entity_id="inception",
    )]
    path = tmp_path / "movie_graph.json"

    save_graph_document_json(graph_document, path)
    loaded = load_graph_document_json(path)

    store = StatefulFakeGraphStore()
    writer = CypherGraphWriter(cast(GraphStore, store))
    writer.write_graph_document(graph_document)
    receipt = writer.write_graph_document(loaded)
    claim = store.get_claim("claim:1")

    assert receipt["claims"] == 1
    assert claim is not None
    assert claim["entity_id"] == "alice"
    assert claim["object_entity_id"] == "inception"
    assert claim["chunk_id"] == "chunk-1"
    assert loaded == graph_document


def test_assembly_review_groups_round_trip_and_legacy_default(tmp_path):
    document = _sample_graph_document()
    document.assembly_review_groups = [{
        "reason": "property_conflict", "decision": "blocked", "entity_ids": ["alice"],
        "conflicts": [{"property": "contact", "observations": [
            {"value": "one", "source": {"document_id": "doc-1", "chunk_id": "chunk-1"}},
            {"value": "two", "source": {"document_id": "doc-1", "chunk_id": "chunk-2"}},
        ]}],
    }]
    path = tmp_path / "review.json"
    save_graph_document_json(document, path)
    assert load_graph_document_json(path) == document
    payload = graph_document_to_dict(document)
    del payload["assembly_review_groups"]
    assert graph_document_from_dict(payload).assembly_review_groups == []
