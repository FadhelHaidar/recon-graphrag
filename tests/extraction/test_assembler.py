"""Tests for graph document assembler."""

from uuid import UUID

from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.types import ExtractedClaim, GraphExtraction
from recon_graphrag.extraction.types import (
    ExtractedNode,
    ExtractedRelationship,
)


def test_assembler_deduplicates_entities():
    assembler = GraphDocumentAssembler()
    chunks = [
        TextChunk(id="c1", text="Alice knows Bob", index=0, metadata={}),
        TextChunk(id="c2", text="Alice likes Bob", index=1, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[],
        ),
        "c2": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="abc",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={"source": "test"},
        graph_name="entity-graph",
    )
    assert len(result.entities) == 2
    assert {e.canonical_key for e in result.entities} == {"p1", "p2"}
    assert {e.human_readable_id for e in result.entities} == {"p1", "p2"}
    assert all(UUID(e.id) for e in result.entities)


def test_assembler_deduplicates_relationships_and_aggregates_weight():
    assembler = GraphDocumentAssembler()
    chunks = [
        TextChunk(id="c1", text="Alice knows Bob", index=0, metadata={}),
        TextChunk(id="c2", text="Alice knows Bob too", index=1, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="p1",
                    target_id="p2",
                    type="KNOWS",
                    properties={"weight": 1.0},
                ),
            ],
        ),
        "c2": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="p1",
                    target_id="p2",
                    type="KNOWS",
                    properties={"weight": 1.0},
                ),
            ],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="abc",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
    )
    assert len(result.relationships) == 1
    rel = result.relationships[0]
    assert rel.observation_count == 2
    assert rel.strength == 1.0  # first extracted weight
    assert rel.properties["weight"] == 2.0  # backward compat alias
    assert rel.properties["canonical_key"] == "p1:KNOWS:p2"
    assert rel.properties["human_readable_id"] == "p1:KNOWS:p2"
    assert set(rel.properties["source_chunk_ids"]) == {"c1", "c2"}


def test_assembler_creates_evidence_links():
    assembler = GraphDocumentAssembler()
    chunks = [
        TextChunk(id="c1", text="Alice knows Bob", index=0, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
            ],
            relationships=[],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="abc",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
    )
    assert len(result.evidence_links) == 1
    assert result.evidence_links[0].chunk_id == "c1"
    assert result.evidence_links[0].entity_id == result.entities[0].id
    assert UUID(result.evidence_links[0].entity_id)
    assert result.evidence_links[0].graph_name == "entity-graph"


def test_assembler_skips_missing_extractions():
    assembler = GraphDocumentAssembler()
    chunks = [
        TextChunk(id="c1", text="Alice knows Bob", index=0, metadata={}),
    ]
    extractions = {}
    result = assembler.assemble(
        document_id="doc1",
        text_hash="abc",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
    )
    assert len(result.entities) == 0
    assert len(result.evidence_links) == 0


def test_assembler_converts_claims_to_claim_records():
    assembler = GraphDocumentAssembler()
    chunks = [
        TextChunk(id="c1", text="Alice is CEO.", index=0, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="person:alice", label="Person", properties={"name": "Alice"}),
            ],
            relationships=[],
        ),
    }
    chunk_claims = {
        "c1": [
            ExtractedClaim(
                subject_entity_id="person:alice",
                claim_type="role",
                description="Alice is the CEO of Acme.",
                status="active",
            ),
            ExtractedClaim(
                subject_entity_id="person:alice",
                claim_type="opinion",
                description="Alice supports the merger.",
                status="active",
            ),
        ],
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="abc",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={"source": "test"},
        graph_name="entity-graph",
        chunk_claims=chunk_claims,
    )
    assert len(result.claims) == 2
    alice_id = result.entities[0].id
    assert all(c.entity_id == alice_id for c in result.claims)
    assert UUID(alice_id)
    assert result.claims[0].claim_type == "role"
    assert result.claims[1].claim_type == "opinion"
    assert result.claims[0].source.document_id == "doc1"
    assert result.claims[0].source.chunk_id == "c1"
    # Claim IDs should be deterministic and unique
    assert result.claims[0].id != result.claims[1].id


def test_assembler_no_claims_when_none_provided():
    assembler = GraphDocumentAssembler()
    chunks = [
        TextChunk(id="c1", text="Alice.", index=0, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[ExtractedNode(id="person:alice", label="Person", properties={})],
            relationships=[],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="abc",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
    )
    assert result.claims == []


def test_assembler_claims_from_multiple_chunks():
    assembler = GraphDocumentAssembler()
    chunks = [
        TextChunk(id="c1", text="Alice.", index=0, metadata={}),
        TextChunk(id="c2", text="Bob.", index=1, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[ExtractedNode(id="person:alice", label="Person", properties={})],
            relationships=[],
        ),
        "c2": GraphExtraction(
            nodes=[ExtractedNode(id="person:bob", label="Person", properties={})],
            relationships=[],
        ),
    }
    chunk_claims = {
        "c1": [
            ExtractedClaim(
                subject_entity_id="person:alice",
                claim_type="role",
                description="Alice is CEO.",
            ),
        ],
        "c2": [
            ExtractedClaim(
                subject_entity_id="person:bob",
                claim_type="role",
                description="Bob is CTO.",
            ),
        ],
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="abc",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
        chunk_claims=chunk_claims,
    )
    assert len(result.claims) == 2
    entity_ids = {c.entity_id for c in result.claims}
    assert entity_ids == {e.id for e in result.entities}
    assert {e.canonical_key for e in result.entities} == {"person:alice", "person:bob"}
