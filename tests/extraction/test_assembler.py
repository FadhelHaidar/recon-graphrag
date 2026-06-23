"""Tests for graph document assembler."""

from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.types import GraphExtraction
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
    assert {e.id for e in result.entities} == {"p1", "p2"}


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
    assert result.evidence_links[0].entity_id == "p1"
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
