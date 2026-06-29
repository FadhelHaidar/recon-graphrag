"""Tests for graph document assembler — core logic and cross-chunk aggregation."""

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
    assert rel.properties["weight"] == 2.0  # Leiden observation weight
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


# ---------------------------------------------------------------------------
# Cross-chunk aggregation characterization tests
# ---------------------------------------------------------------------------


def _make_assembler() -> GraphDocumentAssembler:
    return GraphDocumentAssembler()


def _chunk(cid: str, text: str, index: int = 0) -> TextChunk:
    return TextChunk(id=cid, text=text, index=index, metadata={})


def _node(eid: str, description: str) -> ExtractedNode:
    return ExtractedNode(
        id=eid,
        label="Person",
        properties={"name": "Alice" if eid == "e1" else "Bob", "description": description},
    )


def _rel(source: str = "e1", target: str = "e2", weight: float = 1.0) -> ExtractedRelationship:
    return ExtractedRelationship(
        source_id=source,
        target_id=target,
        type="WORKS_AT",
        properties={"weight": weight},
    )


def make_cross_chunk_graph_document(graph_name: str = "entity-graph"):
    """Build a GraphDocument covering multi-chunk and multi-document aggregation.

    Fixture shape:
    - Document A (chunks c1, c2): same entity e1 with two descriptions; same
      relationship e1-[WORKS_AT]->e2 in both chunks.
    - Document B (chunk c3): repeats the relationship and adds a third e1 description.
    - The same logical IDs are also returned under a different graph_name by callers.
    """
    assembler = _make_assembler()
    chunks = [
        _chunk("c1", "Alice works at Acme.", index=0),
        _chunk("c2", "Alice is a senior engineer.", index=1),
        _chunk("c3", "Alice still works at Acme and leads the team.", index=0),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[_node("e1", "desc from c1"), _node("e2", "desc from c2")],
            relationships=[_rel()],
        ),
        "c2": GraphExtraction(
            nodes=[_node("e1", "desc from c2"), _node("e2", "desc from c2")],
            relationships=[_rel()],
        ),
        "c3": GraphExtraction(
            nodes=[_node("e1", "desc from c3")],
            relationships=[_rel()],
        ),
    }
    return assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={"source": "test"},
        graph_name=graph_name,
    )


def test_entity_descriptions_accumulated_and_consolidated():
    doc = make_cross_chunk_graph_document()
    e1 = next(e for e in doc.entities if e.canonical_key == "e1")
    assert UUID(e1.id)
    # All three unique descriptions are consolidated
    assert "desc from c1" in e1.properties["description"]
    assert "desc from c2" in e1.properties["description"]
    assert "desc from c3" in e1.properties["description"]
    # Observations are preserved with provenance
    assert len(e1.description_observations) == 3
    assert e1.description_observations[0].source.chunk_id == "c1"
    assert e1.description_observations[1].source.chunk_id == "c2"
    assert e1.description_observations[2].source.chunk_id == "c3"


def test_entity_empty_descriptions_excluded_from_consolidation():
    assembler = _make_assembler()
    chunks = [_chunk("c1", "Alice works at Acme.", index=0)]
    extractions = {
        "c1": GraphExtraction(
            nodes=[_node("e1", ""), _node("e2", "Bob works here")],
            relationships=[],
        ),
    }
    doc = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
    )
    e1 = next(e for e in doc.entities if e.canonical_key == "e1")
    # Empty description excluded from consolidation
    assert e1.properties["description"] == ""
    assert len(e1.description_observations) == 1


def test_entity_description_source_includes_document_metadata():
    assembler = _make_assembler()
    chunks = [_chunk("c1", "Alice works at Acme.", index=0)]
    chunks[0].metadata = {"page_start": 5, "page_end": 6, "char_start": 0, "char_end": 20}
    extractions = {
        "c1": GraphExtraction(
            nodes=[ExtractedNode(
                id="e1",
                label="Person",
                properties={"name": "Alice", "description": "CEO"},
            )],
            relationships=[],
        ),
    }
    doc = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={"source": "Annual Report"},
        graph_name="entity-graph",
    )
    e1 = doc.entities[0]
    source = e1.description_observations[0].source
    assert source.document_id == "doc1"
    assert source.chunk_id == "c1"
    assert source.document_name == "Annual Report"
    assert source.page_start == 5
    assert source.page_end == 6
    assert source.char_start == 0
    assert source.char_end == 20


def test_relationship_observation_count_aggregated():
    doc = make_cross_chunk_graph_document()
    assert len(doc.relationships) == 1
    rel = doc.relationships[0]
    assert rel.observation_count == 3
    assert rel.strength == 1.0  # first extracted weight preserved
    assert rel.properties["weight"] == 3.0  # Leiden observation weight


def test_relationship_source_chunk_ids_accumulated():
    doc = make_cross_chunk_graph_document()
    rel = doc.relationships[0]
    assert set(rel.properties["source_chunk_ids"]) == {"c1", "c2", "c3"}


def test_evidence_links_count():
    doc = make_cross_chunk_graph_document()
    # c1 links e1 and e2; c2 links e1 and e2; c3 links e1.
    assert len(doc.evidence_links) == 5


def test_rerun_does_not_inflate_entities():
    assembler = _make_assembler()
    first = make_cross_chunk_graph_document()
    second = assembler.assemble(
        document_id=first.document.id,
        text_hash=first.document.text_hash,
        chunks=first.chunks,
        chunk_extractions={
            chunk.id: GraphExtraction(
                nodes=[
                    ExtractedNode(
                        id=e.id,
                        label=e.type,
                        properties=e.properties,
                    )
                    for e in first.entities
                ],
                relationships=[],
            )
            for chunk in first.chunks
        },
        metadata=first.document.metadata,
        graph_name=first.document.graph_name,
    )
    assert len(second.entities) == 2
    assert {e.id for e in second.entities} == {e.id for e in first.entities}
    assert {e.canonical_key for e in second.entities} == {"e1", "e2"}


def test_graph_name_stamped_on_all_records():
    doc = make_cross_chunk_graph_document(graph_name="other-graph")
    assert doc.document.graph_name == "other-graph"
    assert all(c.graph_name == "other-graph" for c in doc.chunks)
    assert all(e.graph_name == "other-graph" for e in doc.entities)
    assert all(r.graph_name == "other-graph" for r in doc.relationships)
    assert all(link.graph_name == "other-graph" for link in doc.evidence_links)


def test_graph_name_isolation_at_assembler_level():
    """Same IDs under different graph names produce independent GraphDocuments."""
    doc_a = make_cross_chunk_graph_document(graph_name="graph-a")
    doc_b = make_cross_chunk_graph_document(graph_name="graph-b")
    assert doc_a.document.graph_name == "graph-a"
    assert doc_b.document.graph_name == "graph-b"
    assert {e.canonical_key for e in doc_a.entities} == {
        e.canonical_key for e in doc_b.entities
    }
    assert {e.id for e in doc_a.entities} != {e.id for e in doc_b.entities}
