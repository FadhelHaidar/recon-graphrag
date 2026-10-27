"""Tests for graph document assembler — core logic and cross-chunk aggregation."""

from uuid import UUID

import pytest

from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.types import (
    ExtractedClaim,
    ExtractedNode,
    ExtractedRelationship,
    GraphExtraction,
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
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice", "canonical_key": "Person:Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob", "canonical_key": "Person:Bob"}),
            ],
            relationships=[],
        ),
        "c2": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice", "canonical_key": "Person:Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob", "canonical_key": "Person:Bob"}),
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
    # Trusted canonical keys merge observations across chunks.
    assert {e.canonical_key for e in result.entities} == {
        "Person:Alice",
        "Person:Bob",
    }
    assert {e.human_readable_id for e in result.entities} == {"Person:Alice", "Person:Bob"}
    assert all(UUID(e.id) for e in result.entities)


def test_assembler_retains_complementary_properties_and_observations():
    chunks = [_chunk("c1", "Alice email"), _chunk("c2", "Alice phone", 1)]
    first = {
        "canonical_key": "Person:Alice",
        "human_readable_id": "alice-first",
        "name": "Alice",
        "email": "alice@example.com",
        "description": "Alice email",
    }
    second = {
        "canonical_key": "Person:Alice",
        "human_readable_id": "alice-later",
        "name": "ALICE",
        "phone": "555-0100",
        "description": "Alice phone",
    }
    extractions = {
        chunk.id: GraphExtraction(
            nodes=[ExtractedNode(id=f"p{index}", label="Person", properties=properties)]
        )
        for index, (chunk, properties) in enumerate(zip(chunks, [first, second], strict=True))
    }

    result = GraphDocumentAssembler().assemble(
        "doc1", "hash", chunks, extractions, {}, "entity-graph"
    )

    (entity,) = result.entities
    assert entity.properties == {
        **first,
        "phone": "555-0100",
        "title": "Alice",
        "description": "Alice email; Alice phone",
    }
    assert entity.canonical_key == "Person:Alice"
    assert entity.human_readable_id == "alice-first"
    assert entity.type == "Person"
    assert entity.graph_name == "entity-graph"
    assert [obs.description for obs in entity.description_observations] == [
        "Alice email", "Alice phone"
    ]
    assert [obs.source.chunk_id for obs in entity.description_observations] == ["c1", "c2"]
    assert [(link.chunk_id, link.entity_id) for link in result.evidence_links] == [
        ("c1", entity.id), ("c2", entity.id)
    ]
    assert "phone" not in first
    assert second["human_readable_id"] == "alice-later"


@pytest.mark.parametrize(
    ("first", "incoming", "expected"),
    [
        ({}, {"value": "new"}, {"value": "new"}),
        ({"value": None}, {"value": "new"}, {"value": "new"}),
        ({}, {"value": None}, {}),
        ({"value": None}, {"value": None}, {"value": None}),
        ({"value": "first"}, {}, {"value": "first"}),
        ({"value": "first"}, {"value": None}, {"value": "first"}),
        ({"value": "first"}, {"value": "later"}, {"value": "first"}),
        ({}, {"value": 0}, {"value": 0}),
        ({"value": None}, {"value": False}, {"value": False}),
        ({"value": 0}, {"value": 1}, {"value": 0}),
        ({"value": False}, {"value": True}, {"value": False}),
    ],
)
def test_assembler_property_merge_keeps_first_populated_value(first, incoming, expected):
    chunk = _chunk("c1", "Alice")
    nodes = [
        ExtractedNode(
            id=f"p{index}", label="Person",
            properties={"canonical_key": "Person:Alice", **properties},
        )
        for index, properties in enumerate([first, incoming])
    ]
    result = GraphDocumentAssembler().assemble(
        "doc1", "hash", [chunk], {chunk.id: GraphExtraction(nodes=nodes)}, {}, "g"
    )
    (entity,) = result.entities
    assert {key: value for key, value in entity.properties.items() if key == "value"} == expected


def test_assembler_deduplicates_relationships_and_aggregates_weight():
    assembler = GraphDocumentAssembler()
    chunks = [
        TextChunk(id="c1", text="Alice knows Bob", index=0, metadata={}),
        TextChunk(id="c2", text="Alice knows Bob too", index=1, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice", "canonical_key": "Person:Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob", "canonical_key": "Person:Bob"}),
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
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice", "canonical_key": "Person:Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob", "canonical_key": "Person:Bob"}),
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
    assert rel.properties["canonical_key"] == (
        "Person:Alice:KNOWS:Person:Bob"
    )
    assert rel.properties["human_readable_id"] == (
        "Person:Alice:KNOWS:Person:Bob"
    )
    assert rel.properties["source_canonical_key"] == "Person:Alice"
    assert rel.properties["target_canonical_key"] == "Person:Bob"
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
    assert {e.canonical_key for e in result.entities} == {
        "doc1:c1:Person:person:alice",
        "doc1:c2:Person:person:bob",
    }


# Cross-chunk aggregation characterization tests
def _make_assembler() -> GraphDocumentAssembler:
    return GraphDocumentAssembler()


def _chunk(cid: str, text: str, index: int = 0) -> TextChunk:
    return TextChunk(id=cid, text=text, index=index, metadata={})


def _node(eid: str, description: str) -> ExtractedNode:
    return ExtractedNode(
        id=eid,
        label="Person",
        properties={"canonical_key": f"doc1:Person:{eid}", "name": "Alice" if eid == "e1" else "Bob", "description": description},
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
      Both endpoints are present because the validator drops relationships whose
      endpoints are missing from the same chunk extraction.
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
            nodes=[_node("e1", "desc from c3"), _node("e2", "desc from c2")],
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
    e1 = next(e for e in doc.entities if e.canonical_key == "doc1:Person:e1")
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
    e1 = next(e for e in doc.entities if e.canonical_key == "doc1:Person:e1")
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
    # c1 links e1 and e2; c2 links e1 and e2; c3 links e1 and e2.
    assert len(doc.evidence_links) == 6


def test_rerun_does_not_inflate_entities():
    assembler = _make_assembler()
    first = make_cross_chunk_graph_document()
    # first.chunks are ChunkRecord; rewrap as TextChunk for the rerun input.
    rerun_chunks = [
        TextChunk(id=c.id, text=c.text, index=c.index, metadata=c.metadata)
        for c in first.chunks
    ]
    second = assembler.assemble(
        document_id=first.document.id,
        text_hash=first.document.text_hash,
        chunks=rerun_chunks,
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
    assert {e.canonical_key for e in second.entities} == {
        "doc1:Person:e1",
        "doc1:Person:e2",
    }


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


# F7 identity scoping: extracted local IDs are not global identity
def _single_doc(
    chunk_id: str, nodes: list[ExtractedNode], relationships=None
) -> dict[str, GraphExtraction]:
    return {
        chunk_id: GraphExtraction(nodes=nodes, relationships=relationships or []),
    }


def test_reused_local_id_across_documents_stays_separate():
    """The same extracted ID "p1" in two documents is not one entity."""
    assembler = GraphDocumentAssembler()
    nodes = [ExtractedNode(id="p1", label="Person", properties={"name": "Alice"})]
    doc1 = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=[TextChunk(id="doc1:chunk:0", text="...", index=0, metadata={})],
        chunk_extractions=_single_doc("doc1:chunk:0", nodes),
        metadata={},
        graph_name="entity-graph",
    )
    doc2 = assembler.assemble(
        document_id="doc2",
        text_hash="hash",
        chunks=[TextChunk(id="doc2:chunk:0", text="...", index=0, metadata={})],
        chunk_extractions=_single_doc(
            "doc2:chunk:0",
            [ExtractedNode(id="p1", label="Person", properties={"name": "Alice"})],
        ),
        metadata={},
        graph_name="entity-graph",
    )
    assert doc1.entities[0].canonical_key == "doc1:doc1:chunk:0:Person:p1"
    assert doc2.entities[0].canonical_key == "doc2:doc2:chunk:0:Person:p1"
    assert doc1.entities[0].id != doc2.entities[0].id


def test_relationship_endpoints_map_to_extracted_nodes():
    assembler = GraphDocumentAssembler()
    chunks = [TextChunk(id="c1", text="Alice knows Bob", index=0, metadata={})]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="p1", target_id="p2", type="KNOWS", properties={}
                )
            ],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
    )
    alice = next(e for e in result.entities if e.properties["name"] == "Alice")
    bob = next(e for e in result.entities if e.properties["name"] == "Bob")
    rel = result.relationships[0]
    assert rel.source_id == alice.id
    assert rel.target_id == bob.id
    assert rel.properties["source_canonical_key"] == alice.canonical_key
    assert rel.properties["target_canonical_key"] == bob.canonical_key


def test_relationship_falls_back_to_legacy_key_for_unknown_endpoint():
    """Endpoints with no matching node keep the pre-F7 global raw-ID mapping."""
    assembler = GraphDocumentAssembler()
    chunks = [TextChunk(id="c1", text="...", index=0, metadata={})]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="p1", target_id="ghost", type="KNOWS", properties={}
                )
            ],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
    )
    alice = result.entities[0]
    rel = result.relationships[0]
    assert rel.source_id == alice.id
    assert rel.properties["target_canonical_key"] == "ghost"


def test_claim_endpoints_map_to_extracted_nodes():
    assembler = GraphDocumentAssembler()
    chunks = [TextChunk(id="c1", text="Alice is CEO.", index=0, metadata={})]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
            ],
            relationships=[],
        ),
    }
    chunk_claims = {
        "c1": [
            ExtractedClaim(
                subject_entity_id="p1",
                claim_type="role",
                description="Alice is the CEO of Acme.",
                status="active",
            ),
        ],
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
        chunk_claims=chunk_claims,
    )
    assert result.claims[0].entity_id == result.entities[0].id


def test_identity_property_scopes_keys():
    """A configured identity property gives label-scoped trusted keys."""
    assembler = GraphDocumentAssembler()
    chunks = [TextChunk(id="c1", text="...", index=0, metadata={})]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(
                    id="p1",
                    label="Person",
                    properties={"name": "John", "person_no": "P-42"},
                ),
                ExtractedNode(
                    id="p2",
                    label="Person",
                    properties={"name": "John", "person_no": "P-99"},
                ),
                ExtractedNode(
                    id="p3",
                    label="Person",
                    properties={"name": "Johnny", "person_no": "P-42"},
                ),
            ],
            relationships=[],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
        identity_properties={"Person": "person_no"},
    )
    # Same identity value merges (trusted key); same NAME does not collapse.
    assert {e.canonical_key for e in result.entities} == {"Person:P-42", "Person:P-99"}
    assert len(result.entities) == 2


def test_identity_property_without_value_falls_back_to_surrogate():
    assembler = GraphDocumentAssembler()
    chunks = [TextChunk(id="c1", text="...", index=0, metadata={})]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "John"}),
            ],
            relationships=[],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
        identity_properties={"Person": "person_no"},
    )
    assert result.entities[0].canonical_key == "doc1:c1:Person:p1"


def test_explicit_canonical_key_wins_over_surrogate():
    assembler = GraphDocumentAssembler()
    chunks = [TextChunk(id="c1", text="...", index=0, metadata={})]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(
                    id="w1",
                    label="Product",
                    properties={
                        "name": "Widget",
                        "canonical_key": "Product:Widget",
                    },
                ),
            ],
            relationships=[],
        ),
    }
    result = assembler.assemble(
        document_id="doc1",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name="entity-graph",
    )
    assert result.entities[0].canonical_key == "Product:Widget"
    assert result.entities[0].human_readable_id == "Product:Widget"


def test_chunk_local_ids_keep_endpoints_claims_and_citations_separate():
    chunks = [_chunk("c1", "Alice knows Shared"), _chunk("c2", "Bob knows Shared", 1)]
    extractions = {
        chunk.id: GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": name}),
                ExtractedNode(id="p2", label="Person", properties={"canonical_key": "shared"}),
            ],
            relationships=[ExtractedRelationship(source_id="p1", target_id="p2", type="KNOWS", properties={})],
        )
        for chunk, name in zip(chunks, ["Alice", "Bob"], strict=True)
    }
    claims = {
        chunk.id: [ExtractedClaim(subject_entity_id="p1", object_entity_id="p2", claim_type="role", description=chunk.text)]
        for chunk in chunks
    }
    result = GraphDocumentAssembler().assemble("doc", "hash", chunks, extractions, {}, "g", chunk_claims=claims)
    entities = {entity.canonical_key: entity for entity in result.entities}
    assert len(entities) == 3
    shared_id = entities["shared"].id
    assert len(entities["shared"].description_observations) == 2
    for chunk, name in zip(chunks, ["Alice", "Bob"], strict=True):
        entity = entities[f"doc:{chunk.id}:Person:p1"]
        assert entity.properties["name"] == name
        assert entity.description_observations[0].source.chunk_id == chunk.id
        rel = next(rel for rel in result.relationships if rel.source_id == entity.id)
        assert rel.target_id == shared_id
        assert rel.properties["source_chunk_ids"] == [chunk.id]
        claim = next(claim for claim in result.claims if claim.entity_id == entity.id)
        assert claim.object_entity_id == shared_id
        assert claim.source.chunk_id == claim.text_unit_id == chunk.id
        assert {(link.chunk_id, link.entity_id) for link in result.evidence_links if link.chunk_id == chunk.id} == {(chunk.id, entity.id), (chunk.id, shared_id)}


def test_name_conflicts_split_all_variants_and_preserve_endpoints():
    chunks = [_chunk("c1", "first"), _chunk("c2", "second", 1)]
    extractions = {
        chunk.id: GraphExtraction(
            nodes=[
                ExtractedNode("alice", "Person", {
                    "name": "Alice", "canonical_key": "Person:Alice", "contact": value,
                }, canonical_key_trusted=False),
                ExtractedNode("bob", "Person", {"canonical_key": "Person:Bob"}),
            ],
            relationships=[ExtractedRelationship("alice", "bob", "KNOWS")],
        )
        for chunk, value in zip(chunks, ["one", "two"], strict=True)
    }
    claims = {
        chunk.id: [ExtractedClaim("alice", "KNOWS", "observation", object_entity_id="bob")]
        for chunk in chunks
    }
    assembler = GraphDocumentAssembler()
    result = assembler.assemble("doc", "hash", chunks, extractions, {}, "g", claims)
    again = assembler.assemble("doc", "hash", chunks, extractions, {}, "g", claims)
    assert result == again
    alice = [e for e in result.entities if e.properties.get("name") == "Alice"]
    assert len(alice) == 2
    assert all(e.canonical_key and e.canonical_key.startswith("Person:Alice:provisional:") for e in alice)
    bob = next(e for e in result.entities if e.canonical_key == "Person:Bob")
    for chunk, entity, rel, claim in zip(chunks, alice, result.relationships, result.claims, strict=True):
        assert rel.source_id == claim.entity_id == entity.id
        assert rel.target_id == claim.object_entity_id == bob.id
        assert rel.properties["source_canonical_key"] == entity.canonical_key
        assert any(link.chunk_id == chunk.id and link.entity_id == entity.id for link in result.evidence_links)
    (review,) = result.assembly_review_groups
    assert review["decision"] == "blocked"
    assert set(review["entity_ids"]) == {e.id for e in alice}
    assert [o["source"]["chunk_id"] for o in review["conflicts"][0]["observations"]] == ["c1", "c2"]
    assert not any("assembly_review_groups" in e.properties for e in result.entities)


def test_trusted_identity_precedes_keys_and_preserves_scalar_conflicts_and_lists():
    from recon_graphrag.extraction.schema import GraphSchema, NodeType, PropertyType

    schema = GraphSchema([NodeType("Person", properties=[
        PropertyType("tags", "LIST"), PropertyType("identifier"), PropertyType("contact"),
    ], identity_property="identifier")], [])
    chunks = [_chunk("c1", "first"), _chunk("c2", "second", 1), _chunk("c3", "third", 2)]
    extractions = {
        chunk.id: GraphExtraction(nodes=[ExtractedNode("alice", "Person", {
            "canonical_key": "Person:Alice", "name": "Alice", "identifier": identity,
            "contact": contact, "tags": tags,
        })], relationships=[ExtractedRelationship("alice", "alice", "SELF")])
        for chunk, identity, contact, tags in zip(
            chunks, [0, 0, 1], [False, True, False], [["a"], ["a", "b", "b"], ["c"]], strict=True,
        )
    }
    claims = {chunk.id: [ExtractedClaim("alice", "X", "x", object_entity_id="alice")] for chunk in chunks}
    result = GraphDocumentAssembler().assemble("doc", "hash", chunks, extractions, {}, "g", claims, schema=schema)
    assert len(result.entities) == 2
    first, second = result.entities
    assert first.canonical_key == "Person:0"
    assert second.canonical_key == "Person:1"
    assert first.properties["contact"] is False
    assert first.properties["tags"] == ["a", "b"]
    assert all(c.entity_id == c.object_entity_id for c in result.claims)
    assert [c.entity_id for c in result.claims] == [first.id, first.id, second.id]
    assert result.relationships[0].source_id == first.id
    (review,) = result.assembly_review_groups
    assert review["decision"] == "identity_retained"
    assert review["conflicts"][0]["property"] == "contact"
    assert [o["value"] for o in review["conflicts"][0]["observations"]] == [False, True]


@pytest.mark.parametrize("empty", [None, "", [], {}])
def test_empty_observations_do_not_split_name_candidates(empty):
    chunks = [_chunk("c1", "first"), _chunk("c2", "second", 1)]
    extractions = {
        chunk.id: GraphExtraction(nodes=[ExtractedNode("a", "Thing", {
            "canonical_key": "Thing:A", "value": value,
        }, canonical_key_trusted=False)])
        for chunk, value in zip(chunks, [empty, 0], strict=True)
    }
    result = GraphDocumentAssembler().assemble("doc", "h", chunks, extractions, {}, "g")
    assert len(result.entities) == 1
    assert result.entities[0].properties["value"] == 0
    assert result.assembly_review_groups == []


def test_explicit_keys_do_not_collapse_labels_or_identity_trust():
    nodes = [
        ExtractedNode("a", "One", {"canonical_key": "shared"}),
        ExtractedNode("b", "Two", {"canonical_key": "shared"}),
        ExtractedNode("c", "One", {"canonical_key": "shared"}, canonical_key_trusted=False),
    ]
    result = GraphDocumentAssembler().assemble(
        "doc", "h", [_chunk("c1", "text")], {"c1": GraphExtraction(nodes=nodes)}, {}, "g",
    )
    assert len({e.id for e in result.entities}) == 3
    assert len(result.entities) == 3


def test_untrusted_lists_are_not_implicitly_unioned():
    from recon_graphrag.extraction.schema import GraphSchema, NodeType, PropertyType

    schema = GraphSchema([NodeType("Thing", properties=[PropertyType("tags", "LIST")])], [])
    chunks = [_chunk("c1", "first"), _chunk("c2", "second", 1)]
    extractions = {chunk.id: GraphExtraction(nodes=[ExtractedNode("a", "Thing", {
        "canonical_key": "Thing:A", "tags": tags,
    }, canonical_key_trusted=False)]) for chunk, tags in zip(chunks, [["a"], ["b"]], strict=True)}
    result = GraphDocumentAssembler().assemble("doc", "h", chunks, extractions, {}, "g", schema=schema)
    assert result.entities[0].properties["tags"] == ["a"]
    assert result.assembly_review_groups[0]["decision"] == "review_required"


def test_same_chunk_candidate_variants_resolve_distinct_local_ids():
    extraction = GraphExtraction(nodes=[
        ExtractedNode(local, "Thing", {"canonical_key": "Thing:A", "value": value},
                      canonical_key_trusted=False)
        for local, value in [("a1", "one"), ("a2", "two")]
    ], relationships=[ExtractedRelationship("a1", "a2", "RELATED")])
    result = GraphDocumentAssembler().assemble(
        "doc", "h", [_chunk("c1", "text")], {"c1": extraction}, {}, "g",
        {"c1": [ExtractedClaim("a1", "REL", "observed", object_entity_id="a2")]},
    )
    first, second = result.entities
    assert first.id != second.id
    assert result.relationships[0].source_id == result.claims[0].entity_id == first.id
    assert result.relationships[0].target_id == result.claims[0].object_entity_id == second.id
    assert {link.entity_id for link in result.evidence_links} == {first.id, second.id}


def test_list_values_without_list_schema_remain_scalar_conflicts():
    chunks = [_chunk("c1", "one"), _chunk("c2", "two", 1)]
    extractions = {chunk.id: GraphExtraction(nodes=[ExtractedNode("a", "Thing", {
        "canonical_key": "Thing:A", "values": value,
    })]) for chunk, value in zip(chunks, [["a"], ["b"]], strict=True)}
    result = GraphDocumentAssembler().assemble("doc", "h", chunks, extractions, {}, "g")
    assert len(result.entities) == 1
    assert result.entities[0].properties["values"] == ["a"]
    assert result.assembly_review_groups[0]["decision"] == "identity_retained"
    assert [item["value"] for item in result.assembly_review_groups[0]["conflicts"][0]["observations"]] == [["a"], ["b"]]


@pytest.mark.parametrize("collision", ["label", "trust", "provisional", "generated"])
def test_generated_keys_avoid_original_and_assigned_keys(collision):
    from uuid import NAMESPACE_URL, uuid5

    from recon_graphrag.extraction.schema import GraphSchema, NodeType, PropertyType

    schema = GraphSchema([NodeType("Person", properties=[PropertyType("identifier")],
                                   identity_property="identifier")], [])
    if collision == "label":
        nodes = [
            ExtractedNode("a", "Person", {"identifier": "x"}),
            ExtractedNode("b", "Organization", {"canonical_key": "Person:x"}),
            ExtractedNode("c", "Person", {"identifier": "Person:x"}),
        ]
        preserved_key = "Person:Person:x"
    elif collision == "trust":
        nodes = [
            ExtractedNode("a", "Thing", {"canonical_key": "shared"}),
            ExtractedNode("b", "Thing", {"canonical_key": "shared"}, canonical_key_trusted=False),
            ExtractedNode("c", "Thing", {"canonical_key": "shared:untrusted"}),
            ExtractedNode("d", "Thing", {"canonical_key": "shared:untrusted:1"}),
        ]
        preserved_key = "shared:untrusted"
    elif collision == "provisional":
        provisional_key = f"shared:provisional:{uuid5(NAMESPACE_URL, 'doc:c1:1')}"
        nodes = [
            ExtractedNode("a", "Thing", {"canonical_key": provisional_key}),
            ExtractedNode("b", "Thing", {"canonical_key": "shared", "value": "one"},
                          canonical_key_trusted=False),
            ExtractedNode("c", "Thing", {"canonical_key": "shared", "value": "two"},
                          canonical_key_trusted=False),
            ExtractedNode("d", "Thing", {"canonical_key": f"{provisional_key}:1"}),
        ]
        preserved_key = provisional_key
    else:
        nodes = [
            ExtractedNode("a", "A", {"canonical_key": "shared"}),
            ExtractedNode("b", "B", {"canonical_key": "shared"}),
            ExtractedNode("c", "A", {"canonical_key": "shared"}, canonical_key_trusted=False),
            ExtractedNode("d", "A", {"canonical_key": "shared:untrusted"}),
            ExtractedNode("e", "C", {"canonical_key": "shared:untrusted"}),
        ]
        preserved_key = "ordinary"
    nodes.append(ExtractedNode("ordinary", "Other", {"canonical_key": "ordinary"}))
    pairs = list(zip(nodes, nodes[1:] + nodes[:1], strict=True))
    chunks = [_chunk("c1", "all identities"), _chunk("c2", "repeat first", 1)]
    extractions = {
        "c1": GraphExtraction(nodes=nodes, relationships=[
            ExtractedRelationship(source.id, target.id, "RELATED") for source, target in pairs
        ]),
        "c2": GraphExtraction(nodes=[nodes[0]]),
    }
    claims = {"c1": [
        ExtractedClaim(source.id, "REL", "observed", object_entity_id=target.id)
        for source, target in pairs
    ]}
    assembler = GraphDocumentAssembler()
    result = assembler.assemble("doc", "h", chunks, extractions, {}, "g", claims, schema=schema)
    assert result == assembler.assemble("doc", "h", chunks, extractions, {}, "g", claims, schema=schema)
    assert len(result.entities) == len(nodes)
    assert len({entity.id for entity in result.entities}) == len(nodes)
    assert len({entity.canonical_key for entity in result.entities}) == len(nodes)
    by_local_id = {node.id: entity for node, entity in zip(nodes, result.entities, strict=True)}
    assert by_local_id["ordinary"].canonical_key == "ordinary"
    assert preserved_key in {entity.canonical_key for entity in result.entities}
    assert len(result.entities[0].description_observations) == 2
    for (source, target), relationship, claim in zip(pairs, result.relationships, result.claims, strict=True):
        assert relationship.source_id == claim.entity_id == by_local_id[source.id].id
        assert relationship.target_id == claim.object_entity_id == by_local_id[target.id].id
        assert relationship.properties["source_canonical_key"] == by_local_id[source.id].canonical_key
        assert relationship.properties["target_canonical_key"] == by_local_id[target.id].canonical_key
    expected_evidence = {("c1", entity.id) for entity in result.entities}
    expected_evidence.add(("c2", result.entities[0].id))
    assert {(link.chunk_id, link.entity_id) for link in result.evidence_links} == expected_evidence
    if collision == "provisional":
        assert set(result.assembly_review_groups[0]["entity_ids"]) == {
            by_local_id["b"].id, by_local_id["c"].id,
        }


@pytest.mark.parametrize(
    ("description_type", "value", "expected"),
    [
        (None, None, ""),
        (None, "  text  ", "text"),
        ("INTEGER", 0, "0"),
        ("BOOLEAN", False, "False"),
        ("FLOAT", 1.5, "1.5"),
        ("LIST", ["a", 0, False], "['a', 0, False]"),
        ("LIST", [], "[]"),
        ("DATE", "2026-01-01", "2026-01-01"),
        ("DATETIME", "2026-01-01T12:00:00", "2026-01-01T12:00:00"),
    ],
)
def test_validated_descriptions_become_text_observations(description_type, value, expected):
    from recon_graphrag.extraction.schema import GraphSchema, NodeType, PropertyType
    from recon_graphrag.extraction.validator import SchemaValidator

    properties = [PropertyType("identifier"), PropertyType("count", "INTEGER"),
                  PropertyType("enabled", "BOOLEAN"), PropertyType("tags", "LIST")]
    if description_type:
        properties.append(PropertyType("description", description_type))
    schema = GraphSchema([NodeType("Thing", properties=properties,
                                   identity_property="identifier")], [])
    chunks = [_chunk(f"c{i}", "observation", i) for i in range(4)]
    extractions = {
        chunk.id: SchemaValidator().validate(GraphExtraction(nodes=[ExtractedNode(
            "a", "Thing", {"identifier": "same", "description": description,
                           "count": 0, "enabled": False, "tags": ["a", 1, False]},
        )]), schema)
        for chunk, description in zip(chunks, [None, value, value, None], strict=True)
    }
    result = GraphDocumentAssembler().assemble(
        "doc", "h", chunks, extractions, {}, "g", schema=schema,
    )
    (entity,) = result.entities
    assert entity.properties["description"] == expected
    assert [obs.description for obs in entity.description_observations] == [
        "", "" if value is None else str(value), "" if value is None else str(value), "",
    ]
    assert all(isinstance(obs.description, str) for obs in entity.description_observations)
    assert entity.properties["count"] == 0
    assert type(entity.properties["count"]) is int
    assert entity.properties["enabled"] is False
    assert entity.properties["tags"] == ["a", 1, False]
    assert extractions["c1"].nodes[0].properties["description"] == value
    assert result.assembly_review_groups == []


@pytest.mark.parametrize(("identity_type", "identity"), [("INTEGER", 0), ("BOOLEAN", False)])
def test_validated_falsy_identities_are_trusted_across_chunks(identity_type, identity):
    from recon_graphrag.extraction.schema import GraphSchema, NodeType, PropertyType, RelationshipType
    from recon_graphrag.extraction.validator import SchemaValidator

    schema = GraphSchema([NodeType("Thing", properties=[PropertyType("identifier", identity_type)],
                                   identity_property="identifier")], [RelationshipType("SELF")])
    chunks = [_chunk("c1", "first"), _chunk("c2", "second", 1)]
    extractions = {
        chunk.id: SchemaValidator().validate(GraphExtraction(
            nodes=[ExtractedNode("a", "Thing", {"identifier": identity}, canonical_key_trusted=False)],
            relationships=[ExtractedRelationship("a", "a", "SELF")],
        ), schema) for chunk in chunks
    }
    result = GraphDocumentAssembler().assemble("doc", "h", chunks, extractions, {}, "g", schema=schema)
    assert result == GraphDocumentAssembler().assemble("doc", "h", chunks, extractions, {}, "g", schema=schema)
    (entity,) = result.entities
    assert entity.canonical_key == f"Thing:{identity}"
    assert entity.properties["identifier"] is identity
    assert len(entity.description_observations) == 2
    assert result.relationships[0].source_id == result.relationships[0].target_id == entity.id
    assert {(link.chunk_id, link.entity_id) for link in result.evidence_links} == {
        (chunk.id, entity.id) for chunk in chunks
    }
    assert result.assembly_review_groups == []


@pytest.mark.parametrize("properties", [{}, {"identifier": None}, {"identifier": ""}])
def test_validated_empty_identities_remain_chunk_local(properties):
    from recon_graphrag.extraction.schema import GraphSchema, NodeType, PropertyType
    from recon_graphrag.extraction.validator import SchemaValidator

    schema = GraphSchema([NodeType("Thing", properties=[PropertyType("identifier")],
                                   identity_property="identifier")], [])
    chunks = [_chunk("c1", "first"), _chunk("c2", "second", 1)]
    extractions = {
        chunk.id: SchemaValidator().validate(GraphExtraction(nodes=[
            ExtractedNode("a", "Thing", properties, canonical_key_trusted=False),
        ]), schema) for chunk in chunks
    }
    result = GraphDocumentAssembler().assemble("doc", "h", chunks, extractions, {}, "g", schema=schema)
    assert [entity.canonical_key for entity in result.entities] == [
        "doc:c1:Thing:a", "doc:c2:Thing:a",
    ]
