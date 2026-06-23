"""Characterization tests for GraphDocumentAssembler aggregation behavior.

These tests document the current behavior of ``GraphDocumentAssembler`` so that
later phases can change it intentionally. Tests that assert known-defect
behavior are marked with ``pytest.mark.characterization`` and reference the
Phase 2 aggregation/provenance plan.
"""

from __future__ import annotations

import pytest

from recon_graphrag.extraction.assembler import GraphDocumentAssembler
from recon_graphrag.extraction.chunking import TextChunk
from recon_graphrag.extraction.types import (
    ExtractedNode,
    ExtractedRelationship,
    GraphExtraction,
)


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


@pytest.mark.characterization(
    reason="Entity properties are first-wins; Phase 2 will merge observations."
)
def test_entity_description_first_wins_no_merge():
    doc = make_cross_chunk_graph_document()
    e1 = next(e for e in doc.entities if e.id == "e1")
    assert e1.properties["description"] == "desc from c1"


def test_relationship_observation_count_aggregated():
    doc = make_cross_chunk_graph_document()
    assert len(doc.relationships) == 1
    rel = doc.relationships[0]
    assert rel.observation_count == 3
    assert rel.strength == 1.0  # first extracted weight preserved
    assert rel.properties["weight"] == 3.0  # backward compat alias


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
    assert {e.id for e in second.entities} == {"e1", "e2"}


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
    assert {e.id for e in doc_a.entities} == {e.id for e in doc_b.entities}
