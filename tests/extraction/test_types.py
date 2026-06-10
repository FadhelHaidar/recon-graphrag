"""Tests for database-neutral graph data types."""

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    ExtractedNode,
    ExtractedRelationship,
    GraphDocument,
    GraphExtraction,
    RelationshipRecord,
)


def test_extracted_node_defaults():
    node = ExtractedNode(id="n1", label="Person")
    assert node.properties == {}


def test_extracted_relationship_defaults():
    rel = ExtractedRelationship(source_id="n1", target_id="n2", type="KNOWS")
    assert rel.properties == {}


def test_graph_extraction_defaults():
    extraction = GraphExtraction()
    assert extraction.nodes == []
    assert extraction.relationships == []


def test_document_record():
    doc = DocumentRecord(id="doc:1", text_hash="abc123", metadata={"source": "test"})
    assert doc.graph_name == "entity-graph"


def test_chunk_record():
    chunk = ChunkRecord(
        id="chunk:1",
        document_id="doc:1",
        text="hello world",
        index=0,
    )
    assert chunk.graph_name == "entity-graph"


def test_entity_record():
    entity = EntityRecord(id="e1", type="Person", properties={"name": "Alice"})
    assert entity.graph_name == "entity-graph"


def test_relationship_record():
    rel = RelationshipRecord(
        id="r1",
        source_id="e1",
        target_id="e2",
        type="KNOWS",
        properties={"weight": 1.0},
    )
    assert rel.graph_name == "entity-graph"


def test_evidence_link():
    link = EvidenceLink(chunk_id="chunk:1", entity_id="e1")
    assert link.graph_name == "entity-graph"


def test_graph_document_construction():
    doc = DocumentRecord(id="doc:1", text_hash="abc")
    chunk = ChunkRecord(id="c1", document_id="doc:1", text="hello", index=0)
    entity = EntityRecord(id="e1", type="Person", properties={"name": "Alice"})
    rel = RelationshipRecord(
        id="r1", source_id="e1", target_id="e2", type="KNOWS"
    )
    link = EvidenceLink(chunk_id="c1", entity_id="e1")

    graph_doc = GraphDocument(
        document=doc,
        chunks=[chunk],
        entities=[entity],
        relationships=[rel],
        evidence_links=[link],
    )
    assert graph_doc.document.id == "doc:1"
    assert len(graph_doc.chunks) == 1
    assert len(graph_doc.entities) == 1
    assert len(graph_doc.relationships) == 1
    assert len(graph_doc.evidence_links) == 1
