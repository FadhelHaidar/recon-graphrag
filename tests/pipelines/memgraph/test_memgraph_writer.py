"""Unit tests for MemgraphGraphWriter."""

from __future__ import annotations

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)
from recon_graphrag.pipelines.memgraph.writer import MemgraphGraphWriter


class FakeGraphStore:
    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append(query.strip())
        self.params.append(parameters or {})


def test_write_graph_document_issues_all_writes():
    store = FakeGraphStore()
    writer = MemgraphGraphWriter(store)

    document = DocumentRecord(
        id="doc-1",
        text_hash="hash",
        graph_name="entity-graph",
        metadata={"source": "web"},
    )
    chunks = [
        ChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            text="hello",
            index=0,
            graph_name="entity-graph",
            metadata={},
        )
    ]
    entities = [
        EntityRecord(
            id="ent-1",
            type="Person",
            graph_name="entity-graph",
            properties={"name": "Alice"},
        )
    ]
    relationships = [
        RelationshipRecord(
            id="rel-1",
            source_id="ent-1",
            target_id="ent-1",
            type="KNOWS",
            graph_name="entity-graph",
            properties={},
        )
    ]
    evidence_links = [
        EvidenceLink(
            chunk_id="chunk-1",
            entity_id="ent-1",
            graph_name="entity-graph",
        )
    ]
    graph_doc = GraphDocument(
        document=document,
        chunks=chunks,
        entities=entities,
        relationships=relationships,
        evidence_links=evidence_links,
    )

    stats = writer.write_graph_document(graph_doc)
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert stats["entities"] == 1
    assert stats["relationships"] == 1
    assert stats["evidence_links"] == 1

    query_text = "\n".join(store.queries)
    assert "MERGE (d:Document" in query_text
    assert "MERGE (c:Chunk" in query_text
    assert "MERGE (e:__Entity__:`Person`" in query_text
    assert "MERGE (c)-[r:FROM_CHUNK]" in query_text
    assert "MERGE (source)-[r:`KNOWS`]" in query_text
