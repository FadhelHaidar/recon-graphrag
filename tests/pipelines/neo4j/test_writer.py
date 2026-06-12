"""Tests for Neo4jGraphWriter and GraphWriter protocol."""

import pytest

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)
from recon_graphrag.graphdb.neo4j.cypher import escape_cypher_identifier
from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter


class FakeGraphStore:
    def __init__(self):
        self.queries = []
        self.params = []

    def execute_query(self, query, parameters=None):
        self.queries.append(query.strip())
        self.params.append(parameters or {})
        return []

    def create_vector_index(self, **kwargs):
        pass

    def create_fulltext_index(self, **kwargs):
        pass

    def upsert_vectors(self, **kwargs):
        pass


def test_escape_cypher_identifier():
    assert escape_cypher_identifier("Movie") == "`Movie`"
    assert escape_cypher_identifier("M`ovie") == "`M``ovie`"


def test_writer_returns_stats():
    store = FakeGraphStore()
    writer = Neo4jGraphWriter(store)

    doc = GraphDocument(
        document=DocumentRecord(id="doc:1", text_hash="abc"),
        chunks=[
            ChunkRecord(id="c1", document_id="doc:1", text="hello", index=0),
        ],
        entities=[
            EntityRecord(id="e1", type="Person", properties={"name": "Alice"}),
        ],
        relationships=[
            RelationshipRecord(
                id="r1", source_id="e1", target_id="e2", type="KNOWS"
            ),
        ],
        evidence_links=[
            EvidenceLink(chunk_id="c1", entity_id="e1"),
        ],
    )

    stats = writer.write_graph_document(doc)
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert stats["entities"] == 1
    assert stats["relationships"] == 1
    assert stats["evidence_links"] == 1


def test_writer_groups_entities_by_type():
    store = FakeGraphStore()
    writer = Neo4jGraphWriter(store)

    doc = GraphDocument(
        document=DocumentRecord(id="doc:1", text_hash="abc"),
        chunks=[],
        entities=[
            EntityRecord(id="e1", type="Person", properties={"name": "Alice"}),
            EntityRecord(id="e2", type="Person", properties={"name": "Bob"}),
            EntityRecord(id="e3", type="Movie", properties={"name": "Inception"}),
        ],
        relationships=[],
        evidence_links=[],
    )

    writer.write_graph_document(doc)

    entity_queries = [q for q in store.queries if "MERGE (e:__Entity__" in q]
    assert len(entity_queries) == 2  # Person and Movie groups


def test_writer_groups_relationships_by_type():
    store = FakeGraphStore()
    writer = Neo4jGraphWriter(store)

    doc = GraphDocument(
        document=DocumentRecord(id="doc:1", text_hash="abc"),
        chunks=[],
        entities=[
            EntityRecord(id="e1", type="Person", properties={}),
            EntityRecord(id="e2", type="Movie", properties={}),
        ],
        relationships=[
            RelationshipRecord(
                id="r1", source_id="e1", target_id="e2", type="DIRECTED"
            ),
            RelationshipRecord(
                id="r2", source_id="e1", target_id="e2", type="ACTED_IN"
            ),
        ],
        evidence_links=[],
    )

    writer.write_graph_document(doc)

    rel_queries = [q for q in store.queries if "MERGE (source)-[r:" in q]
    assert len(rel_queries) == 2  # DIRECTED and ACTED_IN groups
