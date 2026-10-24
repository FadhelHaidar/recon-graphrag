"""Write receipts and reference validation for the Cypher writer."""

from __future__ import annotations

from typing import cast

import pytest

from recon_graphrag.extraction.types import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.models.artifacts import ClaimRecord, SourceReference
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.writer_scenarios import (
    FakeGraphStore,
    StatefulFakeGraphStore,
    make_graph_document,
)


class ConfirmedCountStore(StatefulFakeGraphStore):
    """Stateful fake whose entity MERGE confirms a caller-chosen row count."""

    def __init__(self, entity_written: int) -> None:
        super().__init__()
        self.entity_written = entity_written

    def execute_query(self, query: str, parameters: dict | None = None):
        result = super().execute_query(query, parameters)
        if "MERGE (e:__Entity__:" in query:
            return [{"written": self.entity_written}]
        return result


def make_writer(store) -> CypherGraphWriter:
    return CypherGraphWriter(cast(GraphStore, store))


def write_queries(store) -> list[str]:
    return [query for query in store.queries if "MERGE" in query]


def make_document(
    entities: list[EntityRecord],
    chunks: list[ChunkRecord],
    relationships: list[RelationshipRecord] | None = None,
    evidence_links: list[EvidenceLink] | None = None,
    claims: list[ClaimRecord] | None = None,
    document_id: str = "doc:1",
) -> GraphDocument:
    return GraphDocument(
        document=DocumentRecord(id=document_id, text_hash="hash"),
        chunks=chunks,
        entities=entities,
        relationships=relationships or [],
        evidence_links=evidence_links or [],
        claims=claims or [],
    )


def test_write_stats_report_the_rows_the_database_confirmed() -> None:
    store = ConfirmedCountStore(entity_written=0)
    stats = make_writer(store).write_graph_document(make_graph_document())

    # Three entities were submitted, none were persisted.
    assert len(make_graph_document().entities) == 3
    assert stats["entities"] == 0
    assert stats["relationships"] == 2


def test_write_stats_fall_back_to_submitted_rows_without_a_count() -> None:
    store = FakeGraphStore()
    stats = make_writer(store).write_graph_document(make_graph_document())

    assert stats == {
        "documents": 1,
        "chunks": 1,
        "entities": 3,
        "relationships": 2,
        "evidence_links": 1,
        "claims": 0,
    }


def test_unknown_relationship_endpoint_fails_before_any_write() -> None:
    store = StatefulFakeGraphStore()
    document = make_document(
        entities=[
            EntityRecord(id="person:1", type="Person", properties={"name": "Alice"})
        ],
        chunks=[ChunkRecord(id="chunk:1", document_id="doc:1", text="hi", index=0)],
        relationships=[
            RelationshipRecord(
                id="rel:1",
                source_id="person:1",
                target_id="person:ghost",
                type="KNOWS",
            )
        ],
    )

    with pytest.raises(ValueError, match="person:ghost"):
        make_writer(store).write_graph_document(document)

    assert write_queries(store) == []


def test_claim_with_an_absent_chunk_fails_before_any_write() -> None:
    store = StatefulFakeGraphStore()
    document = make_document(
        entities=[
            EntityRecord(id="person:1", type="Person", properties={"name": "Alice"})
        ],
        chunks=[ChunkRecord(id="chunk:1", document_id="doc:1", text="hi", index=0)],
        claims=[
            ClaimRecord(
                id="claim:1",
                entity_id="person:1",
                claim_type="status",
                description="active",
                source=SourceReference(document_id="doc:1", chunk_id="chunk:missing"),
            )
        ],
    )

    with pytest.raises(ValueError, match="chunk:missing"):
        make_writer(store).write_graph_document(document)

    assert write_queries(store) == []


def test_endpoint_owned_by_another_document_is_accepted() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        make_document(
            document_id="doc:a",
            entities=[
                EntityRecord(id="person:1", type="Person", properties={"name": "Alice"})
            ],
            chunks=[
                ChunkRecord(id="doc:a:c0", document_id="doc:a", text="hi", index=0)
            ],
        )
    )

    stats = writer.write_graph_document(
        make_document(
            document_id="doc:b",
            entities=[
                EntityRecord(
                    id="org:1", type="Organization", properties={"name": "Acme"}
                )
            ],
            chunks=[
                ChunkRecord(id="doc:b:c0", document_id="doc:b", text="hi", index=0)
            ],
            relationships=[
                RelationshipRecord(
                    id="rel:b",
                    source_id="person:1",
                    target_id="org:1",
                    type="WORKS_AT",
                )
            ],
            evidence_links=[EvidenceLink(chunk_id="doc:b:c0", entity_id="person:1")],
            claims=[
                ClaimRecord(
                    id="claim:b",
                    entity_id="person:1",
                    claim_type="status",
                    description="active",
                    source=SourceReference(
                        document_id="doc:b", chunk_id="doc:b:c0"
                    ),
                )
            ],
        )
    )

    assert stats["entities"] == 1
    assert stats["relationships"] == 1
    assert stats["evidence_links"] == 1
    assert stats["claims"] == 1
