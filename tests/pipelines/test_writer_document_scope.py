"""B01: the writer rejects child records that leave the document scope.

State reads and reconciliation use the document graph, but each child row writes
its own graph name. A foreign child therefore wrote outside the replaced graph
without an error.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.test_writer_replacement import build_document
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore

DOCUMENT_GRAPH = "tenant-a"
FOREIGN_GRAPH = "tenant-b"


def make_writer(store: StatefulFakeGraphStore) -> CypherGraphWriter:
    return CypherGraphWriter(cast(GraphStore, store))


def scoped_document() -> GraphDocument:
    return build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", "Alice leads Acme."), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
        claims=[("claim:1", "e1", "doc:1:c0")],
        graph_name=DOCUMENT_GRAPH,
    )


def test_scoped_document_is_valid() -> None:
    store = StatefulFakeGraphStore()
    receipt = make_writer(store).write_graph_document(scoped_document())

    assert receipt["chunks"] == 1
    assert receipt["entities"] == 2
    assert store.chunk_ids == {"doc:1:c0"}


def foreign_child(kind: str) -> GraphDocument:
    """Return a scoped document with exactly one child in the foreign graph."""
    document = scoped_document()
    if kind == "chunk":
        document.chunks[0].graph_name = FOREIGN_GRAPH
    elif kind == "entity":
        document.entities[0].graph_name = FOREIGN_GRAPH
    elif kind == "relationship":
        document.relationships[0].graph_name = FOREIGN_GRAPH
    elif kind == "evidence":
        document.evidence_links[0].graph_name = FOREIGN_GRAPH
    elif kind == "claim":
        document.claims[0].graph_name = FOREIGN_GRAPH
    else:
        # Default dataclass graph name inside a custom-scoped document.
        document.entities[0].graph_name = "entity-graph"
    return document


@pytest.mark.parametrize(
    ("kind", "record_kind"),
    [
        ("chunk", "ChunkRecord"),
        ("entity", "EntityRecord"),
        ("relationship", "RelationshipRecord"),
        ("evidence", "EvidenceLink"),
        ("claim", "ClaimRecord"),
        ("default", "EntityRecord"),
    ],
)
def test_foreign_graph_name_child_writes_are_rejected(kind, record_kind) -> None:
    store = StatefulFakeGraphStore()

    with pytest.raises(ValueError) as error:
        make_writer(store).write_graph_document(foreign_child(kind))

    message = str(error.value)
    assert record_kind in message
    assert DOCUMENT_GRAPH in message
    assert "targets graph" in message
    assert store.queries == []


def test_chunk_from_another_document_is_rejected() -> None:
    store = StatefulFakeGraphStore()
    document = scoped_document()
    document.chunks[0].document_id = "doc:other"

    with pytest.raises(ValueError) as error:
        make_writer(store).write_graph_document(document)

    message = str(error.value)
    assert "ChunkRecord" in message
    assert "doc:other" in message
    assert "doc:1" in message
    assert store.queries == []


def test_claim_source_document_must_match() -> None:
    store = StatefulFakeGraphStore()
    document = scoped_document()
    claim = document.claims[0]
    document.claims[0] = replace(
        claim, source=replace(claim.source, document_id="doc:other")
    )

    with pytest.raises(ValueError) as error:
        make_writer(store).write_graph_document(document)

    message = str(error.value)
    assert "ClaimRecord" in message
    assert "doc:other" in message
    assert store.queries == []


def test_shared_entity_reference_keeps_working() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            document_id="doc:1",
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", "Alice leads Acme.")],
            graph_name=DOCUMENT_GRAPH,
        )
    )

    receipt = writer.write_graph_document(
        build_document(
            document_id="doc:2",
            chunks=["doc:2:c0"],
            entities=[],
            evidence=[("doc:2:c0", "e1")],
            claims=[("claim:1", "e1", "doc:2:c0")],
            graph_name=DOCUMENT_GRAPH,
        )
    )

    assert receipt["claims"] == 1
    assert receipt["evidence_links"] == 1


def test_scope_violation_is_checked_on_every_attempt() -> None:
    store = StatefulFakeGraphStore()
    document = scoped_document()
    document.entities[0].graph_name = FOREIGN_GRAPH

    for _ in range(2):
        with pytest.raises(ValueError, match="EntityRecord"):
            make_writer(store).write_graph_document(document)
    assert store.queries == []
