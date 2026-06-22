"""Characterization tests for backend writer MERGE behavior.

These tests document the current merge/replacement behavior of
``Neo4jGraphWriter`` and ``MemgraphGraphWriter``. Known defects are marked with
``pytest.mark.characterization`` and reference the Phase 2 aggregation/provenance
plan.
"""

from __future__ import annotations

import pytest

from recon_graphrag.pipelines.memgraph.writer import MemgraphGraphWriter
from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter
from tests.pipelines.writer_scenarios import (
    StatefulFakeGraphStore,
    make_entity_graph_document,
    make_relationship_graph_document,
)


@pytest.fixture(params=[Neo4jGraphWriter, MemgraphGraphWriter], ids=["neo4j", "memgraph"])
def writer_cls(request):
    return request.param


def test_entity_merge_replaces_properties(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_entity_graph_document("e1", "Alice", document_id="doc:a")
    )
    writer.write_graph_document(
        make_entity_graph_document("e1", "Alicia", document_id="doc:b")
    )

    node = store.get_node("e1")
    assert node is not None
    assert node["name"] == "Alicia"


@pytest.mark.characterization(
    reason="Writer SET replaces weight; Phase 2 will make aggregation idempotent."
)
def test_relationship_merge_replaces_weight(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_relationship_graph_document(
            "e1", "e2", "WORKS_AT", 1.0, document_id="doc:a"
        )
    )
    writer.write_graph_document(
        make_relationship_graph_document(
            "e1", "e2", "WORKS_AT", 1.0, document_id="doc:b"
        )
    )

    rel = store.get_relationship("e1", "e2", "WORKS_AT")
    assert rel is not None
    # Current behavior: second write overwrites the weight.
    assert rel["weight"] == 1.0


@pytest.mark.characterization(
    reason="Writer SET replaces source_chunk_ids; Phase 2 will accumulate."
)
def test_relationship_source_chunk_ids_overwritten(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_relationship_graph_document(
            "e1", "e2", "WORKS_AT", 1.0, ["c1"], document_id="doc:a"
        )
    )
    writer.write_graph_document(
        make_relationship_graph_document(
            "e1", "e2", "WORKS_AT", 1.0, ["c2"], document_id="doc:b"
        )
    )

    rel = store.get_relationship("e1", "e2", "WORKS_AT")
    assert rel is not None
    # Current behavior: second write overwrites source_chunk_ids.
    assert rel["source_chunk_ids"] == ["c2"]


@pytest.mark.characterization(
    reason="MERGE key lacks graph_name; Phase 2 will scope identity by graph."
)
def test_identical_ids_collide_across_graph_names(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_entity_graph_document("e1", "Alice", graph_name="graph-a", document_id="doc:a")
    )
    writer.write_graph_document(
        make_entity_graph_document("e1", "Alicia", graph_name="graph-b", document_id="doc:b")
    )

    # Current behavior: one node with the last-written property values.
    assert store.node_count == 1
    node = store.get_node("e1")
    assert node["graph_name"] == "graph-b"
    assert node["name"] == "Alicia"
