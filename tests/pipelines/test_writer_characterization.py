"""Tests for backend writer MERGE behavior."""

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


def test_entity_merge_unions_descriptions(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_entity_graph_document(
            "e1", "Alice", document_id="doc:a", description="Alice leads Acme."
        )
    )
    writer.write_graph_document(
        make_entity_graph_document(
            "e1", "Alicia", document_id="doc:b", description="Alice founded Beta."
        )
    )

    node = store.get_node("e1")
    assert node is not None
    assert node["descriptions"] == ["Alice leads Acme.", "Alice founded Beta."]
    assert node["observation_count"] == 2
    assert node["description"] == "Alice leads Acme.\nAlice founded Beta."


def test_relationship_merge_unions_source_chunks_and_weight(writer_cls):
    store = StatefulFakeGraphStore()
    writer = writer_cls(store)

    writer.write_graph_document(
        make_relationship_graph_document(
            "person:alice",
            "org:acme",
            "WORKS_AT",
            weight=1.0,
            source_chunk_ids=["chunk:a"],
            document_id="doc:a",
        )
    )
    writer.write_graph_document(
        make_relationship_graph_document(
            "person:alice",
            "org:acme",
            "WORKS_AT",
            weight=1.0,
            source_chunk_ids=["chunk:b"],
            document_id="doc:b",
        )
    )

    rel = store.get_relationship("person:alice", "org:acme", "WORKS_AT")
    assert rel is not None
    assert rel["source_chunk_ids"] == ["chunk:a", "chunk:b"]
    assert rel["observation_count"] == 2
    assert rel["weight"] == 2.0
