"""Tests for backend writer MERGE behavior."""

from __future__ import annotations

import pytest

from recon_graphrag.pipelines.memgraph.writer import MemgraphGraphWriter
from recon_graphrag.pipelines.neo4j.writer import Neo4jGraphWriter
from tests.pipelines.writer_scenarios import (
    StatefulFakeGraphStore,
    make_entity_graph_document,
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
