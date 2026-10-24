"""Tests for the Neo4j index manager and internal entity resolver."""

from typing import Any, cast

import pytest

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.neo4j.index_manager import IndexManager


class FakeGraphStore:
    def __init__(self, apoc_available=True):
        self.calls = []
        self.vector_indexes = []
        self.fulltext_indexes = []
        self.apoc_available = apoc_available

    def execute_query(self, query, parameters=None):
        self.calls.append((query.strip(), parameters or {}))
        if "apoc.version" in query and not self.apoc_available:
            raise RuntimeError("There is no procedure with the name apoc.version")
        if "apoc.version" in query:
            return [{"version": "5.0"}]
        if "apoc.refactor.mergeNodes" in query:
            return [{"merged_groups": 2}]
        if "MATCH (e:__Entity__)" in query and "elementId(e) AS node_id" in query:
            return [
                {
                    "node_id": "4:a",
                    "entity_id": "e1",
                    "graph_name": "movie-graph",
                    "resolve_value": "OpenAI",
                    "labels": ["__Entity__", "Organization"],
                    "properties": {},
                },
                {
                    "node_id": "4:b",
                    "entity_id": "e2",
                    "graph_name": "movie-graph",
                    "resolve_value": "openai",
                    "labels": ["__Entity__", "Organization"],
                    "properties": {},
                },
            ]
        return []

    def create_vector_index(self, **kwargs):
        self.vector_indexes.append(kwargs)

    def create_fulltext_index(self, **kwargs):
        self.fulltext_indexes.append(kwargs)

    def upsert_vectors(self, **kwargs):
        pass


def test_index_manager_create_indexes_uses_graph_store_methods():
    store = FakeGraphStore()
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    manager.create_indexes()

    assert len(store.vector_indexes) == 2
    assert store.vector_indexes[0]["dimensions"] == 42
    assert store.fulltext_indexes == [
        {
            "name": "entity-names",
            "label": "__Entity__",
            "node_properties": ["name"],
        }
    ]
    constraint_calls = [call for call in store.calls if "CREATE CONSTRAINT" in call[0]]
    assert len(constraint_calls) == 4
    assert (
        "(c.graph_name, c.level, c.id) IS UNIQUE" in [c[0] for c in constraint_calls][0]
    )
    # F8: per-label uid uniqueness for Document/Chunk/__Entity__.
    assert sum("REQUIRE n.uid IS UNIQUE" in c[0] for c in constraint_calls) == 3


@pytest.mark.asyncio
async def test_index_manager_resolve_entities_normalized_forwards_graph_name():
    store = FakeGraphStore(apoc_available=True)
    manager = IndexManager(cast(GraphStore, store))

    result = await manager.resolve_entities_normalized(
        graph_name="movie-graph",
        dry_run=True,
    )

    assert result["strategy"] == "normalized"
    assert result["merged_groups"] == 1
    load_call = next(
        call
        for call in store.calls
        if "MATCH (e:__Entity__)" in call[0] and "elementId(e) AS node_id" in call[0]
    )
    assert load_call[1]["graph_name"] == "movie-graph"


@pytest.mark.asyncio
async def test_index_manager_resolve_entities_hybrid_uses_embedder():
    store = FakeGraphStore(apoc_available=True)
    embedder = cast(Any, object())
    manager = IndexManager(cast(GraphStore, store), embedder=embedder)
    captured = {}

    class CapturingResolver:
        def __init__(self, graph_store):
            self.graph_store = graph_store

        async def resolve_hybrid(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"strategy": "hybrid"}

    import recon_graphrag.graphdb.neo4j.index_manager as im

    original_resolver = im._Neo4jEntityResolver
    im._Neo4jEntityResolver = CapturingResolver
    try:
        await manager.resolve_entities_hybrid(
            graph_name="movie-graph",
            dry_run=True,
        )
    finally:
        im._Neo4jEntityResolver = original_resolver

    assert captured["kwargs"]["embedder"] is embedder
