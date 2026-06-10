"""Tests for index manager and internal entity resolver."""

import pytest

from recon_graphrag.graph.index_manager import ExactMatchEntityResolver, IndexManager


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
        return []

    def create_vector_index(self, **kwargs):
        self.vector_indexes.append(kwargs)

    def create_fulltext_index(self, **kwargs):
        self.fulltext_indexes.append(kwargs)

    def upsert_vectors(self, **kwargs):
        pass

    @property
    def driver(self):
        return None


def test_index_manager_create_indexes_uses_graph_store_methods():
    store = FakeGraphStore()
    manager = IndexManager(store, embedding_dim=42)

    manager.create_indexes()

    assert len(store.vector_indexes) == 3
    assert store.vector_indexes[0]["dimensions"] == 42
    assert store.fulltext_indexes == [
        {
            "name": "entity-names",
            "label": "__Entity__",
            "node_properties": ["name"],
        }
    ]


@pytest.mark.asyncio
async def test_resolver_skips_when_apoc_is_unavailable():
    store = FakeGraphStore(apoc_available=False)
    resolver = ExactMatchEntityResolver(store)

    result = await resolver.run()

    assert result["skipped"] is True
    assert result["merged_groups"] == 0


@pytest.mark.asyncio
async def test_resolver_merges_duplicate_entities_with_apoc():
    store = FakeGraphStore(apoc_available=True)
    resolver = ExactMatchEntityResolver(store)

    result = await resolver.run()

    assert result == {"skipped": False, "merged_groups": 2}
    merge_query = store.calls[1][0]
    assert "MATCH (e:__Entity__)" in merge_query
    assert "e.`name` AS resolve_value" in merge_query
    assert "graph_name, domain_label, resolve_value" in merge_query
    assert "apoc.refactor.mergeNodes" in merge_query
