"""Unit tests for the FalkorDB IndexManager."""

from __future__ import annotations

from recon_graphrag.graphdb.falkordb.index_manager import IndexManager
from recon_graphrag.models.types import IndexConfig


class FakeGraphStore:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._vectors = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.calls.append((query.strip(), parameters or {}))

    def create_vector_index(self, **kwargs):
        self.calls.append(("create_vector_index", kwargs))

    def create_fulltext_index(self, **kwargs):
        self.calls.append(("create_fulltext_index", kwargs))


def test_create_indexes_creates_all_expected_indexes():
    store = FakeGraphStore()
    config = IndexConfig()
    mgr = IndexManager(store, embedding_dim=768, index_config=config)
    mgr.create_indexes()

    calls = [call for call in store.calls if call[0] == "create_vector_index"]
    assert len(calls) == 3
    assert calls[0][1]["name"] == config.chunk_vector_index
    assert calls[0][1]["dimensions"] == 768
    assert calls[1][1]["name"] == config.entity_vector_index
    assert calls[2][1]["name"] == config.community_vector_index

    fulltext_calls = [call for call in store.calls if call[0] == "create_fulltext_index"]
    assert len(fulltext_calls) == 1
    assert fulltext_calls[0][1]["name"] == config.entity_fulltext_index
    assert fulltext_calls[0][1]["node_properties"] == ["name"]
