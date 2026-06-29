"""Tests for the Memgraph index manager."""

from __future__ import annotations

from recon_graphrag.graphdb.memgraph.index_manager import IndexManager


class FakeGraphStore:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.vector_indexes: list[dict] = []
        self.fulltext_indexes: list[dict] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.calls.append((query.strip(), parameters or {}))
        return []

    def create_vector_index(self, **kwargs):
        self.vector_indexes.append(kwargs)

    def create_fulltext_index(self, **kwargs):
        self.fulltext_indexes.append(kwargs)

    def upsert_vectors(self, **kwargs):
        pass


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


def test_index_manager_create_indexes_adds_uid_constraint():
    store = FakeGraphStore()
    manager = IndexManager(store, embedding_dim=42)

    manager.create_indexes()

    constraint_calls = [call for call in store.calls if "CREATE CONSTRAINT" in call[0]]
    assert len(constraint_calls) == 1
    query = constraint_calls[0][0]
    assert "c.uid IS UNIQUE" in query


def test_index_manager_constraint_swallows_already_exists():
    store = FakeGraphStore()

    def raise_already_exists(query: str, parameters: dict | None = None):
        raise RuntimeError("Constraint already exists")

    store.execute_query = raise_already_exists
    manager = IndexManager(store, embedding_dim=42)

    manager.create_indexes()  # should not raise


def test_index_manager_constraint_prints_unexpected_error(capsys):
    store = FakeGraphStore()

    def raise_unexpected(query: str, parameters: dict | None = None):
        raise RuntimeError("some other error")

    store.execute_query = raise_unexpected
    manager = IndexManager(store, embedding_dim=42)

    manager.create_indexes()

    captured = capsys.readouterr()
    assert "community uniqueness constraint failed" in captured.out
    assert "some other error" in captured.out
