"""Tests for the Memgraph index manager."""

from __future__ import annotations

from typing import cast

import pytest

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.memgraph.index_manager import IndexManager


class FakeGraphStore:
    def __init__(
        self,
        duplicates: dict[str, list[dict]] | None = None,
        pending_backfill: int = 0,
        constraint_error: Exception | None = None,
    ):
        self.calls: list[tuple[str, dict]] = []
        self.vector_indexes: list[dict] = []
        self.fulltext_indexes: list[dict] = []
        self.duplicates = duplicates or {}
        self.pending_backfill = pending_backfill
        self.constraint_error = constraint_error

    def execute_query(self, query: str, parameters: dict | None = None):
        stripped = query.strip()
        self.calls.append((stripped, parameters or {}))
        if "duplicates > 1" in stripped and "count(*) AS duplicates" in stripped:
            label = stripped.split("MATCH (n:")[-1].split(")")[0]
            return self.duplicates.get(label, [])
        if "RETURN count(n) AS backfilled" in stripped:
            return [{"backfilled": self.pending_backfill}]
        if self.constraint_error and "ASSERT" in stripped:
            raise self.constraint_error
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


def test_index_manager_create_indexes_adds_uid_constraints():
    store = FakeGraphStore()
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    manager.create_indexes()

    constraint_calls = [call for call in store.calls if "CREATE CONSTRAINT" in call[0]]
    # Community uid + Document/Chunk/__Entity__ uid constraints.
    assert len(constraint_calls) == 4
    queries = [call[0] for call in constraint_calls]
    assert any("c.uid IS UNIQUE" in query for query in queries)
    for label in ("Document", "Chunk", "__Entity__"):
        assert any(
            "ASSERT n.uid IS UNIQUE" in query and f"(n:{label})" in query
            for query in queries
        )


def test_index_manager_preflights_then_backfills_then_constrains():
    store = FakeGraphStore(pending_backfill=3)
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    manager.create_indexes()

    for label in ("Document", "Chunk", "__Entity__"):
        sequence = [
            call[0]
            for call in store.calls
            if f"(n:{label})" in call[0] and "Community" not in call[0]
        ]
        assert len(sequence) == 3, sequence
        assert "duplicates > 1" in sequence[0]
        assert "WHERE n.uid IS NULL" in sequence[1]
        assert "ASSERT n.uid IS UNIQUE" in sequence[2]


def test_index_manager_backfills_missing_uid():
    store = FakeGraphStore(pending_backfill=7)
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    manager.create_indexes()

    backfills = [
        call for call in store.calls if "RETURN count(n) AS backfilled" in call[0]
    ]
    assert len(backfills) == 3
    for query, _ in backfills:
        assert "SET n.uid = toString(size(n.graph_name))" in query


def test_index_manager_constraint_swallows_already_exists():
    store = FakeGraphStore()

    def raise_already_exists(query: str, parameters: dict | None = None):
        if "CREATE CONSTRAINT" in query:
            raise RuntimeError("Constraint already exists")
        return []

    store.execute_query = raise_already_exists
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    manager.create_indexes()  # should not raise


def test_index_manager_constraint_raises_unexpected_error():
    store = FakeGraphStore()

    def raise_unexpected(query: str, parameters: dict | None = None):
        if "CREATE CONSTRAINT" in query:
            raise RuntimeError("some other error")
        return []

    store.execute_query = raise_unexpected
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(RuntimeError, match="some other error"):
        manager.create_indexes()


def test_index_manager_duplicate_identity_raises_with_diagnostics():
    store = FakeGraphStore(
        duplicates={
            "__Entity__": [{"graph_name": "movie-graph", "id": "dup", "duplicates": 2}]
        }
    )
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(ValueError) as excinfo:
        manager.create_indexes()

    message = str(excinfo.value)
    assert "__Entity__" in message
    assert "movie-graph" in message
    assert "dup" in message
    assert "2" in message


def test_index_manager_no_duplicates_creates_constraints():
    store = FakeGraphStore()
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    manager.create_indexes()

    assert (
        len(
            [
                call
                for call in store.calls
                if "CREATE CONSTRAINT" in call[0] and "IS UNIQUE" in call[0]
            ]
        )
        == 4
    )
