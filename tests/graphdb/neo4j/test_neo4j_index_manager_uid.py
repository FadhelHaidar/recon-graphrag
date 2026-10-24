"""F8 tests for the Neo4j index manager: uid constraints, preflight, backfill."""

from typing import cast

import pytest

from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.neo4j.index_manager import IndexManager


class FakeGraphStore:
    """Fake store preflight/backfill behavior for uid constraint setup."""

    def __init__(self, apoc_available=True, constraint_error=None, pending_backfill=0):
        self.calls = []
        self.vector_indexes = []
        self.fulltext_indexes = []
        self.duplicates: dict[str, list[dict]] = {}
        self._nodes = []
        self.apoc_available = apoc_available
        self.constraint_error = constraint_error
        self.pending_backfill = pending_backfill

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
        if "duplicates > 1" in query and "count(*) AS duplicates" in query:
            label = query.split("MATCH (n:")[-1].split(")")[0]
            return self.duplicates.get(label, [])
        if "RETURN count(n) AS backfilled" in query:
            return [{"backfilled": self.pending_backfill}]
        if self.constraint_error and "n.uid IS UNIQUE" in query:
            raise self.constraint_error
        return []

    def create_vector_index(self, **kwargs):
        self.vector_indexes.append(kwargs)

    def create_fulltext_index(self, **kwargs):
        self.fulltext_indexes.append(kwargs)

    def upsert_vectors(self, **kwargs):
        pass


def _make_manager(**kwargs) -> tuple[IndexManager, FakeGraphStore]:
    store = FakeGraphStore(
        pending_backfill=kwargs.pop("pending_backfill", 0),
        constraint_error=kwargs.pop("constraint_error", None),
    )
    if "duplicates" in kwargs:
        store.duplicates = kwargs.pop("duplicates")
    assert not kwargs
    return IndexManager(cast(GraphStore, store), embedding_dim=42), store


def test_create_indexes_preflights_then_backfills_then_constrains():
    manager, store = _make_manager(pending_backfill=3)
    manager.create_indexes()

    for label in ("Document", "Chunk", "__Entity__"):
        sequence = [
            c[0]
            for c in store.calls
            if f"(n:{label})" in c[0] and "Community" not in c[0]
        ]
        assert len(sequence) == 3, sequence
        assert "duplicates > 1" in sequence[0]
        assert "WHERE n.uid IS NULL" in sequence[1]
        assert "REQUIRE n.uid IS UNIQUE" in sequence[2]


def test_create_indexes_creates_three_uid_constraints_and_community():
    manager, store = _make_manager()
    manager.create_indexes()

    uid_calls = [
        q for q, _ in store.calls if "CREATE CONSTRAINT" in q and "uid IS UNIQUE" in q
    ]
    assert len(uid_calls) == 3
    for label in ("Document", "Chunk", "__Entity__"):
        assert any(f"(n:{label})" in c and "IF NOT EXISTS" in c for c in uid_calls)
    community = [
        q
        for q, _ in store.calls
        if "CREATE CONSTRAINT" in q and "IS UNIQUE" in q and "uid" not in q
    ]
    assert len(community) == 1
    assert "(c.graph_name, c.level, c.id) IS UNIQUE" in community[0]


def test_create_indexes_backfills_missing_uid():
    manager, store = _make_manager(pending_backfill=7)
    manager.create_indexes()

    backfills = [c for c in store.calls if "RETURN count(n) AS backfilled" in c[0]]
    assert len(backfills) == 3
    for query, _ in backfills:
        assert "SET n.uid = toString(size(n.graph_name))" in query


@pytest.mark.asyncio
async def test_preflight_duplicate_identity_raises_with_diagnostics():
    manager, _ = _make_manager(
        duplicates={
            "__Entity__": [{"graph_name": "movie-graph", "id": "dup", "duplicates": 2}]
        }
    )

    with pytest.raises(ValueError) as excinfo:
        manager.create_indexes()

    message = str(excinfo.value)
    assert "__Entity__" in message
    assert "movie-graph" in message
    assert "dup" in message
    assert "2" in message


def test_preflight_no_duplicates_creates_constraints():
    manager, store = _make_manager()
    manager.create_indexes()

    assert (
        len(
            [
                q
                for q, _ in store.calls
                if "CREATE CONSTRAINT" in q and "uid IS UNIQUE" in q
            ]
        )
        == 3
    )


def test_constraint_errors_propagate_with_context():
    manager, _ = _make_manager(constraint_error=RuntimeError("Unable to acquire lock"))

    with pytest.raises(RuntimeError, match="Unable to acquire lock"):
        manager.create_indexes()


def test_unexpected_preflight_error_propagates():
    manager, store = _make_manager()

    def raise_host_error(query, parameters=None):
        if "duplicates > 1" in query:
            raise RuntimeError("connection reset by peer")
        return []

    store.execute_query = raise_host_error

    with pytest.raises(RuntimeError, match="connection reset by peer"):
        manager.create_indexes()
