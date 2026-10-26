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


class ExistingIndexStore(FakeGraphStore):
    def __init__(self, existing=None, inspect_error=None):
        super().__init__()
        self.existing = existing or []
        self.inspect_error = inspect_error

    def execute_query(self, query: str, parameters: dict | None = None):
        if query.strip() == "SHOW INDEX INFO":
            self.calls.append((query.strip(), parameters or {}))
            if self.inspect_error:
                raise self.inspect_error
            return self.existing
        return super().execute_query(query, parameters)


COMPATIBLE_INDEXES = [
    {
        "index name": "entity-embeddings",
        "index type": "vector",
        "label": "__Entity__",
        "property": "embedding",
    },
    {
        "index name": "entity-names",
        "index type": "text",
        "label": "__Entity__",
        "property": "name",
    },
    {
        "index name": "community-report-embeddings",
        "index type": "vector",
        "label": "Community",
        "property": "report_embedding",
    },
]


def test_create_indexes_rejects_unverifiable_vector_indexes():
    store = ExistingIndexStore(COMPATIBLE_INDEXES)
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(ValueError, match="Cannot verify dimensions and similarity"):
        manager.create_indexes()

    assert store.vector_indexes == []
    assert store.fulltext_indexes == []


def test_create_indexes_never_drops_existing_indexes():
    store = ExistingIndexStore([COMPATIBLE_INDEXES[1]])
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    manager.create_indexes()

    assert not any("DROP VECTOR INDEX" in q or "DROP TEXT INDEX" in q for q, _ in store.calls)


def test_create_indexes_raises_on_incompatible_definition():
    incompatible = [dict(COMPATIBLE_INDEXES[0], property="other_embedding")]
    store = ExistingIndexStore(incompatible)
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(ValueError) as excinfo:
        manager.create_indexes()

    message = str(excinfo.value)
    assert "entity-embeddings" in message
    assert "other_embedding" in message
    assert not any("DROP VECTOR INDEX" in q for q, _ in store.calls)


def test_create_indexes_stops_when_inspection_fails():
    store = ExistingIndexStore(inspect_error=RuntimeError("SHOW INDEX INFO unsupported"))
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(RuntimeError, match="Cannot verify existing Memgraph indexes"):
        manager.create_indexes()

    assert store.vector_indexes == []
    assert not any("DROP VECTOR INDEX" in q for q, _ in store.calls)


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


def test_create_indexes_rejects_same_schema_under_another_name():
    """Retrieval targets indexes by name, so a renamed index is not the request."""
    renamed = [dict(COMPATIBLE_INDEXES[1], **{"index name": "old-name"})]
    store = ExistingIndexStore(renamed)
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(ValueError) as excinfo:
        manager.create_indexes()

    message = str(excinfo.value)
    assert "entity-names" in message
    assert "old-name" in message
    assert not any("DROP VECTOR INDEX" in q or "DROP TEXT INDEX" in q for q, _ in store.calls)


# Rows exactly as Memgraph 2.x reports them: no vector index name, no
# dimensions, labels prefixed with ':', text name embedded in the type string.
MEMGRAPH_REPORTED_INDEXES = [
    {
        "index type": "label+property_vector",
        "label": ":Community",
        "property": "report_embedding",
        "count": 0,
    },
    {
        "index type": "label+property_vector",
        "label": ":__Entity__",
        "property": "embedding",
        "count": 0,
    },
    {
        "index type": "label_text (name: entity-names)",
        "label": "__Entity__",
        "property": ["name"],
        "count": 5,
    },
]


def test_index_signature_normalizes_real_memgraph_rows():
    assert IndexManager._index_signature(MEMGRAPH_REPORTED_INDEXES[2]) == (
        "entity-names",
        "text",
        "__Entity__",
        ["name"],
    )
    assert IndexManager._index_signature(MEMGRAPH_REPORTED_INDEXES[1]) == (
        None,
        "vector",
        "__Entity__",
        ["embedding"],
    )


def test_create_indexes_fails_closed_on_real_memgraph_vector_index():
    """Memgraph cannot report vector dimensions, so an existing index is unverifiable."""
    store = ExistingIndexStore(MEMGRAPH_REPORTED_INDEXES)
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(ValueError) as excinfo:
        manager.create_indexes()

    message = str(excinfo.value)
    assert "entity-embeddings" in message
    assert "Cannot verify dimensions and similarity" in message
    assert not any("DROP" in query for query, _ in store.calls)
    assert store.vector_indexes == []


class VectorInfoStore(ExistingIndexStore):
    """Store whose Memgraph answers SHOW VECTOR INDEX INFO."""

    def __init__(self, existing=None, vector_info=None):
        super().__init__(existing)
        self.vector_info = vector_info or []

    def execute_query(self, query: str, parameters: dict | None = None):
        if query.strip() == "SHOW VECTOR INDEX INFO":
            self.calls.append((query.strip(), parameters or {}))
            return self.vector_info
        return super().execute_query(query, parameters)


def _vector_info(dimension: int = 42, metric: str = "cos"):
    return [
        {
            "index_name": "entity-embeddings",
            "label": ":__Entity__",
            "property": "embedding",
            "dimension": dimension,
            "metric": metric,
        },
        {
            "index_name": "community-report-embeddings",
            "label": ":Community",
            "property": "report_embedding",
            "dimension": dimension,
            "metric": metric,
        },
    ]


def test_create_indexes_reuses_compatible_vector_indexes():
    store = VectorInfoStore(MEMGRAPH_REPORTED_INDEXES, _vector_info())
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    manager.create_indexes()

    assert store.vector_indexes == []
    assert not any("DROP" in query for query, _ in store.calls)


def test_create_indexes_reports_reported_dimension_mismatch():
    store = VectorInfoStore(MEMGRAPH_REPORTED_INDEXES, _vector_info(dimension=1536))
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(ValueError, match="dimensions=1536, expected 42"):
        manager.create_indexes()

    assert store.vector_indexes == []


def test_create_indexes_reports_reported_similarity_mismatch():
    store = VectorInfoStore(MEMGRAPH_REPORTED_INDEXES, _vector_info(metric="l2"))
    manager = IndexManager(cast(GraphStore, store), embedding_dim=42)

    with pytest.raises(ValueError, match="similarity='l2', expected 'cos'"):
        manager.create_indexes()

    assert store.vector_indexes == []
