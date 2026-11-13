"""B05: a missing or malformed relationship ID fails before any mutation.

Reconciliation sorts relationship IDs to delete edges that lost their last
contribution. Mixed ``None`` and string IDs raised ``TypeError`` after writes had
started, so the caller saw an unrelated crash and, without a transaction, a
partial write.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.test_writer_replacement import (
    _mutation_queries,
    build_document,
)
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore

RELATIONSHIP_QUERY_MARKERS = (
    "UNWIND $documents AS row",
    "UNWIND $chunks AS row",
    "MERGE (e:__Entity__",
    "UNWIND $relationships AS row",
    "UNWIND $links AS row",
    "UNWIND $claims AS row",
    "UNWIND $claim_ids AS id",
    "UNWIND $chunk_ids AS id",
    "UNWIND $rel_ids AS id",
    "DETACH DELETE e",
    "SET e.description_sources = row.description_sources",
)


def make_writer(store: StatefulFakeGraphStore) -> CypherGraphWriter:
    return CypherGraphWriter(cast(GraphStore, store))


def seed_relationship_document(store: StatefulFakeGraphStore) -> None:
    make_writer(store).write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
            relationships=[("WORKS_AT", "e1", "e2", None)],
        )
    )


def mutations_after(store: StatefulFakeGraphStore, index: int) -> list[str]:
    return [
        query
        for query in store.queries[index:]
        if any(marker in query for marker in RELATIONSHIP_QUERY_MARKERS)
    ]


def test_missing_stored_relationship_id_fails_before_mutation() -> None:
    store = StatefulFakeGraphStore()
    seed_relationship_document(store)
    # Imported data can omit the domain ID property.
    for relationship in store._relationships.values():
        relationship["id"] = None
    queries_before = len(store.queries)

    with pytest.raises(ValueError) as error:
        make_writer(store).write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", None), ("e2", "Acme", None)],
                relationships=[],
            )
        )

    message = str(error.value)
    assert "entity-graph" in message
    assert "WORKS_AT" in message
    assert "e1" in message and "e2" in message
    assert "id" in message
    assert mutations_after(store, queries_before) == []
    assert len(store._relationships) == 1


def test_mixed_relationship_id_types_fail_before_sorting() -> None:
    store = StatefulFakeGraphStore()
    make_writer(store).write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None), ("e3", "Globex", None)],
            relationships=[
                ("WORKS_AT", "e1", "e2", None),
                ("WORKS_AT", "e1", "e3", None),
            ],
        )
    )
    stored = list(store._relationships.values())
    assert len(stored) == 2
    stored[0]["id"] = None
    queries_before = len(store.queries)

    # Both edges lose their last contribution, so cleanup would sort one None
    # and one string ID. Pre-fix this raised TypeError after writes started.
    with pytest.raises(ValueError, match="unusable id"):
        make_writer(store).write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", None), ("e2", "Acme", None), ("e3", "Globex", None)],
                relationships=[],
            )
        )

    assert mutations_after(store, queries_before) == []
    assert len(store._relationships) == 2


def test_empty_incoming_relationship_id_is_rejected() -> None:
    store = StatefulFakeGraphStore()
    document = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
    )
    document.relationships[0].id = ""

    with pytest.raises(ValueError, match="submitted relationship"):
        make_writer(store).write_graph_document(document)

    assert store.queries == []
    assert store._relationships == {}


def test_non_string_incoming_relationship_id_is_rejected() -> None:
    store = StatefulFakeGraphStore()
    document = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
    )
    document.relationships[0].id = 42  # type: ignore[assignment]

    with pytest.raises(ValueError, match="submitted relationship"):
        make_writer(store).write_graph_document(document)

    assert store.queries == []


def test_custom_string_relationship_id_remains_supported() -> None:
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    document = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
    )
    document.relationships[0].id = "custom:works-at:alice"
    writer.write_graph_document(document)

    replacement = build_document(
        chunks=["doc:1:c0"],
        entities=[("e1", "Alice", None), ("e2", "Acme", None)],
        relationships=[("WORKS_AT", "e1", "e2", None)],
    )
    replacement.relationships[0].id = "custom:works-at:alice"
    receipt = writer.write_graph_document(replacement)

    assert receipt["relationships"] == 1
    assert [rel["id"] for rel in store._relationships.values()] == [
        "custom:works-at:alice"
    ]


def test_unrelated_malformed_relationship_is_not_scanned() -> None:
    store = StatefulFakeGraphStore()
    seed_relationship_document(store)
    for relationship in store._relationships.values():
        relationship["id"] = None

    receipt = make_writer(store).write_graph_document(
        build_document(
            document_id="doc:2",
            chunks=["doc:2:c0"],
            entities=[("e3", "Globex", None)],
        )
    )

    assert receipt["documents"] == 1
    assert store.chunk_ids == {"doc:1:c0", "doc:2:c0"}


class TransientError(Exception):
    """A retryable backend abort, matched by class name in the writer."""


class _AbortingExecutor:
    def __init__(self, store: AbortingStore, on_abort) -> None:
        self._store = store
        self._on_abort = on_abort

    def execute_query(self, query: str, parameters: dict | None = None):
        if not self._store.aborted and "UNWIND $documents AS row" in query:
            self._store.aborted = True
            self._on_abort()
            raise TransientError("simulated transient abort")
        return self._store.execute_query(query, parameters)


class AbortingStore(StatefulFakeGraphStore):
    """First attempt aborts retryably; later attempts read current state."""

    def __init__(self, on_abort) -> None:
        super().__init__()
        self._on_abort = on_abort
        self.attempts = 0
        self.aborted = False

    def _transaction(self):
        return self

    def __enter__(self):
        self.attempts += 1
        return _AbortingExecutor(self, self._on_abort)

    def __exit__(self, *exc_info):
        return False


def test_malformed_relationship_preflight_is_retry_local() -> None:
    holder: dict[str, AbortingStore] = {}

    def corrupt_stored_id() -> None:
        for relationship in holder["store"]._relationships.values():
            relationship["id"] = None

    store = AbortingStore(corrupt_stored_id)
    holder["store"] = store
    writer = make_writer(store)
    writer.write_graph_document(
        build_document(
            chunks=["doc:1:c0"],
            entities=[("e1", "Alice", None), ("e2", "Acme", None)],
            relationships=[("WORKS_AT", "e1", "e2", None)],
        )
    )
    store.attempts = 0
    store.aborted = False

    # Attempt one reads a valid ID and aborts retryably. The stored ID becomes
    # malformed before attempt two, which must validate its own fresh read.
    with pytest.raises(ValueError, match="stored relationship"):
        writer.write_graph_document(
            build_document(
                chunks=["doc:1:c0"],
                entities=[("e1", "Alice", None), ("e2", "Acme", None)],
                relationships=[],
            )
        )

    assert store.attempts == 2


# --- B02: one logical relationship per graph/source/type/target ---


def _stored_row(
    *,
    rel_id: str | None,
    source_id: str = "person:1",
    target_id: str = "person:2",
    rel_type: str = "KNOWS",
    chunk_ids: tuple[str, ...] = ("chunk:1",),
) -> dict:
    return {
        "id": rel_id,
        "type": rel_type,
        "source_id": source_id,
        "target_id": target_id,
        "graph_name": "entity-graph",
        "source_chunk_ids": list(chunk_ids),
        "descriptions": [],
        "description_sources": [],
        "strength_sources": [],
        "description": None,
        "description_summary_status": None,
        "strength": None,
        "properties": {},
    }


def _state(relationships: list[dict]) -> Any:
    return SimpleNamespace(
        chunk_ids=["chunk:1"],
        evidence=[],
        claims=[],
        relationships=relationships,
        entities={},
    )


def _logical_document() -> GraphDocument:
    return build_document(
        chunks=["chunk:1"],
        entities=[("person:1", "Alice", None), ("person:2", "Bob", None)],
        relationships=[("KNOWS", "person:1", "person:2", None)],
    )


def test_relationship_rows_keep_the_stored_id_for_the_same_logical_edge() -> None:
    """A different incoming ID must not rename the stored logical edge."""
    writer = make_writer(StatefulFakeGraphStore())
    document = _logical_document()

    rows = writer._relationship_rows(document, _state([_stored_row(rel_id="stored-id")]))

    assert [row["id"] for row in rows] == ["stored-id"]


def test_relationship_rows_keep_the_incoming_id_for_a_new_logical_edge() -> None:
    writer = make_writer(StatefulFakeGraphStore())
    document = _logical_document()

    rows = writer._relationship_rows(
        document,
        _state([_stored_row(rel_id="other-id", target_id="person:3")]),
    )

    assert [row["id"] for row in rows] == ["person:1:KNOWS:person:2"]


def test_duplicate_stored_relationships_are_rejected_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two stored edges for one logical key must not collapse into one row."""
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    document = _logical_document()
    state = _state([_stored_row(rel_id="first")])
    duplicates = [
        {
            "source_id": "person:1",
            "target_id": "person:2",
            "type": "KNOWS",
            "id": rel_id,
            "rel_graph_name": "entity-graph",
        }
        for rel_id in ("first", "second")
    ]
    monkeypatch.setattr(writer, "_read_document_state", lambda _document: state)
    monkeypatch.setattr(
        writer, "_read_relationship_targets", lambda _document: duplicates
    )

    with pytest.raises(ValueError, match="ambiguous"):
        writer.write_graph_document(document)

    assert store.relationship_count == 0
    assert store.node_count == 0


def test_stored_id_for_an_incompatible_logical_key_is_rejected() -> None:
    """One stored ID must not identify two different endpoint/type keys."""
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    first = build_document(
        chunks=["chunk:1"],
        entities=[("person:1", "Alice", None), ("person:3", "Cara", None)],
        relationships=[("KNOWS", "person:1", "person:3", None)],
    )
    first = replace(
        first,
        relationships=[replace(first.relationships[0], id="shared-id")],
    )
    writer.write_graph_document(first)

    document = build_document(
        chunks=["chunk:2"],
        entities=[("person:1", "Alice", None), ("person:2", "Bob", None)],
        relationships=[("KNOWS", "person:1", "person:2", None)],
    )
    document = replace(
        document,
        relationships=[replace(document.relationships[0], id="shared-id")],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        writer.write_graph_document(document)

    assert store.relationship_count == 1
    assert [rel["id"] for rel in store._relationships.values()] == ["shared-id"]


def test_two_new_keys_sharing_one_incoming_id_are_rejected() -> None:
    """One submitted id must not name two logical edges."""
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    document = build_document(
        chunks=["chunk:1"],
        entities=[
            ("person:1", "Alice", None),
            ("person:2", "Bob", None),
            ("person:3", "Cara", None),
        ],
        relationships=[
            ("KNOWS", "person:1", "person:2", None),
            ("KNOWS", "person:1", "person:3", None),
        ],
    )
    document = replace(
        document,
        relationships=[
            replace(rel, id="shared-id") for rel in document.relationships
        ],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        writer.write_graph_document(document)

    assert _mutation_queries(store) == []
    assert store.relationship_count == 0


def test_repeated_incoming_key_writes_one_edge() -> None:
    """A document may repeat a logical key without creating parallel edges."""
    store = StatefulFakeGraphStore()
    writer = make_writer(store)
    document = build_document(
        chunks=["chunk:1"],
        entities=[("person:1", "Alice", None), ("person:2", "Bob", None)],
        relationships=[
            ("KNOWS", "person:1", "person:2", "first description"),
            ("KNOWS", "person:1", "person:2", "second description"),
        ],
    )

    writer.write_graph_document(document)

    assert store.relationship_count == 1
    assert store._relationships[("entity-graph", "person:1", "person:2", "KNOWS")][
        "id"
    ] == "person:1:KNOWS:person:2"
