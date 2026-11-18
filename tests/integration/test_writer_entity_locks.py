"""Cross-backend regressions: transient entity locks serialize writer races.

Each test runs one writer to a controlled pause after its authoritative state
read. A competing writer must then wait or abort and retry, so the committed
contributions of distinct documents survive. Run on disposable databases:

    RUN_DATABASE_INTEGRATION_TESTS=1 uv run pytest \
        tests/integration/test_writer_entity_locks.py -m database

Use one index dimension for the whole disposable run. This module passes
``EMBEDDING_DIM`` to ``create_indexes()`` so it matches the selected database.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, cast
from uuid import uuid4

import pytest

from recon_graphrag.extraction.types import GraphDocument
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.pipelines.writer import (
    CypherGraphWriter,
    _is_retryable_write_error,
)
from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env
from tests.pipelines.test_writer_replacement import build_document

pytestmark = [pytest.mark.integration, pytest.mark.database]

_STATE_ENTITY_MARK = "RETURN e.id AS id,"
_LOCK_MARK = "__recon_write_lock"


@pytest.fixture(scope="module", params=["neo4j", "memgraph"])
def backend(request):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "writer entity-lock concurrency regressions",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    store.create_indexes(embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")))
    try:
        yield store
    finally:
        driver = getattr(store, "driver", None)
        if driver is not None:
            driver.close()


@pytest.fixture
def case(backend):
    graph = f"writer-lock-{uuid4().hex}"
    try:
        yield backend, graph
    finally:
        cleanup_graph(backend, graph)


class _PauseAfterStateRead:
    """Proxy store that pauses one writer after its authoritative state read."""

    def __init__(self, store, entered: threading.Event, release: threading.Event):
        self._store = store
        self._entered = entered
        self._release = release
        self._paused = False

    def __getattr__(self, name):
        return getattr(self._store, name)

    def execute_query(self, query, parameters=None):
        return self._store.execute_query(query, parameters)

    def _transaction(self):
        inner = self._store
        owner = self

        class _Transaction:
            def __enter__(self):
                self._context = inner._transaction()
                self._executor = self._context.__enter__()
                return self

            def __exit__(self, *exc):
                return self._context.__exit__(*exc)

            def execute_query(self, query, parameters=None):
                rows = self._executor.execute_query(query, parameters)
                if _STATE_ENTITY_MARK in query and not owner._paused:
                    owner._paused = True
                    owner._entered.set()
                    if not owner._release.wait(30):
                        raise TimeoutError("race release timed out")
                return rows

        return _Transaction()


class _SignalFirstLock:
    """Proxy store that reports the first entity-lock query of one writer."""

    def __init__(self, store, attempted: threading.Event):
        self._store = store
        self._attempted = attempted

    def __getattr__(self, name):
        return getattr(self._store, name)

    def execute_query(self, query, parameters=None):
        return self._store.execute_query(query, parameters)

    def _transaction(self):
        inner = self._store
        owner = self

        class _Transaction:
            def __enter__(self):
                self._context = inner._transaction()
                self._executor = self._context.__enter__()
                return self

            def __exit__(self, *exc):
                return self._context.__exit__(*exc)

            def execute_query(self, query, parameters=None):
                if _LOCK_MARK in query:
                    owner._attempted.set()
                return self._executor.execute_query(query, parameters)

        return _Transaction()


def _run_controlled_race(
    store: GraphStore, paused_document: GraphDocument, other_document: GraphDocument
) -> tuple[dict[str, dict], dict[str, BaseException], dict[str, GraphDocument]]:
    """Pause one writer after its state read, then start the competing writer."""
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    outcomes: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}
    documents = {"paused": paused_document, "other": other_document}

    def run(name, writer, document):
        try:
            outcomes[name] = writer.write_graph_document(document)
        except BaseException as error:  # noqa: BLE001 - asserted below
            errors[name] = error

    paused_writer = CypherGraphWriter(
        cast(GraphStore, _PauseAfterStateRead(store, entered, release))
    )
    other_writer = CypherGraphWriter(
        cast(GraphStore, _SignalFirstLock(store, attempted))
    )
    paused_thread = threading.Thread(
        target=run, args=("paused", paused_writer, paused_document)
    )
    paused_thread.start()
    assert entered.wait(30), "paused writer did not reach its state read"
    other_thread = threading.Thread(
        target=run, args=("other", other_writer, other_document)
    )
    other_thread.start()
    assert attempted.wait(30), "other writer did not reach its lock batch"
    # The bounded retry budget allows about 0.15 s of backoff. Release the
    # first writer inside that window so a conflict can retry, not exhaust.
    time.sleep(0.1)
    release.set()
    paused_thread.join(60)
    other_thread.join(60)
    assert not paused_thread.is_alive(), "paused writer did not finish"
    assert not other_thread.is_alive(), "other writer did not finish"
    return outcomes, errors, documents


def _retry_bounded_conflict(
    store: GraphStore,
    outcomes: dict[str, dict],
    errors: dict[str, BaseException],
    documents: dict[str, GraphDocument],
) -> None:
    """Allow one writer to surface a conflict after bounded retries, then retry."""
    if not errors:
        return
    assert len(errors) == 1, {
        name: f"{type(error).__name__}: {error}" for name, error in errors.items()
    }
    name, error = next(iter(errors.items()))
    assert _is_retryable_write_error(error), (
        f"unexpected permanent error from {name}: {type(error).__name__}: {error}"
    )
    CypherGraphWriter(store).write_graph_document(documents[name])


def _edge_document(
    owner: str, graph: str, description: str, strength: float
) -> GraphDocument:
    document = build_document(
        chunks=[f"{owner}:c0"],
        document_id=owner,
        graph_name=graph,
        entities=[
            ("e1", "Alice", f"{owner} sees Alice"),
            ("e2", "Bob", f"{owner} sees Bob"),
        ],
        relationships=[("KNOWS", "e1", "e2", description)],
    )
    document.relationships[0].strength = strength
    return document


def _edge_addition(
    owner: str, graph: str, description: str, strength: float
) -> GraphDocument:
    document = build_document(
        chunks=[f"{owner}:c0"],
        document_id=owner,
        graph_name=graph,
        entities=[],
        relationships=[("KNOWS", "e1", "e2", description)],
    )
    document.relationships[0].strength = strength
    return document


def _edge_snapshot(store: GraphStore, graph: str) -> list[dict]:
    rows = store.execute_query(
        """
        MATCH (s:__Entity__ {graph_name: $graph_name})-[r]->(t:__Entity__ {graph_name: $graph_name})
        WHERE r.graph_name = $graph_name
        RETURN type(r) AS type, properties(r) AS props
        """,
        {"graph_name": graph},
    )
    return [row["props"] for row in rows if row["type"] == "KNOWS"]


def _contributions(edge: dict, field: str) -> dict[str, Any]:
    return {
        json.loads(value)[0]: json.loads(value)[1]
        for value in edge.get(field) or []
    }


def _assert_no_lock_residue(store: GraphStore, graph: str) -> None:
    rows = store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name})
        RETURN e.id AS id, e.__recon_write_lock AS lock
        """,
        {"graph_name": graph},
    )
    residual = {row["id"]: row["lock"] for row in rows if row["lock"] is not None}
    assert residual == {}


def _assert_single_edge(
    store: GraphStore,
    graph: str,
    *,
    chunk_ids: set[str],
    descriptions: dict[str, Any],
    strengths: dict[str, Any],
    strength: float,
) -> None:
    edges = _edge_snapshot(store, graph)
    assert len(edges) == 1, edges
    edge = edges[0]
    assert edge["id"] == "e1:KNOWS:e2"
    assert set(edge["source_chunk_ids"]) == chunk_ids
    assert _contributions(edge, "description_sources") == descriptions
    assert _contributions(edge, "strength_sources") == strengths
    assert edge["strength"] == strength
    _assert_no_lock_residue(store, graph)


def test_same_edge_creation_race_keeps_both_contributions(case):
    store, graph = case
    outcomes, errors, documents = _run_controlled_race(
        store,
        _edge_document("A", graph, "A created the edge", 9.0),
        _edge_document("B", graph, "B created the edge", 3.0),
    )
    assert outcomes or errors
    _retry_bounded_conflict(store, outcomes, errors, documents)

    _assert_single_edge(
        store,
        graph,
        chunk_ids={"A:c0", "B:c0"},
        descriptions={"A": "A created the edge", "B": "B created the edge"},
        strengths={"A": 9.0, "B": 3.0},
        strength=9.0,
    )
    entity_rows = store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name, id: 'e1'})
        RETURN e.descriptions AS descriptions
        """,
        {"graph_name": graph},
    )
    assert set(entity_rows[0]["descriptions"]) == {
        "A sees Alice",
        "B sees Alice",
    }


def test_same_edge_update_race_keeps_all_contributions(case):
    store, graph = case
    CypherGraphWriter(store).write_graph_document(
        _edge_document("seed", graph, "seed text", 1.0)
    )

    outcomes, errors, documents = _run_controlled_race(
        store,
        _edge_addition("A", graph, "A updated the edge", 9.0),
        _edge_addition("B", graph, "B updated the edge", 3.0),
    )
    assert outcomes or errors
    _retry_bounded_conflict(store, outcomes, errors, documents)

    _assert_single_edge(
        store,
        graph,
        chunk_ids={"seed:c0", "A:c0", "B:c0"},
        descriptions={
            "seed": "seed text",
            "A": "A updated the edge",
            "B": "B updated the edge",
        },
        strengths={"seed": 1.0, "A": 9.0, "B": 3.0},
        strength=9.0,
    )


@pytest.mark.parametrize(
    ("paused_document", "other_document"),
    [
        ("removal", "addition"),
        ("addition", "removal"),
    ],
    ids=["removal-first", "addition-first"],
)
def test_edge_removal_race_keeps_the_concurrent_addition(
    case, paused_document, other_document
):
    store, graph = case
    CypherGraphWriter(store).write_graph_document(
        _edge_document("A", graph, "A edge text", 9.0)
    )
    removal = build_document(
        chunks=[], document_id="A", graph_name=graph, entities=[], relationships=[]
    )
    addition = _edge_document("B", graph, "B edge text", 3.0)
    documents = {"removal": removal, "addition": addition}

    outcomes, errors, raced = _run_controlled_race(
        store, documents[paused_document], documents[other_document]
    )
    assert outcomes or errors
    _retry_bounded_conflict(store, outcomes, errors, raced)

    _assert_single_edge(
        store,
        graph,
        chunk_ids={"B:c0"},
        descriptions={"B": "B edge text"},
        strengths={"B": 3.0},
        strength=3.0,
    )
    entity_rows = store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name, id: 'e1'})
        RETURN e.descriptions AS descriptions
        """,
        {"graph_name": graph},
    )
    assert entity_rows[0]["descriptions"] == ["B sees Alice"]
