"""Opt-in filtered-search expansion regressions on disposable backends.

The tests use deterministic vectors and the synthetic corpus. They never call a
provider. Entity tests create scoped indexes on a test-only label. Report tests
redirect the product read to a fixture-owned index and label, so deleted nodes
in a shared index cannot affect the result.
"""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest

from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env
from tests.integration.synthetic_e2e_support import SYNTHETIC_PAGES

pytestmark = [pytest.mark.integration, pytest.mark.database]

REPORT_INDEX = "community-report-embeddings"
REPORT_DIMENSION = 2
TARGET_NAMES = ["Alice Rivera", "Sentinel", "Payments API"]
KEYWORD_QUERY = "Sentinel Alice Payments"


@dataclass(frozen=True)
class LiveSearchGraph:
    store: Any
    backend: str
    graph: str
    leak_graph: str
    entity_label: str
    entity_vector_index: str
    entity_text_index: str
    single_graph: str
    single_label: str
    single_vector_index: str


@dataclass(frozen=True)
class SingleReportGraph:
    store: Any
    graph: str


def _axis_vector(dimension: int, axis: int) -> list[float]:
    vector = [0.0] * dimension
    vector[axis] = 1.0
    return vector


def _mixed_vector(dimension: int, offset: int) -> list[float]:
    """Return a distinct vector that is closer to the first axis."""
    vector = [0.0] * dimension
    vector[0] = 0.8
    if dimension >= 2:
        vector[1] = 0.6 + offset * 0.001
    return vector


def _off_axis_vector(dimension: int, offset: int) -> list[float]:
    """Return a distinct vector that is closer to the second axis."""
    vector = [0.0] * dimension
    vector[1] = 1.0
    if dimension >= 2:
        vector[0] = offset * 0.01
    return vector


def _near_axis_vector(dimension: int, offset: int) -> list[float]:
    """Return a distinct vector that is closer to the first axis."""
    vector = [0.0] * dimension
    vector[0] = 1.0
    if dimension >= 2:
        vector[1] = offset * 0.01
    return vector


def _redirect_report_index(
    monkeypatch: pytest.MonkeyPatch,
    store: Any,
    report_index: str,
    report_label: str,
) -> None:
    """Point the product report query at fixture-owned index and label names."""
    original = store.execute_query

    def redirected(query: str, parameters: dict | None = None) -> list[dict]:
        params = dict(parameters or {})
        if params.get("index_name") == REPORT_INDEX:
            params["index_name"] = report_index
        if REPORT_INDEX in query:
            query = query.replace(REPORT_INDEX, report_index)
        if "c:Community" in query:
            query = query.replace("c:Community", f"c:{report_label}")
        return original(query, params)

    monkeypatch.setattr(store, "execute_query", redirected)


def _wait_for_neo4j_indexes(store: Any, backend: str, *index_names: str) -> None:
    if backend != "neo4j":
        return
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        rows = store.execute_query(
            "SHOW INDEXES YIELD name, type, state "
            "WHERE type = 'VECTOR' AND name IN $names",
            {"names": list(index_names)},
        )
        online = {
            row["name"] for row in rows if str(row.get("state")).upper() == "ONLINE"
        }
        if set(index_names) <= online:
            return
        time.sleep(0.05)
    pytest.fail(f"Neo4j vector indexes did not come online: {index_names}")


def _drop_index(store: Any, backend: str, name: str) -> None:
    statements = (
        (f"DROP INDEX `{name}` IF EXISTS",)
        if backend == "neo4j"
        else (f"DROP VECTOR INDEX `{name}`", f"DROP TEXT INDEX `{name}`")
    )
    for statement in statements:
        with suppress(Exception):
            store.execute_query(statement)


def _seed_entities(
    store: Any, leak_graph: str, graph: str, entity_label: str
) -> None:
    rows = [
        {
            "graph_name": leak_graph,
            "id": f"leak-{index}",
            "name": "Sentinel " * 20,
            "embedding": _mixed_vector(2, index),
        }
        for index in range(60)
    ]
    rows.extend(
        {
            "graph_name": graph,
            "id": f"target-{index}",
            "name": name,
            "embedding": _off_axis_vector(2, index),
        }
        for index, name in enumerate(TARGET_NAMES)
    )
    store.execute_query(
        f"""
        UNWIND $rows AS row
        CREATE (n:{entity_label} {{
          graph_name: row.graph_name,
          id: row.id,
          name: row.name
        }})
        SET n.embedding = row.embedding
        """,
        {"rows": rows},
    )


def _seed_single_entity(store: Any, graph: str, entity_label: str) -> None:
    store.execute_query(
        f"""
        CREATE (n:{entity_label} {{
          graph_name: $graph_name,
          id: 'single-target',
          name: 'single-target'
        }})
        SET n.embedding = $embedding
        """,
        {"graph_name": graph, "embedding": _axis_vector(2, 1)},
    )


def _seed_reports(
    store: Any,
    leak_graph: str,
    graph: str,
    dimension: int,
    report_label: str,
) -> None:
    report_text = SYNTHETIC_PAGES[0]["text"].strip()
    rows = [
        {
            "graph_name": leak_graph,
            "id": f"leak-community-{index}",
            "level": 0,
            "report_text": report_text,
            "embedding": _mixed_vector(dimension, index),
        }
        for index in range(20)
    ]
    rows.extend(
        {
            "graph_name": graph,
            "id": f"target-community-{index}",
            "level": 0,
            "report_text": report_text,
            "embedding": _off_axis_vector(dimension, index),
        }
        for index in range(3)
    )
    rows.append(
        {
            "graph_name": graph,
            "id": "target-community-top",
            "level": 0,
            "report_text": report_text,
            "embedding": _near_axis_vector(dimension, 0),
        }
    )
    rows.append(
        {
            "graph_name": graph,
            "id": "target-community-level-one",
            "level": 1,
            "report_text": report_text,
            "embedding": _near_axis_vector(dimension, 1),
        }
    )
    store.execute_query(
        f"""
        UNWIND $rows AS row
        CREATE (c:{report_label} {{
          graph_name: row.graph_name,
          id: row.id,
          level: row.level,
          report_text: row.report_text,
          report_json: '{{}}',
          rating: 5.0
        }})
        SET c.report_embedding = row.embedding
        """,
        {"rows": rows},
    )


def _mapped_entity_names(
    store: Any, backend: str, graph: str, entity_label: str
) -> dict[Any, str]:
    id_expr = "elementId(n)" if backend == "neo4j" else "id(n)"
    rows = store.execute_query(
        f"MATCH (n:{entity_label} {{graph_name: $graph_name}}) "
        f"RETURN {id_expr} AS id, n.name AS name",
        {"graph_name": graph},
    )
    return {row["id"]: row["name"] for row in rows}


def _record_queries(monkeypatch: pytest.MonkeyPatch, store: Any) -> list[str]:
    queries: list[str] = []
    original = store.execute_query

    def recording(query: str, parameters: dict | None = None) -> list[dict]:
        queries.append(query)
        return original(query, parameters)

    monkeypatch.setattr(store, "execute_query", recording)
    return queries


def _wait_for_single_vector_match(
    store: Any,
    index_name: str,
    entity_label: str,
    graph: str,
    query_vector: list[float],
) -> None:
    """Wait for the approximate index to return its only scoped node."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        rows = store.vector_search(
            index_name,
            query_vector,
            1,
            label=entity_label,
            filters={"graph_name": graph},
        )
        if rows:
            return
        time.sleep(0.05)
    pytest.fail("Vector index did not serve its single scoped match")


def _wait_for_single_report_match(store: Any, graph: str) -> None:
    """Wait for the approximate report index to serve its only scoped node."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        rows = store.vector_search_community_reports(
            _axis_vector(REPORT_DIMENSION, 0),
            graph,
            top_k=1,
            level=0,
        )
        if rows:
            return
        time.sleep(0.05)
    pytest.fail("Report vector index did not serve its single scoped match")


def _memgraph_text_search_skip_reason(store: Any, backend: str) -> str | None:
    if backend != "memgraph":
        return None
    try:
        rows = store.execute_query(
            "CALL mg.procedures() YIELD name, signature "
            "WHERE name = 'text_search.search' RETURN signature"
        )
    except Exception as exc:
        return f"Cannot inspect Memgraph text_search.search: {exc}"
    if not rows:
        return "Memgraph text_search.search is unavailable; MAGE is required."
    signature = str(rows[0].get("signature", ""))
    if "config = {} :: MAP" in signature:
        return (
            "Memgraph MAGE text_search.search expects a config MAP; the product "
            "query needs a compatibility change outside issue #8."
        )
    return None


@pytest.fixture(params=["neo4j", "memgraph"])
def live_search_graph(request, monkeypatch):
    backend = request.param
    prefix = backend.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "filtered search expansion",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if backend == "neo4j" else get_memgraph_store()
    suffix = uuid4().hex
    graph = f"f8-filtered-target-{suffix}"
    leak_graph = f"f8-filtered-leak-{suffix}"
    single_graph = f"f8-filtered-single-{suffix}"
    entity_label = f"F8Entity{suffix}"
    single_label = f"F8Single{suffix}"
    entity_vector_index = f"f8-entity-vector-{suffix}"
    entity_text_index = f"f8-entity-text-{suffix}"
    single_vector_index = f"f8-single-vector-{suffix}"
    report_index = f"f8-report-vector-{suffix}"
    report_label = f"F8Community{suffix}"
    try:
        _redirect_report_index(monkeypatch, store, report_index, report_label)
        store.create_vector_index(
            report_index,
            report_label,
            "report_embedding",
            dimensions=REPORT_DIMENSION,
        )
        store.create_vector_index(
            entity_vector_index,
            entity_label,
            "embedding",
            dimensions=2,
        )
        store.create_vector_index(
            single_vector_index,
            single_label,
            "embedding",
            dimensions=2,
        )
        store.create_fulltext_index(entity_text_index, entity_label, ["name"])
        _seed_entities(store, leak_graph, graph, entity_label)
        _seed_single_entity(store, single_graph, single_label)
        _seed_reports(store, leak_graph, graph, REPORT_DIMENSION, report_label)
        _wait_for_neo4j_indexes(
            store,
            backend,
            report_index,
            entity_vector_index,
            single_vector_index,
        )
        yield LiveSearchGraph(
            store=store,
            backend=backend,
            graph=graph,
            leak_graph=leak_graph,
            entity_label=entity_label,
            entity_vector_index=entity_vector_index,
            entity_text_index=entity_text_index,
            single_graph=single_graph,
            single_label=single_label,
            single_vector_index=single_vector_index,
        )
    finally:
        cleanup_graph(store, graph)
        cleanup_graph(store, leak_graph)
        cleanup_graph(store, single_graph)
        _drop_index(store, backend, entity_vector_index)
        _drop_index(store, backend, entity_text_index)
        _drop_index(store, backend, single_vector_index)
        _drop_index(store, backend, report_index)
        store.driver.close()


@pytest.fixture(params=["neo4j", "memgraph"])
def single_report_graph(request, monkeypatch):
    backend = request.param
    prefix = backend.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "filtered report search expansion",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if backend == "neo4j" else get_memgraph_store()
    suffix = uuid4().hex
    graph = f"f8-single-report-{suffix}"
    report_index = f"f8-single-report-vector-{suffix}"
    report_label = f"F8SingleReport{suffix}"
    try:
        _redirect_report_index(monkeypatch, store, report_index, report_label)
        store.create_vector_index(
            report_index,
            report_label,
            "report_embedding",
            dimensions=REPORT_DIMENSION,
        )
        store.execute_query(
            f"""
            CREATE (c:{report_label} {{
              graph_name: $graph_name,
              id: 'single-report',
              level: 0,
              report_text: $report_text,
              report_json: '{{}}',
              rating: 5.0
            }})
            SET c.report_embedding = $embedding
            """,
            {
                "graph_name": graph,
                "report_text": SYNTHETIC_PAGES[0]["text"].strip(),
                "embedding": _axis_vector(REPORT_DIMENSION, 0),
            },
        )
        _wait_for_neo4j_indexes(store, backend, report_index)
        yield SingleReportGraph(store=store, graph=graph)
    finally:
        cleanup_graph(store, graph)
        _drop_index(store, backend, report_index)
        store.driver.close()


def test_entity_vector_search_recovers_starved_graph_scope(live_search_graph):
    store = live_search_graph.store
    names = _mapped_entity_names(
        store,
        live_search_graph.backend,
        live_search_graph.graph,
        live_search_graph.entity_label,
    )

    rows = store.vector_search(
        index_name=live_search_graph.entity_vector_index,
        query_vector=_axis_vector(2, 0),
        k=5,
        label=live_search_graph.entity_label,
        filters={"graph_name": live_search_graph.graph},
    )

    assert {names[row["id"]] for row in rows} == set(TARGET_NAMES)
    if live_search_graph.backend == "neo4j":
        assert all(isinstance(row["id"], str) for row in rows)
    else:
        assert all(isinstance(row["id"], int) for row in rows)


def test_entity_keyword_search_recovers_starved_graph_scope(live_search_graph):
    store = live_search_graph.store
    skip_reason = _memgraph_text_search_skip_reason(
        store, live_search_graph.backend
    )
    if skip_reason is not None:
        pytest.skip(skip_reason)
    names = _mapped_entity_names(
        store,
        live_search_graph.backend,
        live_search_graph.graph,
        live_search_graph.entity_label,
    )

    rows = store.keyword_search(
        index_name=live_search_graph.entity_text_index,
        query_text=KEYWORD_QUERY,
        k=5,
        label=live_search_graph.entity_label,
        filters={"graph_name": live_search_graph.graph},
    )

    assert {names[row["id"]] for row in rows} == set(TARGET_NAMES)


def test_entity_vector_search_initial_match_does_not_expand(
    live_search_graph, monkeypatch
):
    store = live_search_graph.store
    query_vector = _axis_vector(2, 1)
    _wait_for_single_vector_match(
        store,
        live_search_graph.single_vector_index,
        live_search_graph.single_label,
        live_search_graph.single_graph,
        query_vector,
    )
    queries = _record_queries(monkeypatch, store)

    rows = store.vector_search(
        index_name=live_search_graph.single_vector_index,
        query_vector=query_vector,
        k=1,
        label=live_search_graph.single_label,
        filters={"graph_name": live_search_graph.single_graph},
    )

    assert len(rows) == 1
    assert len(queries) == 1


def test_entity_vector_search_sparse_cap_does_not_expand(
    live_search_graph, monkeypatch
):
    store = live_search_graph.store
    queries = _record_queries(monkeypatch, store)

    rows = store.vector_search(
        index_name=live_search_graph.entity_vector_index,
        query_vector=_axis_vector(2, 0),
        k=200,
        label=live_search_graph.entity_label,
        filters={"graph_name": live_search_graph.graph},
    )

    assert len(rows) == 3
    assert len(queries) == 1


def test_report_vector_search_recovers_starved_graph_and_level_scope(
    live_search_graph,
):
    store = live_search_graph.store

    rows = store.vector_search_community_reports(
        _axis_vector(REPORT_DIMENSION, 0),
        live_search_graph.graph,
        top_k=3,
        level=0,
    )

    ids = {row["id"] for row in rows}
    assert len(rows) == 3
    assert "target-community-top" in ids
    assert ids <= {
        "target-community-0",
        "target-community-1",
        "target-community-2",
        "target-community-top",
    }
    assert {row["level"] for row in rows} == {0}
    assert {row["report_text"] for row in rows} == {
        SYNTHETIC_PAGES[0]["text"].strip()
    }
    assert all(row["id"].startswith("target-community-") for row in rows)


def test_report_vector_search_initial_match_does_not_expand(
    single_report_graph, monkeypatch
):
    store = single_report_graph.store
    _wait_for_single_report_match(store, single_report_graph.graph)
    queries = _record_queries(monkeypatch, store)

    rows = store.vector_search_community_reports(
        _axis_vector(REPORT_DIMENSION, 0),
        single_report_graph.graph,
        top_k=1,
        level=0,
    )

    assert [row["id"] for row in rows] == ["single-report"]
    assert len(queries) == 1


def test_report_vector_search_sparse_cap_does_not_expand(
    live_search_graph, monkeypatch
):
    store = live_search_graph.store
    queries = _record_queries(monkeypatch, store)

    rows = store.vector_search_community_reports(
        _axis_vector(REPORT_DIMENSION, 0),
        live_search_graph.graph,
        top_k=100,
        level=0,
    )

    assert len(rows) == 4
    assert len(queries) == 1
