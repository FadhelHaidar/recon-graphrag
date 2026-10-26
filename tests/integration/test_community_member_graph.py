"""Bounded community member expansion on disposable backends, without providers."""

from uuid import uuid4

import pytest

from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env

pytestmark = [pytest.mark.integration, pytest.mark.database]


def _seed(store, graph: str) -> None:
    """Two hierarchy levels, a cross-graph twin, and an edge-free community."""
    store.execute_query(
        """
        CREATE (a:__Entity__:Person {graph_name:$g, id:'a', name:'Alice'})
        CREATE (b:__Entity__:Person {graph_name:$g, id:'b', name:'Bob'})
        CREATE (c:__Entity__:Company {graph_name:$g, id:'c', name:'Acme'})
        CREATE (lone:__Entity__:Person {graph_name:$g, id:'lone', name:'Solo'})
        CREATE (a)-[:WORKS_AT {weight:4.0, description:'leads'}]->(c)
        CREATE (b)-[:ADVISES {weight:2.0}]->(c)
        CREATE (c)-[:INVESTED_IN {weight:1.0}]->(a)
        CREATE (parent:Community {graph_name:$g, id:'c0', level:0})
        CREATE (child:Community {graph_name:$g, id:'c1', level:1})
        CREATE (lonecomm:Community {graph_name:$g, id:'c9', level:0})
        CREATE (a)-[:IN_COMMUNITY]->(parent)
        CREATE (b)-[:IN_COMMUNITY]->(parent)
        CREATE (c)-[:IN_COMMUNITY]->(parent)
        CREATE (c)-[:IN_COMMUNITY]->(child)
        CREATE (lone)-[:IN_COMMUNITY]->(lonecomm)
        """,
        {"g": graph},
    )
    # Same ids in another graph must never leak into the expansion.
    store.execute_query(
        """
        CREATE (x:__Entity__:Person {graph_name:$other, id:'a', name:'Other Alice'})
        CREATE (y:__Entity__:Person {graph_name:$other, id:'b', name:'Other Bob'})
        CREATE (comm:Community {graph_name:$other, id:'c0', level:0})
        CREATE (x)-[:WORKS_AT]->(y)
        CREATE (x)-[:IN_COMMUNITY]->(comm)
        """,
        {"other": graph + "-other"},
    )


@pytest.fixture(params=["neo4j", "memgraph"])
def backend(request):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS",
        [f"{prefix}_URL"],
        "community member expansion",
        fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    graph = f"f9-members-{uuid4().hex}"
    other = graph + "-other"
    try:
        _seed(store, graph)
        yield store, graph
    finally:
        cleanup_graph(store, graph)
        cleanup_graph(store, other)
        store.driver.close()


def _expand(store, graph, community_id="c0", level=0, member_limit=25):
    rows = store.fetch_community_member_graph(
        graph, community_id, level=level, member_limit=member_limit
    )
    assert len(rows) == 1, f"one row per community, got {rows}"
    return rows[0]


def test_members_and_direction_preserved(backend):
    store, graph = backend
    row = _expand(store, graph)

    assert row["member_count"] == 3
    # Degree-ranked (c has three entity edges), then by id on ties.
    assert [m["id"] for m in row["members"]] == ["c", "a", "b"]
    assert {m["id"]: m["name"] for m in row["members"]} == {
        "a": "Alice", "b": "Bob", "c": "Acme",
    }
    assert "Company" in row["members"][0]["labels"]

    edges = {(e["source"], e["type"], e["target"]): e for e in row["edges"]}
    # Direction is preserved as stored, never reversed or duplicated.
    assert set(edges) == {
        ("a", "WORKS_AT", "c"),
        ("b", "ADVISES", "c"),
        ("c", "INVESTED_IN", "a"),
    }
    assert edges[("a", "WORKS_AT", "c")]["description"] == "leads"
    assert edges[("b", "ADVISES", "c")]["weight"] == 2.0
    # Every edge endpoint is a selected member; Community nodes never appear.
    assert all(node_id in {"a", "b", "c"} for edge in row["edges"]
               for node_id in (edge["source"], edge["target"]))


def test_higher_level_community_and_member_limit(backend):
    store, graph = backend

    capped = _expand(store, graph, member_limit=1)
    assert capped["member_count"] == 1
    assert [m["id"] for m in capped["members"]] == ["c"]  # highest degree first
    assert capped["edges"] == []  # no other member selected, so no edges

    child = _expand(store, graph, community_id="c1", level=1)
    assert [m["id"] for m in child["members"]] == ["c"]
    assert child["edges"] == []  # c's edges all leave the level-1 community

    # Right id, wrong level: hierarchical membership is level-scoped.
    assert store.fetch_community_member_graph(graph, "c1", level=0) == []
    assert store.fetch_community_member_graph(graph, "missing", level=0) == []


def test_edge_free_community_still_returns_members(backend):
    store, graph = backend
    row = _expand(store, graph, community_id="c9")
    assert [m["id"] for m in row["members"]] == ["lone"]
    assert row["edges"] == []


def test_graph_isolation(backend):
    store, graph = backend
    row = _expand(store, graph)
    assert {m["name"] for m in row["members"]} == {"Alice", "Bob", "Acme"}
    # Expanding in the twin graph sees only its own entities.
    twin = _expand(store, graph + "-other")
    assert [m["id"] for m in twin["members"]] == ["a"]
    assert [m["name"] for m in twin["members"]] == ["Other Alice"]
