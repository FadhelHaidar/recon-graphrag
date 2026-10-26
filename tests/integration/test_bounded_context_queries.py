"""Opt-in default-query regressions on disposable backends, without providers."""

from uuid import uuid4

import pytest

from tests.integration.factories import get_memgraph_store, get_neo4j_store
from tests.integration.support import cleanup_graph, require_integration_env
from tests.integration.synthetic_e2e_support import SYNTHETIC_PAGES

pytestmark = [pytest.mark.integration, pytest.mark.database]


@pytest.fixture(params=["neo4j", "memgraph"])
def context_graph(request, degree):
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS", [f"{prefix}_URL"],
        "bounded retrieval regressions", fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    graph = f"f9-retrieval-{uuid4().hex}"
    try:
        store.execute_query(
            """
            UNWIND ['a', 'b', 'isolated'] AS sid
            CREATE (:__Entity__:Person {graph_name:$g, id:sid, name:sid})
            """, {"g": graph},
        )
        store.execute_query(
            """
            MATCH (seed:Person {graph_name:$g}) WHERE seed.id <> 'isolated'
            UNWIND range(1, $degree) AS i
            CREATE (other:__Entity__:Person {graph_name:$g, id:seed.id + toString(i), name:'neighbor'})
            CREATE (seed)-[:KNOWS {weight:i}]->(other)
            CREATE (other)-[:KNOWS {weight:i}]->(seed)
            CREATE (chunk:Chunk {graph_name:$g, id:seed.id + ':chunk:' + toString(i), text:$text})
            CREATE (chunk)-[:FROM_CHUNK]->(seed)
            CREATE (chunk)-[:FROM_CHUNK]->(seed)
            """, {"g": graph, "degree": degree, "text": SYNTHETIC_PAGES[0]["text"]},
        )
        id_expr = "elementId(n)" if request.param == "neo4j" else "toString(id(n))"
        matches = store.execute_query(
            f"MATCH (n:Person {{graph_name:$g}}) WHERE n.id IN ['a', 'b', 'isolated'] "
            f"RETURN {id_expr} AS id, 1.0 AS score ORDER BY n.id", {"g": graph},
        )
        yield store, graph, matches
    finally:
        cleanup_graph(store, graph)
        store.driver.close()


@pytest.mark.parametrize("rel_limit", [0, 2])
@pytest.mark.parametrize("degree", [4, 40])
@pytest.mark.parametrize("mode", ["local", "drift"])
@pytest.mark.parametrize("source_limit", [0, 1, 3])
def test_limits_are_per_seed_and_keep_both_directions(context_graph, degree, rel_limit, mode, source_limit):
    store, graph, matches = context_graph
    params = {"rel_limit": rel_limit, "source_limit": source_limit}
    rows = store.fetch_entity_context(matches, mode=mode, graph_name=graph, query_params=params)

    assert {r["entity_id"] for r in rows} == {"a", "b", "isolated"}
    # With LIMIT 0 both UNION branches return the same empty connection list.
    assert len(rows) == (6 if rel_limit else 3)
    for row in rows:
        seed = row["entity_id"]
        if seed == "isolated":
            assert row["relationship_records"] == []
            assert row["source_chunk_ids"] == []
            assert row["source_text"] == []
            continue
        expected_ids = sorted(f"{seed}:chunk:{i}" for i in range(1, degree + 1))[:source_limit]
        assert row["source_chunk_ids"] == expected_ids
        assert row["source_text"] == [SYNTHETIC_PAGES[0]["text"]] * source_limit
        assert [r["weight"] for r in row["relationship_records"]] == list(
            range(degree, degree - rel_limit, -1)
        )
    for seed in ("a", "b"):
        assert {r["dir"] for row in rows if row["entity_id"] == seed
                for r in row["relationship_records"]} == ({"in", "out"} if rel_limit else set())

    assert store.fetch_entity_context(
        matches, mode=mode, graph_name=graph, query_params=params
    ) == rows


@pytest.fixture(params=["neo4j", "memgraph"])
def ranked_context_graph(request):
    """One entity whose source chunks carry known vectors, on both backends."""
    prefix = request.param.upper()
    require_integration_env(
        "RUN_DATABASE_INTEGRATION_TESTS", [f"{prefix}_URL"],
        "ranked source selection", fail_on_missing=True,
    )
    store = get_neo4j_store() if request.param == "neo4j" else get_memgraph_store()
    graph = f"f9-ranking-{uuid4().hex}"
    try:
        store.execute_query(
            "CREATE (:__Entity__:Person {graph_name:$g, id:'a', name:'a'})",
            {"g": graph},
        )
        store.execute_query(
            """
            MATCH (seed:Person {graph_name:$g, id:'a'})
            UNWIND range(1, 4) AS i
            CREATE (chunk:Chunk {graph_name:$g, id:'a:chunk:' + toString(i),
                                 text:'text ' + toString(i)})
            CREATE (chunk)-[:FROM_CHUNK]->(seed)
            """, {"g": graph},
        )
        chunk_id_expr = "elementId(c)" if request.param == "neo4j" else "id(c)"
        by_chunk = {
            row["chunk_id"]: row["id"]
            for row in store.execute_query(
                f"MATCH (c:Chunk {{graph_name:$g}}) "
                f"RETURN {chunk_id_expr} AS id, c.id AS chunk_id",
                {"g": graph},
            )
        }
        # The query vector below points at [1.0, 0.0]: chunk 2 is aligned,
        # chunk 3 is diagonal, chunk 1 is orthogonal, chunk 4 has no vector.
        store.upsert_vectors(
            [by_chunk[f"a:chunk:{i}"] for i in (1, 2, 3)],
            "embedding",
            [[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]],
        )
        match_id_expr = (
            "elementId(n)" if request.param == "neo4j" else "toString(id(n))"
        )
        matches = store.execute_query(
            f"MATCH (n:Person {{graph_name:$g}}) "
            f"RETURN {match_id_expr} AS id, 1.0 AS score",
            {"g": graph},
        )
        yield store, graph, matches
    finally:
        cleanup_graph(store, graph)
        store.driver.close()


def _sources(store, graph, matches, query_params):
    rows = store.fetch_entity_context(
        matches, mode="local", graph_name=graph, query_params=query_params
    )
    return rows[0]


def test_sources_are_ranked_by_chunk_similarity(ranked_context_graph):
    store, graph, matches = ranked_context_graph

    row = _sources(store, graph, matches, {"query_vector": [1.0, 0.0]})

    # Aligned first, then diagonal, then the two scoring 0.0 in chunk-id order.
    assert row["source_chunk_ids"] == [
        "a:chunk:2", "a:chunk:3", "a:chunk:1", "a:chunk:4",
    ]
    assert row["source_text"] == ["text 2", "text 3", "text 1", "text 4"]


def test_ranking_keeps_citations_aligned_under_a_source_limit(ranked_context_graph):
    store, graph, matches = ranked_context_graph

    row = _sources(
        store, graph, matches, {"query_vector": [1.0, 0.0], "source_limit": 2}
    )

    assert row["source_chunk_ids"] == ["a:chunk:2", "a:chunk:3"]
    assert row["source_text"] == ["text 2", "text 3"]


@pytest.mark.parametrize("query_vector", [None, [1.0, 0.0, 0.0]])
def test_without_a_usable_vector_order_stays_chunk_id(ranked_context_graph, query_vector):
    store, graph, matches = ranked_context_graph

    params = {} if query_vector is None else {"query_vector": query_vector}
    row = _sources(store, graph, matches, params)

    # Unembedded chunks and mismatched dimensions score 0.0 rather than erroring,
    # so the previous chunk-id ordering is preserved exactly.
    assert row["source_chunk_ids"] == [
        "a:chunk:1", "a:chunk:2", "a:chunk:3", "a:chunk:4",
    ]
    assert row["source_text"] == ["text 1", "text 2", "text 3", "text 4"]
