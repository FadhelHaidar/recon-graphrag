"""Live atomicity checks for one Memgraph entity-resolution merge group."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]

from recon_graphrag.graphdb.entity_resolution import _EntityRecord
from recon_graphrag.graphdb.memgraph.entity_resolution import _MemgraphEntityResolver
from tests.integration.database_scenarios import cleanup_graph, entity_count
from tests.integration.factories import get_memgraph_store
from tests.integration.support import require_integration_env

RUN_FLAG = "RUN_DATABASE_INTEGRATION_TESTS"
GRAPH_NAME = "memgraph-merge-atomicity-integration"
REQUIRED_ENV = ["MEMGRAPH_URL"]


@pytest.fixture
def memgraph_store():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Memgraph merge atomicity tests",
        fail_on_missing=True,
    )
    store = get_memgraph_store()
    cleanup_graph(store, GRAPH_NAME)
    try:
        yield store
    finally:
        try:
            cleanup_graph(store, GRAPH_NAME)
        finally:
            cast(Any, store.driver).close()


class _FailAfterExecutor:
    def __init__(self, executor, marker: str):
        self._executor = executor
        self._marker = marker

    def execute_query(self, query: str, parameters: dict | None = None):
        rows = self._executor.execute_query(query, parameters)
        if self._marker in query:
            raise RuntimeError("injected merge failure")
        return rows


class _FailingMergeStore:
    """Run real Bolt transactions and raise after one marker statement."""

    def __init__(self, store, marker: str):
        self._store = store
        self._marker = marker

    def execute_query(self, query: str, parameters: dict | None = None):
        return self._store.execute_query(query, parameters)

    @contextmanager
    def _transaction(self):
        with self._store._transaction() as executor:
            yield _FailAfterExecutor(executor, self._marker)


def _seed_merge_graph(store) -> None:
    store.execute_query(
        """
        CREATE (a:__Entity__:Person {
            id: $a, graph_name: $graph_name,
            name: 'Alice Smith', description: 'Long name'
        })
        CREATE (b:__Entity__:Person {
            id: $b, graph_name: $graph_name,
            name: 'Alice', description: 'Short name'
        })
        CREATE (c:__Entity__:Person {
            id: $c, graph_name: $graph_name,
            name: 'Bob', description: 'Neighbour'
        })
        CREATE (a)-[:KNOWS {
            graph_name: $graph_name,
            source_chunk_ids: ['chunk-a'],
            descriptions: ['a knows bob'],
            strength: 0.9
        }]->(c)
        CREATE (b)-[:KNOWS {
            graph_name: $graph_name,
            source_chunk_ids: ['chunk-b'],
            descriptions: ['b knows bob'],
            strength: 0.5
        }]->(c)
        CREATE (c)-[:KNOWS {
            graph_name: $graph_name,
            source_chunk_ids: ['chunk-c'],
            descriptions: ['bob knows alice'],
            strength: 0.4
        }]->(a)
        CREATE (claim:Claim {
            id: $claim, graph_name: $graph_name, object_entity_id: $b
        })
        CREATE (claim)-[:SUBJECT_OF]->(a)
        """,
        {
            "graph_name": GRAPH_NAME,
            "a": f"{GRAPH_NAME}:alice-smith",
            "b": f"{GRAPH_NAME}:alice",
            "c": f"{GRAPH_NAME}:bob",
            "claim": f"{GRAPH_NAME}:claim",
        },
    )


def _snapshot(store) -> tuple[list[dict], list[dict], list[dict]]:
    nodes = store.execute_query(
        """
        MATCH (n {graph_name: $graph_name})
        RETURN n.id AS id, labels(n) AS labels, properties(n) AS props
        ORDER BY id
        """,
        {"graph_name": GRAPH_NAME},
    )
    relationships = store.execute_query(
        """
        MATCH (a {graph_name: $graph_name})-[r]->(b {graph_name: $graph_name})
        RETURN a.id AS source, b.id AS target, type(r) AS type,
               properties(r) AS props
        ORDER BY source, type, target
        """,
        {"graph_name": GRAPH_NAME},
    )
    claims = store.execute_query(
        """
        MATCH (c:Claim {graph_name: $graph_name})
        RETURN c.id AS id, c.object_entity_id AS object_entity_id
        ORDER BY id
        """,
        {"graph_name": GRAPH_NAME},
    )
    return nodes, relationships, claims


def _merge_group(
    store,
) -> tuple[_MemgraphEntityResolver, list[_EntityRecord]]:
    resolver = _MemgraphEntityResolver(store)
    entities = resolver._load_entities(GRAPH_NAME, "name")
    duplicates = [
        entity
        for entity in entities
        if entity.entity_id in {
            f"{GRAPH_NAME}:alice-smith",
            f"{GRAPH_NAME}:alice",
        }
    ]
    return resolver, duplicates


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.parametrize(
    "marker",
    [
        "SET n += $props",
        "MERGE (canonical)-[new_r:`KNOWS`]->(target)",
        "SET c.object_entity_id",
        "RETURN count(*) AS deleted",
    ],
)
def test_failure_after_each_merge_phase_rolls_back_every_change(
    memgraph_store, marker,
):
    _seed_merge_graph(memgraph_store)
    before = _snapshot(memgraph_store)
    failing = _FailingMergeStore(memgraph_store, marker)
    resolver, duplicates = _merge_group(failing)

    with pytest.raises(RuntimeError, match="injected merge failure"):
        resolver._merge_groups([duplicates], "name")

    assert _snapshot(memgraph_store) == before


@pytest.mark.integration
@pytest.mark.database
def test_committed_group_keeps_both_contributions_and_rewrites_the_claim(
    memgraph_store,
):
    _seed_merge_graph(memgraph_store)
    resolver, duplicates = _merge_group(memgraph_store)

    merged_nodes = resolver._merge_groups([duplicates], "name")

    assert merged_nodes == 2
    assert entity_count(memgraph_store, GRAPH_NAME) == 2
    assert memgraph_store.execute_query(
        """
        MATCH (c:Claim {graph_name: $graph_name})
        RETURN c.object_entity_id AS object_entity_id
        """,
        {"graph_name": GRAPH_NAME},
    )[0]["object_entity_id"] == f"{GRAPH_NAME}:alice-smith"
    contributions = memgraph_store.execute_query(
        """
        MATCH (:__Entity__ {id: $source, graph_name: $graph_name})
              -[r:KNOWS]->
              (:__Entity__ {id: $target, graph_name: $graph_name})
        RETURN r.source_chunk_ids AS source_chunk_ids,
               r.descriptions AS descriptions
        """,
        {
            "graph_name": GRAPH_NAME,
            "source": f"{GRAPH_NAME}:alice-smith",
            "target": f"{GRAPH_NAME}:bob",
        },
    )
    assert contributions
    assert set(contributions[0]["source_chunk_ids"]) == {"chunk-a", "chunk-b"}
    assert set(contributions[0]["descriptions"]) == {
        "a knows bob",
        "b knows bob",
    }
    assert not memgraph_store.execute_query(
        "MATCH (n:__Entity__ {id: $id, graph_name: $graph_name}) RETURN n",
        {"graph_name": GRAPH_NAME, "id": f"{GRAPH_NAME}:alice"},
    )


@pytest.mark.integration
@pytest.mark.database
def test_stale_group_is_skipped_without_writes_or_count(memgraph_store):
    _seed_merge_graph(memgraph_store)
    resolver, duplicates = _merge_group(memgraph_store)
    memgraph_store.execute_query(
        "MATCH (n:__Entity__ {id: $id, graph_name: $graph_name}) DETACH DELETE n",
        {"graph_name": GRAPH_NAME, "id": f"{GRAPH_NAME}:alice"},
    )
    before = _snapshot(memgraph_store)

    merged_nodes = resolver._merge_groups([duplicates], "name")

    assert merged_nodes == 0
    assert _snapshot(memgraph_store) == before
