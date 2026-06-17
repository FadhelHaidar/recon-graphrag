"""Unit tests for Memgraph entity-resolution merge behavior."""

from __future__ import annotations

from recon_graphrag.graphdb.memgraph.entity_resolution import (
    _EntityRecord,
    _MemgraphEntityResolver,
)


class FakeGraphStore:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        params = parameters or {}
        self.queries.append((query.strip(), params))

        if "RETURN DISTINCT type(r) AS rel_type" in query:
            return [{"rel_type": "ACTED_IN", "node_id": 2}]

        if "RETURN id(n) AS node_id, properties(n) AS props" in query:
            return [
                {
                    "node_id": 1,
                    "props": {
                        "name": "Christopher Nolan",
                        "aliases": ["C. Nolan"],
                        "source": "a",
                    },
                },
                {
                    "node_id": 2,
                    "props": {
                        "name": "Chris Nolan",
                        "aliases": ["Nolan"],
                        "source": "b",
                    },
                },
            ]

        return [{"merged_id": 1}]


def test_combine_properties_combines_conflicts_and_aliases():
    resolver = _MemgraphEntityResolver(FakeGraphStore())

    props = resolver._combine_properties([1, 2])

    assert props["name"] == ["Christopher Nolan", "Chris Nolan"]
    assert props["aliases"] == ["C. Nolan", "Nolan"]
    assert props["source"] == ["a", "b"]


def test_rewire_relationships_uses_merge_and_self_loop_guards():
    store = FakeGraphStore()
    resolver = _MemgraphEntityResolver(store)

    resolver._rewire_relationships(canonical_id=1, other_ids=[2])

    query_text = "\n".join(query for query, _ in store.queries)
    assert "MERGE (canonical)-[new_r:`ACTED_IN`]->(target)" in query_text
    assert "MERGE (source)-[new_r:`ACTED_IN`]->(canonical)" in query_text
    assert "WHERE NOT id(target) IN $merged_ids" in query_text
    assert "WHERE NOT id(source) IN $merged_ids" in query_text
    assert "AND id(target) IN $merged_ids" in query_text
    assert "AND id(source) IN $merged_ids" in query_text


def test_merge_groups_reports_neo4j_like_processed_node_count():
    store = FakeGraphStore()
    resolver = _MemgraphEntityResolver(store)
    group = [
        _EntityRecord(1, "person-1", "entity-graph", "Person", "Christopher Nolan", "christophernolan"),
        _EntityRecord(2, "person-2", "entity-graph", "Person", "Chris Nolan", "chrisnolan"),
    ]

    merged_nodes = resolver._merge_groups([group], "name")

    assert merged_nodes == 2
    props = next(params["props"] for query, params in store.queries if "SET n += $props" in query)
    assert props["name"] == "Christopher Nolan"
    assert props["aliases"] == ["C. Nolan", "Nolan", "Chris Nolan"]
