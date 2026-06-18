"""Unit tests for Memgraph entity-resolution merge behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.graphdb.memgraph.entity_resolution import (
    _EntityRecord,
    _MemgraphEntityResolver,
)


class FakeGraphStore:
    def __init__(self, rows=None):
        self.queries: list[tuple[str, dict]] = []
        self._rows = rows or []

    def execute_query(self, query: str, parameters: dict | None = None):
        params = parameters or {}
        self.queries.append((query.strip(), params))

        if "MATCH (e:__Entity__)" in query and "id(e) AS node_id" in query:
            return self._rows

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


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_keeps_llm_review_when_auto_merge_disabled():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"same_entity": true, "confidence": 0.96, '
                '"reason": "Names refer to the same person.", '
                '"merge_allowed": true}'
            )
        )
    )
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(
        strategy="hybrid",
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        allow_ai_auto_merge=False,
    )

    assert result["merged_groups"] == 0
    assert result["merged_nodes"] == 0
    assert len(result["review_groups"]) == 1
    assert result["review_groups"][0]["llm_review"]["merge_allowed"] is True
    assert result["ai_merged_review_groups"] == []


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_auto_merges_llm_approved_review():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {},
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"same_entity": true, "confidence": 0.96, '
                '"reason": "Names refer to the same person.", '
                '"merge_allowed": true}'
            )
        )
    )
    store = FakeGraphStore(rows=rows)
    resolver = _MemgraphEntityResolver(store)

    result = await resolver.resolve(
        strategy="hybrid",
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        allow_ai_auto_merge=True,
    )

    assert result["merged_groups"] == 1
    assert result["merged_nodes"] == 2
    assert result["review_groups"] == []
    assert result["ai_merged_review_groups"][0]["decision"] == "merge"
    merge_query_text = "\n".join(query for query, _ in store.queries)
    assert "WHERE id(n) IN $other_ids" in merge_query_text
    assert "DELETE n" in merge_query_text
