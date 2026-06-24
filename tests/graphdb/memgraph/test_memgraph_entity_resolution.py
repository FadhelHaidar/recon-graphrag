"""Unit tests for Memgraph entity-resolution merge behavior."""

from __future__ import annotations

import json
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
async def test_memgraph_resolver_normalized_merges_case_variants():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "openai",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(strategy="normalized", dry_run=True)

    assert result["merged_groups"] == 1


@pytest.mark.asyncio
async def test_memgraph_resolver_does_not_merge_across_graphs():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g2",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        },
    ]
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(strategy="exact", dry_run=True)

    assert result["merged_groups"] == 0


@pytest.mark.asyncio
async def test_memgraph_resolver_fuzzy_produces_review_candidate():
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
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(
        strategy="fuzzy",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
    )

    assert result["merged_groups"] == 0
    assert result["review_groups"][0]["decision"] == "review"


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_skips_missing_ai_dependencies():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Acme",
            "labels": ["__Entity__", "Organization"],
            "properties": {},
        }
    ]
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(strategy="hybrid", dry_run=True)

    assert result["signals"]["embeddings"] == "skipped_no_embedder"
    assert result["signals"]["llm"] == "skipped_no_llm"


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
async def test_memgraph_resolver_hybrid_sends_property_context_to_llm():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "John Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {
                "name": "John Smith",
                "description": "Senior engineer on the Apollo API team.",
                "canonical_key": "person:john-smith",
                "department": "Platform",
                "embedding": [0.1, 0.2],
            },
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Jon Smith",
            "labels": ["__Entity__", "Person"],
            "properties": {
                "name": "Jon Smith",
                "description": "Apollo API platform engineer.",
                "department": "Platform",
            },
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"same_entity": true, "confidence": 0.91, "merge_allowed": false}'
        )
    )
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(
        strategy="hybrid",
        dry_run=True,
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        context_properties={"Person": ["department"]},
    )

    payload = json.loads(llm.ainvoke.await_args.args[0])
    profiles = payload["candidate"]["entities"]
    assert profiles[0]["description"] == "Senior engineer on the Apollo API team."
    assert profiles[0]["properties"] == {"department": "Platform"}
    assert "embedding" not in json.dumps(profiles)
    assert result["review_groups"][0]["entities"] == profiles


@pytest.mark.asyncio
async def test_memgraph_resolver_hybrid_conflict_properties_block_auto_merge():
    rows = [
        {
            "node_id": 1,
            "entity_id": "e1",
            "graph_name": "g1",
            "resolve_value": "Titanic",
            "labels": ["__Entity__", "Movie"],
            "properties": {"name": "Titanic", "year": "1953"},
        },
        {
            "node_id": 2,
            "entity_id": "e2",
            "graph_name": "g1",
            "resolve_value": "Titanic",
            "labels": ["__Entity__", "Movie"],
            "properties": {"name": "Titanic", "year": "1997"},
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"same_entity": true, "confidence": 1.0, "merge_allowed": true}'
        )
    )
    resolver = _MemgraphEntityResolver(FakeGraphStore(rows=rows))

    result = await resolver.resolve(
        strategy="hybrid",
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=llm,
        allow_ai_auto_merge=True,
        conflict_properties={"Movie": ["year"]},
    )

    llm.ainvoke.assert_not_awaited()
    assert result["merged_groups"] == 0
    assert result["review_groups"][0]["decision"] == "blocked"
    assert result["review_groups"][0]["conflicts"][0]["property"] == "year"


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
