"""Unit tests for FalkorDB entity resolution."""

from __future__ import annotations

import pytest

from recon_graphrag.graphdb.falkordb.entity_resolution import (
    _FalkorDBEntityResolver,
    _normalize_name,
)


class FakeGraphStore:
    def __init__(self, entities=None):
        self.entities = entities or []
        self.queries: list[tuple[str, dict]] = []
        self._next_id = 100

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append((query.strip(), parameters or {}))

        if "MATCH (e:__Entity__)" in query and "RETURN" in query:
            return self.entities

        if "properties(n)" in query:
            return [
                {
                    "node_id": parameters["node_ids"][0],
                    "props": {"name": "OpenAI", "graph_name": "g"},
                }
            ]

        if "MATCH (n)-[r]-(other)" in query and "type(r)" in query:
            return [{"rel_type": "KNOWS", "node_id": parameters["other_ids"][0]}]

        return []


@pytest.mark.asyncio
async def test_normalized_strategy_merges_duplicates():
    entities = [
        {
            "node_id": "1",
            "entity_id": "e1",
            "graph_name": "g",
            "resolve_value": "OpenAI",
            "labels": ["__Entity__", "Organization"],
            "properties": {"name": "OpenAI"},
        },
        {
            "node_id": "2",
            "entity_id": "e2",
            "graph_name": "g",
            "resolve_value": "Open AI",
            "labels": ["__Entity__", "Organization"],
            "properties": {"name": "Open AI"},
        },
    ]
    store = FakeGraphStore(entities=entities)
    resolver = _FalkorDBEntityResolver(store)

    result = await resolver.resolve(
        graph_name="g",
        strategy="normalized",
        resolve_property="name",
    )

    assert result["strategy"] == "normalized"
    assert result["merged_groups"] == 1
    assert result["merged_nodes"] == 1


def test_normalize_name_collapses_whitespace_and_case():
    assert _normalize_name("Open AI Inc.") == "openai"
    assert _normalize_name("  Acme  Corp  ") == "acme"
