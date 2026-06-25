"""Shared entity-resolution normalization contract tests."""

import pytest

from recon_graphrag.graphdb.entity_resolution import (
    BaseEntityResolver,
    _EntityRecord,
)
from recon_graphrag.graphdb.memgraph.entity_resolution import (
    _normalize_name as normalize_memgraph_name,
)
from recon_graphrag.graphdb.neo4j.entity_resolution import (
    _normalize_name as normalize_neo4j_name,
)


@pytest.mark.parametrize(
    "normalize_name",
    [normalize_neo4j_name, normalize_memgraph_name],
)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("OpenAI", "openai"),
        ("Open AI", "openai"),
        ("U.S.A.", "usa"),
        ("Microsoft Corp.", "microsoft"),
        ("Microsoft Corporation", "microsoft"),
        ("Acme Inc.", "acme"),
        ("Acme Ltd", "acme"),
    ],
)
def test_normalize_name_contract(normalize_name, value, expected):
    assert normalize_name(value) == expected


class ContractResolver(BaseEntityResolver):
    def __init__(self, rows):
        super().__init__(graph_store=None)
        self.rows = rows
        self.merged_groups = []

    def _load_entities(self, graph_name: str, resolve_property: str):
        return self.rows

    def _merge_groups(self, groups, resolve_property: str) -> int:
        self.merged_groups = groups
        return sum(len(group) for group in groups)


@pytest.mark.asyncio
async def test_shared_resolver_preserves_hybrid_auto_merge_result_shape():
    rows = [
        _EntityRecord(1, "e1", "g1", "Person", "John Smith", "johnsmith", {}),
        _EntityRecord(2, "e2", "g1", "Person", "Jon Smith", "jonsmith", {}),
    ]

    class LLM:
        async def ainvoke(self, prompt):
            class Response:
                content = (
                    '{"same_entity": true, "confidence": 0.96, '
                    '"reason": "Names refer to the same person.", '
                    '"merge_allowed": true}'
                )

            return Response()

    resolver = ContractResolver(rows)
    result = await resolver.resolve(
        strategy="hybrid",
        merge_threshold=95.0,
        review_threshold=85.0,
        llm=LLM(),
        allow_ai_auto_merge=True,
    )

    assert result["skipped"] is False
    assert result["strategy"] == "hybrid"
    assert result["merged_groups"] == 1
    assert result["merged_nodes"] == 2
    assert result["candidate_groups"] == 1
    assert result["review_groups"] == []
    assert result["ai_merged_review_groups"][0]["decision"] == "merge"
    assert result["signals"] == {
        "normalized": "used",
        "fuzzy": "used",
        "aliases": "skipped_no_aliases",
        "embeddings": "skipped_no_embedder",
        "llm": "used",
    }
