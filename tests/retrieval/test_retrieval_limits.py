"""F9 Phase 1: per-seed relationship/source limits in the shared retrieval query.

The retrieval queries bound relationship and source expansion inside Cypher so a
hub entity's degree cannot decide how many rows cross into Python. These tests
pin the query shape, the store-side limit defaults, and the retriever wiring.
"""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore
from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval._queries import (
    DEFAULT_RETRIEVAL_LIMIT,
    drift_retrieval_query,
    local_retrieval_query,
    set_retrieval_limit_defaults,
)
from recon_graphrag.retrieval.search_drift import DriftSearchRetriever
from recon_graphrag.retrieval.search_local import LocalSearchRetriever


class CapturingStore:
    """Graph store double that records the params handed to fetch_entity_context."""

    def __init__(self):
        self.context_params: list[dict] = []

    def vector_search(self, index_name, query_vector, k, label=None, filters=None):
        return [{"id": "a", "score": 0.8}]

    def keyword_search(self, index_name, query_text, k, label=None, filters=None):
        return [{"id": "a", "score": 1.0}]

    def fetch_entity_context(
        self, matches, retrieval_query=None, query_params=None, mode="local", graph_name=None
    ):
        self.context_params.append(dict(query_params or {}))
        return [
            {
                "title": "Alice (Person)",
                "relationships": [f"rel_{i}" for i in range(20)],
                "relationship_records": [{"weight": 1.0} for _ in range(20)],
                "source_text": ["chunk text"],
                "source_chunk_ids": ["chunk:1"],
                "score": matches[0]["score"],
            }
        ]

    def execute_query(self, query, parameters=None):
        return []

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        return []

    def get_claims_for_entities(self, graph_name, entity_ids):
        return []

    def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
        return []


class FakeEmbedder:
    async def async_embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeLLM:
    async def ainvoke(self, prompt):
        return LLMResponse(content=json.dumps({"answer": "ok", "score": 80, "follow_ups": []}))


# --- Query shape -----------------------------------------------------------


@pytest.mark.parametrize("order_by_score", [True, False])
@pytest.mark.parametrize("build", [local_retrieval_query, drift_retrieval_query])
def test_query_bounds_both_expansion_directions(build, order_by_score):
    query = build(order_by_score=order_by_score)
    assert query.count("LIMIT $rel_limit") == 2  # outgoing + incoming
    assert "LIMIT $source_limit" in query
    assert "ORDER BY ORDER BY" not in query
    assert "LIMIT $rel_limit\n    WITH node\n" not in query


@pytest.mark.parametrize("build", [local_retrieval_query, drift_retrieval_query])
def test_sources_are_limited_in_a_correlated_subquery(build):
    query = build(order_by_score=False)
    assert """CALL (node) {
    OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(chunk:Chunk)
    WHERE chunk.graph_name = node.graph_name
    WITH DISTINCT chunk
    ORDER BY chunk.id ASC
    LIMIT $source_limit
    RETURN collect(chunk) AS selected_chunks
}""" in query


@pytest.mark.parametrize("build", [local_retrieval_query, drift_retrieval_query])
def test_query_orders_relationships_before_limiting(build):
    query = build(order_by_score=False)
    # Deterministic rank before each expansion limit: weight desc, then stable ids.
    assert query.count("ORDER BY coalesce(r.weight, r.observation_count, r.strength, 1.0) DESC") == 2
    for block in query.split("LIMIT $rel_limit")[:2]:
        assert "ORDER BY coalesce(r.weight" in block


@pytest.mark.parametrize("build", [local_retrieval_query, drift_retrieval_query])
def test_query_keeps_source_text_and_ids_aligned(build):
    query = build(order_by_score=False)
    assert "collect(chunk) AS selected_chunks" in query
    assert "[s IN selected_chunks | s.text] AS source_text" in query
    assert "[s IN selected_chunks | s.id] AS source_chunk_ids" in query
    # Graph scoping is unchanged by the limits.
    assert query.count("neighbor.graph_name = node.graph_name") == 2
    assert "chunk.graph_name = node.graph_name" in query


def test_store_limit_defaults_are_unlimited_and_caller_can_narrow():
    parameters: dict = {}
    set_retrieval_limit_defaults(parameters)
    assert parameters == {"rel_limit": DEFAULT_RETRIEVAL_LIMIT, "source_limit": DEFAULT_RETRIEVAL_LIMIT}

    narrowed = {"rel_limit": 4, "source_limit": 2}
    set_retrieval_limit_defaults(narrowed)
    assert narrowed == {"rel_limit": 4, "source_limit": 2}


@pytest.mark.parametrize(
    "module_path",
    [
        "recon_graphrag.retrieval.neo4j.queries",
        "recon_graphrag.retrieval.memgraph.queries",
    ],
)
def test_backend_default_constants_stay_referenceable(module_path):
    """Exported DEFAULT_* constants keep parsing with the new limit params."""
    import importlib

    module = importlib.import_module(module_path)
    for name in ("DEFAULT_LOCAL_RETRIEVAL_QUERY", "DEFAULT_DRIFT_RETRIEVAL_QUERY"):
        query = getattr(module, name)
        assert "LIMIT $rel_limit" in query
        assert "LIMIT $source_limit" in query


@pytest.mark.parametrize("store_type", [Neo4jGraphStore, MemgraphGraphStore])
@pytest.mark.parametrize("mode", ["local", "drift"])
@pytest.mark.parametrize("limits", [{}, {"rel_limit": 2, "source_limit": 1}])
def test_real_store_merges_limits_without_overriding_reserved_parameters(store_type, mode, limits):
    store = store_type(MagicMock())
    store.execute_query = MagicMock(return_value=[])
    matches = [{"id": "1", "score": 0.8}]
    supplied = {**limits, "matches": [], "graph_name": "foreign"}

    store.fetch_entity_context(matches, query_params=supplied, mode=mode, graph_name="owned")

    query, parameters = store.execute_query.call_args.args
    assert parameters["matches"] == matches
    assert parameters["graph_name"] == "owned"
    assert parameters["rel_limit"] == limits.get("rel_limit", DEFAULT_RETRIEVAL_LIMIT)
    assert parameters["source_limit"] == limits.get("source_limit", DEFAULT_RETRIEVAL_LIMIT)
    assert "ORDER BY ORDER BY" not in query
    assert supplied == {**limits, "matches": [], "graph_name": "foreign"}


# --- Retriever wiring ------------------------------------------------------


async def test_local_search_pushes_limits_into_query_params():
    store = CapturingStore()
    retriever = LocalSearchRetriever(cast(Any, store), cast(Any, FakeLLM()), cast(Any, FakeEmbedder()), use_mixed_context=False)

    await retriever.search("who?", synthesize_response=False)

    assert store.context_params[-1]["rel_limit"] == 10
    assert store.context_params[-1]["source_limit"] == 3


def _caps(params: dict) -> dict:
    """Relationship/source caps only; the query vector is not a cap."""
    return {k: v for k, v in params.items() if k in {"rel_limit", "source_limit"}}


def test_ranked_template_orders_sources_by_chunk_similarity():
    for query in (
        local_retrieval_query(order_by_score=False, rank_sources=True),
        drift_retrieval_query(order_by_score=False, rank_sources=True),
    ):
        assert "$query_vector" in query
        assert "END AS source_score" in query
        # Similarity first, so chunk-id order is only the deterministic tie-break.
        assert "ORDER BY source_score DESC, chunk.id ASC" in query


def test_ranked_template_differs_only_in_source_selection():
    unranked = local_retrieval_query(order_by_score=False)
    ranked = local_retrieval_query(order_by_score=False, rank_sources=True)
    removed = [line for line in unranked.splitlines() if line not in ranked]

    # Only the ordering decision changes; the dedupe step is preserved.
    assert removed == ["    ORDER BY chunk.id ASC"]
    assert "    WITH DISTINCT chunk" in ranked
    assert "END AS source_score" in ranked
    # Scoring happens after dedupe, so a chunk reached twice is not ranked twice.
    assert ranked.index("WITH DISTINCT chunk") < ranked.index("END AS source_score")


def test_unranked_template_never_mentions_scores():
    assert "source_score" not in local_retrieval_query(order_by_score=False)
    assert "source_score" not in drift_retrieval_query(order_by_score=False)


async def test_local_search_limits_are_configurable_and_overridable():
    store = CapturingStore()
    retriever = LocalSearchRetriever(
        cast(Any, store),
        cast(Any, FakeLLM()),
        cast(Any, FakeEmbedder()),
        use_mixed_context=False,
        top_k_relationships=4,
        top_k_sources=2,
    )

    await retriever.search("who?", synthesize_response=False)
    assert _caps(store.context_params[-1]) == {"rel_limit": 4, "source_limit": 2}
    assert store.context_params[-1]["query_vector"] == [0.1, 0.2, 0.3]

    await retriever.search(
        "who?", query_params={"rel_limit": 1}, synthesize_response=False
    )
    assert _caps(store.context_params[-1]) == {"rel_limit": 1, "source_limit": 2}


async def test_local_search_mixed_path_sends_the_same_limits():
    store = CapturingStore()
    retriever = LocalSearchRetriever(cast(Any, store), cast(Any, FakeLLM()), cast(Any, FakeEmbedder()), use_mixed_context=True)

    result = await retriever.search(
        "who?", synthesize_response=False, token_budget=500
    )

    assert store.context_params[-1]["rel_limit"] == 10
    assert store.context_params[-1]["source_limit"] == 3
    assert result.mode == "local"


async def test_drift_search_pushes_limits_on_traversal():
    store = CapturingStore()
    retriever = DriftSearchRetriever(cast(Any, store), cast(Any, FakeLLM()), cast(Any, FakeEmbedder()))

    await retriever.search("who?", synthesize_response=False)

    assert store.context_params, "DRIFT traversal never fetched entity context"
    for params in store.context_params:
        assert params["rel_limit"] == 10
        assert params["source_limit"] == 2


async def test_drift_search_limits_are_configurable():
    store = CapturingStore()
    retriever = DriftSearchRetriever(
        cast(Any, store),
        cast(Any, FakeLLM()),
        cast(Any, FakeEmbedder()),
        top_k_relationships=6,
        top_k_sources=1,
    )

    await retriever.search("who?", synthesize_response=False)

    assert store.context_params
    for params in store.context_params:
        assert _caps(params) == {"rel_limit": 6, "source_limit": 1}
