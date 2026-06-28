"""Focused entity-resolution integration tests for Memgraph."""

from __future__ import annotations

import pytest

from tests.integration.factories import get_embedder, get_llm, get_memgraph_store
from tests.integration.database_scenarios import (
    assert_hybrid_ai_review,
    assert_hybrid_alias_dry_run,
    assert_normalized_entity_resolution,
    cleanup_graph,
)
from tests.integration.support import (
    require_integration_env,
    require_selected_provider_env,
)


RUN_FLAG = "RUN_DATABASE_INTEGRATION_TESTS"
RUN_AI_FLAG = "RUN_ENTITY_RESOLUTION_AI_TESTS"
GRAPH_NAME = "memgraph-entity-resolution-integration"
REQUIRED_ENV = ["MEMGRAPH_URL"]


@pytest.fixture
def memgraph_store():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Memgraph entity resolution tests",
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
            store.driver.close()


@pytest.mark.integration
@pytest.mark.database
async def test_memgraph_normalized_entity_resolution_merges_real_nodes(memgraph_store):
    await assert_normalized_entity_resolution(memgraph_store, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
async def test_memgraph_hybrid_alias_dry_run_returns_candidate_without_merging(
    memgraph_store,
):
    await assert_hybrid_alias_dry_run(memgraph_store, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
async def test_memgraph_hybrid_uses_real_embedder_and_llm_for_review(memgraph_store):
    require_integration_env(
        RUN_AI_FLAG,
        [],
        "real Memgraph LLM/embedder resolution test",
    )
    llm_provider, embedder_provider = require_selected_provider_env(
        "Memgraph entity-resolution AI test"
    )
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)
    try:
        await assert_hybrid_ai_review(
            memgraph_store,
            GRAPH_NAME,
            llm,
            embedder,
        )
    finally:
        for resource in (llm, embedder):
            close = getattr(resource, "aclose", None)
            if callable(close):
                await close()
