"""Focused entity-resolution integration tests for Neo4j."""

from __future__ import annotations

import pytest

from examples.config import get_embedder, get_llm, get_neo4j_store
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
GRAPH_NAME = "neo4j-entity-resolution-integration"
REQUIRED_ENV = ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"]


def preflight_neo4j_apoc(store) -> None:
    try:
        store.execute_query("RETURN apoc.version() AS version")
    except Exception as exc:
        pytest.fail(f"Neo4j APOC preflight failed: {exc}")


@pytest.fixture
def neo4j_store():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Neo4j entity resolution tests",
        fail_on_missing=True,
    )
    store = get_neo4j_store()
    preflight_neo4j_apoc(store)
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
async def test_neo4j_normalized_entity_resolution_merges_real_nodes(neo4j_store):
    await assert_normalized_entity_resolution(neo4j_store, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
async def test_neo4j_hybrid_alias_dry_run_returns_candidate_without_merging(
    neo4j_store,
):
    await assert_hybrid_alias_dry_run(neo4j_store, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
async def test_neo4j_hybrid_uses_real_embedder_and_llm_for_review(neo4j_store):
    require_integration_env(
        RUN_AI_FLAG,
        [],
        "real Neo4j LLM/embedder resolution test",
    )
    llm_provider, embedder_provider = require_selected_provider_env(
        "Neo4j entity-resolution AI test"
    )
    llm = get_llm(llm_provider)
    embedder = get_embedder(embedder_provider)
    try:
        await assert_hybrid_ai_review(
            neo4j_store,
            GRAPH_NAME,
            llm,
            embedder,
        )
    finally:
        for resource in (llm, embedder):
            close = getattr(resource, "aclose", None)
            if callable(close):
                await close()
