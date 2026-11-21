"""Focused entity-resolution integration tests for Neo4j."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from neo4j.exceptions import Neo4jError

from recon_graphrag.graphdb.store_base import TransactionExecutor
from tests.integration.database_scenarios import (
    assert_hybrid_ai_review,
    assert_hybrid_alias_dry_run,
    assert_normalized_entity_resolution,
    assert_resolution_keeps_list_source_chunk_ids,
    cleanup_graph,
)
from tests.integration.factories import get_embedder, get_llm, get_neo4j_store
from tests.integration.support import (
    require_integration_env,
    require_selected_provider_env,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable


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
async def test_neo4j_resolution_keeps_list_source_chunk_ids(neo4j_store):
    await assert_resolution_keeps_list_source_chunk_ids(neo4j_store, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.parametrize("failed_write", [None, "SET r.source_chunk_ids", "SET n.aliases"])
async def test_neo4j_merge_group_is_atomic(
    neo4j_store, monkeypatch, failed_write,
):
    params = {"g": GRAPH_NAME}
    neo4j_store.execute_query(
        """
        CREATE (a:__Entity__:Organization {graph_name: $g, id: 'winner', name: 'Open AI'})
        CREATE (b:__Entity__:Organization {graph_name: $g, id: 'loser', name: 'OpenAI'})
        CREATE (q:__Entity__:Quote {graph_name: $g, id: 'quote', name: 'Q009'})
        CREATE (a)-[:OFFERED {graph_name: $g, source_chunk_ids: ['chunk-1']}]->(q)
        CREATE (b)-[:OFFERED {graph_name: $g, source_chunk_ids: ['chunk-1']}]->(q)
        CREATE (:Claim {graph_name: $g, id: 'claim', object_entity_id: 'loser'})
        """,
        params,
    )
    snapshot_query = """
        MATCH (a:Organization {graph_name: $g})-[r:OFFERED]->(q:Quote {graph_name: $g})
        MATCH (c:Claim {graph_name: $g, id: 'claim'})
        RETURN properties(a) AS entity, properties(r) AS edge, c.object_entity_id AS target
        ORDER BY a.id
    """
    before = neo4j_store.execute_query(snapshot_query, params)
    assert len(before) == 2

    if failed_write is None:
        result = await neo4j_store.resolve_entities_normalized(graph_name=GRAPH_NAME)
        assert result["merged_groups"] == 1
        assert result["merged_nodes"] == 2
        (merged,) = neo4j_store.execute_query(snapshot_query, params)
        assert merged["entity"]["id"] == merged["target"] == "winner"
        assert merged["entity"]["aliases"] == ["OpenAI"]
        assert merged["edge"]["source_chunk_ids"] == ["chunk-1"]
        assert merged["edge"]["observation_count"] == 1
        return

    def inject_failure(execute, query, parameters):
        if failed_write in query:
            # Inspect the merge on the same connection before failing the next write.
            merged = execute(snapshot_query, params)
            assert len(merged) == 1 and merged[0]["target"] == "winner"
            return execute("RETURN 1 / toInteger($zero)", {"zero": 0})
        return execute(query, parameters)

    auto_commit = neo4j_store.execute_query
    in_transaction = TransactionExecutor.execute_query
    monkeypatch.setattr(
        neo4j_store, "execute_query",
        lambda query, parameters=None: inject_failure(auto_commit, query, parameters),
    )
    monkeypatch.setattr(
        TransactionExecutor, "execute_query",
        lambda tx, query, parameters=None: inject_failure(
            in_transaction.__get__(tx), query, parameters,
        ),
    )

    with pytest.raises(Neo4jError, match="/ by zero"):
        await neo4j_store.resolve_entities_normalized(graph_name=GRAPH_NAME)
    assert neo4j_store.execute_query(snapshot_query, params) == before


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
                await cast("Awaitable[None]", close())
