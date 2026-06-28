"""Focused community-detection integration test for Neo4j."""

from __future__ import annotations

import pytest

from examples.config import get_neo4j_store
from tests.integration.database_scenarios import (
    assert_weighted_community_detection,
    cleanup_graph,
)
from tests.integration.support import require_integration_env


RUN_FLAG = "RUN_DATABASE_INTEGRATION_TESTS"
GRAPH_NAME = "neo4j-community-weight-integration"
REQUIRED_ENV = ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"]


def preflight_neo4j_graph_data_science(store) -> None:
    for label, query in (
        ("APOC", "RETURN apoc.version() AS version"),
        ("GDS", "RETURN gds.version() AS version"),
    ):
        try:
            store.execute_query(query)
        except Exception as exc:
            pytest.fail(f"Neo4j {label} preflight failed: {exc}")


def cleanup_neo4j_community_graph(store) -> None:
    cleanup_graph(store, GRAPH_NAME)
    store.execute_query(
        "CALL gds.graph.drop($graph_name, false)",
        {"graph_name": GRAPH_NAME},
    )


@pytest.fixture
def neo4j_store():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Neo4j community detection tests",
        fail_on_missing=True,
    )
    store = get_neo4j_store()
    preflight_neo4j_graph_data_science(store)
    cleanup_neo4j_community_graph(store)
    try:
        yield store
    finally:
        try:
            cleanup_neo4j_community_graph(store)
        finally:
            store.driver.close()


@pytest.mark.integration
@pytest.mark.database
def test_neo4j_leiden_uses_configured_relationship_weight_property(neo4j_store):
    assert_weighted_community_detection(neo4j_store, GRAPH_NAME)
