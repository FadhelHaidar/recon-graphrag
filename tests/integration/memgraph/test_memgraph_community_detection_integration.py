"""Focused community-detection integration test for Memgraph."""

from __future__ import annotations

import pytest

from examples.config import get_memgraph_store
from tests.integration.database_scenarios import (
    assert_weighted_community_detection,
    cleanup_graph,
)
from tests.integration.support import require_integration_env


RUN_FLAG = "RUN_DATABASE_INTEGRATION_TESTS"
GRAPH_NAME = "memgraph-community-weight-integration"
REQUIRED_ENV = ["MEMGRAPH_URL"]


def preflight_memgraph_mage(store) -> None:
    procedures = store.execute_query("CALL mg.procedures() YIELD name RETURN name")
    if not any(
        row.get("name") == "leiden_community_detection.get_subgraph"
        for row in procedures
    ):
        pytest.fail("Memgraph MAGE Leiden procedure is not available.")


@pytest.fixture
def memgraph_store():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Memgraph community detection tests",
        fail_on_missing=True,
    )
    store = get_memgraph_store()
    preflight_memgraph_mage(store)
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
def test_memgraph_leiden_uses_configured_relationship_weight_property(memgraph_store):
    assert_weighted_community_detection(memgraph_store, GRAPH_NAME)
