"""Focused persistence smoke test for the Neo4j backend."""

from __future__ import annotations

import pytest

from examples.config import get_neo4j_store
from tests.integration.database_scenarios import (
    assert_cross_document_rerun_idempotent,
    assert_graph_document_write,
    cleanup_graph,
)
from tests.integration.support import require_integration_env


RUN_FLAG = "RUN_DATABASE_INTEGRATION_TESTS"
GRAPH_NAME = "neo4j-store-smoke"
REQUIRED_ENV = ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD"]


@pytest.fixture
def neo4j_store():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Neo4j store integration tests",
        fail_on_missing=True,
    )
    store = get_neo4j_store()
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
def test_neo4j_writes_graph_document_and_reports_counts(neo4j_store):
    assert_graph_document_write(neo4j_store, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
def test_neo4j_cross_document_rerun_is_idempotent(neo4j_store):
    assert_cross_document_rerun_idempotent(neo4j_store, GRAPH_NAME)
