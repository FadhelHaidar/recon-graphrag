"""Integration smoke test for the Memgraph backend.

Requires a running Memgraph instance with MAGE and opt-in via the environment.
"""

from __future__ import annotations

import pytest

from examples.config import get_memgraph_store
from tests.integration.database_scenarios import (
    assert_cross_document_rerun_idempotent,
    assert_graph_document_write,
    assert_graph_name_isolation,
    cleanup_graph,
)
from tests.integration.support import require_integration_env

RUN_FLAG = "RUN_MEMGRAPH_INTEGRATION_TESTS"
REQUIRED_ENV = ["MEMGRAPH_URL"]
GRAPH_NAME = "memgraph-store-smoke"


def _memgraph_env_or_skip() -> None:
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Memgraph store integration tests",
        fail_on_missing=True,
    )


@pytest.fixture
def memgraph_store():
    _memgraph_env_or_skip()

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
def test_memgraph_writes_graph_document_and_reports_counts(memgraph_store):
    assert_graph_document_write(memgraph_store, GRAPH_NAME)


@pytest.mark.integration
def test_memgraph_cross_document_rerun_is_idempotent(memgraph_store):
    assert_cross_document_rerun_idempotent(memgraph_store, GRAPH_NAME)


@pytest.mark.characterization
@pytest.mark.integration
def test_memgraph_graph_name_isolation_characterization(memgraph_store):
    # This test documents the current cross-graph collision defect.
    # Phase 2 will update writer MERGE keys to scope by graph_name.
    assert_graph_name_isolation(
        memgraph_store,
        graph_name_a=f"{GRAPH_NAME}-a",
        graph_name_b=f"{GRAPH_NAME}-b",
    )
