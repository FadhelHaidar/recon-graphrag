"""Live publication integration tests for Memgraph.

These tests are opt-in and were not run in Stage 3. They require a disposable
Memgraph database with MAGE.
"""

from __future__ import annotations

import pytest

from recon_graphrag.communities.memgraph.detection import CommunityDetector
from recon_graphrag.graphdb._community_publication import (
    new_community_generation_token,
    replace_community_hierarchy,
    verify_memgraph_community_generation_constraint,
)
from tests.integration.community_publication_scenarios import (
    PUBLICATION_RELATIONSHIP,
    assert_controlled_publication_race,
    assert_first_concurrent_publication,
    assert_missing_constraint_fails_closed,
    assert_publication_rollback,
    assert_replacement_concurrent_publication,
    controlled_layout_rows,
)
from tests.integration.factories import get_memgraph_store
from tests.integration.support import require_integration_env

RUN_FLAG = "RUN_DATABASE_INTEGRATION_TESTS"
GRAPH_NAME = "memgraph-community-publication-integration"
REQUIRED_ENV = ["MEMGRAPH_URL"]


def preflight_memgraph_mage(store) -> None:
    procedures = store.execute_query("CALL mg.procedures() YIELD name RETURN name")
    if not any(
        row.get("name") == "leiden_community_detection.get_subgraph"
        for row in procedures
    ):
        pytest.fail("Memgraph MAGE Leiden procedure is not available.")


def cleanup_memgraph_publication_graph(store) -> None:
    store.execute_query(
        "MATCH (n {graph_name: $graph_name}) DETACH DELETE n",
        {"graph_name": GRAPH_NAME},
    )


def publish_controlled_layout(store):
    """Return a publisher that writes one controlled layout.

    The publisher uses the production constraint check and replacement
    transaction. It only supplies deterministic layout rows instead of a MAGE
    result, so two concurrent publications have distinct membership.
    """
    detector = CommunityDetector(
        store,
        relationship_types=[PUBLICATION_RELATIONSHIP],
        graph_name=GRAPH_NAME,
        random_seed=42,
    )

    def publish(layout: str, barrier=None):
        rows = store.execute_query(
            """
            MATCH (e:__Entity__ {graph_name: $graph_name})
            RETURN e.id AS id, id(e) AS entity_id
            ORDER BY e.id
            """,
            {"graph_name": GRAPH_NAME},
        )
        entity_ids = {row["id"]: row["entity_id"] for row in rows}
        assignments = controlled_layout_rows(GRAPH_NAME, layout)
        paths = {
            entity_ids[entity_id]: [int(community_id)]
            for entity_id, community_id in assignments.items()
        }
        community_rows, membership_rows, parent_rows = (
            detector._build_hierarchy_rows(paths)
        )
        token = new_community_generation_token()

        def verify(st):
            if barrier is not None:
                barrier.wait(timeout=30)
            verify_memgraph_community_generation_constraint(st)

        replace_community_hierarchy(
            store,
            graph_name=GRAPH_NAME,
            token=token,
            verify_constraint=verify,
            write_hierarchy=lambda executor, generation: (
                detector._replace_community_rows(
                    executor,
                    community_rows,
                    membership_rows,
                    parent_rows,
                    generation,
                )
            ),
            read_stats=lambda executor: [],
        )
        return layout, token

    return publish


@pytest.fixture
def memgraph_store():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Memgraph community publication tests",
        fail_on_missing=True,
    )
    store = get_memgraph_store()
    preflight_memgraph_mage(store)
    cleanup_memgraph_publication_graph(store)
    store.create_indexes()
    try:
        yield store
    finally:
        try:
            cleanup_memgraph_publication_graph(store)
        finally:
            store.driver.close()


@pytest.mark.integration
@pytest.mark.database
def test_concurrent_first_publication_keeps_one_generation(memgraph_store):
    assert_first_concurrent_publication(memgraph_store, CommunityDetector, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
def test_concurrent_replacement_keeps_one_generation(memgraph_store):
    assert_replacement_concurrent_publication(
        memgraph_store, CommunityDetector, GRAPH_NAME
    )


@pytest.mark.integration
@pytest.mark.database
def test_controlled_first_publication_keeps_one_exact_layout(memgraph_store):
    assert_controlled_publication_race(
        memgraph_store,
        GRAPH_NAME,
        publish_controlled_layout(memgraph_store),
        existing=False,
    )


@pytest.mark.integration
@pytest.mark.database
def test_controlled_replacement_keeps_one_exact_layout(memgraph_store):
    assert_controlled_publication_race(
        memgraph_store,
        GRAPH_NAME,
        publish_controlled_layout(memgraph_store),
        existing=True,
    )


@pytest.mark.integration
@pytest.mark.database
def test_failed_publication_rolls_back_hierarchy_and_token(
    memgraph_store, monkeypatch
):
    assert_publication_rollback(
        memgraph_store, CommunityDetector, GRAPH_NAME, monkeypatch
    )


@pytest.mark.integration
@pytest.mark.database
def test_missing_generation_constraint_fails_closed(memgraph_store):
    assert_missing_constraint_fails_closed(
        memgraph_store,
        CommunityDetector,
        GRAPH_NAME,
        "DROP CONSTRAINT ON (g:CommunityGeneration) ASSERT g.graph_name IS UNIQUE",
    )
