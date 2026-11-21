"""Live publication integration tests for Neo4j.

These tests are opt-in and were not run in Stage 3. They require a disposable
Neo4j database with APOC and GDS.
"""

from __future__ import annotations

import pytest

from recon_graphrag.communities.neo4j.detection import CommunityDetector
from recon_graphrag.graphdb._community_publication import (
    new_community_generation_token,
    replace_community_hierarchy,
    verify_neo4j_community_generation_constraint,
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
from tests.integration.factories import get_neo4j_store
from tests.integration.support import require_integration_env

RUN_FLAG = "RUN_DATABASE_INTEGRATION_TESTS"
GRAPH_NAME = "neo4j-community-publication-integration"
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


def cleanup_neo4j_publication_graph(store) -> None:
    store.execute_query(
        "MATCH (n {graph_name: $graph_name}) DETACH DELETE n",
        {"graph_name": GRAPH_NAME},
    )


def publish_controlled_layout(store):
    """Return a publisher that writes one controlled layout.

    The publisher uses the production constraint check and replacement
    transaction. It only supplies deterministic layout rows instead of a GDS
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
            RETURN e.id AS id, elementId(e) AS element_id
            ORDER BY e.id
            """,
            {"graph_name": GRAPH_NAME},
        )
        element_ids = {row["id"]: row["element_id"] for row in rows}
        assignments = controlled_layout_rows(GRAPH_NAME, layout)
        leiden_results = [
            {
                "entity_element_id": element_ids[entity_id],
                "communityId": community_id,
                "intermediateCommunityIds": None,
            }
            for entity_id, community_id in assignments.items()
        ]
        membership_rows, parent_edges = detector._build_hierarchy_rows(
            leiden_results
        )
        token = new_community_generation_token()

        def verify(st):
            if barrier is not None:
                barrier.wait(timeout=30)
            verify_neo4j_community_generation_constraint(st)

        replace_community_hierarchy(
            store,
            graph_name=GRAPH_NAME,
            token=token,
            verify_constraint=verify,
            write_hierarchy=lambda executor, generation: (
                detector._replace_community_rows(
                    executor, membership_rows, parent_edges, generation
                )
            ),
            read_stats=lambda executor: [],
        )
        return layout, token

    return publish


@pytest.fixture
def neo4j_store():
    require_integration_env(
        RUN_FLAG,
        REQUIRED_ENV,
        "Neo4j community publication tests",
        fail_on_missing=True,
    )
    store = get_neo4j_store()
    preflight_neo4j_graph_data_science(store)
    cleanup_neo4j_publication_graph(store)
    store.create_indexes()
    try:
        yield store
    finally:
        try:
            cleanup_neo4j_publication_graph(store)
        finally:
            store.driver.close()


@pytest.mark.integration
@pytest.mark.database
def test_concurrent_first_publication_keeps_one_generation(neo4j_store):
    assert_first_concurrent_publication(neo4j_store, CommunityDetector, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
def test_concurrent_replacement_keeps_one_generation(neo4j_store):
    assert_replacement_concurrent_publication(neo4j_store, CommunityDetector, GRAPH_NAME)


@pytest.mark.integration
@pytest.mark.database
def test_controlled_first_publication_keeps_one_exact_layout(neo4j_store):
    assert_controlled_publication_race(
        neo4j_store,
        GRAPH_NAME,
        publish_controlled_layout(neo4j_store),
        existing=False,
    )


@pytest.mark.integration
@pytest.mark.database
def test_controlled_replacement_keeps_one_exact_layout(neo4j_store):
    assert_controlled_publication_race(
        neo4j_store,
        GRAPH_NAME,
        publish_controlled_layout(neo4j_store),
        existing=True,
    )


@pytest.mark.integration
@pytest.mark.database
def test_failed_publication_rolls_back_hierarchy_and_token(neo4j_store, monkeypatch):
    assert_publication_rollback(neo4j_store, CommunityDetector, GRAPH_NAME, monkeypatch)


@pytest.mark.integration
@pytest.mark.database
def test_missing_generation_constraint_fails_closed(neo4j_store):
    assert_missing_constraint_fails_closed(
        neo4j_store,
        CommunityDetector,
        GRAPH_NAME,
        "DROP CONSTRAINT community_generation_unique IF EXISTS",
    )
