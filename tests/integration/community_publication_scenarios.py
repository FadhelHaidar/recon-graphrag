"""Live-database scenarios for community generation publication."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

PUBLICATION_RELATIONSHIP = "PUBLICATION_REL"

# Two controlled layouts assign the same four seeded entities differently. A
# publication race must leave one complete layout, never a mix of both.
CONTROLLED_LAYOUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "single": {"7001": ("a", "b", "c", "d")},
    "split": {
        "7002": ("a", "b"),
        "7003": ("c", "d"),
    },
}


def controlled_layout_rows(graph_name: str, layout: str) -> dict[str, str]:
    """Return entity ID to community ID for one controlled layout."""
    return {
        f"{graph_name}:{entity_name}": community_id
        for community_id, entity_names in CONTROLLED_LAYOUTS[layout].items()
        for entity_name in entity_names
    }


@contextmanager
def assert_raises(error_type: type[BaseException], message: str):
    """Assert that a block raises ``error_type`` with ``message``."""
    try:
        yield
    except error_type as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected {error_type.__name__}")


def seed_publication_graph(store: Any, graph_name: str) -> None:
    """Create one connected entity graph for publication tests."""
    store.execute_query(
        """
        UNWIND $rows AS row
        CREATE (e:__Entity__:PublicationTestEntity)
        SET e.id = row.id,
            e.graph_name = $graph_name,
            e.name = row.name,
            e.description = ''
        """,
        {
            "graph_name": graph_name,
            "rows": [
                {"id": f"{graph_name}:a", "name": "A"},
                {"id": f"{graph_name}:b", "name": "B"},
                {"id": f"{graph_name}:c", "name": "C"},
                {"id": f"{graph_name}:d", "name": "D"},
            ],
        },
    )
    store.execute_query(
        """
        UNWIND $rows AS row
        MATCH (source:__Entity__ {graph_name: $graph_name, id: row.source})
        MATCH (target:__Entity__ {graph_name: $graph_name, id: row.target})
        CREATE (source)-[r:PUBLICATION_REL]->(target)
        SET r.graph_name = $graph_name
        """,
        {
            "graph_name": graph_name,
            "rows": [
                {"source": f"{graph_name}:a", "target": f"{graph_name}:b"},
                {"source": f"{graph_name}:c", "target": f"{graph_name}:d"},
                {"source": f"{graph_name}:b", "target": f"{graph_name}:c"},
            ],
        },
    )


def run_concurrently(operations: list[Callable[[], Any]]) -> tuple[list, list]:
    """Start every operation at one barrier and return results and errors."""
    barrier = threading.Barrier(len(operations))
    results: list = []
    errors: list = []
    lock = threading.Lock()

    def run(operation: Callable[[], Any]) -> None:
        try:
            barrier.wait(timeout=60)
            value = operation()
        except BaseException as error:  # noqa: BLE001 - reported to the caller
            with lock:
                errors.append(error)
        else:
            with lock:
                results.append(value)

    threads = [threading.Thread(target=run, args=(operation,)) for operation in operations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads)
    return results, errors


def assert_publication_consistent(store: Any, graph_name: str) -> None:
    """Assert one generation owns every community and every membership."""
    rows = store.execute_query(
        """
        MATCH (c:Community {graph_name: $graph_name})
        RETURN count(c) AS communities,
               collect(DISTINCT c.generation) AS generations
        """,
        {"graph_name": graph_name},
    )
    assert rows and rows[0]["communities"] > 0
    generations = rows[0]["generations"]
    assert len(generations) == 1
    generation = generations[0]
    assert generation, "every published community must carry a generation"

    anchors = store.execute_query(
        """
        MATCH (g:CommunityGeneration {graph_name: $graph_name})
        RETURN count(g) AS anchors, collect(g.token) AS tokens
        """,
        {"graph_name": graph_name},
    )
    assert anchors[0]["anchors"] == 1
    assert anchors[0]["tokens"] == [generation]

    mixed = store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name})-[:IN_COMMUNITY]->
              (c:Community {graph_name: $graph_name})
        WITH e, c.level AS level, count(DISTINCT c) AS memberships
        WHERE memberships > 1
        RETURN count(*) AS mixed
        """,
        {"graph_name": graph_name},
    )
    assert mixed[0]["mixed"] == 0


def publication_snapshot(store: Any, graph_name: str) -> dict:
    """Return the generation-owned state that a failed publication must preserve."""
    anchors = store.execute_query(
        """
        MATCH (g:CommunityGeneration {graph_name: $graph_name})
        RETURN g.token AS token
        """,
        {"graph_name": graph_name},
    )
    communities = store.execute_query(
        """
        MATCH (c:Community {graph_name: $graph_name})
        RETURN c.level AS level, c.id AS id, c.generation AS generation
        ORDER BY level, id
        """,
        {"graph_name": graph_name},
    )
    memberships = store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name})-[:IN_COMMUNITY]->
              (c:Community {graph_name: $graph_name})
        RETURN e.id AS entity_id, c.level AS level, c.id AS community_id
        ORDER BY entity_id, level, community_id
        """,
        {"graph_name": graph_name},
    )
    return {
        "token": anchors[0]["token"],
        "communities": communities,
        "memberships": memberships,
    }


def assert_publication_rollback(
    store: Any,
    detector_class: Any,
    graph_name: str,
    monkeypatch: Any,
) -> None:
    """Inject a failure after deletion and assert the prior hierarchy survives."""
    seed_publication_graph(store, graph_name)
    store.detect_communities(
        graph_name=graph_name,
        relationship_types=[PUBLICATION_RELATIONSHIP],
        random_seed=42,
    )
    before = publication_snapshot(store, graph_name)
    detector = detector_class(
        store,
        relationship_types=[PUBLICATION_RELATIONSHIP],
        graph_name=graph_name,
        random_seed=42,
    )

    def fail_after_delete(executor, *rows):
        detector._cleanup_communities(executor)
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(detector, "_replace_community_rows", fail_after_delete)

    with assert_raises(RuntimeError, "injected publication failure"):
        detector.detect()

    assert publication_snapshot(store, graph_name) == before


def assert_missing_constraint_fails_closed(
    store: Any,
    detector_class: Any,
    graph_name: str,
    drop_constraint_query: str,
) -> None:
    """Drop the anchor constraint and assert detection refuses to publish."""
    seed_publication_graph(store, graph_name)
    store.execute_query(drop_constraint_query)
    try:
        with assert_raises(RuntimeError, "is missing"):
            detector_class(
                store,
                relationship_types=[PUBLICATION_RELATIONSHIP],
                graph_name=graph_name,
                random_seed=42,
            ).detect()

        communities = store.execute_query(
            "MATCH (c:Community {graph_name: $graph_name}) "
            "RETURN count(c) AS count",
            {"graph_name": graph_name},
        )
        anchors = store.execute_query(
            "MATCH (g:CommunityGeneration {graph_name: $graph_name}) "
            "RETURN count(g) AS count",
            {"graph_name": graph_name},
        )
        assert communities[0]["count"] == 0
        assert anchors[0]["count"] == 0
    finally:
        store.create_indexes()


def assert_first_concurrent_publication(
    store: Any, detector_class: Any, graph_name: str
) -> None:
    seed_publication_graph(store, graph_name)
    operations = [
        lambda: detector_class(
            store,
            relationship_types=[PUBLICATION_RELATIONSHIP],
            graph_name=graph_name,
            random_seed=42,
        ).detect()
        for _ in range(2)
    ]

    results, errors = run_concurrently(operations)

    assert not errors, errors
    assert len(results) == 2
    assert_publication_consistent(store, graph_name)


def assert_replacement_concurrent_publication(
    store: Any, detector_class: Any, graph_name: str
) -> None:
    seed_publication_graph(store, graph_name)
    store.detect_communities(
        graph_name=graph_name,
        relationship_types=[PUBLICATION_RELATIONSHIP],
        random_seed=42,
    )
    operations = [
        lambda: detector_class(
            store,
            relationship_types=[PUBLICATION_RELATIONSHIP],
            graph_name=graph_name,
            random_seed=42,
        ).detect()
        for _ in range(2)
    ]

    results, errors = run_concurrently(operations)

    assert not errors, errors
    assert len(results) == 2
    assert_publication_consistent(store, graph_name)


def assert_controlled_publication_race(
    store: Any,
    graph_name: str,
    publish_layout: Callable[[str, Any], tuple[str, str]],
    *,
    existing: bool,
) -> None:
    """Race two distinct layouts through the real publication transaction.

    ``publish_layout(layout, barrier)`` writes one controlled layout with the
    production publishing path and returns ``(layout, token)``. The barrier
    starts both replacement transactions together. The final state must equal
    one complete layout and carry that layout's token.
    """
    seed_publication_graph(store, graph_name)
    if existing:
        publish_layout("split", None)

    barrier = threading.Barrier(len(CONTROLLED_LAYOUTS))
    operations = [
        lambda layout=layout: publish_layout(layout, barrier)
        for layout in CONTROLLED_LAYOUTS
    ]
    results, errors = run_concurrently(operations)

    assert not errors, errors
    assert len(results) == len(CONTROLLED_LAYOUTS)
    tokens = dict(results)
    assert set(tokens) == set(CONTROLLED_LAYOUTS)

    assert_publication_consistent(store, graph_name)
    anchors = store.execute_query(
        """
        MATCH (g:CommunityGeneration {graph_name: $graph_name})
        RETURN g.token AS token
        """,
        {"graph_name": graph_name},
    )
    assert len(anchors) == 1
    final_token = anchors[0]["token"]
    assert final_token in tokens.values()
    winning_layout = next(
        layout for layout, token in tokens.items() if token == final_token
    )

    membership_rows = store.execute_query(
        """
        MATCH (e:__Entity__ {graph_name: $graph_name})-[:IN_COMMUNITY]->
              (c:Community {graph_name: $graph_name, level: 0})
        RETURN e.id AS entity_id, c.id AS community_id
        ORDER BY entity_id
        """,
        {"graph_name": graph_name},
    )
    actual = {
        row["entity_id"]: row["community_id"] for row in membership_rows
    }
    assert actual == controlled_layout_rows(graph_name, winning_layout)

    community_rows = store.execute_query(
        """
        MATCH (c:Community {graph_name: $graph_name, level: 0})
        RETURN count(c) AS count
        """,
        {"graph_name": graph_name},
    )
    assert community_rows[0]["count"] == len(CONTROLLED_LAYOUTS[winning_layout])
