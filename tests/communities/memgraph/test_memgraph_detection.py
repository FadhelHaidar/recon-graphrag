"""Unit tests for Memgraph community detection."""

from __future__ import annotations

from recon_graphrag.communities.memgraph.detection import CommunityDetector


class FakeGraphStore:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append((query.strip(), parameters or {}))
        params = parameters or {}

        if "leiden_community_detection.get" in query:
            return [
                {
                    "entity_id": 1,
                    "community_id": 10,
                    "communities": [10],
                },
                {
                    "entity_id": 2,
                    "community_id": 10,
                    "communities": [10],
                },
                {
                    "entity_id": 3,
                    "community_id": 87,
                    "communities": [10, 52, 87],
                },
            ]

        if "RETURN DISTINCT type(r) AS t" in query:
            return [{"t": "KNOWS"}]

        if "RETURN count(e) AS cnt" in query:
            return [{"cnt": 3}]

        if "RETURN count(r) AS cnt" in query:
            return [{"cnt": 4}]

        if "c.id AS community_id" in query:
            return [
                {
                    "community_id": "10",
                    "level": 0,
                    "entity_count": 3,
                    "child_community_count": 1,
                },
                {
                    "community_id": "87",
                    "level": 2,
                    "entity_count": 1,
                    "child_community_count": 0,
                },
            ]

        return []


def test_detect_communities_returns_hierarchical_communities():
    store = FakeGraphStore()
    detector = CommunityDetector(store, graph_name="test-graph", max_levels=3)

    stats = detector.detect()
    assert len(stats) == 2

    query_text = " ".join(q for q, _ in store.queries)
    assert "leiden_community_detection.get_subgraph" in query_text
    assert "$weight_property" in query_text
    assert "MERGE (c:`Community`" in query_text
    assert "c.uid = row.uid" in query_text
    assert "MERGE (e)-[rel:IN_COMMUNITY]" in query_text
    assert "MERGE (child)-[rel:PARENT_COMMUNITY]" in query_text


def test_community_rows_include_uid():
    store = FakeGraphStore()
    detector = CommunityDetector(store, graph_name="test-graph", max_levels=3)

    detector.detect()

    community_create_query = next(
        q for q, _ in store.queries if "MERGE (c:`Community`" in q
    )
    assert "c.uid = row.uid" in community_create_query

    # Verify the rows passed to the query contain uid values keyed by graph/level/id.
    params = next(p for q, p in store.queries if "MERGE (c:`Community`" in q)
    for row in params["rows"]:
        expected_uid = f"{row['graph_name']}:{row['level']}:{row['community_id']}"
        assert row["uid"] == expected_uid


def test_leiden_query_uses_get_subgraph_with_weight_parameter():
    store = FakeGraphStore()
    detector = CommunityDetector(
        store,
        graph_name="test-graph",
        relationship_weight_property="strength",
    )

    detector.detect()

    leiden_query = next(q for q, _ in store.queries if "leiden_community_detection" in q)
    params = next(p for q, p in store.queries if "leiden_community_detection" in q)
    assert "leiden_community_detection.get_subgraph" in leiden_query
    assert "$weight_property" in leiden_query
    assert params["weight_property"] == "strength"
    # Tolerance is passed as the 6th positional argument, which MAGE maps to
    # resolution_parameter. Numeric parameters should still be passed as
    # parameters, not embedded literals.
    assert "$gamma" in leiden_query
    assert "$theta" in leiden_query
    assert "$tolerance" in leiden_query


def test_community_rows_are_deterministically_sorted():
    store = FakeGraphStore()
    detector = CommunityDetector(store, graph_name="test-graph", max_levels=3)

    detector.detect()

    params = next(p for q, p in store.queries if "MERGE (c:`Community`" in q)
    rows = params["rows"]
    sort_keys = [(row["level"], row["community_id"]) for row in rows]
    assert sort_keys == sorted(sort_keys)


def test_detect_communities_requires_entity_relationships():
    class NoRelationshipStore(FakeGraphStore):
        def execute_query(self, query: str, parameters: dict | None = None):
            if "RETURN count(e) AS cnt" in query:
                return [{"cnt": 3}]
            if "RETURN DISTINCT type(r) AS t" in query:
                return []
            return super().execute_query(query, parameters)

    detector = CommunityDetector(NoRelationshipStore(), graph_name="test-graph")

    try:
        detector.detect()
    except RuntimeError as exc:
        assert "No valid entity-to-entity relationship types found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_random_seed_is_not_forwarded_to_mage():
    """MAGE does not expose a random seed; keep it for API symmetry only."""
    store = FakeGraphStore()
    detector = CommunityDetector(store, graph_name="test-graph", random_seed=12345)

    detector.detect()

    leiden_query = next(q for q, _ in store.queries if "leiden_community_detection" in q)
    assert "seed" not in leiden_query.lower()
