"""Unit tests for FalkorDB community detection."""

from __future__ import annotations

from recon_graphrag.communities.falkordb.detection import CommunityDetector


class FakeGraphStore:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append((query.strip(), parameters or {}))
        params = parameters or {}

        if "RETURN id(e) AS id" in query:
            return [
                {"id": "1"},
                {"id": "2"},
                {"id": "3"},
                {"id": "4"},
            ]

        if "RETURN DISTINCT type(r) AS t" in query:
            return [{"t": "KNOWS"}]

        if "RETURN id(source)" in query:
            return [
                {"source_id": "1", "target_id": "2", "rel_type": "KNOWS", "rel_props": {}},
                {"source_id": "2", "target_id": "3", "rel_type": "KNOWS", "rel_props": {}},
                {"source_id": "4", "target_id": "4", "rel_type": "KNOWS", "rel_props": {}},
            ]

        if "c.id AS community_id" in query:
            return [
                {
                    "community_id": "1",
                    "level": 0,
                    "entity_count": 3,
                    "child_community_count": 0,
                },
                {
                    "community_id": "4",
                    "level": 0,
                    "entity_count": 1,
                    "child_community_count": 0,
                },
            ]

        return []


def test_detect_communities_returns_single_level_communities():
    store = FakeGraphStore()
    detector = CommunityDetector(store, graph_name="test-graph")

    stats = detector.detect()
    assert len(stats) == 2
    assert stats[0]["level"] == 0
    assert stats[0]["child_community_count"] == 0

    # Verify both community creation and membership queries were issued.
    query_text = " ".join(q for q, _ in store.queries)
    assert "MERGE (c:`Community`" in query_text
    assert "MERGE (e)-[rel:IN_COMMUNITY]" in query_text
