"""Tests for scoped community detection queries."""

import pytest

from recon_graphrag.communities.neo4j.detection import CommunityDetector


class FakeGraphStore:
    def __init__(self):
        self.queries = []
        self.params = []

    def execute_query(self, query, parameters=None):
        self.queries.append(query.strip())
        self.params.append(parameters or {})
        if "RETURN count(e) AS cnt" in query:
            return [{"cnt": 2}]
        if "RETURN DISTINCT type(r) AS t" in query:
            return [{"t": "ACTED_IN"}, {"t": "HAS_PROBLEM"}]
        if "gds.graph.project" in query:
            return [{"graphName": "movie-graph", "nodeCount": 2, "relationshipCount": 1}]
        return []


def test_projection_is_scoped_by_graph_name():
    store = FakeGraphStore()
    detector = CommunityDetector(
        store,
        relationship_types=["ACTED_IN", "DIRECTED"],
        graph_name="movie-graph",
    )

    detector._project_graph()

    count_query = store.queries[0]
    type_query = store.queries[1]
    projection_query = store.queries[2]

    assert "MATCH (e:`__Entity__` {graph_name: $graph_name})" in count_query
    assert "WHERE r.graph_name = $graph_name" in type_query
    assert "MATCH (source:`__Entity__` {graph_name: $graph_name})" in projection_query
    assert "target:`__Entity__` {graph_name: $graph_name}" in projection_query
    assert "undirectedRelationshipTypes: $relationship_types" in projection_query
    assert store.params[2] == {
        "graph_name": "movie-graph",
        "relationship_types": ["ACTED_IN"],
    }
