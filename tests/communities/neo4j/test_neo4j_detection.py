"""Unit tests for Neo4j community detection."""

from contextlib import contextmanager
from typing import Any, cast

from recon_graphrag.communities.neo4j.detection import CommunityDetector
from recon_graphrag.graphdb.base import GraphStore


class FakeGraphStore:
    def __init__(self):
        self.queries = []
        self.params = []

    @contextmanager
    def _transaction(self) -> Any:
        yield self

    def execute_query(self, query, parameters=None):
        self.queries.append(query.strip())
        self.params.append(parameters or {})
        if "RETURN count(e) AS cnt" in query:
            return [{"cnt": 2}]
        if "RETURN DISTINCT type(r) AS t" in query:
            return [{"t": "ACTED_IN"}, {"t": "HAS_PROBLEM"}]
        if "gds.graph.project" in query:
            return [{"graphName": "movie-graph", "nodeCount": 2, "relationshipCount": 1}]
        if "gds.leiden.stream" in query:
            return [
                {
                    "entity_element_id": "4:a",
                    "communityId": 20,
                    "intermediateCommunityIds": [10, 20],
                },
                {
                    "entity_element_id": "4:b",
                    "communityId": 20,
                    "intermediateCommunityIds": [10, 20],
                },
            ]
        if "c.id AS community_id" in query:
            return [
                {
                    "community_id": "10",
                    "level": 0,
                    "entity_count": 2,
                    "child_community_count": 0,
                }
            ]
        return []


def test_projection_is_scoped_by_graph_name():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store),
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


def test_weighted_projection_includes_relationship_weight_property():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store),
        relationship_types=["ACTED_IN", "DIRECTED"],
        graph_name="movie-graph",
        relationship_weight_property="weight",
    )

    detector._project_graph()

    projection_query = store.queries[2]
    params = store.params[2]

    assert "relationshipProperties" in projection_query
    assert "apoc.map.fromLists" in projection_query
    assert "$relationship_weight_property" in projection_query
    assert "r.`weight`" in projection_query
    assert params == {
        "graph_name": "movie-graph",
        "relationship_types": ["ACTED_IN"],
        "relationship_weight_property": "weight",
    }


def test_unweighted_projection_omits_relationship_properties():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store),
        relationship_types=["ACTED_IN", "DIRECTED"],
        graph_name="movie-graph",
    )

    detector._project_graph()

    projection_query = store.queries[2]
    params = store.params[2]

    assert "relationshipProperties" not in projection_query
    assert "apoc.map.fromLists" not in projection_query
    assert params == {
        "graph_name": "movie-graph",
        "relationship_types": ["ACTED_IN"],
    }


def test_detect_communities_writes_hierarchy_and_returns_stats():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store),
        relationship_types=["ACTED_IN"],
        graph_name="movie-graph",
        max_levels=2,
    )

    stats = detector.detect()

    assert stats[0]["community_id"] == "10"
    query_text = "\n".join(store.queries)
    assert "gds.leiden.stream" in query_text
    assert "MERGE (e)-[rel:IN_COMMUNITY]" in query_text
    assert "MERGE (child)-[rel:PARENT_COMMUNITY]" in query_text


def test_detect_communities_requires_entity_relationships():
    class NoRelationshipStore(FakeGraphStore):
        def execute_query(self, query, parameters=None):
            if "RETURN DISTINCT type(r) AS t" in query:
                return []
            return super().execute_query(query, parameters)

    detector = CommunityDetector(cast(GraphStore, NoRelationshipStore()), graph_name="movie-graph")

    try:
        detector.detect()
    except RuntimeError as exc:
        assert "No valid entity-to-entity relationship types found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_random_seed_is_forwarded_to_gds():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph", random_seed=12345)

    detector._run_leiden()

    query = store.queries[-1]
    params = store.params[-1]
    assert "randomSeed: $random_seed" in query
    assert params["random_seed"] == 12345


class _RecordingTransaction:
    def __init__(self, sink):
        self.sink = sink
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute_query(self, query, parameters=None):
        self.queries.append(query.strip())
        return []


class TransactionalStore(FakeGraphStore):
    """Fake store exposing the store's real `_transaction()` seam."""

    def __init__(self, fail_on=None):
        super().__init__()
        self.transactions = []
        self.fail_on = fail_on

    def execute_query(self, query, parameters=None):
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("leiden unavailable")
        return super().execute_query(query, parameters)

    @contextmanager
    def _transaction(self):
        transaction = _RecordingTransaction(self.transactions)
        self.transactions.append(transaction)
        yield transaction


def test_hierarchy_writes_run_in_one_transaction():
    store = TransactionalStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    detector.detect()

    assert len(store.transactions) == 1
    written = "\n".join(store.transactions[0].queries)
    assert "DETACH DELETE c" in written
    assert "MERGE (c:`Community`" in written
    assert "MERGE (e)-[rel:IN_COMMUNITY]" in written
    assert "MERGE (child)-[rel:PARENT_COMMUNITY]" in written
    # Nothing destructive escapes the transaction.
    assert not any("DETACH DELETE" in query for query in store.queries)


def test_leiden_failure_leaves_existing_communities_untouched():
    store = TransactionalStore(fail_on="gds.leiden.stream")
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    try:
        detector.detect()
    except RuntimeError as exc:
        assert "leiden unavailable" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert store.transactions == []
    assert not any("DETACH DELETE" in query for query in store.queries)


def test_missing_entities_leaves_existing_communities_untouched():
    class EmptyStore(TransactionalStore):
        def execute_query(self, query, parameters=None):
            if "RETURN count(e) AS cnt" in query:
                self.queries.append(query.strip())
                return [{"cnt": 0}]
            return super().execute_query(query, parameters)

    store = EmptyStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    try:
        detector.detect()
    except RuntimeError as exc:
        assert "No __Entity__ nodes found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert store.transactions == []
    assert not any("DETACH DELETE" in query for query in store.queries)
