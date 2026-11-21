"""Unit tests for Neo4j community detection."""

import threading
from contextlib import contextmanager
from typing import Any, cast

from recon_graphrag.communities.neo4j.detection import CommunityDetector
from recon_graphrag.graphdb.base import GraphStore

GENERATION_CONSTRAINT = {
    "name": "community_generation_unique",
    "type": "UNIQUENESS",
    "labelsOrTypes": ["CommunityGeneration"],
    "properties": ["graph_name"],
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
        if "SHOW CONSTRAINTS" in query:
            return [dict(GENERATION_CONSTRAINT)]
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

    detector._project_graph("run-projection")

    count_query = store.queries[0]
    type_query = store.queries[1]
    projection_query = store.queries[2]

    assert "MATCH (e:`__Entity__` {graph_name: $graph_name})" in count_query
    assert "WHERE r.graph_name = $graph_name" in type_query
    assert "MATCH (source:`__Entity__` {graph_name: $graph_name})" in projection_query
    assert "target:`__Entity__` {graph_name: $graph_name}" in projection_query
    assert "gds.graph.project(\n            $projection_name," in projection_query
    assert "undirectedRelationshipTypes: $relationship_types" in projection_query
    assert store.params[2] == {
        "graph_name": "movie-graph",
        "projection_name": "run-projection",
        "relationship_types": ["ACTED_IN"],
    }


def test_weighted_projection_includes_relationship_weight_property():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store),
        relationship_types=["ACTED_IN", "DIRECTED"],
        graph_name="movie-graph",
        relationship_weight_property="weight",
    )

    detector._project_graph("run-projection")

    projection_query = store.queries[2]
    params = store.params[2]

    assert "relationshipProperties" in projection_query
    assert "apoc.map.fromLists" in projection_query
    assert "$relationship_weight_property" in projection_query
    assert "r.`weight`" in projection_query
    assert params == {
        "graph_name": "movie-graph",
        "projection_name": "run-projection",
        "relationship_types": ["ACTED_IN"],
        "relationship_weight_property": "weight",
    }


def test_unweighted_projection_omits_relationship_properties():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store),
        relationship_types=["ACTED_IN", "DIRECTED"],
        graph_name="movie-graph",
    )

    detector._project_graph("run-projection")

    projection_query = store.queries[2]
    params = store.params[2]

    assert "relationshipProperties" not in projection_query
    assert "apoc.map.fromLists" not in projection_query
    assert params == {
        "graph_name": "movie-graph",
        "projection_name": "run-projection",
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
    assert "c.generation = $generation" in query_text


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

    detector._run_leiden("run-projection")

    query = store.queries[-1]
    params = store.params[-1]
    assert "CALL gds.leiden.stream($projection_name" in query
    assert "randomSeed: $random_seed" in query
    assert params["projection_name"] == "run-projection"
    assert "graph_name" not in params
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


class ProjectionRecordingStore(FakeGraphStore):
    """Fake store that records the GDS projections each run creates and drops."""

    def __init__(self, project_relationship_count: int = 1):
        super().__init__()
        self.created_projections: list[str] = []
        self.dropped_projections: list[str] = []
        self.leiden_projections: list[str] = []
        self.project_relationship_count = project_relationship_count
        self.project_barrier: threading.Barrier | None = None
        self.project_error: Exception | None = None
        self.leiden_error: Exception | None = None
        self.drop_error: Exception | None = None
        self._lock = threading.Lock()

    def execute_query(self, query: str, parameters: dict[str, Any] | None = None):
        params = parameters or {}
        if "gds.graph.project" in query:
            with self._lock:
                self.created_projections.append(params["projection_name"])
            if self.project_barrier is not None:
                self.project_barrier.wait(timeout=10)
            if self.project_error is not None:
                raise self.project_error
            return [
                {
                    "graphName": params["projection_name"],
                    "nodeCount": 2,
                    "relationshipCount": self.project_relationship_count,
                }
            ]
        if "gds.graph.drop" in query:
            with self._lock:
                self.dropped_projections.append(params["projection_name"])
            if self.drop_error is not None:
                raise self.drop_error
            return []
        if "gds.leiden.stream" in query:
            with self._lock:
                self.leiden_projections.append(params["projection_name"])
            if self.leiden_error is not None:
                raise self.leiden_error
        return super().execute_query(query, parameters)


def test_each_detection_uses_a_unique_projection_and_drops_only_that_projection():
    store = ProjectionRecordingStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    detector.detect()
    detector.detect()

    assert len(store.created_projections) == 2
    assert len(set(store.created_projections)) == 2
    assert store.leiden_projections == store.created_projections
    assert store.dropped_projections == store.created_projections
    assert all(name.startswith("movie-graph-") for name in store.created_projections)
    assert "movie-graph" not in store.dropped_projections


def test_concurrent_detections_on_one_detector_use_independent_projections():
    store = ProjectionRecordingStore()
    store.project_barrier = threading.Barrier(2)
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")
    errors = []

    def run_detection():
        try:
            detector.detect()
        except Exception as exc:  # noqa: BLE001 - surfaced through assertion
            errors.append(exc)

    threads = [threading.Thread(target=run_detection) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(store.created_projections) == 2
    assert len(set(store.created_projections)) == 2
    assert sorted(store.leiden_projections) == sorted(store.created_projections)
    assert sorted(store.dropped_projections) == sorted(store.created_projections)


def test_failed_projection_creation_cleans_up_its_projection():
    store = ProjectionRecordingStore()
    store.project_error = RuntimeError("projection failed")
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    with assert_raises(RuntimeError, "projection failed"):
        detector.detect()

    assert len(store.created_projections) == 1
    assert store.dropped_projections == store.created_projections


def test_projection_validation_failure_cleans_up_its_projection():
    store = ProjectionRecordingStore(project_relationship_count=0)
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    with assert_raises(RuntimeError, "zero relationships"):
        detector.detect()

    assert len(store.created_projections) == 1
    assert store.dropped_projections == store.created_projections


def test_cleanup_failure_does_not_replace_primary_failure():
    cases = [
        (RuntimeError("projection failed"), None, "projection failed"),
        (None, RuntimeError("leiden failed"), "leiden failed"),
    ]
    for project_error, leiden_error, expected_error in cases:
        store = ProjectionRecordingStore()
        store.project_error = project_error
        store.leiden_error = leiden_error
        store.drop_error = RuntimeError("cleanup failed")
        detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

        with assert_raises(RuntimeError, expected_error):
            detector.detect()

        assert len(store.dropped_projections) == 1


def test_successful_detection_surfaces_projection_cleanup_failure():
    store = ProjectionRecordingStore()
    store.drop_error = RuntimeError("cleanup failed")
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    with assert_raises(RuntimeError, "cleanup failed"):
        detector.detect()


def test_detection_returns_stats_and_the_invocation_generation_token():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    stats, token = detector._detect_with_generation()

    assert stats[0]["community_id"] == "10"
    membership_params = [
        parameters
        for query, parameters in zip(store.queries, store.params, strict=True)
        if "c.generation = $generation" in query
    ]
    assert len(membership_params) == 1
    assert membership_params[0]["generation"] == token


def test_two_detections_mint_distinct_generation_tokens():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    _, first = detector._detect_with_generation()
    _, second = detector._detect_with_generation()

    assert first != second


def test_public_detect_returns_only_community_stats():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    result = detector.detect()

    assert result == [
        {
            "community_id": "10",
            "level": 0,
            "entity_count": 2,
            "child_community_count": 0,
        }
    ]


def test_publication_publishes_the_state_token_before_replacing_communities():
    store = TransactionalStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    detector.detect()

    queries = store.transactions[0].queries
    assert "MERGE (g:CommunityGeneration" in queries[0]
    assert "DETACH DELETE c" in queries[1]
    assert "c.generation = $generation" in "\n".join(queries)


def test_publication_cleanup_keeps_the_generation_anchor():
    store = TransactionalStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    detector.detect()

    delete_queries = [
        query for query in store.transactions[0].queries if "DETACH DELETE" in query
    ]
    assert len(delete_queries) == 1
    assert "CommunityGeneration" not in delete_queries[0]


class MissingGenerationConstraintStore(FakeGraphStore):
    def execute_query(self, query, parameters=None):
        if "SHOW CONSTRAINTS" in query:
            self.queries.append(query.strip())
            self.params.append(parameters or {})
            return []
        return super().execute_query(query, parameters)


def test_missing_generation_constraint_fails_closed_before_publication():
    store = MissingGenerationConstraintStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    with assert_raises(RuntimeError, "is missing"):
        detector.detect()

    assert not any("MERGE (g:CommunityGeneration" in query for query in store.queries)


def test_differently_defined_generation_constraint_fails_closed():
    class IncompatibleConstraintStore(FakeGraphStore):
        def execute_query(self, query, parameters=None):
            if "SHOW CONSTRAINTS" in query:
                self.queries.append(query.strip())
                self.params.append(parameters or {})
                return [dict(GENERATION_CONSTRAINT, properties=["token"])]
            return super().execute_query(query, parameters)

    store = IncompatibleConstraintStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    with assert_raises(RuntimeError, "incompatible definition"):
        detector.detect()

    assert not any("MERGE (g:CommunityGeneration" in query for query in store.queries)


class TransientError(Exception):
    """Stand-in for a retryable replacement conflict."""


class ConflictStore(FakeGraphStore):
    def __init__(self):
        super().__init__()
        self.conflicts_remaining = 1
        self.transaction_attempts = 0
        self.transaction_queries = []

    @contextmanager
    def _transaction(self):
        self.transaction_attempts += 1
        transaction = ConflictTransaction(self)
        self.transaction_queries.append(transaction.queries)
        yield transaction
        if self.conflicts_remaining:
            self.conflicts_remaining -= 1
            raise TransientError("conflicting transactions")


class ConflictTransaction:
    def __init__(self, delegate):
        self.delegate = delegate
        self.queries = []

    def execute_query(self, query, parameters=None):
        self.queries.append((query.strip(), parameters or {}))
        return self.delegate.execute_query(query, parameters)


def test_publication_retries_conflict_without_rerunning_detection():
    store = ConflictStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    stats = detector.detect()

    assert stats[0]["community_id"] == "10"
    assert store.transaction_attempts == 2
    assert sum("gds.leiden.stream" in query for query in store.queries) == 1
    state_tokens = [
        parameters["token"]
        for queries in store.transaction_queries
        for query, parameters in queries
        if "MERGE (g:CommunityGeneration" in query
    ]
    assert len(state_tokens) == 2
    assert state_tokens[0] == state_tokens[1]


def test_non_retryable_publication_failure_does_not_retry():
    class FailingWriterStore(TransactionalStore):
        @contextmanager
        def _transaction(self):
            transaction = _RecordingTransaction(self.transactions)
            original = transaction.execute_query

            def execute_query(query, parameters=None):
                if "MERGE (g:CommunityGeneration" in query:
                    raise ValueError("invalid publication")
                return original(query, parameters)

            transaction.execute_query = execute_query  # type: ignore[method-assign]
            self.transactions.append(transaction)
            yield transaction

    store = FailingWriterStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    with assert_raises(ValueError, "invalid publication"):
        detector.detect()

    assert len(store.transactions) == 1


def test_stats_are_read_inside_the_publication_transaction():
    class StatsTransaction:
        def __init__(self, delegate):
            self.delegate = delegate
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute_query(self, query, parameters=None):
            self.queries.append((query.strip(), parameters or {}))
            if "c.id AS community_id" in query:
                return [{"community_id": "from-transaction", "level": 0}]
            return self.delegate.execute_query(query, parameters)

    class StatsStore(FakeGraphStore):
        def __init__(self):
            super().__init__()
            self.transactions = []

        @contextmanager
        def _transaction(self):
            transaction = StatsTransaction(self)
            self.transactions.append(transaction)
            yield transaction

    store = StatsStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="movie-graph")

    stats = detector.detect()

    assert stats[0]["community_id"] == "from-transaction"
    assert any("c.id AS community_id" in query for query, _ in store.transactions[0].queries)
