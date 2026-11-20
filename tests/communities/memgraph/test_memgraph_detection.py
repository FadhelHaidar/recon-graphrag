"""Unit tests for Memgraph community detection."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

from recon_graphrag.communities.memgraph.detection import CommunityDetector
from recon_graphrag.graphdb.base import GraphStore


@contextmanager
def assert_raises(error_type: type[BaseException], message: str):
    """Assert that a block raises ``error_type`` with ``message``."""
    try:
        yield
    except error_type as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected {error_type.__name__}")


GENERATION_CONSTRAINT = {
    "constraint type": "unique",
    "label": ":CommunityGeneration",
    "properties": ["graph_name"],
}


class FakeGraphStore:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    @contextmanager
    def _transaction(self) -> Any:
        yield self

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append((query.strip(), parameters or {}))

        if "SHOW CONSTRAINT INFO" in query:
            return [dict(GENERATION_CONSTRAINT)]

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
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph", max_levels=3)

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
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph", max_levels=3)

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
    detector = CommunityDetector(cast(GraphStore, store),
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
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph", max_levels=3)

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

    detector = CommunityDetector(cast(GraphStore, NoRelationshipStore()), graph_name="test-graph")

    try:
        detector.detect()
    except RuntimeError as exc:
        assert "No valid entity-to-entity relationship types found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_random_seed_is_not_forwarded_to_mage():
    """MAGE does not expose a random seed; keep it for API symmetry only."""
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph", random_seed=12345)

    detector.detect()

    leiden_query = next(q for q, _ in store.queries if "leiden_community_detection" in q)
    assert "seed" not in leiden_query.lower()


def test_detection_queries_are_scoped_by_graph_name():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="scoped-graph")

    detector.detect()

    scoped_calls = [
        params for query, params in store.queries if "$graph_name" in query
    ]
    assert scoped_calls
    assert all(params["graph_name"] == "scoped-graph" for params in scoped_calls)


class _RecordingTransaction:
    def __init__(self):
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append(query.strip())
        return []


class TransactionalStore(FakeGraphStore):
    """Fake store exposing the store's real `_transaction()` seam."""

    def __init__(self, fail_on: str | None = None):
        super().__init__()
        self.transactions: list[_RecordingTransaction] = []
        self.fail_on = fail_on

    def execute_query(self, query: str, parameters: dict | None = None):
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("mage leiden unavailable")
        return super().execute_query(query, parameters)

    @contextmanager
    def _transaction(self):
        transaction = _RecordingTransaction()
        self.transactions.append(transaction)
        yield transaction


def test_hierarchy_writes_run_in_one_transaction():
    store = TransactionalStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    detector.detect()

    assert len(store.transactions) == 1
    written = "\n".join(store.transactions[0].queries)
    assert "DETACH DELETE c" in written
    assert "MERGE (c:`Community`" in written
    assert "MERGE (e)-[rel:IN_COMMUNITY]" in written
    assert "MERGE (child)-[rel:PARENT_COMMUNITY]" in written
    assert not any("DETACH DELETE" in query for query, _ in store.queries)


def test_leiden_failure_leaves_existing_communities_untouched():
    store = TransactionalStore(fail_on="leiden_community_detection.get")
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    try:
        detector.detect()
    except RuntimeError as exc:
        assert "mage leiden unavailable" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert store.transactions == []
    assert not any("DETACH DELETE" in query for query, _ in store.queries)


def test_detection_returns_stats_and_the_invocation_generation_token():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    stats, token = detector._detect_with_generation()

    assert stats[0]["community_id"] == "10"
    community_params = [
        parameters
        for query, parameters in store.queries
        if "MERGE (c:`Community`" in query
    ]
    assert len(community_params) == 1
    assert community_params[0]["generation"] == token
    assert "c.generation = $generation" in next(
        query for query, _ in store.queries if "MERGE (c:`Community`" in query
    )


def test_two_detections_mint_distinct_generation_tokens():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    _, first = detector._detect_with_generation()
    _, second = detector._detect_with_generation()

    assert first != second


def test_public_detect_returns_only_community_stats():
    store = FakeGraphStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    result = detector.detect()

    assert result == [
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


def test_publication_publishes_the_state_token_before_replacing_communities():
    store = TransactionalStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    detector.detect()

    queries = store.transactions[0].queries
    assert "MERGE (g:CommunityGeneration" in queries[0]
    assert "DETACH DELETE c" in queries[1]
    assert any("c.generation = $generation" in query for query in queries)


def test_publication_cleanup_keeps_the_generation_anchor():
    store = TransactionalStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    detector.detect()

    delete_queries = [
        query for query in store.transactions[0].queries if "DETACH DELETE" in query
    ]
    assert len(delete_queries) == 1
    assert "CommunityGeneration" not in delete_queries[0]


class MissingGenerationConstraintStore(FakeGraphStore):
    def execute_query(self, query: str, parameters: dict | None = None):
        if "SHOW CONSTRAINT INFO" in query:
            self.queries.append((query.strip(), parameters or {}))
            return []
        return super().execute_query(query, parameters)


def test_missing_generation_constraint_fails_closed_before_publication():
    store = MissingGenerationConstraintStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    with assert_raises(RuntimeError, "is missing"):
        detector.detect()

    assert not any(
        "MERGE (g:CommunityGeneration" in query for query, _ in store.queries
    )


def test_differently_defined_generation_constraint_fails_closed():
    class IncompatibleConstraintStore(FakeGraphStore):
        def execute_query(self, query: str, parameters: dict | None = None):
            if "SHOW CONSTRAINT INFO" in query:
                self.queries.append((query.strip(), parameters or {}))
                return [dict(GENERATION_CONSTRAINT, properties=["token"])]
            return super().execute_query(query, parameters)

    store = IncompatibleConstraintStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    with assert_raises(RuntimeError, "is missing"):
        detector.detect()

    assert not any(
        "MERGE (g:CommunityGeneration" in query for query, _ in store.queries
    )


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

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append((query.strip(), parameters or {}))
        return self.delegate.execute_query(query, parameters)


def test_publication_retries_conflict_without_rerunning_detection():
    store = ConflictStore()
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    stats = detector.detect()

    assert stats[0]["community_id"] == "10"
    assert store.transaction_attempts == 2
    assert (
        sum("leiden_community_detection.get" in query for query, _ in store.queries)
        == 1
    )
    state_tokens = [
        parameters["token"]
        for queries in store.transaction_queries
        for query, parameters in queries
        if "MERGE (g:CommunityGeneration" in query
    ]
    assert len(state_tokens) == 2
    assert state_tokens[0] == state_tokens[1]


def test_stats_are_read_inside_the_publication_transaction():
    class StatsTransaction:
        def __init__(self, delegate):
            self.delegate = delegate
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute_query(self, query: str, parameters: dict | None = None):
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
    detector = CommunityDetector(cast(GraphStore, store), graph_name="test-graph")

    stats = detector.detect()

    assert stats[0]["community_id"] == "from-transaction"
    assert any("c.id AS community_id" in query for query, _ in store.transactions[0].queries)
