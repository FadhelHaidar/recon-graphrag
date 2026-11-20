"""Deterministic tests for community generation publication."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from recon_graphrag.graphdb._community_publication import (
    COMMUNITY_GENERATION_CONSTRAINT_NAME,
    replace_community_hierarchy,
    verify_memgraph_community_generation_constraint,
    verify_neo4j_community_generation_constraint,
)

NEO4J_CONSTRAINT = {
    "name": COMMUNITY_GENERATION_CONSTRAINT_NAME,
    "type": "UNIQUENESS",
    "labelsOrTypes": ["CommunityGeneration"],
    "properties": ["graph_name"],
}
MEMGRAPH_CONSTRAINT = {
    "constraint type": "unique",
    "label": ":CommunityGeneration",
    "properties": ["graph_name"],
}


class TransientError(Exception):
    """Stand-in for a Memgraph retryable transaction conflict."""


class RecordingTransaction:
    def __init__(self, fail_on_commit: bool):
        self.fail_on_commit = fail_on_commit
        self.queries: list[tuple[str, dict]] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        self.queries.append((query.strip(), parameters or {}))
        return []


class PublicationStore:
    def __init__(
        self,
        constraints: list[dict] | None = None,
        *,
        commit_conflicts: int = 0,
        write_error: Exception | None = None,
        constraint_error: Exception | None = None,
    ):
        self.constraints = (
            constraints if constraints is not None else [NEO4J_CONSTRAINT]
        )
        self.commit_conflicts = commit_conflicts
        self.write_error = write_error
        self.constraint_error = constraint_error
        self.transactions: list[RecordingTransaction] = []

    def execute_query(self, query: str, parameters: dict | None = None):
        if "SHOW CONSTRAINTS" in query or "SHOW CONSTRAINT INFO" in query:
            if self.constraint_error is not None:
                raise self.constraint_error
            return self.constraints
        return []

    @contextmanager
    def _transaction(self):
        transaction = RecordingTransaction(fail_on_commit=self.commit_conflicts > 0)
        self.transactions.append(transaction)
        yield transaction
        if transaction.fail_on_commit:
            self.commit_conflicts -= 1
            raise TransientError("conflicting transactions")


def _publish(store: PublicationStore, token: str = "token-1", writer=None):
    return replace_community_hierarchy(
        store,
        graph_name="graph-a",
        token=token,
        verify_constraint=verify_neo4j_community_generation_constraint,
        write_hierarchy=writer or (lambda executor, generation: None),
        read_stats=lambda executor: [{"community_id": "c", "level": 0}],
    )


def test_neo4j_exact_constraint_is_accepted():
    store = PublicationStore([NEO4J_CONSTRAINT])

    stats = _publish(store)

    assert stats == [{"community_id": "c", "level": 0}]


def test_neo4j_missing_constraint_fails_closed_before_publication():
    store = PublicationStore([])

    with pytest.raises(RuntimeError, match="is missing"):
        _publish(store)

    assert store.transactions == []


def test_neo4j_differently_defined_named_constraint_fails_closed():
    store = PublicationStore(
        [dict(NEO4J_CONSTRAINT, properties=["token"])]
    )

    with pytest.raises(RuntimeError, match="incompatible definition"):
        _publish(store)

    assert store.transactions == []


def test_neo4j_non_uniqueness_constraint_fails_closed():
    store = PublicationStore([dict(NEO4J_CONSTRAINT, type="NODE_KEY")])

    with pytest.raises(RuntimeError, match="incompatible definition"):
        _publish(store)

    assert store.transactions == []


def test_neo4j_constraint_inspection_error_fails_closed():
    store = PublicationStore([], constraint_error=RuntimeError("permission denied"))

    with pytest.raises(RuntimeError, match="Cannot verify"):
        _publish(store)

    assert store.transactions == []


def test_memgraph_exact_constraint_is_accepted():
    store = PublicationStore([MEMGRAPH_CONSTRAINT])

    verify_memgraph_community_generation_constraint(store)


def test_memgraph_missing_constraint_fails_closed():
    store = PublicationStore([])

    with pytest.raises(RuntimeError, match="is missing"):
        verify_memgraph_community_generation_constraint(store)


def test_memgraph_non_unique_constraint_fails_closed():
    store = PublicationStore(
        [dict(MEMGRAPH_CONSTRAINT, **{"constraint type": "existence"})]
    )

    with pytest.raises(RuntimeError, match="incompatible type"):
        verify_memgraph_community_generation_constraint(store)


def test_state_token_is_the_first_transaction_statement():
    store = PublicationStore()

    def writer(executor, generation):
        executor.execute_query("DETACH DELETE old communities")

    _publish(store, token="token-9", writer=writer)

    queries = store.transactions[0].queries
    assert "MERGE (g:CommunityGeneration" in queries[0][0]
    assert queries[0][1] == {"graph_name": "graph-a", "token": "token-9"}
    assert "DETACH DELETE" in queries[1][0]


def test_publication_retries_conflict_with_the_same_token():
    store = PublicationStore(commit_conflicts=1)

    stats = _publish(store, token="stable-token")

    assert stats == [{"community_id": "c", "level": 0}]
    assert len(store.transactions) == 2
    tokens = [
        parameters["token"]
        for transaction in store.transactions
        for query, parameters in transaction.queries
        if "MERGE (g:CommunityGeneration" in query
    ]
    assert tokens == ["stable-token", "stable-token"]


def test_publication_stops_after_bounded_retries():
    store = PublicationStore(commit_conflicts=10)

    with pytest.raises(TransientError, match="conflicting transactions"):
        _publish(store)

    assert len(store.transactions) == 3


def test_publication_does_not_retry_non_retryable_failure():
    store = PublicationStore()

    def writer(executor, generation):
        raise ValueError("invalid hierarchy")

    with pytest.raises(ValueError, match="invalid hierarchy"):
        _publish(store, writer=writer)

    assert len(store.transactions) == 1


def test_stats_are_read_inside_the_publication_transaction():
    store = PublicationStore()
    observed: list[Any] = []

    def read_stats(executor):
        observed.append(executor)
        return [{"community_id": "c", "level": 0}]

    replace_community_hierarchy(
        store,
        graph_name="graph-a",
        token="token-1",
        verify_constraint=verify_neo4j_community_generation_constraint,
        write_hierarchy=lambda executor, generation: None,
        read_stats=read_stats,
    )

    assert observed == [store.transactions[0]]
