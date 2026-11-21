"""Atomic merge-group transactions for the Memgraph entity resolver."""

from __future__ import annotations

from contextlib import contextmanager

import pytest  # pyright: ignore[reportMissingImports]

from recon_graphrag.graphdb.entity_resolution import _EntityRecord
from recon_graphrag.graphdb.memgraph.entity_resolution import _MemgraphEntityResolver


class DeadlockError(Exception):
    code = "Neo.TransientError.Transaction.DeadlockDetected"


class MergeTransaction:
    def __init__(self, store: MergeStore):
        self.store = store
        self.queries: list[tuple[str, dict]] = []
        self.committed = False
        self.rolled_back = False

    def run(self, query: str, parameters: dict | None = None):
        self.queries.append((query, parameters or {}))
        return self.store.run_query(query, parameters or {})

    def commit(self) -> None:
        if self.store.commit_error is not None:
            raise self.store.commit_error
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class MergeExecutor:
    def __init__(self, transaction: MergeTransaction):
        self._transaction = transaction

    def execute_query(self, query: str, parameters: dict | None = None):
        return self._transaction.run(query, parameters)


class MergeStore:
    """Transactional fake that records statements per transaction."""

    def __init__(
        self,
        entities: list[_EntityRecord],
        *,
        rel_types: list[str] | None = None,
        stale_ids: set[int] | None = None,
        deleted: int | None = None,
    ):
        self.node_props = {
            entity.node_id: {
                **entity.properties,
                "id": entity.entity_id,
                "graph_name": entity.graph_name,
            }
            for entity in entities
        }
        self.rel_types = rel_types or []
        self.stale_ids = stale_ids or set()
        self.deleted = deleted
        self.transactions: list[MergeTransaction] = []
        self.active: MergeTransaction | None = None
        self.fail_on: str | None = None
        self.error: BaseException | None = None
        self.failures = 0
        self.fail_inside_transaction_only = False
        self.commit_error: BaseException | None = None

    @contextmanager
    def _transaction(self):
        transaction = MergeTransaction(self)
        self.transactions.append(transaction)
        self.active = transaction
        try:
            yield MergeExecutor(transaction)
        except BaseException:
            transaction.rollback()
            raise
        else:
            transaction.commit()
        finally:
            self.active = None

    def execute_query(self, query: str, parameters: dict | None = None):
        if self.active is not None:
            return self.active.run(query, parameters)
        return self.run_query(query, parameters or {})

    def run_query(self, query: str, parameters: dict):
        if (
            self.fail_on is not None
            and self.fail_on in query
            and self.failures
            and (self.active is not None or not self.fail_inside_transaction_only)
        ):
            self.failures -= 1
            raise self.error or RuntimeError("injected failure")

        if "RETURN id(n) AS node_id, properties(n) AS props" in query:
            return [
                {"node_id": node_id, "props": dict(props)}
                for node_id, props in self.node_props.items()
                if node_id in parameters.get("node_ids", [])
                and node_id not in self.stale_ids
            ]

        if "AS scope_type" in query:
            return []

        if "properties(r) AS rel_props" in query:
            return []

        if "RETURN DISTINCT type(r) AS rel_type" in query:
            return [
                {"rel_type": rel_type, "node_id": node_id}
                for node_id in parameters.get("other_ids", [])
                for rel_type in self.rel_types
            ]

        if "SET n += $props" in query:
            return [{"merged_id": parameters.get("canonical_id")}]

        if "RETURN count(*) AS deleted" in query:
            expected = len(parameters.get("other_ids", []))
            return [{"deleted": expected if self.deleted is None else self.deleted}]

        return []


def entity(
    node_id: int,
    entity_id: str,
    name: str,
    graph_name: str = "g:1",
) -> _EntityRecord:
    return _EntityRecord(
        node_id=node_id,
        entity_id=entity_id,
        graph_name=graph_name,
        domain_label="Person",
        resolve_value=name,
        normalized_value=name.casefold(),
        properties={"id": entity_id, "graph_name": graph_name},
    )


def group() -> list[_EntityRecord]:
    return [
        entity(1, "canonical", "Alice Smith"),
        entity(2, "duplicate", "Alice"),
    ]


def merge(store: object, groups: list[list[_EntityRecord]] | None = None) -> int:
    return _MemgraphEntityResolver(store)._merge_groups(groups or [group()], "name")


def query_text(transaction: MergeTransaction) -> str:
    return "\n".join(query for query, _ in transaction.queries)


def test_happy_path_commits_one_transaction_and_counts_after_commit():
    store = MergeStore(group(), rel_types=["KNOWS"])

    merged_nodes = merge(store)

    assert merged_nodes == 2
    assert len(store.transactions) == 1
    assert store.transactions[0].committed is True
    text = query_text(store.transactions[0])
    assert "MATCH (n:__Entity__)" in text
    assert "AS scope_type" in text
    assert "SET n += $props" in text
    assert "MERGE (canonical)-[new_r:`KNOWS`]->(target)" in text
    assert "SET c.object_entity_id" in text
    assert "DELETE n" in text
    assert "DETACH DELETE" not in text
    assert text.index("SET c.object_entity_id") < text.index("DELETE n")


@pytest.mark.parametrize(
    "fail_on",
    [
        "SET n += $props",
        "MERGE (canonical)-[new_r:`KNOWS`]->(target)",
        "SET c.object_entity_id",
        "RETURN count(*) AS deleted",
    ],
)
def test_failure_after_each_mutation_phase_rolls_back(fail_on):
    store = MergeStore(group(), rel_types=["KNOWS"])
    store.fail_on = fail_on
    store.error = RuntimeError(f"{fail_on} failed")
    store.failures = 99

    with pytest.raises(RuntimeError, match="failed"):
        merge(store)

    assert len(store.transactions) == 1
    assert store.transactions[0].rolled_back is True
    assert store.transactions[0].committed is False
    assert not store.transactions[0].queries[-1][0].strip().startswith("COMMIT")


def test_each_group_uses_its_own_transaction():
    second = [
        entity(3, "canonical-2", "Bob Jones"),
        entity(4, "duplicate-2", "Bob"),
    ]
    store = MergeStore([*group(), *second])

    merged_nodes = merge(store, [group(), second])

    assert merged_nodes == 4
    assert len(store.transactions) == 2
    assert all(transaction.committed for transaction in store.transactions)
    for transaction in store.transactions:
        text = query_text(transaction)
        assert text.count("SET n += $props") == 1
        assert text.count("DELETE n") == 1


def test_retry_rereads_membership_and_preflight_in_each_attempt():
    store = MergeStore(group())
    store.fail_on = "SET n += $props"
    store.error = DeadlockError("write conflict")
    store.failures = 1

    merged_nodes = merge(store)

    assert merged_nodes == 2
    assert len(store.transactions) == 2
    first, second = store.transactions
    assert first.rolled_back is True
    assert first.committed is False
    assert second.committed is True
    for transaction in (first, second):
        text = query_text(transaction)
        assert "RETURN id(n) AS node_id, properties(n) AS props" in text
        assert "AS scope_type" in text
    assert any("SET n += $props" in query for query, _ in second.queries)


def test_persistent_retryable_error_stops_after_bounded_attempts():
    store = MergeStore(group())
    store.fail_on = "SET n += $props"
    store.error = DeadlockError("write conflict")
    store.failures = 99

    with pytest.raises(DeadlockError):
        merge(store)

    assert len(store.transactions) == 3
    assert all(transaction.rolled_back for transaction in store.transactions)
    assert all(not transaction.committed for transaction in store.transactions)


def test_non_retryable_error_is_not_retried():
    store = MergeStore(group())
    store.fail_on = "AS scope_type"
    store.error = ValueError("foreign or unscoped incident relationships")
    store.failures = 99
    # The all-group preflight also uses this query outside a transaction.
    # Fail only the fresh per-attempt preflight to test the retry decision.
    store.fail_inside_transaction_only = True

    with pytest.raises(ValueError, match="foreign or unscoped"):
        merge(store)

    assert len(store.transactions) == 1
    assert store.transactions[0].rolled_back is True


def test_stale_member_skips_the_group_and_counts_zero(caplog):
    store = MergeStore(group(), stale_ids={2})

    merged_nodes = merge(store)

    assert merged_nodes == 0
    assert len(store.transactions) == 1
    assert store.transactions[0].committed is True
    assert "SET n += $props" not in query_text(store.transactions[0])
    assert "DELETE n" not in query_text(store.transactions[0])
    assert "skipping stale merge group" in caplog.text


def test_recycled_identity_skips_the_group(caplog):
    store = MergeStore(group())
    store.node_props[2]["id"] = "other-entity"

    merged_nodes = merge(store)

    assert merged_nodes == 0
    assert "SET n += $props" not in query_text(store.transactions[0])
    assert "skipping stale merge group" in caplog.text


def test_residual_relationship_fails_the_delete_invariant():
    store = MergeStore(group(), deleted=0)

    with pytest.raises(RuntimeError, match="Merge invariant violated"):
        merge(store)

    assert store.transactions[0].rolled_back is True
    text = query_text(store.transactions[0])
    assert "DELETE n" in text
    assert "DETACH DELETE" not in text


def test_missing_transaction_hook_fails_before_any_query():
    class NoTransactionStore:
        def __init__(self):
            self.queries: list[str] = []

        def execute_query(self, query: str, parameters: dict | None = None):
            self.queries.append(query)
            return []

    store = NoTransactionStore()

    with pytest.raises(RuntimeError, match="_transaction"):
        merge(store)

    assert store.queries == []


def test_commit_failure_is_surfaced_without_a_success_count():
    store = MergeStore(group())
    store.commit_error = RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        merge(store)

    assert len(store.transactions) == 1
    assert store.transactions[0].committed is False
