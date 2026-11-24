"""Per-document transaction boundaries for the Cypher writer."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, cast

import pytest

from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore
from recon_graphrag.pipelines.writer import CypherGraphWriter
from tests.pipelines.writer_scenarios import (
    StatefulFakeGraphStore,
    make_graph_document,
)


class FakeTransaction:
    def __init__(self, session: FakeSession):
        self._session = session
        self.queries: list[str] = []
        self.committed = False
        self.rolled_back = False

    def run(self, query: str, parameters: dict | None = None):
        self.queries.append(query)
        return self._session.respond(query, parameters)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class FakeSession:
    """Records statements; fails on queries containing ``fail_on``."""

    def __init__(self, fail_on: str | None = None):
        self.queries: list[str] = []
        self.fail_on = fail_on
        self.transactions: list[FakeTransaction] = []

    def run(self, query: str, parameters: dict | None = None):
        self.queries.append(query)
        return self.respond(query, parameters)

    def respond(self, query: str, parameters: dict | None = None):
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("connection dropped")
        params = parameters or {}
        if "UNWIND $rows AS row" in query and "__recon_write_lock" in query:
            rows = params["rows"]
            return [{"locked": [row["id"] for row in rows], "requested": len(rows)}]
        if "REMOVE e.__recon_write_lock" in query:
            return [{"released": len(params["ids"])}]
        return []

    def begin_transaction(self) -> FakeTransaction:
        transaction = FakeTransaction(self)
        self.transactions.append(transaction)
        return transaction

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeDriver:
    def __init__(self, session: FakeSession):
        self.session_obj = session

    def session(self, **kwargs):
        return self.session_obj


@pytest.fixture(params=["neo4j", "memgraph"])
def store_and_session(request):
    session = FakeSession()
    driver = cast(Any, FakeDriver(session))
    if request.param == "neo4j":
        return Neo4jGraphStore(driver), session
    return MemgraphGraphStore(driver), session


@pytest.mark.parametrize("failing_write", [None, "MERGE (e:__Entity__"])
def test_document_write_is_one_transaction(store_and_session, failing_write):
    store, session = store_and_session
    session.fail_on = failing_write

    if failing_write is None:
        store.write_graph_document(make_graph_document())
    else:
        with pytest.raises(RuntimeError, match="connection dropped"):
            store.write_graph_document(make_graph_document())

    assert len(session.transactions) == 1
    transaction = session.transactions[0]
    assert transaction.committed is (failing_write is None)
    assert transaction.rolled_back is (failing_write is not None)
    assert not session.queries, "no writer statement may use autocommit"
    assert any("MERGE (d:Document" in query for query in transaction.queries)


@pytest.mark.parametrize("fail_on", [None])
def test_validation_error_rolls_back_before_writing(store_and_session, fail_on):
    from recon_graphrag.extraction.types import (
        ChunkRecord,
        DocumentRecord,
        EntityRecord,
        GraphDocument,
        RelationshipRecord,
    )

    store, session = store_and_session
    document = GraphDocument(
        document=DocumentRecord(id="doc:1", text_hash="hash"),
        chunks=[ChunkRecord(id="chunk:1", document_id="doc:1", text="hi", index=0)],
        entities=[
            EntityRecord(id="person:1", type="Person", properties={"name": "Alice"})
        ],
        relationships=[
            RelationshipRecord(
                id="rel:1",
                source_id="person:1",
                target_id="person:ghost",
                type="KNOWS",
            )
        ],
        evidence_links=[],
        claims=[],
    )

    with pytest.raises(ValueError, match="person:ghost"):
        store.write_graph_document(document)

    assert session.transactions[0].rolled_back is True
    assert not session.queries
    assert [
        query
        for query in session.transactions[0].queries
        if "MERGE" in query and "__recon_write_lock" not in query
    ] == []


class DeadlockError(Exception):
    """Stand-in for neo4j.exceptions.TransientError on a lock cycle."""

    code = "Neo.TransientError.Transaction.DeadlockDetected"


class FlakySession(FakeSession):
    """Fails ``failures`` write attempts with a deadlock, then succeeds."""

    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures

    def respond(self, query: str, parameters: dict | None = None):
        if self.failures and "MERGE (e:__Entity__" in query:
            self.failures -= 1
            raise DeadlockError("DeadlockDetected: would form a cycle")
        return super().respond(query, parameters)


def _flaky_store(name: str, failures: int) -> tuple[Any, FlakySession]:
    session = FlakySession(failures)
    driver = cast(Any, FakeDriver(session))
    store = Neo4jGraphStore(driver) if name == "neo4j" else MemgraphGraphStore(driver)
    return store, session


@pytest.mark.parametrize("name", ["neo4j", "memgraph"])
def test_deadlock_is_retried_inside_the_writer(name):
    store, session = _flaky_store(name, failures=1)

    store.write_graph_document(make_graph_document())

    assert len(session.transactions) == 2
    assert session.transactions[0].rolled_back is True
    assert session.transactions[1].committed is True
    assert not session.queries
    assert any("MERGE (d:Document" in query for query in session.transactions[1].queries)


def test_lock_batch_precedes_the_authoritative_state_read(store_and_session):
    store, session = store_and_session

    store.write_graph_document(make_graph_document())

    assert not session.queries
    queries = session.transactions[0].queries
    lock_index = next(
        index for index, query in enumerate(queries) if "__recon_write_lock" in query
    )
    state_index = next(
        index
        for index, query in enumerate(queries)
        if "RETURN e.id AS id," in query and "e.type AS type" in query
    )
    target_index = next(
        index
        for index, query in enumerate(queries)
        if "UNWIND $relationship_keys AS row" in query
    )
    assert lock_index < state_index < target_index
    lock_query = queries[lock_index]
    # The importing-WITH form works on older Neo4j 5.x; the variable-scope
    # CALL (row) form needs a newer server.
    assert re.search(r"CALL \{\s+WITH row\b", lock_query)
    assert "CALL (row)" not in lock_query
    assert "REMOVE e.__recon_write_lock" in queries[-1]


class TransactionalStatefulStore(StatefulFakeGraphStore):
    """Stateful fake with the private transaction hook the writer probes."""

    def __init__(self) -> None:
        super().__init__()
        self.transaction_count = 0

    @contextmanager
    def _transaction(self):
        self.transaction_count += 1
        yield self


def test_replacement_lock_set_covers_previous_entities():
    from tests.pipelines.test_writer_replacement import build_document

    store = TransactionalStatefulStore()
    writer = CypherGraphWriter(cast(Any, store))
    writer.write_graph_document(
        build_document(
            chunks=["a:c0"],
            document_id="a",
            entities=[("e1", "Alice", "Old fact."), ("e2", "Bob", None)],
        )
    )
    store.queries.clear()
    store.lock_batches.clear()

    writer.write_graph_document(
        build_document(
            chunks=["a:c1"],
            document_id="a",
            entities=[("e2", "Bob", "New fact.")],
        )
    )

    locked = {row["id"] for row in store.lock_batches[0]}
    assert locked == {"e1", "e2"}
    assert store.lock_releases and "e1" in store.lock_releases[-1]
    lock_index = next(
        index
        for index, query in enumerate(store.queries)
        if "UNWIND $rows AS row" in query and "__recon_write_lock" in query
    )
    state_index = next(
        index
        for index, query in enumerate(store.queries)
        if "RETURN e.id AS id," in query and "e.type AS type" in query
    )
    assert lock_index < state_index


def test_auto_commit_store_never_runs_a_lock_query():
    store = StatefulFakeGraphStore()

    CypherGraphWriter(cast(Any, store)).write_graph_document(make_graph_document())

    assert not any("__recon_write_lock" in query for query in store.queries)
    assert store.lock_batches == []


def test_missing_reference_is_not_created_by_the_lock_batch():
    from tests.pipelines.test_writer_replacement import build_document

    store = TransactionalStatefulStore()
    writer = CypherGraphWriter(cast(Any, store))
    document = build_document(
        chunks=["chunk:1"],
        entities=[("e1", "Alice", None)],
        relationships=[("KNOWS", "e1", "ghost", None)],
    )

    with pytest.raises(ValueError, match="ghost"):
        writer.write_graph_document(document)

    assert store.get_node("ghost") is None
    assert not any("UNWIND $relationships AS row" in query for query in store.queries)


class OmitPriorLockStore(TransactionalStatefulStore):
    def __init__(self) -> None:
        super().__init__()
        self.omitted: str | None = None

    def _write_entity_locks(self, params: dict) -> list:
        result = super()._write_entity_locks(params)
        if self.omitted is not None:
            result[0]["locked"] = [
                entity_id
                for entity_id in result[0]["locked"]
                if entity_id != self.omitted
            ]
        return result


def test_unlocked_prior_entity_retries_then_surfaces_a_conflict():
    from tests.pipelines.test_writer_replacement import build_document

    store = OmitPriorLockStore()
    writer = CypherGraphWriter(cast(Any, store))
    writer.write_graph_document(
        build_document(
            chunks=["a:c0"],
            document_id="a",
            entities=[("e1", "Alice", "Old fact.")],
        )
    )
    store.omitted = "e1"
    store.transaction_count = 0
    mark = len(store.queries)

    with pytest.raises(Exception, match="lock"):
        writer.write_graph_document(
            build_document(chunks=["a:c1"], document_id="a")
        )

    assert store.transaction_count == 3
    assert not any(
        "UNWIND $documents AS row" in query for query in store.queries[mark:]
    )


def test_transient_lock_does_not_invalidate_a_retained_summary():
    from tests.pipelines.test_writer_replacement import build_document

    store = TransactionalStatefulStore()
    writer = CypherGraphWriter(cast(Any, store))
    document = build_document(
        chunks=["chunk"], entities=[("alice", "Alice", "Source fact")]
    )
    document.entities[0].properties["role"] = "Engineer"
    writer.write_graph_document(document)
    node = store.get_node("alice")
    assert node is not None
    node.update(description="Previous summary", description_summary_status="success")

    writer.write_graph_document(document)

    assert node["description_summary_status"] == "success"
    assert node["description"] == "Previous summary"


def test_lock_property_does_not_pollute_the_written_entity():
    from tests.pipelines.test_writer_replacement import build_document

    store = TransactionalStatefulStore()
    writer = CypherGraphWriter(cast(Any, store))
    writer.write_graph_document(
        build_document(chunks=["a:c0"], document_id="a", entities=[])
    )
    writer.write_graph_document(
        build_document(
            chunks=["a:c0"],
            document_id="a",
            entities=[("e1", "Alice", "A fact.")],
        )
    )

    node = store.get_node("e1")
    assert node is not None
    assert "__recon_write_lock" not in node
    assert store.lock_releases


@pytest.mark.parametrize("name", ["neo4j", "memgraph"])
def test_persistent_deadlock_surfaces_after_bounded_retries(name):
    store, session = _flaky_store(name, failures=99)

    with pytest.raises(DeadlockError):
        store.write_graph_document(make_graph_document())

    assert len(session.transactions) == 3
    assert all(transaction.rolled_back for transaction in session.transactions)


class ConstraintError(Exception):
    """Stand-in for a unique-constraint failure on :__Entity__(uid)."""

    code = "Neo.ClientError.Schema.ConstraintValidationFailed"


class MemgraphConstraintError(Exception):
    """Memgraph reports the same race as a generic ClientError message."""

    code = "Memgraph.ClientError.MemgraphError.MemgraphError"

    def __str__(self) -> str:
        return "Unable to commit due to unique constraint violation on :__Entity__(uid)"


class ConstraintSession(FakeSession):
    def __init__(self, error: type[Exception], failures: int):
        super().__init__()
        self.error = error
        self.failures = failures

    def respond(self, query: str, parameters: dict | None = None):
        if self.failures and "MERGE (e:__Entity__" in query:
            self.failures -= 1
            raise self.error()
        return super().respond(query, parameters)


@pytest.mark.parametrize(
    "error", [ConstraintError, MemgraphConstraintError], ids=["neo4j", "memgraph"]
)
def test_unique_constraint_race_is_retried(error):
    session = ConstraintSession(error, failures=1)
    store = Neo4jGraphStore(cast(Any, FakeDriver(session)))

    store.write_graph_document(make_graph_document())

    assert len(session.transactions) == 2
    assert session.transactions[0].rolled_back is True
    assert session.transactions[1].committed is True


def test_unrelated_constraint_error_is_not_retried():
    session = ConstraintSession(ConstraintError, failures=99)
    session.error = ValueError  # no code, no constraint wording
    store = Neo4jGraphStore(cast(Any, FakeDriver(session)))

    with pytest.raises(ValueError):
        store.write_graph_document(make_graph_document())

    assert len(session.transactions) == 1


class DeadlockRetryError(Exception):
    """Stand-in for a retryable Neo4j deadlock."""

    code = "Neo.TransientError.Transaction.DeadlockDetected"


class RetryOnceStatefulStore(StatefulFakeGraphStore):
    """Fails the first entity write, emulating a rollback-safe retry."""

    def __init__(self) -> None:
        super().__init__()
        self.failures = 0

    def execute_query(self, query: str, parameters: dict | None = None):
        if self.failures and "MERGE (e:__Entity__" in query:
            self.failures -= 1
            raise DeadlockRetryError("DeadlockDetected: would form a cycle")
        return super().execute_query(query, parameters)

    @contextmanager
    def _transaction(self):
        yield self


def test_replacement_rows_are_rebuilt_for_every_attempt():
    from recon_graphrag.pipelines.writer import CypherGraphWriter
    from tests.pipelines.test_writer_replacement import build_document

    store = RetryOnceStatefulStore()
    writer = CypherGraphWriter(cast(Any, store))
    writer.write_graph_document(
        build_document(chunks=["a:c0"], entities=[("e1", "Alice", "Old fact.")])
    )
    store.failures = 1
    store.lock_batches.clear()
    store.lock_releases.clear()

    calls: list[tuple[Any, list[dict[str, Any]]]] = []
    original = writer._rebuild_entity_rows

    def counting(graph_document, state, retained_rows):
        rows = original(graph_document, state, retained_rows)
        calls.append((state, rows))
        return rows

    writer._rebuild_entity_rows = counting
    writer.write_graph_document(
        build_document(chunks=["a:c0"], entities=[("e2", "Bob", "New fact.")])
    )

    assert len(calls) == 2, (
        "each attempt must rebuild replacement rows exactly once"
    )
    assert calls[0][0] is not calls[1][0], (
        "each attempt must read its own pre-write document state"
    )
    assert [row["id"] for row in calls[0][1]] == ["e1"]
    assert calls[0][1] == calls[1][1], (
        "the retry must recompute the same rows from unchanged state"
    )
    assert calls[1][0].entities["e1"]["descriptions"] == ["Old fact."]
    assert len(store.lock_batches) == 2, (
        "each attempt must acquire its own entity locks"
    )
    assert len(store.lock_releases) == 1, (
        "only the successful attempt must release its locks"
    )
