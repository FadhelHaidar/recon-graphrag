"""Per-document transaction boundaries for the Cypher writer."""

from __future__ import annotations

from typing import Any, cast

import pytest

from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore
from tests.pipelines.writer_scenarios import make_graph_document


class FakeTransaction:
    def __init__(self, session: FakeSession):
        self._session = session
        self.committed = False
        self.rolled_back = False

    def run(self, query: str, parameters: dict | None = None):
        return self._session.run(query, parameters)

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
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("connection dropped")
        self.queries.append(query)
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
    assert session.queries, "writer statements must run inside the transaction"


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
    assert [query for query in session.queries if "MERGE" in query] == []


class DeadlockError(Exception):
    """Stand-in for neo4j.exceptions.TransientError on a lock cycle."""

    code = "Neo.TransientError.Transaction.DeadlockDetected"


class FlakySession(FakeSession):
    """Fails ``failures`` write attempts with a deadlock, then succeeds."""

    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures

    def run(self, query: str, parameters: dict | None = None):
        if self.failures and "MERGE (e:__Entity__" in query:
            self.failures -= 1
            raise DeadlockError("DeadlockDetected: would form a cycle")
        return super().run(query, parameters)


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
    assert session.queries, "the retry must actually write the document"


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

    def run(self, query: str, parameters: dict | None = None):
        if self.failures and "MERGE (e:__Entity__" in query:
            self.failures -= 1
            raise self.error()
        return super().run(query, parameters)


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
