"""Reject incompatible assertions with one caller-supplied ID before writes."""

from dataclasses import replace

import pytest

from tests.pipelines.test_writer_replacement import build_document, make_writer
from tests.pipelines.writer_scenarios import StatefulFakeGraphStore


def test_incompatible_duplicate_claim_ids_fail_before_database_queries():
    store = StatefulFakeGraphStore()
    document = build_document(chunks=["chunk"], entities=[("alice", "Alice", None)],
                              claims=[("stable-claim", "alice", "chunk")])
    document.claims.append(replace(document.claims[0], start_date="2025-01-01"))
    with pytest.raises(ValueError, match="incompatible assertion"):
        make_writer(store).write_graph_document(document)
    assert not store.queries


def test_status_update_with_stable_id_remains_supported():
    store = StatefulFakeGraphStore()
    document = build_document(chunks=["chunk"], entities=[("alice", "Alice", None)],
                              claims=[("stable-claim", "alice", "chunk")])
    document.claims.append(replace(document.claims[0], status="resolved"))
    make_writer(store).write_graph_document(document)
    stored = store.get_claim("stable-claim")
    assert stored is not None
    assert stored["status"] == "resolved"
