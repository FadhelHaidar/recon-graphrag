"""Inspect vector write parameters; live backend tests own Cypher semantics."""

from unittest.mock import MagicMock

import pytest

from recon_graphrag.entity_text import entity_embedding_fingerprint
from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore


@pytest.fixture(params=[Neo4jGraphStore, MemgraphGraphStore], ids=["neo4j", "memgraph"])
def store(request):
    store = request.param(None)
    store.execute_query = MagicMock(return_value=[])
    return store


@pytest.mark.parametrize("matches", [True, False])
def test_null_expected_snapshot_attached_only_for_matching_text(store, matches):
    snapshot = {"id": "1", "labels": ["Person"], "name": None, "description": "Alice"}
    fingerprint = entity_embedding_fingerprint(labels=["Person"], name=None, description="Alice")
    store.execute_query.side_effect = [[snapshot], []]
    supplied = fingerprint if matches else "stale"
    store.upsert_vectors(["1"], "embedding", [[0.1]], content_fingerprints=[supplied], model_fingerprint="model")
    read, write = store.execute_query.call_args_list
    assert read.args[1] == {"node_ids": ["1"]}
    assert "n.embedding_input_fingerprint IS NULL" in read.args[0]
    row = {"id": "1", "vector": [0.1], "content_fingerprint": supplied}
    if matches:
        row["legacy_input"] = snapshot
    assert write.args[1]["rows"] == [row]
    assert write.args[1]["model_fingerprint"] == "model"


@pytest.mark.parametrize("model", [None, "supplied-model"])
def test_legacy_vector_parameters_clear_unknown_certification(store, model):
    store.upsert_vectors(["1"], "embedding", [[0.1]], model_fingerprint=model)
    store.execute_query.assert_called_once()
    query, params = store.execute_query.call_args.args
    assert params["rows"] == [{"id": "1", "vector": [0.1], "content_fingerprint": None}]
    assert params["model_fingerprint"] == model
    assert "n.embedding_content_fingerprint = row.content_fingerprint" in query
    assert "n.embedding_model_fingerprint = $model_fingerprint" in query
    assert "n.embedding_input_fingerprint = coalesce(n.embedding_input_fingerprint, row.content_fingerprint)" in query


def test_neo4j_locks_before_freshness_guard(store):
    if not isinstance(store, Neo4jGraphStore):
        pytest.skip("Neo4j-specific locking query")
    store.upsert_vectors(["1"], "embedding", [[0.1]])
    execute = store.execute_query
    assert isinstance(execute, MagicMock)
    query = execute.call_args.args[0]
    lock = "SET n.embedding_input_fingerprint = n.embedding_input_fingerprint + ''"
    assert query.index("WHERE elementId(n) = row.id") < query.index(lock)
    assert query.index(lock) < query.index("WHERE (row.content_fingerprint IS NULL")
    assert "WITH n, row\n            CALL db.create.setNodeVectorProperty" in query


@pytest.mark.parametrize("vectors,fingerprints,message", [
    ([], None, "node_ids and vectors must have the same length"),
    ([[0.1]], [], "node_ids and content_fingerprints must have the same length"),
])
def test_vector_lengths_validated_before_queries(store, vectors, fingerprints, message):
    with pytest.raises(ValueError, match=message):
        store.upsert_vectors(["1"], "embedding", vectors, content_fingerprints=fingerprints)
    store.execute_query.assert_not_called()


def test_empty_vector_batch_does_not_query(store):
    store.upsert_vectors([], "embedding", [], content_fingerprints=[])
    store.execute_query.assert_not_called()
