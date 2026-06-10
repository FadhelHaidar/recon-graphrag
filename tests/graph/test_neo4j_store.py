"""Tests for direct Neo4jGraphStore index and vector operations."""

import pytest

from recon_graphrag.graph.neo4j_store import Neo4jGraphStore


class FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, parameters=None):
        self.calls.append((query.strip(), parameters or {}))
        return []


class FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self, **kwargs):
        self.session_kwargs = kwargs
        return FakeSession(self.calls)


def test_create_vector_index_executes_direct_cypher_with_escaped_identifiers():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.create_vector_index(
        name="entity`embeddings",
        label="__Entity__",
        embedding_property="embed`ding",
        dimensions=1536,
        similarity_fn="cosine",
    )

    query, params = driver.calls[0]
    assert "CREATE VECTOR INDEX `entity``embeddings` IF NOT EXISTS" in query
    assert "FOR (n:`__Entity__`)" in query
    assert "ON (n.`embed``ding`)" in query
    assert "`vector.dimensions`: 1536" in query
    assert "`vector.similarity_function`: 'cosine'" in query
    assert params == {}


def test_create_fulltext_index_executes_direct_cypher_with_properties():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.create_fulltext_index(
        name="entity-names",
        label="__Entity__",
        node_properties=["name", "description"],
    )

    query, params = driver.calls[0]
    assert "CREATE FULLTEXT INDEX `entity-names` IF NOT EXISTS" in query
    assert "FOR (n:`__Entity__`)" in query
    assert "ON EACH [n.`name`, n.`description`]" in query
    assert params == {}


def test_upsert_vectors_matches_nodes_by_element_id():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.upsert_vectors(["4:a", "4:b"], "embedding", [[0.1], [0.2]])

    query, params = driver.calls[0]
    assert "WHERE elementId(n) = row.id" in query
    assert "db.create.setNodeVectorProperty" in query
    assert params == {
        "rows": [
            {"id": "4:a", "vector": [0.1]},
            {"id": "4:b", "vector": [0.2]},
        ],
        "embedding_property": "embedding",
    }


def test_upsert_vectors_rejects_mismatched_lengths():
    store = Neo4jGraphStore(FakeDriver())

    with pytest.raises(ValueError):
        store.upsert_vectors(["4:a"], "embedding", [])
