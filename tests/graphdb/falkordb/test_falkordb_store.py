"""Unit tests for FalkorDBGraphStore."""

from __future__ import annotations

import pytest

from recon_graphrag.graphdb.falkordb.store import FalkorDBGraphStore
from recon_graphrag.models.types import IndexConfig


class FakeResult:
    def __init__(self, header, result_set):
        self.header = header
        self.result_set = result_set

    @property
    def columns(self):
        """Backward-compatible alias for tests that use old API."""
        return [entry[1] for entry in self.header]


class FakeGraph:
    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []
        self._next_id = 0

    def query(self, cypher: str, params: dict | None = None):
        self.queries.append(cypher.strip())
        self.params.append(params or {})
        return FakeResult([], [])

    def select_graph(self, name: str):
        return self


class FakeFalkorDB:
    def __init__(self):
        self.graphs: dict[str, FakeGraph] = {}

    def select_graph(self, name: str) -> FakeGraph:
        return self.graphs.setdefault(name, FakeGraph())


@pytest.fixture
def fake_db():
    return FakeFalkorDB()


@pytest.fixture
def store(fake_db):
    return FalkorDBGraphStore(fake_db, graph_name="test-graph")


def test_execute_query_translates_result(store, fake_db):
    graph = fake_db.select_graph("test-graph")
    graph.query = lambda cypher, params=None: FakeResult(
        [[0, "cnt"]], [[42]]
    )

    result = store.execute_query("MATCH (n) RETURN count(n) AS cnt")
    assert result == [{"cnt": 42}]


def test_write_graph_document_delegates_to_writer(store, fake_db):
    from recon_graphrag.extraction.types import (
        ChunkRecord,
        DocumentRecord,
        EvidenceLink,
        GraphDocument,
        RelationshipRecord,
    )

    document = DocumentRecord(
        id="doc-1",
        text_hash="hash",
        graph_name="test-graph",
        metadata={},
    )
    chunks = [
        ChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            text="hello world",
            index=0,
            graph_name="test-graph",
            metadata={},
        )
    ]
    graph_doc = GraphDocument(
        document=document,
        chunks=chunks,
        entities=[],
        relationships=[],
        evidence_links=[],
    )

    result = store.write_graph_document(graph_doc)
    assert result["documents"] == 1
    assert result["chunks"] == 1


def test_create_vector_index_registers_index(store, fake_db):
    graph = fake_db.select_graph("test-graph")
    original_query = graph.query

    def recording_query(cypher, params=None):
        graph.queries.append(cypher.strip())
        graph.params.append(params or {})
        return FakeResult([], [])

    graph.query = recording_query

    store.create_vector_index(
        name="entity-embeddings",
        label="__Entity__",
        embedding_property="embedding",
        dimensions=1536,
    )
    assert store._vector_index_registry["entity-embeddings"] == ("__Entity__", "embedding")
    assert any("CREATE VECTOR INDEX" in q for q in graph.queries)


def test_create_fulltext_index_registers_index(store, fake_db):
    graph = fake_db.select_graph("test-graph")

    def recording_query(cypher, params=None):
        graph.queries.append(cypher.strip())
        graph.params.append(params or {})
        return FakeResult([], [])

    graph.query = recording_query

    store.create_fulltext_index(
        name="entity-names",
        label="__Entity__",
        node_properties=["name"],
    )
    assert store._fulltext_index_registry["entity-names"] == "__Entity__"
    assert any("db.idx.fulltext.createNodeIndex" in q for q in graph.queries)


def test_upsert_vectors_runs_unwind_set(store, fake_db):
    graph = fake_db.select_graph("test-graph")

    def recording_query(cypher, params=None):
        graph.queries.append(cypher.strip())
        graph.params.append(params or {})
        return FakeResult([], [])

    graph.query = recording_query

    store.upsert_vectors(
        node_ids=["1", "2"],
        embedding_property="embedding",
        vectors=[[0.1, 0.2], [0.3, 0.4]],
    )
    assert any("UNWIND $rows" in q for q in graph.queries)
    assert any("SET n.embedding = vecf32(row.vector)" in q for q in graph.queries)


def test_vector_search_runs_vector_procedure(store, fake_db):
    graph = fake_db.select_graph("test-graph")
    graph.query = lambda cypher, params=None: FakeResult(
        [[0, "id"], [1, "score"]], [["1", 0.9], ["2", 0.8]]
    )
    store._vector_index_registry["entity-embeddings"] = ("__Entity__", "embedding")

    result = store.vector_search(
        index_name="entity-embeddings",
        query_vector=[0.1, 0.2],
        k=5,
    )
    assert result == [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}]


def test_keyword_search_runs_fulltext_procedure(store, fake_db):
    graph = fake_db.select_graph("test-graph")
    graph.query = lambda cypher, params=None: FakeResult(
        [[0, "id"], [1, "score"]], [["1", 0.9]]
    )
    store._fulltext_index_registry["entity-names"] = "__Entity__"

    result = store.keyword_search(
        index_name="entity-names",
        query_text="Alice",
        k=5,
    )
    assert result == [{"id": "1", "score": 0.9}]


def test_fetch_entity_context_aggregates_flat_rows(store, fake_db):
    graph = fake_db.select_graph("test-graph")

    def fake_query(cypher, params=None):
        return FakeResult(
            [
                [0, "title"],
                [1, "relationship"],
                [2, "source_text"],
                [3, "score"],
            ],
            [
                ["Alice (Person)", "Person: Alice -[KNOWS]-> Person: Bob", ["text1"], 0.9],
                ["Alice (Person)", "Person: Alice -[WORKS_AT]-> Organization: Acme", ["text1"], 0.9],
            ],
        )

    graph.query = fake_query
    result = store.fetch_entity_context(
        matches=[{"id": "1", "score": 0.9}],
        mode="local",
    )
    assert len(result) == 1
    assert result[0]["title"] == "Alice (Person)"
    assert len(result[0]["relationships"]) == 2
    assert result[0]["source_text"] == ["text1"]


def test_validate_graph_build_runs_counts(store, fake_db):
    graph = fake_db.select_graph("test-graph")

    def fake_query(cypher, params=None):
        if "MATCH (e:__Entity__)" in cypher:
            return FakeResult([[0, "cnt"]], [[10]])
        if "MATCH (c:Chunk)" in cypher:
            return FakeResult([[0, "cnt"]], [[5]])
        if "FROM_CHUNK" in cypher:
            return FakeResult([[0, "cnt"]], [[3]])
        return FakeResult([[0, "cnt"]], [[2]])

    graph.query = fake_query
    result = store.validate_graph_build()
    assert result["entity_count"] == 10
    assert result["chunk_count"] == 5
    assert result["evidence_link_count"] == 3
    assert result["entity_relationship_count"] == 2
