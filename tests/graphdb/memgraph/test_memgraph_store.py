"""Unit tests for MemgraphGraphStore."""

from __future__ import annotations

import pytest

import recon_graphrag.graphdb.memgraph.entity_resolution as entity_resolution
from recon_graphrag.graphdb.memgraph.store import (
    MemgraphGraphStore,
    _format_tantivy_query,
)
from recon_graphrag.models.artifacts import CommunityReport
from recon_graphrag.models.types import IndexConfig


class FakeRecord:
    def __init__(self, data: dict):
        self._data = data

    def __iter__(self):
        return iter(self._data.items())

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


class FakeResult:
    def __init__(self, records: list[dict]):
        self._records = [FakeRecord(r) for r in records]

    def __iter__(self):
        return iter(self._records)


class FakeSession:
    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []
        self._next_result: FakeResult = FakeResult([])

    def run(self, query: str, parameters: dict | None = None):
        self.queries.append(query.strip())
        self.params.append(parameters or {})
        return self._next_result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()

    def session(self, database=None):
        return self.session_obj


@pytest.fixture
def fake_driver():
    return FakeDriver()


@pytest.fixture
def store(fake_driver):
    return MemgraphGraphStore(fake_driver, database="memgraph")


def test_execute_query_translates_result(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult([{"cnt": 42}])

    result = store.execute_query("MATCH (n) RETURN count(n) AS cnt")
    assert result == [{"cnt": 42}]


@pytest.mark.asyncio
async def test_resolve_entities_forwards_full_advanced_parameter_set(
    store, monkeypatch
):
    captured = {}

    class CapturingResolver:
        def __init__(self, graph_store):
            captured["graph_store"] = graph_store

        async def resolve(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"strategy": kwargs["strategy"]}

    monkeypatch.setattr(entity_resolution, "_MemgraphEntityResolver", CapturingResolver)
    embedder = object()
    llm = object()
    aliases = {"Person": {"Bob Iger": ["Robert Iger"]}}
    context_properties = {"Person": ["description", "birth_date"]}
    conflict_properties = {"Movie": ["year"]}

    result = await store.resolve_entities(
        graph_name="movie-graph",
        strategy="hybrid",
        resolve_property="title",
        dry_run=True,
        merge_threshold=91.0,
        review_threshold=72.0,
        max_candidates_per_entity=7,
        aliases=aliases,
        embedder=embedder,
        llm=llm,
        llm_guidance="Use all supplied profile context.",
        allow_ai_auto_merge=True,
        context_properties=context_properties,
        conflict_properties=conflict_properties,
        context_mode="config_only",
    )

    assert result == {"strategy": "hybrid"}
    assert captured["graph_store"] is store
    assert captured["kwargs"] == {
        "graph_name": "movie-graph",
        "strategy": "hybrid",
        "resolve_property": "title",
        "dry_run": True,
        "merge_threshold": 91.0,
        "review_threshold": 72.0,
        "max_candidates_per_entity": 7,
        "aliases": aliases,
        "embedder": embedder,
        "llm": llm,
        "llm_guidance": "Use all supplied profile context.",
        "allow_ai_auto_merge": True,
        "context_properties": context_properties,
        "conflict_properties": conflict_properties,
        "context_mode": "config_only",
    }


def test_write_graph_document_delegates_to_writer(store, fake_driver):
    from recon_graphrag.extraction.types import (
        ChunkRecord,
        DocumentRecord,
        GraphDocument,
    )

    document = DocumentRecord(
        id="doc-1",
        text_hash="hash",
        graph_name="entity-graph",
        metadata={},
    )
    chunks = [
        ChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            text="hello world",
            index=0,
            graph_name="entity-graph",
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


def test_create_vector_index_runs_create_query(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult([])

    store.create_vector_index(
        name="entity-embeddings",
        label="__Entity__",
        embedding_property="embedding",
        dimensions=1536,
    )
    query_text = "\n".join(fake_driver.session_obj.queries)
    assert "CREATE VECTOR INDEX" in query_text
    assert "dimension: 1536" in query_text


def test_create_fulltext_index_runs_create_query(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult([])

    store.create_fulltext_index(
        name="entity-names",
        label="__Entity__",
        node_properties=["name"],
    )
    query_text = "\n".join(fake_driver.session_obj.queries)
    assert "CREATE TEXT INDEX" in query_text
    assert "__Entity__" in query_text
    assert "name" in query_text


def test_upsert_vectors_runs_unwind_set(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult([])

    store.upsert_vectors(
        node_ids=[1, 2],
        embedding_property="embedding",
        vectors=[[0.1, 0.2], [0.3, 0.4]],
    )
    query_text = "\n".join(fake_driver.session_obj.queries)
    assert "UNWIND $rows" in query_text
    assert "SET n.`embedding` = row.vector" in query_text


def test_upsert_vectors_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.upsert_vectors([1], "embedding", [])


def test_vector_search_runs_vector_procedure(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult(
        [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]
    )

    result = store.vector_search(
        index_name="entity-embeddings",
        query_vector=[0.1, 0.2],
        k=5,
    )
    assert result == [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]
    query_text = fake_driver.session_obj.queries[-1]
    assert "vector_search.search" in query_text


def test_vector_search_overfetches_when_label_filtering(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult([])

    store.vector_search(
        index_name="entity-embeddings",
        query_vector=[0.1, 0.2],
        k=5,
        label="__Entity__",
    )

    query_text = fake_driver.session_obj.queries[-1]
    assert "WHERE node:`__Entity__`" in query_text
    assert fake_driver.session_obj.params[-1]["k"] == 25
    assert fake_driver.session_obj.params[-1]["top_k"] == 5


def test_keyword_search_runs_text_search_procedure(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult(
        [{"id": 1, "score": 0.9}]
    )

    result = store.keyword_search(
        index_name="entity-names",
        query_text="Alice",
        k=5,
    )
    assert result == [{"id": 1, "score": 0.9}]
    query_text = fake_driver.session_obj.queries[-1]
    assert "text_search.search" in query_text
    assert fake_driver.session_obj.params[-1]["query_text"] == '"Alice"'


def test_keyword_search_fields_natural_language_query_for_tantivy(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult([])

    store.keyword_search(
        index_name="entity-names",
        query_text="Who directed Inception?",
        k=5,
    )

    assert fake_driver.session_obj.params[-1]["query_text"] == (
        '"Who" OR "directed" OR "Inception"'
    )


def test_keyword_search_handles_quoted_phrases_and_label_filter(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult([])

    store.keyword_search(
        index_name="entity-names",
        query_text='"Christopher Nolan" + Oppenheimer?',
        k=4,
        label="__Entity__",
    )

    query_text = fake_driver.session_obj.queries[-1]
    assert "WHERE node:`__Entity__`" in query_text
    assert fake_driver.session_obj.params[-1]["query_text"] == (
        '"Christopher Nolan" OR "Oppenheimer"'
    )
    assert fake_driver.session_obj.params[-1]["k"] == 20


def test_tantivy_query_falls_back_to_quoted_field_query_for_punctuation():
    assert _format_tantivy_query("???") == '"???"'


def test_tantivy_query_quotes_boolean_operator_words():
    assert _format_tantivy_query("Nolan and Murphy") == (
        '"Nolan" OR "and" OR "Murphy"'
    )


def test_keyword_search_returns_empty_when_tantivy_rejects_query(store, fake_driver):
    def raise_tantivy_error(query, parameters=None):
        fake_driver.session_obj.queries.append(query.strip())
        fake_driver.session_obj.params.append(parameters or {})
        raise RuntimeError("text_search.search: Tantivy error: Syntax Error")

    fake_driver.session_obj.run = raise_tantivy_error

    result = store.keyword_search(
        index_name="entity-names",
        query_text="Which movies were directed by Christopher Nolan and feature Cillian Murphy?",
        k=5,
    )

    assert result == []


def test_store_community_report_sets_structured_fields(store, fake_driver):
    report = CommunityReport(
        id="report:community:1:0",
        community_id="community:1",
        level=0,
        title="Title",
        summary="Summary",
    )

    store.store_community_report(report, "graph-a")

    query = fake_driver.session_obj.queries[-1]
    params = fake_driver.session_obj.params[-1]
    assert "c.report_json" in query
    assert "c.report_status = 'success'" in query
    assert params["graph_name"] == "graph-a"
    assert params["cid"] == "community:1"
    assert params["title"] == "Title"


def test_mark_community_report_failed_sets_status(store, fake_driver):
    store.mark_community_report_failed("graph-a", "community:1", 0, "bad json")

    query = fake_driver.session_obj.queries[-1]
    params = fake_driver.session_obj.params[-1]
    assert "c.report_status = 'failed'" in query
    assert params == {
        "graph_name": "graph-a",
        "cid": "community:1",
        "level": 0,
        "error": "bad json",
    }


def test_claim_and_citation_reads_are_graph_scoped(store, fake_driver):
    store.get_claims_for_entities("graph-a", ["person:alice"])
    query = fake_driver.session_obj.queries[-1]
    params = fake_driver.session_obj.params[-1]
    assert "graph_name: $graph_name" in query
    assert params["graph_name"] == "graph-a"

    store.resolve_chunk_citations("graph-a", ["chunk:1"])
    query = fake_driver.session_obj.queries[-1]
    params = fake_driver.session_obj.params[-1]
    assert "Chunk {id: cid, graph_name: $graph_name}" in query
    assert "properties(c) AS chunk_metadata" in query
    assert "properties(d) AS document_metadata" in query
    assert params == {"graph_name": "graph-a", "chunk_ids": ["chunk:1"]}


def test_local_context_uses_integer_id(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult(
        [
            {
                "title": "Alice (Person)",
                "relationships": [
                    "Person: Alice -[KNOWS]-> Person: Bob",
                    "Person: Alice -[WORKS_AT]-> Organization: Acme",
                ],
                "source_text": ["text1"],
                "score": 0.9,
            }
        ]
    )

    result = store.fetch_entity_context(
        matches=[{"id": "1", "score": 0.9}],
        mode="local",
    )
    assert len(result) == 1
    assert result[0]["title"] == "Alice (Person)"
    query_text = fake_driver.session_obj.queries[-1]
    assert "id(node) = toInteger(match.id)" in query_text


def test_validate_graph_build_runs_counts(store, fake_driver):
    def pick_result(query: str, params: dict):
        if "MATCH (e:__Entity__) RETURN count(e) AS cnt" in query:
            return FakeResult([{"cnt": 10}])
        if "MATCH (c:Chunk) RETURN count(c) AS cnt" in query:
            return FakeResult([{"cnt": 5}])
        if "FROM_CHUNK" in query:
            return FakeResult([{"cnt": 3}])
        if "MATCH (:__Entity__)-[r]-(:__Entity__)" in query:
            return FakeResult([{"cnt": 2}])
        if "MATCH (c:Community)" in query and "summary" not in query:
            return FakeResult([{"cnt": 7}])
        if "MATCH (c:Community)" in query and "summary" in query:
            return FakeResult([{"cnt": 6}])
        if "MATCH (e:__Entity__)-[r]-(e)" in query:
            return FakeResult([{"cnt": 0}])
        return FakeResult([])

    def fake_run(query, parameters=None):
        fake_driver.session_obj.queries.append(query.strip())
        fake_driver.session_obj.params.append(parameters or {})
        return pick_result(query, parameters or {})

    fake_driver.session_obj.run = fake_run

    result = store.validate_graph_build()
    assert result["entity_count"] == 10
    assert result["chunk_count"] == 5
    assert result["evidence_link_count"] == 3
    assert result["entity_relationship_count"] == 2
    assert result["community_count"] == 7
    assert result["community_summary_count"] == 6
    assert result["entity_self_loop_count"] == 0


def test_create_indexes_runs_uid_constraint(store, fake_driver):
    fake_driver.session_obj._next_result = FakeResult([])

    store.create_indexes(IndexConfig(), embedding_dim=1536)

    query_text = "\n".join(fake_driver.session_obj.queries)
    assert "CREATE CONSTRAINT" in query_text
    assert "c.uid IS UNIQUE" in query_text
