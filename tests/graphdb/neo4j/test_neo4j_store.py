"""Tests for direct Neo4jGraphStore index and vector operations."""

import pytest

import recon_graphrag.graphdb.neo4j.entity_resolution as entity_resolution
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore
from recon_graphrag.models.artifacts import CommunityReport


class FakeSession:
    def __init__(self, calls, records):
        self.calls = calls
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, parameters=None):
        self.calls.append((query.strip(), parameters or {}))
        return self.records


class FakeDriver:
    def __init__(self):
        self.calls = []
        self.records = []

    def session(self, **kwargs):
        self.session_kwargs = kwargs
        return FakeSession(self.calls, self.records)


def test_execute_query_translates_records():
    driver = FakeDriver()
    driver.records = [{"cnt": 42}]
    store = Neo4jGraphStore(driver)

    result = store.execute_query("MATCH (n) RETURN count(n) AS cnt")

    assert result == [{"cnt": 42}]


@pytest.mark.asyncio
async def test_resolve_entities_forwards_full_advanced_parameter_set(monkeypatch):
    captured = {}

    class CapturingResolver:
        def __init__(self, graph_store):
            captured["graph_store"] = graph_store

        async def resolve(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"strategy": kwargs["strategy"]}

    monkeypatch.setattr(entity_resolution, "_Neo4jEntityResolver", CapturingResolver)
    driver = FakeDriver()
    store = Neo4jGraphStore(driver, database="neo4j")
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


def test_vector_search_runs_vector_procedure_with_label_filter():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.vector_search(
        index_name="entity-embeddings",
        query_vector=[0.1, 0.2],
        k=5,
        label="__Entity__",
    )

    query, params = driver.calls[-1]
    assert "db.index.vector.queryNodes" in query
    assert "WHERE node:`__Entity__`" in query
    assert params["k"] == 5
    assert params["top_k"] == 5


def test_keyword_search_runs_fulltext_procedure_with_label_filter():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.keyword_search(
        index_name="entity-names",
        query_text="Alice",
        k=5,
        label="__Entity__",
    )

    query, params = driver.calls[-1]
    assert "db.index.fulltext.queryNodes" in query
    assert "WHERE node:`__Entity__`" in query
    assert params["query_text"] == "Alice"


def test_search_communities_overfetches_before_filtering():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.search_communities(
        index_name="community-embeddings",
        query_vector=[0.1, 0.2],
        graph_name="entity-graph",
        top_k=3,
        level=1,
    )

    _, params = driver.calls[-1]
    assert params["k"] == 15
    assert params["top_k"] == 3
    assert params["graph_name"] == "entity-graph"
    assert params["level"] == 1


def test_store_community_report_sets_structured_fields():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)
    report = CommunityReport(
        id="report:community:1:0",
        community_id="community:1",
        level=0,
        title="Title",
        summary="Summary",
    )

    store.store_community_report(report, "graph-a")

    query, params = driver.calls[-1]
    assert "c.report_json" in query
    assert "c.report_status = 'success'" in query
    assert params["graph_name"] == "graph-a"
    assert params["cid"] == "community:1"
    assert params["title"] == "Title"


def test_mark_community_report_failed_sets_status():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.mark_community_report_failed("graph-a", "community:1", 0, "bad json")

    query, params = driver.calls[-1]
    assert "c.report_status = 'failed'" in query
    assert params == {
        "graph_name": "graph-a",
        "cid": "community:1",
        "level": 0,
        "error": "bad json",
    }


def test_claim_and_citation_reads_are_graph_scoped():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.get_claims_for_entities("graph-a", ["person:alice"])
    query, params = driver.calls[-1]
    assert "graph_name: $graph_name" in query
    assert params["graph_name"] == "graph-a"

    store.resolve_chunk_citations("graph-a", ["chunk:1"])
    query, params = driver.calls[-1]
    assert "Chunk {id: cid, graph_name: $graph_name}" in query
    assert "properties(c) AS chunk_metadata" in query
    assert "properties(d) AS document_metadata" in query
    assert params == {"graph_name": "graph-a", "chunk_ids": ["chunk:1"]}


def test_local_context_uses_element_id():
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)

    store.fetch_entity_context(matches=[{"id": "4:a", "score": 0.9}])

    query, params = driver.calls[-1]
    assert "elementId(node) = match.id" in query
    assert params["matches"] == [{"id": "4:a", "score": 0.9}]


def test_validate_graph_build_returns_counts():
    driver = FakeDriver()

    def fake_run(query, parameters=None):
        driver.calls.append((query.strip(), parameters or {}))
        if "MATCH (e:__Entity__) RETURN count(e) AS cnt" in query:
            return [{"cnt": 10}]
        if "MATCH (c:Chunk) RETURN count(c) AS cnt" in query:
            return [{"cnt": 5}]
        if "FROM_CHUNK" in query:
            return [{"cnt": 3}]
        if "MATCH (:__Entity__)-[r]-(:__Entity__)" in query:
            return [{"cnt": 2}]
        return []

    class CountingSession(FakeSession):
        def run(self, query, parameters=None):
            return fake_run(query, parameters)

    driver.session = lambda **kwargs: CountingSession(driver.calls, [])
    store = Neo4jGraphStore(driver)

    result = store.validate_graph_build()

    assert result == {
        "entity_count": 10,
        "chunk_count": 5,
        "evidence_link_count": 3,
        "entity_relationship_count": 2,
    }
