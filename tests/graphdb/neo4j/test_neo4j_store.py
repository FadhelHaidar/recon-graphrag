"""Tests for direct Neo4jGraphStore index and vector operations."""

from typing import cast

import neo4j
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
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    result = store.execute_query("MATCH (n) RETURN count(n) AS cnt")

    assert result == [{"cnt": 42}]


@pytest.mark.asyncio
async def test_resolve_entities_hybrid_forwards_full_advanced_parameter_set(monkeypatch):
    captured = {}

    class CapturingResolver:
        def __init__(self, graph_store):
            captured["graph_store"] = graph_store

        async def resolve_hybrid(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"strategy": "hybrid"}

    monkeypatch.setattr(entity_resolution, "_Neo4jEntityResolver", CapturingResolver)
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver), database="neo4j")
    embedder = object()
    llm = object()
    aliases = {"Person": {"Bob Iger": ["Robert Iger"]}}
    context_properties = {"Person": ["description", "birth_date"]}
    conflict_properties = {"Movie": ["year"]}

    result = await store.resolve_entities_hybrid(
        graph_name="movie-graph",
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


@pytest.mark.asyncio
async def test_resolve_entities_exact_forwards_basic_parameters(monkeypatch):
    captured = {}

    class CapturingResolver:
        def __init__(self, graph_store):
            captured["graph_store"] = graph_store

        async def resolve_exact(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"strategy": "exact"}

    monkeypatch.setattr(entity_resolution, "_Neo4jEntityResolver", CapturingResolver)
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver), database="neo4j")

    result = await store.resolve_entities_exact(
        graph_name="movie-graph",
        resolve_property="title",
        dry_run=True,
    )

    assert result == {"strategy": "exact"}
    assert captured["graph_store"] is store
    assert captured["kwargs"] == {
        "graph_name": "movie-graph",
        "resolve_property": "title",
        "dry_run": True,
    }


@pytest.mark.asyncio
async def test_resolve_entities_normalized_forwards_basic_parameters(monkeypatch):
    captured = {}

    class CapturingResolver:
        def __init__(self, graph_store):
            captured["graph_store"] = graph_store

        async def resolve_normalized(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"strategy": "normalized"}

    monkeypatch.setattr(entity_resolution, "_Neo4jEntityResolver", CapturingResolver)
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver), database="neo4j")

    result = await store.resolve_entities_normalized(
        graph_name="movie-graph",
        resolve_property="title",
        dry_run=True,
    )

    assert result == {"strategy": "normalized"}
    assert captured["graph_store"] is store
    assert captured["kwargs"] == {
        "graph_name": "movie-graph",
        "resolve_property": "title",
        "dry_run": True,
        "conflict_properties": None,
    }


@pytest.mark.asyncio
async def test_resolve_entities_fuzzy_forwards_fuzzy_parameters(monkeypatch):
    captured = {}

    class CapturingResolver:
        def __init__(self, graph_store):
            captured["graph_store"] = graph_store

        async def resolve_fuzzy(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"strategy": "fuzzy"}

    monkeypatch.setattr(entity_resolution, "_Neo4jEntityResolver", CapturingResolver)
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver), database="neo4j")

    result = await store.resolve_entities_fuzzy(
        graph_name="movie-graph",
        resolve_property="title",
        dry_run=True,
        merge_threshold=91.0,
        review_threshold=72.0,
        max_candidates_per_entity=7,
    )

    assert result == {"strategy": "fuzzy"}
    assert captured["graph_store"] is store
    assert captured["kwargs"] == {
        "graph_name": "movie-graph",
        "resolve_property": "title",
        "dry_run": True,
        "merge_threshold": 91.0,
        "review_threshold": 72.0,
        "max_candidates_per_entity": 7,
    }


def test_create_vector_index_executes_direct_cypher_with_escaped_identifiers():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

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
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

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
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.upsert_vectors(["4:a", "4:b"], "embedding", [[0.1], [0.2]])

    query, params = driver.calls[0]
    assert "WHERE elementId(n) = row.id" in query
    assert "db.create.setNodeVectorProperty" in query
    assert params == {
        "rows": [
            {"id": "4:a", "vector": [0.1], "content_fingerprint": None},
            {"id": "4:b", "vector": [0.2], "content_fingerprint": None},
        ],
        "embedding_property": "embedding",
        "model_fingerprint": None,
    }


def test_get_unembedded_entities_selects_missing_or_stale_vectors():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.get_unembedded_entities("g", limit=5, model_fingerprint="model-1")

    query, params = driver.calls[0]
    assert "e.embedding IS NULL" in query
    assert "e.embedding_content_fingerprint" in query
    assert "e.embedding_input_fingerprint" in query
    assert "e.embedding_model_fingerprint" in query
    assert params == {"graph_name": "g", "limit": 5, "model_fingerprint": "model-1"}


def test_upsert_vectors_records_content_and_model_fingerprints():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.upsert_vectors(
        ["4:a"],
        "embedding",
        [[0.1]],
        content_fingerprints=["content-1"],
        model_fingerprint="model-1",
    )

    query, params = driver.calls[-1]
    assert "n.embedding_content_fingerprint" in query
    assert "n.embedding_input_fingerprint" in query
    assert "n.embedding_model_fingerprint" in query
    assert params["rows"] == [
        {"id": "4:a", "vector": [0.1], "content_fingerprint": "content-1"}
    ]
    assert params["model_fingerprint"] == "model-1"


def test_upsert_vectors_rejects_mismatched_lengths():
    store = Neo4jGraphStore(cast(neo4j.Driver, FakeDriver()))

    with pytest.raises(ValueError):
        store.upsert_vectors(["4:a"], "embedding", [])


def test_vector_search_runs_vector_procedure_with_label_filter():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.vector_search(
        index_name="entity-embeddings",
        query_vector=[0.1, 0.2],
        k=5,
        label="__Entity__",
    )

    query, params = driver.calls[-1]
    assert "db.index.vector.queryNodes" in query
    assert "WHERE node:`__Entity__`" in query
    # label filter is post-applied, so we over-fetch before trimming to top_k
    assert params["k"] == 25
    assert params["top_k"] == 5


def test_vector_search_no_filters_fetches_exactly_k():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.vector_search(
        index_name="entity-embeddings",
        query_vector=[0.1, 0.2],
        k=5,
    )

    query, params = driver.calls[-1]
    assert "WHERE" not in query
    assert params["k"] == 5
    assert params["top_k"] == 5


def test_community_report_vector_search_overfetches_and_scopes():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.vector_search_community_reports([0.1], "graph-a", top_k=3, level=0)

    query, params = driver.calls[-1]
    assert "db.index.vector.queryNodes" in query
    assert "c.graph_name = $graph_name" in query
    assert params["candidate_k"] == 15
    assert params["level"] == 0


def test_keyword_search_runs_fulltext_procedure_with_label_filter():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

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


def test_vector_search_graph_filter_overfetches_and_scopes():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.vector_search(
        index_name="entity-embeddings",
        query_vector=[0.1],
        k=5,
        filters={"graph_name": "graph-a"},
    )

    query, params = driver.calls[-1]
    assert "node.graph_name = $graph_name" in query
    assert params["graph_name"] == "graph-a"
    assert params["k"] == 25  # over-fetch before post-filter trims to top_k
    assert params["top_k"] == 5


def test_keyword_search_graph_filter_overfetches_and_scopes():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.keyword_search(
        index_name="entity-names",
        query_text="Alice",
        k=5,
        filters={"graph_name": "graph-a"},
    )

    query, params = driver.calls[-1]
    assert "node.graph_name = $graph_name" in query
    assert params["graph_name"] == "graph-a"
    assert params["k"] == 25
    assert params["top_k"] == 5


def test_store_community_report_sets_structured_fields():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))
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
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

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
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

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
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

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
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    result = store.validate_graph_build()

    assert result == {
        "entity_count": 10,
        "chunk_count": 5,
        "evidence_link_count": 3,
        "entity_relationship_count": 2,
    }


def test_fetch_entity_context_ranks_sources_only_with_a_query_vector():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))
    matches = [{"id": "e1", "score": 0.9}]

    store.fetch_entity_context(matches, graph_name="g")
    unranked_query, unranked_params = driver.calls[-1]
    assert "source_score" not in unranked_query
    assert "ORDER BY chunk.id ASC" in unranked_query
    assert "query_vector" not in unranked_params

    store.fetch_entity_context(
        matches, query_params={"query_vector": [1.0, 0.0]}, graph_name="g"
    )
    ranked_query, ranked_params = driver.calls[-1]
    assert "source_score" in ranked_query
    assert ranked_params["query_vector"] == [1.0, 0.0]


def test_fetch_entity_context_ranks_drift_sources_too():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.fetch_entity_context(
        [{"id": "e1", "score": 0.9}],
        query_params={"query_vector": [1.0, 0.0]},
        mode="drift",
        graph_name="g",
    )

    assert "source_score" in driver.calls[-1][0]


def test_fetch_entity_context_custom_query_wins_over_ranking():
    driver = FakeDriver()
    store = Neo4jGraphStore(cast(neo4j.Driver, driver))

    store.fetch_entity_context(
        [{"id": "e1", "score": 0.9}],
        retrieval_query="RETURN node AS context_row",
        query_params={"query_vector": [1.0, 0.0]},
        graph_name="g",
    )

    query = driver.calls[-1][0]
    assert "RETURN node AS context_row" in query
    assert "source_score" not in query
