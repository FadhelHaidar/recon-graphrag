"""Tests for GraphBuilderPipeline."""

import asyncio
import logging
import re
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.extraction.chunking import TextChunker
from recon_graphrag.extraction.description_summarizer import DescriptionSummarizer
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from recon_graphrag.graphdb.base import GraphStore
from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore
from recon_graphrag.observability import UsageTrackingLLM, track_usage
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline
from recon_graphrag.pipelines.writer import CypherGraphWriter



class FakeGraphStore:
    def __init__(self):
        self.queries = []
        self.resolve_kwargs = None

    def execute_query(self, query, parameters=None):
        self.queries.append(query.strip())
        return []

    def write_graph_document(self, graph_document):
        return {
            "documents": 1,
            "chunks": len(graph_document.chunks),
            "entities": len(graph_document.entities),
            "relationships": len(graph_document.relationships),
            "evidence_links": len(graph_document.evidence_links),
        }

    def backfill_descriptions(self):
        pass

    async def resolve_entities_exact(self, graph_name="entity-graph", resolve_property="name", dry_run=False, **kwargs):
        self.resolve_kwargs = {"strategy": "exact", "graph_name": graph_name, **kwargs}
        return {"skipped": False, "merged_groups": 0, "strategy": "exact"}

    async def resolve_entities_normalized(self, graph_name="entity-graph", resolve_property="name", dry_run=False, **kwargs):
        self.resolve_kwargs = {"strategy": "normalized", "graph_name": graph_name, **kwargs}
        return {"skipped": False, "merged_groups": 0, "strategy": "normalized"}

    async def resolve_entities_fuzzy(self, graph_name="entity-graph", resolve_property="name", dry_run=False,
                                     merge_threshold=95.0, review_threshold=85.0, max_candidates_per_entity=20, **kwargs):
        self.resolve_kwargs = {"strategy": "fuzzy", "graph_name": graph_name, **kwargs}
        return {"skipped": False, "merged_groups": 0, "strategy": "fuzzy"}

    async def resolve_entities_hybrid(self, graph_name="entity-graph", resolve_property="name", dry_run=False,
                                      merge_threshold=95.0, review_threshold=85.0, max_candidates_per_entity=20,
                                      aliases=None, embedder=None, llm=None, llm_guidance=None,
                                      allow_ai_auto_merge=False, context_properties=None, conflict_properties=None,
                                      context_mode="safe_defaults", **kwargs):
        self.resolve_kwargs = {
            "strategy": "hybrid",
            "graph_name": graph_name,
            "embedder": embedder,
            "llm": llm,
            "aliases": aliases,
            "llm_guidance": llm_guidance,
            "allow_ai_auto_merge": allow_ai_auto_merge,
            "context_properties": context_properties,
            "conflict_properties": conflict_properties,
            "context_mode": context_mode,
            **kwargs,
        }
        return {"skipped": False, "merged_groups": 0, "strategy": "hybrid"}

    def get_unembedded_entities(self, graph_name, limit=500):
        return []

    def get_entities_needing_summary(self, graph_name, limit=500):
        return []

    def get_relationships_needing_summary(self, graph_name, limit=500):
        return []

    def persist_entity_summaries(self, graph_name, summaries):
        pass

    def persist_relationship_summaries(self, graph_name, summaries):
        pass

    def upsert_vectors(self, node_ids, embedding_property, vectors):
        pass

    def validate_graph_build(self):
        return {
            "entity_count": 2,
            "chunk_count": 1,
            "evidence_link_count": 2,
            "entity_relationship_count": 1,
        }

    def create_indexes(self, config=None, embedding_dim=1536):
        pass

    def drop_indexes(self, config=None):
        pass

    def create_vector_index(self, name, label, embedding_property, dimensions, similarity_fn="cosine"):
        pass

    def create_fulltext_index(self, name, label, node_properties):
        pass

    def vector_search(self, index_name, query_vector, k, label=None, filters=None):
        return []

    def keyword_search(self, index_name, query_text, k, label=None, filters=None):
        return []

    def fetch_entity_context(self, matches, retrieval_query=None, query_params=None, mode="local", graph_name=None):
        return []

    def detect_communities(self, graph_name, relationship_types=None, max_levels=3, gamma=1.0,
                           theta=0.01, tolerance=1e-4, relationship_weight_property=None,
                           random_seed=None, entity_label="__Entity__", community_label="Community"):
        return []

    def get_communities(self, graph_name, level=None):
        return []

    def get_community_stats(self, graph_name):
        return []

    def store_community_report(self, report, graph_name):
        pass

    def mark_community_report_failed(self, graph_name, community_id, level, error):
        pass

    def get_community_ranked_context(self, graph_name, community_id, level=0):
        return []

    def get_child_community_reports(self, graph_name, community_id, level, child_level):
        return []

    def get_claims_for_entities(self, graph_name, entity_ids):
        return []

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        return []

    def get_community_reports_by_keys(self, graph_name, keys, top_k):
        return []

    def get_unembedded_community_reports(self, graph_name, limit=500):
        return []

    def upsert_community_report_vectors(self, node_ids, vectors, graph_name, levels):
        pass

    def vector_search_community_reports(self, query_vector, graph_name, top_k=3, level=None):
        return []

    def get_entity_count(self):
        return 2

    def get_chunk_count(self):
        return 1

    def get_evidence_link_count(self):
        return 2

    def get_relationship_count(self):
        return 1


class MyGraphStore(FakeGraphStore):
    pass


class RealWriterFakeGraphStore(FakeGraphStore):
    """FakeGraphStore whose writes go through the real CypherGraphWriter."""

    def __init__(self):
        super().__init__()
        self.params = []

    def execute_query(self, query, parameters=None):
        self.params.append(parameters or {})
        return super().execute_query(query, parameters)

    def write_graph_document(self, graph_document):
        return CypherGraphWriter(cast(GraphStore, self)).write_graph_document(
            graph_document
        )


def _write_rows(store, query_fragment):
    for query, parameters in zip(store.queries, store.params, strict=True):
        if query_fragment in query:
            return next(iter(parameters.values()))
    raise AssertionError(f"no writer query matching {query_fragment!r}")


class CustomBackend(FakeGraphStore):
    pass


@pytest.fixture
def movie_schema():
    return GraphSchema(
        node_types=[
            NodeType(
                label="Person",
                properties=[PropertyType(name="name", type="STRING")],
            ),
            NodeType(
                label="Movie",
                properties=[
                    PropertyType(name="title", type="STRING"),
                    PropertyType(name="year", type="STRING"),
                ],
                identity_property="title",
            ),
        ],
        relationship_types=[
            RelationshipType(label="DIRECTED"),
            RelationshipType(label="ACTED_IN"),
        ],
        patterns=[
            ("Person", "DIRECTED", "Movie"),
            ("Person", "ACTED_IN", "Movie"),
        ],
    )


@pytest.fixture
def fake_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='''
    {
        "nodes": [
            {"id": "p1", "label": "Person", "properties": {"name": "Alice"}},
            {"id": "m1", "label": "Movie", "properties": {"title": "Inception", "year": "2010"}}
        ],
        "relationships": [
            {"source_id": "p1", "target_id": "m1", "type": "DIRECTED"}
        ]
    }
    ''', usage=None))
    return llm


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(return_value=[0.1] * 1536)
    return embedder


@pytest.fixture
def fake_writer():
    writer = MagicMock()
    writer.write_graph_document = MagicMock(return_value={
        "documents": 1,
        "chunks": 1,
        "entities": 2,
        "relationships": 1,
        "evidence_links": 2,
    })
    return writer


@pytest.mark.asyncio
async def test_build_from_text_orchestration(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    result = await pipeline.build_from_text(
        "Alice directed Inception in 2010.",
        metadata={"source": "test"},
        schema=movie_schema,
    )

    assert "extraction" in result
    assert "validation" in result
    assert result["extraction"]["chunks"] > 0
    fake_writer.write_graph_document.assert_called_once()
    written_doc = fake_writer.write_graph_document.call_args[0][0]
    assert len(written_doc.entities) > 0
    assert len(written_doc.relationships) > 0


@pytest.mark.asyncio
async def test_build_from_text_logs_derived_graph_store_name(
    movie_schema, fake_llm, fake_embedder, fake_writer, caplog
):
    store = MyGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    with caplog.at_level(logging.INFO):
        await pipeline.build_from_text(
            "Alice directed Inception.", schema=movie_schema
        )

    assert "write complete to My:" in caplog.text


@pytest.mark.asyncio
async def test_build_from_documents_with_page_windows(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    pages = ["Page one", "Page two", "Page three"]
    results = await pipeline.build_from_documents(
        [{"pages": pages, "metadata": {"source": "test"}}],
        schema=movie_schema,
        window_size=2,
        window_overlap=1,
    )

    assert len(results) == 1
    result = results[0]
    assert "extraction" in result
    assert result["extraction"]["chunks"] == 2  # windows: 1-2, 2-3


@pytest.mark.asyncio
async def test_build_from_documents_preserves_page_metadata(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    pages = [
        {"text": "Page one", "metadata": {"record_id": "page-1"}},
        {"text": "Page two", "metadata": {"record_id": "page-2"}},
        {"text": "Page three", "metadata": {"record_id": "page-3"}},
    ]
    await pipeline.build_from_documents(
        [{"pages": pages, "metadata": {"source": "document-source", "collection": "movies"}}],
        schema=movie_schema,
        window_size=2,
        window_overlap=1,
    )

    graph_document = fake_writer.write_graph_document.call_args.args[0]
    assert graph_document.document.metadata["source"] == "document-source"
    assert graph_document.chunks[0].text == "Page one\n\nPage two"
    assert graph_document.chunks[0].metadata["record_ids"] == ["page-1", "page-2"]
    assert graph_document.chunks[1].metadata["record_ids"] == ["page-2", "page-3"]


@pytest.mark.asyncio
async def test_build_from_documents_mixed_text_and_pages(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    results = await pipeline.build_from_documents(
        [
            {"text": "Alice directed Inception.", "metadata": {"source": "text-doc"}},
            {
                "pages": [
                    {"text": "Page one", "metadata": {"record_id": "page-1"}},
                    {"text": "Page two", "metadata": {"record_id": "page-2"}},
                ],
                "metadata": {"source": "page-doc"},
            },
        ],
        schema=movie_schema,
        window_size=2,
        window_overlap=1,
    )

    assert len(results) == 2
    assert fake_writer.write_graph_document.call_count == 2
    assert results[0]["extraction"]["chunks"] == 1
    assert results[1]["extraction"]["chunks"] == 1


@pytest.mark.asyncio
async def test_build_from_documents_token_usage_is_per_document(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    """token_usage on each result must reflect only that document's own
    calls, not the whole batch's total stamped onto every result."""
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=False,
    )

    long_text = "Alice directed Inception in 2010 with Christopher Nolan. " * 5
    short_text = "Bob acted in a movie."

    results = await pipeline.build_from_documents(
        [
            {"text": long_text, "metadata": {"source": "long-doc"}},
            {"text": short_text, "metadata": {"source": "short-doc"}},
        ],
        schema=movie_schema,
        chunk_size=15,
        chunk_overlap=5,
        chunk_unit="characters",
        max_gleanings=0,
    )

    assert len(results) == 2
    long_result, short_result = results
    assert long_result["extraction"]["chunks"] > short_result["extraction"]["chunks"]

    long_calls = long_result["token_usage"]["totals"]["calls"]
    short_calls = short_result["token_usage"]["totals"]["calls"]
    assert long_calls == long_result["extraction"]["chunks"]
    assert short_calls == short_result["extraction"]["chunks"]
    assert long_calls != short_calls


@pytest.mark.asyncio
async def test_build_from_documents_envelope_validation(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    with pytest.raises(ValueError, match="must be a dict"):
        await pipeline.build_from_documents(["not a dict"], schema=movie_schema)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="both 'text' and 'pages'"):
        await pipeline.build_from_documents(
            [{"text": "x", "pages": ["p"]}], schema=movie_schema
        )

    with pytest.raises(ValueError, match="either 'text' or 'pages'"):
        await pipeline.build_from_documents([{"metadata": {}}], schema=movie_schema)

    with pytest.raises(ValueError, match="'text' must be a string"):
        await pipeline.build_from_documents([{"text": 123}], schema=movie_schema)

    with pytest.raises(ValueError, match="'pages' must be a list"):
        await pipeline.build_from_documents(
            [{"pages": "not-a-list"}], schema=movie_schema
        )

    with pytest.raises(ValueError, match="'metadata' must be a dict"):
        await pipeline.build_from_documents(
            [{"text": "x", "metadata": "bad"}], schema=movie_schema
        )

    with pytest.raises(ValueError, match="schema must be a GraphSchema"):
        await pipeline.build_from_documents(
            [{"text": "x"}], schema=None  # type: ignore[arg-type]
        )

    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
async def test_build_from_documents_with_token_chunking(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    class FakeTokenCounter:
        def count(self, text: str) -> int:
            return len(text.split())

        def truncate(self, text: str, max_tokens: int) -> str:
            words = text.split()
            return " ".join(words[:max_tokens])

    results = await pipeline.build_from_documents(
        [
            {
                "text": "Alice directed Inception in 2010 with Christopher Nolan. " * 3,
                "metadata": {"source": "token-test"},
            }
        ],
        schema=movie_schema,
        chunk_size=5,
        chunk_overlap=1,
        chunk_unit="tokens",
        token_counter=FakeTokenCounter(),
    )

    assert len(results) == 1
    assert results[0]["extraction"]["chunks"] > 1
    fake_writer.write_graph_document.assert_called_once()


@pytest.mark.asyncio
async def test_build_from_text_with_tiktoken_default(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    result = await pipeline.build_from_text(
        "Alice directed Inception in 2010 with Christopher Nolan. " * 3,
        schema=movie_schema,
        chunk_size=5,
        chunk_overlap=1,
        chunk_unit="tokens",
    )

    assert result["extraction"]["chunks"] > 1
    fake_writer.write_graph_document.assert_called_once()


def test_make_document_id_with_source():
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
    )
    doc_id = pipeline._make_document_id("hello", {"source": "My Doc"})
    assert re.fullmatch(r"doc:[0-9a-f]{16}", doc_id), doc_id


def test_make_document_id_slash_and_dash_sources_are_distinct():
    """Slug slugs collapsed "A/B" and "A-B" to the same id; hashing must not."""
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
    )
    slash_id = pipeline._make_document_id("t", {"source": "A/B"})
    dash_id = pipeline._make_document_id("t", {"source": "A-B"})
    assert slash_id != dash_id


def test_make_document_id_non_ascii_sources_are_distinct():
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
    )
    arabic_id = pipeline._make_document_id("t", {"source": "نص عربي"})
    cjk_id = pipeline._make_document_id("t", {"source": "中文文档"})
    assert arabic_id != cjk_id
    assert re.fullmatch(r"doc:[0-9a-f]{16}", arabic_id)


def test_make_document_id_same_source_is_stable():
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
    )
    first = pipeline._make_document_id("text one", {"source": "Report"})
    second = pipeline._make_document_id("text two", {"source": "Report"})
    # same source → same document identity even as text differs between chunks
    assert first == second


def test_make_document_id_explicit_document_id_wins():
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
    )
    assert (
        pipeline._make_document_id("hello", {"source": "My Doc", "document_id": "doc:test"})
        == "doc:test"
    )


@pytest.mark.parametrize(
    ("store_cls", "expected_name"),
    [
        (Neo4jGraphStore, "Neo4j"),
        (MemgraphGraphStore, "Memgraph"),
    ],
)
def test_graph_store_name_uses_backend_class_name(store_cls, expected_name):
    pipeline = GraphBuilderPipeline(
        graph_store=store_cls(driver=MagicMock()),
        llm=MagicMock(),
        embedder=MagicMock(),
    )

    assert pipeline._graph_store_name() == expected_name


def test_graph_store_name_trims_graph_store_suffix():
    pipeline = GraphBuilderPipeline(
        graph_store=MyGraphStore(),
        llm=MagicMock(),
        embedder=MagicMock(),
    )

    assert pipeline._graph_store_name() == "My"


def test_graph_store_name_keeps_custom_class_without_graph_store_suffix():
    pipeline = GraphBuilderPipeline(
        graph_store=CustomBackend(),
        llm=MagicMock(),
        embedder=MagicMock(),
    )

    assert pipeline._graph_store_name() == "CustomBackend"


def test_make_document_id_without_source():
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
    )
    doc_id = pipeline._make_document_id("hello world", {})
    assert doc_id.startswith("doc:")
    assert len(doc_id) == 20  # doc: + 16 hex chars


def test_hash_text():
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
    )
    h1 = pipeline._hash_text("hello")
    h2 = pipeline._hash_text("hello")
    h3 = pipeline._hash_text("world")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


@pytest.mark.asyncio
async def test_hybrid_entity_resolution_forwards_llm_and_embedder(movie_schema):
    store = FakeGraphStore()
    llm = MagicMock()
    embedder = MagicMock()
    aliases = {"Person": {"John Smith": ["Jon Smith"]}}
    guidance = "Only merge people with clear evidence."
    context_properties = {"Person": ["description", "department"]}
    conflict_properties = {"Movie": ["year"]}
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=llm,
        embedder=embedder,
        entity_resolution_strategy="hybrid",
        entity_resolution_aliases=aliases,
        entity_resolution_llm_guidance=guidance,
        entity_resolution_context_properties=context_properties,
        entity_resolution_conflict_properties=conflict_properties,
        allow_ai_auto_merge=True,
    )

    await pipeline._resolve_entities()

    # pipeline.llm is llm auto-wrapped with token-usage tracking (default
    # on); entity resolution should forward whatever the pipeline holds.
    assert store.resolve_kwargs == {
        "strategy": "hybrid",
        "graph_name": "entity-graph",
        "embedder": embedder,
        "llm": pipeline.llm,
        "aliases": aliases,
        "llm_guidance": guidance,
        "allow_ai_auto_merge": True,
        "context_properties": context_properties,
        "conflict_properties": conflict_properties,
        "context_mode": "safe_defaults",
    }


@pytest.mark.asyncio
async def test_extract_chunks_concurrently(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    text = "Alice directed Inception in 2010 with Christopher Nolan. " * 3
    result = await pipeline.build_from_text(
        text,
        metadata={"source": "test"},
        schema=movie_schema,
        extraction_concurrency=3,
        max_gleanings=0,
        chunk_size=15,
        chunk_overlap=5,
    )

    assert result["extraction"]["chunks"] > 1
    assert fake_llm.ainvoke.call_count == result["extraction"]["chunks"]
    fake_writer.write_graph_document.assert_called_once()


@pytest.mark.asyncio
async def test_concurrency_limit_respected(
    movie_schema, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    chunker = TextChunker(chunk_size=15, chunk_overlap=5)
    chunks = chunker.chunk_text(
        "Alice directed Inception in 2010 with Christopher Nolan. " * 3,
        document_id="doc:test",
        metadata={},
    )
    assert len(chunks) >= 4

    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def tracked_invoke(prompt):
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        async with lock:
            active -= 1
        return MagicMock(content='{"nodes": [], "relationships": []}')

    pipeline.llm.inner.ainvoke = AsyncMock(side_effect=tracked_invoke)  # type: ignore[attr-defined]

    await pipeline._extract_and_write_chunks(
        document_id="doc:test",
        text_hash="hash",
        chunks=chunks,
        metadata={},
        schema=movie_schema,
        extraction_concurrency=2,
        max_gleanings=1,
        extract_claims=False,
    )

    assert max_active <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("all_fail", [False, True], ids=["one-of-two", "all"])
async def test_incomplete_extraction_never_calls_writer(
    movie_schema, fake_embedder, fake_writer, all_fail
):
    llm = MagicMock()
    responses = [
        RuntimeError("private provider error"),
        RuntimeError("private provider error") if all_fail else
        MagicMock(content='{"nodes": [], "relationships": []}', usage=None),
    ]
    if not all_fail:
        responses.reverse()  # One completed chunk must still not be written.
    started = 0
    both_started = asyncio.Event()

    async def invoke(prompt):
        nonlocal started
        index = started
        started += 1
        if started == 2:
            both_started.set()
        await both_started.wait()
        response = responses[index]
        if isinstance(response, Exception):
            raise response
        return response

    llm.ainvoke = AsyncMock(side_effect=invoke)
    # Type-only cast preserves the partial duck store's runtime capabilities.
    pipeline = GraphBuilderPipeline(
        cast(GraphStore, FakeGraphStore()), llm, fake_embedder, graph_writer=fake_writer,
        perform_entity_resolution=False, embed_entities=False,
    )
    with pytest.raises(RuntimeError, match=r"Extraction failed .*document=.*chunk="):
        await pipeline.build_from_documents(
            [{"pages": ["first", "second"]}], schema=movie_schema,
            window_size=1, window_overlap=0, max_gleanings=0,
        )
    assert llm.ainvoke.await_count == 2
    fake_writer.write_graph_document.assert_not_called()


def test_build_from_text_stamps_graph_name_on_all_records():
    """Characterization: every assembled record carries the pipeline graph_name."""
    from recon_graphrag.extraction.chunking import TextChunk
    from recon_graphrag.extraction.types import (
        ExtractedNode,
        ExtractedRelationship,
        GraphExtraction,
    )

    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
        graph_name="custom-graph",
    )

    chunks = [
        TextChunk(id="c1", text="Alice knows Bob", index=0, metadata={}),
        TextChunk(id="c2", text="Alice knows Bob", index=1, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="p1", target_id="p2", type="KNOWS", properties={"weight": 1.0}
                )
            ],
        ),
        "c2": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="p1", target_id="p2", type="KNOWS", properties={"weight": 1.0}
                )
            ],
        ),
    }

    doc = pipeline.assembler.assemble(
        document_id="doc:test",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name=pipeline.graph_name,
    )

    assert doc.document.graph_name == "custom-graph"
    assert all(c.graph_name == "custom-graph" for c in doc.chunks)
    assert all(e.graph_name == "custom-graph" for e in doc.entities)
    assert all(r.graph_name == "custom-graph" for r in doc.relationships)
    assert all(link.graph_name == "custom-graph" for link in doc.evidence_links)


def test_pipeline_rerun_does_not_inflate_assembled_records():
    """Characterization: assembler-level rerun produces the same record counts."""
    from recon_graphrag.extraction.chunking import TextChunk
    from recon_graphrag.extraction.types import (
        ExtractedNode,
        ExtractedRelationship,
        GraphExtraction,
    )

    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
    )

    chunks = [
        TextChunk(id="c1", text="Alice knows Bob", index=0, metadata={}),
    ]
    extractions = {
        "c1": GraphExtraction(
            nodes=[
                ExtractedNode(id="p1", label="Person", properties={"name": "Alice"}),
                ExtractedNode(id="p2", label="Person", properties={"name": "Bob"}),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="p1", target_id="p2", type="KNOWS", properties={"weight": 1.0}
                )
            ],
        ),
    }

    first = pipeline.assembler.assemble(
        document_id="doc:test",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name=pipeline.graph_name,
    )
    second = pipeline.assembler.assemble(
        document_id="doc:test",
        text_hash="hash",
        chunks=chunks,
        chunk_extractions=extractions,
        metadata={},
        graph_name=pipeline.graph_name,
    )

    assert len(first.entities) == len(second.entities) == 2
    assert len(first.relationships) == len(second.relationships) == 1
    assert len(first.evidence_links) == len(second.evidence_links) == 2


@pytest.mark.asyncio
async def test_build_from_text_uses_custom_string_prompts(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=False,
        extraction_prompt="CUSTOM EXTRACTION INSTRUCTION",
    )

    await pipeline.build_from_text(
        "Alice directed Inception.",
        metadata={"source": "test"},
        schema=movie_schema,
        max_gleanings=0,
    )

    prompt = fake_llm.ainvoke.call_args[0][0]
    assert "CUSTOM EXTRACTION INSTRUCTION" in prompt


def test_pipeline_configures_prompt_builder_from_strings():
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
        extraction_prompt="E",
        assessment_prompt="A",
        continuation_prompt="C",
        claim_prompt="CL",
        entity_summary_prompt="ES",
        relationship_summary_prompt="RS",
    )

    builder = pipeline.prompt_builder
    assert builder.extraction_prompt == "E"
    assert builder.assessment_prompt == "A"
    assert builder.continuation_prompt == "C"
    assert builder.claim_prompt == "CL"
    assert builder.entity_summary_prompt == "ES"
    assert builder.relationship_summary_prompt == "RS"
    assert pipeline.extractor.prompt_builder is builder


@pytest.mark.asyncio
async def test_description_summarizer_receives_configured_prompt_builder(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=True,
        entity_summary_prompt="CUSTOM ENTITY SUMMARY",
        relationship_summary_prompt="CUSTOM RELATIONSHIP SUMMARY",
    )

    # The summarizer receives the builder configured from the string prompts.
    summarizer = DescriptionSummarizer(
        pipeline.llm,
        pipeline.graph_store,
        pipeline.graph_name,
        prompt_builder=pipeline.prompt_builder,
    )
    assert summarizer.prompts.entity_summary_prompt == "CUSTOM ENTITY SUMMARY"
    assert summarizer.prompts.relationship_summary_prompt == "CUSTOM RELATIONSHIP SUMMARY"


def test_token_usage_tracking_default_on(fake_embedder, fake_writer):
    """Token-usage tracking is on by default and can be opted out of; a
    pre-wrapped LLM is never double-wrapped (track_usage's idempotency)."""
    store = FakeGraphStore()

    default_pipeline = GraphBuilderPipeline(
        graph_store=store, llm=MagicMock(), embedder=fake_embedder, graph_writer=fake_writer,
    )
    assert isinstance(default_pipeline.llm, UsageTrackingLLM)
    # The extractor must share the exact same wrapped instance, not the
    # original unwrapped llm, or extraction calls go untracked.
    assert default_pipeline.extractor.llm is default_pipeline.llm

    opted_out_pipeline = GraphBuilderPipeline(
        graph_store=store, llm=MagicMock(), embedder=fake_embedder, graph_writer=fake_writer,
        track_token_usage=False,
    )
    assert not isinstance(opted_out_pipeline.llm, UsageTrackingLLM)

    pre_wrapped = track_usage(MagicMock())
    already_wrapped_pipeline = GraphBuilderPipeline(
        graph_store=store, llm=pre_wrapped, embedder=fake_embedder, graph_writer=fake_writer,
    )
    assert already_wrapped_pipeline.llm is pre_wrapped



@pytest.fixture
def extraction_safety_pipeline(fake_llm, fake_embedder, fake_writer):
    # Type-only cast preserves the partial duck store's runtime capabilities.
    return GraphBuilderPipeline(
        cast(GraphStore, FakeGraphStore()), fake_llm, fake_embedder, graph_writer=fake_writer,
        perform_entity_resolution=False, embed_entities=False,
        summarize_descriptions=False, track_token_usage=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["initial", "assessment", "continuation", "claims"])
@pytest.mark.parametrize("provider_error_text", [
    "nodes[0].properties: expected an object",
    "nodes[0].properties.private_source: raw provider secret",
])
async def test_required_extraction_pass_failure_is_safe_and_prevents_write(
    extraction_safety_pipeline, movie_schema, fake_writer, failure, caplog,
    provider_error_text,
):
    import traceback

    pipeline = extraction_safety_pipeline
    initial = MagicMock(content='{"nodes": [{"id": "p", "label": "Person"}]}')
    prefix = {
        "initial": [], "assessment": [initial],
        "continuation": [initial, MagicMock(content="yes")],
        "claims": [initial],
    }[failure]
    # Even a provider error matching a parser error must not be copied.
    pipeline.llm.ainvoke = AsyncMock(side_effect=[*prefix, ValueError(provider_error_text)])
    with pytest.raises(RuntimeError) as caught:
        await pipeline.build_from_text(
            "sensitive source text", metadata={"source": "private-source-name"},
            schema=movie_schema, max_gleanings=0 if failure == "claims" else 1,
            extract_claims=failure == "claims",
        )
    displayed = "".join(traceback.format_exception(caught.value))
    assert "document=" in str(caught.value)
    assert "chunk=1" in str(caught.value)
    expected = "Claims extraction failed" if failure == "claims" else "Extraction failed"
    assert expected in str(caught.value)
    assert "field=" not in str(caught.value)
    for private in (provider_error_text, "sensitive source text", "private-source-name", "private_source"):
        assert private not in displayed
        assert private not in caplog.text
    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["name", "unit-price", "年齢"])
async def test_validation_failure_has_safe_document_chunk_and_field_context(
    extraction_safety_pipeline, fake_writer, field
):
    import json

    pipeline = extraction_safety_pipeline
    schema = GraphSchema([NodeType("Person", properties=[PropertyType(field, "INTEGER")])], [])
    pipeline.llm.ainvoke = AsyncMock(return_value=MagicMock(content=json.dumps({
        "nodes": [{"id": "p", "label": "Person", "properties": {field: {"private": "value"}}}],
    })))
    with pytest.raises(ValueError) as caught:
        await pipeline.build_from_text("source", schema=schema, max_gleanings=0)
    assert "document=" in str(caught.value)
    assert f"chunk=1 field=nodes[0].properties.{field}: expected INTEGER" in str(caught.value)
    assert "private" not in str(caught.value)
    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 3])
async def test_failed_batch_retains_only_fully_committed_document(
    extraction_safety_pipeline, movie_schema, fake_writer, concurrency
):
    pipeline = extraction_safety_pipeline
    committed = asyncio.Event()
    original_write = fake_writer.write_graph_document.side_effect

    def write(document):
        committed.set()
        return {"documents": 1, "extension_stat": 9}

    fake_writer.write_graph_document.side_effect = write

    async def extract(*, text, **kwargs):
        from recon_graphrag.extraction.types import GraphExtraction
        if text == "complete":
            return GraphExtraction()
        await committed.wait()
        raise RuntimeError("private error")

    pipeline.extractor.extract = AsyncMock(side_effect=extract)
    pipeline._finalize_graph = AsyncMock()
    with pytest.raises(RuntimeError, match="Extraction failed"):
        await pipeline.build_from_documents(
            [{"text": "complete"}, {"text": "incomplete"}], schema=movie_schema,
            document_concurrency=concurrency, max_gleanings=0,
        )
    fake_writer.write_graph_document.assert_called_once()
    assert fake_writer.write_graph_document.call_args.args[0].chunks[0].text == "complete"
    pipeline._finalize_graph.assert_not_awaited()
    fake_writer.write_graph_document.side_effect = original_write


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["chunks", "documents"])
@pytest.mark.parametrize("stop", ["failure", "cancel", "provider-cancel"])
async def test_owned_extraction_tasks_are_drained_and_queued_calls_stop(
    extraction_safety_pipeline, movie_schema, fake_writer, scope, stop
):
    pipeline = extraction_safety_pipeline
    both_started = asyncio.Event()
    trigger = asyncio.Event()
    cleaned = asyncio.Event()
    provider_tasks = []

    async def invoke(prompt):
        provider_tasks.append(asyncio.current_task())
        if len(provider_tasks) == 1:
            await trigger.wait()
            if stop == "provider-cancel":
                raise asyncio.CancelledError
            raise RuntimeError("private provider error")
        both_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # Awaited cleanup must finish before the build raises/returns.
            await asyncio.sleep(0)
            cleaned.set()

    pipeline.llm.ainvoke = AsyncMock(side_effect=invoke)
    if scope == "chunks":
        documents = [{"pages": ["first", "second", "queued"]}]
    else:
        documents = [{"text": text} for text in ("first", "second", "queued")]
    build = asyncio.create_task(pipeline.build_from_documents(
        documents, schema=movie_schema, max_gleanings=0,
        extraction_concurrency=2, document_concurrency=2,
        window_size=1, window_overlap=0,
    ))
    await asyncio.wait_for(both_started.wait(), timeout=2)
    if stop == "cancel":
        build.cancel()
    else:
        trigger.set()
    expected = RuntimeError if stop == "failure" else asyncio.CancelledError
    with pytest.raises(expected):
        await asyncio.wait_for(build, timeout=2)
    assert cleaned.is_set()
    assert len(provider_tasks) == 2  # Third chunk/document stayed queued.
    assert all(task.done() for task in provider_tasks)
    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["nodes", "relationships"])
@pytest.mark.parametrize("extraction_stage", ["initial", "continuation"])
async def test_parser_property_failure_preserves_safe_field_context(
    extraction_safety_pipeline, movie_schema, fake_writer, kind, extraction_stage
):
    import json

    pipeline = extraction_safety_pipeline
    item = (
        {"id": "private-id", "label": "Person"} if kind == "nodes" else
        {"source_id": "private-id", "target_id": "private-id", "type": "DIRECTED"}
    )
    invalid = MagicMock(content=json.dumps({kind: [{**item, "properties": []}]}))
    prefix = [] if extraction_stage == "initial" else [
        MagicMock(content='{"nodes": []}'), MagicMock(content="yes"),
    ]
    pipeline.llm.ainvoke = AsyncMock(side_effect=[*prefix, invalid])
    with pytest.raises(ValueError) as caught:
        await pipeline.build_from_text(
            "private-source-text", metadata={"source": "private-source-name"},
            schema=movie_schema, max_gleanings=0 if extraction_stage == "initial" else 1,
        )
    message = str(caught.value)
    assert "Validation failed (document=" in message
    assert f"chunk=1 field={kind}[0].properties: expected an object" in message
    for private in ("private-id", "private-source-text", "private-source-name"):
        assert private not in message
    fake_writer.write_graph_document.assert_not_called()


@pytest.mark.asyncio
async def test_batch_gate_closes_before_failed_document_finishes_chunk_cleanup(
    extraction_safety_pipeline, movie_schema, fake_writer, monkeypatch
):
    from recon_graphrag.extraction.types import GraphExtraction

    pipeline = extraction_safety_pipeline
    sibling_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()
    completed_document_finished = asyncio.Event()
    provider_tasks = []
    submitted = []
    ingest_text = pipeline._ingest_text

    async def ingest(*args, **kwargs):
        if kwargs["text"] == "B":
            # B's document task releases the semaphore before this callback.
            # C gets its turn at the batch gate before the event waiter resumes.
            task = asyncio.current_task()
            assert task is not None
            task.add_done_callback(lambda _: completed_document_finished.set())
        return await ingest_text(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_ingest_text", ingest)

    async def extract(*, text, **kwargs):
        provider_tasks.append(asyncio.current_task())
        submitted.append(text)
        if text == "A-fails":
            await sibling_started.wait()
            raise RuntimeError("private provider error")
        if text == "A-drains":
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await cleanup_release.wait()
                cleanup_finished.set()
        if text == "B":
            await cleanup_started.wait()
        return GraphExtraction()

    pipeline.extractor.extract = AsyncMock(side_effect=extract)
    pipeline._finalize_graph = AsyncMock()
    build = asyncio.create_task(pipeline.build_from_documents(
        [{"pages": ["A-fails", "A-drains"]}, {"text": "B"}, {"text": "C"}],
        schema=movie_schema, document_concurrency=2, extraction_concurrency=2,
        max_gleanings=0, window_size=1, window_overlap=0,
    ))
    try:
        await asyncio.wait_for(completed_document_finished.wait(), timeout=2)
        assert cleanup_started.is_set()
        assert not cleanup_finished.is_set()
        assert not build.done()
        assert submitted == ["A-fails", "A-drains", "B"]
        fake_writer.write_graph_document.assert_called_once()
        assert fake_writer.write_graph_document.call_args.args[0].chunks[0].text == "B"
    finally:
        cleanup_release.set()
        with pytest.raises(RuntimeError, match="Extraction failed"):
            await asyncio.wait_for(build, timeout=2)
    assert cleanup_finished.is_set()
    assert all(task.done() for task in provider_tasks)
    fake_writer.write_graph_document.assert_called_once()
    pipeline._finalize_graph.assert_not_awaited()
