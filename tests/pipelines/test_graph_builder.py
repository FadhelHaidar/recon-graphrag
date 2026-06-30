"""Tests for GraphBuilderPipeline."""

from unittest.mock import AsyncMock, MagicMock

import asyncio
import pytest

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from recon_graphrag.extraction.chunking import TextChunker
from recon_graphrag.graphdb.memgraph.store import MemgraphGraphStore
from recon_graphrag.graphdb.neo4j.store import Neo4jGraphStore
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline


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

    async def resolve_entities(self, **kwargs):
        self.resolve_kwargs = kwargs
        return {"skipped": False, "merged_groups": 0}

    def get_unembedded_entities(self, limit=500):
        return []

    def get_entities_needing_summary(self, graph_name, limit=500):
        return []

    def get_relationships_needing_summary(self, graph_name, limit=500):
        return []

    def persist_entity_summaries(self, graph_name, summaries):
        pass

    def persist_relationship_summaries(self, graph_name, summaries):
        pass

    def upsert_vectors(self, ids, property_name, vectors):
        pass

    def validate_graph_build(self):
        return {
            "entity_count": 2,
            "chunk_count": 1,
            "evidence_link_count": 2,
            "entity_relationship_count": 1,
        }


class MyGraphStore(FakeGraphStore):
    pass


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
    '''))
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


def _make_test_schema():
    return GraphSchema(
        node_types=[
            NodeType(
                label="Person",
                properties=[PropertyType(name="name", type="STRING")],
            ),
        ],
        relationship_types=[],
    )


@pytest.mark.asyncio
async def test_build_from_text_orchestration(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        schema=movie_schema,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    result = await pipeline.build_from_text(
        "Alice directed Inception in 2010.",
        metadata={"source": "test"},
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
    movie_schema, fake_llm, fake_embedder, fake_writer, capsys
):
    store = MyGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        schema=movie_schema,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    await pipeline.build_from_text("Alice directed Inception.")

    captured = capsys.readouterr()
    assert "Writing graph document to My " in captured.out


@pytest.mark.asyncio
async def test_build_from_documents_with_page_windows(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        schema=movie_schema,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    pages = ["Page one", "Page two", "Page three"]
    results = await pipeline.build_from_documents(
        [{"pages": pages, "metadata": {"source": "test"}}],
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
        schema=movie_schema,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    pages = [
        {"text": "Page one", "metadata": {"record_id": "page-1"}},
        {"text": "Page two", "metadata": {"record_id": "page-2"}},
        {"text": "Page three", "metadata": {"record_id": "page-3"}},
    ]
    results = await pipeline.build_from_documents(
        [{"pages": pages, "metadata": {"source": "document-source", "collection": "movies"}}],
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
        schema=movie_schema,
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
        window_size=2,
        window_overlap=1,
    )

    assert len(results) == 2
    assert fake_writer.write_graph_document.call_count == 2
    assert results[0]["extraction"]["chunks"] == 1
    assert results[1]["extraction"]["chunks"] == 1


@pytest.mark.asyncio
async def test_build_from_documents_envelope_validation(
    movie_schema, fake_llm, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        schema=movie_schema,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    with pytest.raises(ValueError, match="must be a dict"):
        await pipeline.build_from_documents(["not a dict"])

    with pytest.raises(ValueError, match="both 'text' and 'pages'"):
        await pipeline.build_from_documents([{"text": "x", "pages": ["p"]}])

    with pytest.raises(ValueError, match="either 'text' or 'pages'"):
        await pipeline.build_from_documents([{"metadata": {}}])

    with pytest.raises(ValueError, match="'text' must be a string"):
        await pipeline.build_from_documents([{"text": 123}])

    with pytest.raises(ValueError, match="'pages' must be a list"):
        await pipeline.build_from_documents([{"pages": "not-a-list"}])

    with pytest.raises(ValueError, match="'metadata' must be a dict"):
        await pipeline.build_from_documents([{"text": "x", "metadata": "bad"}])

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
        schema=movie_schema,
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
        schema=movie_schema,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    result = await pipeline.build_from_text(
        "Alice directed Inception in 2010 with Christopher Nolan. " * 3,
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
        schema=_make_test_schema(),
    )
    doc_id = pipeline._make_document_id("hello", {"source": "My Doc"})
    assert doc_id == "doc:my-doc"


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
        schema=_make_test_schema(),
    )

    assert pipeline._graph_store_name() == expected_name


def test_graph_store_name_trims_graph_store_suffix():
    pipeline = GraphBuilderPipeline(
        graph_store=MyGraphStore(),
        llm=MagicMock(),
        embedder=MagicMock(),
        schema=_make_test_schema(),
    )

    assert pipeline._graph_store_name() == "My"


def test_graph_store_name_keeps_custom_class_without_graph_store_suffix():
    pipeline = GraphBuilderPipeline(
        graph_store=CustomBackend(),
        llm=MagicMock(),
        embedder=MagicMock(),
        schema=_make_test_schema(),
    )

    assert pipeline._graph_store_name() == "CustomBackend"


def test_make_document_id_without_source():
    store = FakeGraphStore()
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=MagicMock(),
        embedder=MagicMock(),
        schema=_make_test_schema(),
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
        schema=_make_test_schema(),
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
        schema=movie_schema,
        entity_resolution_strategy="hybrid",
        entity_resolution_aliases=aliases,
        entity_resolution_llm_guidance=guidance,
        entity_resolution_context_properties=context_properties,
        entity_resolution_conflict_properties=conflict_properties,
        allow_ai_auto_merge=True,
    )

    await pipeline._resolve_entities()

    assert store.resolve_kwargs == {
        "graph_name": "entity-graph",
        "strategy": "hybrid",
        "embedder": embedder,
        "llm": llm,
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
        schema=movie_schema,
        graph_writer=fake_writer,
        extraction_concurrency=3,
        max_gleanings=0,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    text = "Alice directed Inception in 2010 with Christopher Nolan. " * 3
    result = await pipeline.build_from_text(
        text,
        metadata={"source": "test"},
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
        schema=movie_schema,
        graph_writer=fake_writer,
        extraction_concurrency=2,
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

    pipeline.llm.ainvoke = AsyncMock(side_effect=tracked_invoke)

    await pipeline._extract_and_write_chunks(
        document_id="doc:test",
        text_hash="hash",
        chunks=chunks,
        metadata={},
    )

    assert max_active <= 2


@pytest.mark.asyncio
async def test_partial_extraction_failure_continues(
    movie_schema, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    llm = MagicMock()
    call_count = 0

    async def fail_second_extraction(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("provider error")
        return MagicMock(content='{"nodes": [], "relationships": []}')

    llm.ainvoke = AsyncMock(side_effect=fail_second_extraction)

    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=llm,
        embedder=fake_embedder,
        schema=movie_schema,
        graph_writer=fake_writer,
        extraction_concurrency=3,
        max_gleanings=0,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    text = "Alice directed Inception in 2010 with Christopher Nolan. " * 3
    chunker = TextChunker(chunk_size=15, chunk_overlap=5)
    expected_chunks = len(
        chunker.chunk_text(text, document_id="doc:test", metadata={})
    )
    result = await pipeline.build_from_text(
        text,
        metadata={"source": "test"},
        chunk_size=15,
        chunk_overlap=5,
        chunk_unit="characters",
    )

    assert result["extraction"]["chunks"] == expected_chunks
    assert llm.ainvoke.await_count == expected_chunks
    assert fake_writer.write_graph_document.call_count == 1


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
        schema=_make_test_schema(),
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
        schema=_make_test_schema(),
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
