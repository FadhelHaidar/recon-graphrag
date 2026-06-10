"""Tests for GraphBuilderPipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline


class FakeGraphStore:
    def __init__(self):
        self.queries = []

    def execute_query(self, query, parameters=None):
        self.queries.append(query.strip())
        return []

    def create_vector_index(self, **kwargs):
        pass

    def create_fulltext_index(self, **kwargs):
        pass

    def upsert_vectors(self, **kwargs):
        pass

    @property
    def driver(self):
        return None


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


@pytest.mark.asyncio
async def test_build_from_pages_with_windows(
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
    result = await pipeline.build_from_pages(
        pages=pages,
        metadata={"source": "test"},
        window_size=2,
        window_overlap=1,
    )

    assert "extraction" in result
    assert result["extraction"]["chunks"] == 2  # windows: 1-2, 2-3


@pytest.mark.asyncio
async def test_build_from_documents(
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

    results = await pipeline.build_from_documents([
        {"text": "Alice directed Inception.", "metadata": {"source": "a"}},
        {"text": "Bob acted in Inception.", "metadata": {"source": "b"}},
    ])

    assert len(results) == 2
    assert fake_writer.write_graph_document.call_count == 2


@pytest.mark.asyncio
async def test_build_from_text_with_entity_embedding_loop(
    movie_schema, fake_llm, fake_embedder
):
    store = FakeGraphStore()
    # Writer calls: documents, chunks, entities, evidence_links, relationships
    # Then: backfill descriptions, validation
    # Then: embed_entities loop (2 batches)
    store.execute_query = MagicMock(side_effect=[
        [],  # write documents
        [],  # write chunks
        [],  # write entities
        [],  # write evidence links
        [],  # write relationships
        [],  # backfill descriptions
        [],  # validation
        [  # first batch of unembedded entities
            {"id": "e1", "labels": ["Person"], "name": "Alice", "description": ""},
            {"id": "e2", "labels": ["Movie"], "name": "Inception", "description": ""},
        ],
        [],  # second batch (empty) -> break loop
    ])

    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=fake_llm,
        embedder=fake_embedder,
        schema=movie_schema,
        perform_entity_resolution=False,
        embed_entities=True,
    )

    result = await pipeline.build_from_text("Alice directed Inception.")
    assert "extraction" in result


@pytest.mark.asyncio
async def test_build_from_text_raises_when_all_extractions_fail(
    movie_schema, fake_embedder, fake_writer
):
    store = FakeGraphStore()
    failing_llm = MagicMock()
    failing_llm.ainvoke = AsyncMock(side_effect=RuntimeError("provider failed"))

    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=failing_llm,
        embedder=fake_embedder,
        schema=movie_schema,
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
    )

    with pytest.raises(RuntimeError, match="Extraction failed for all"):
        await pipeline.build_from_text("Alice directed Inception.")

    fake_writer.write_graph_document.assert_not_called()


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
