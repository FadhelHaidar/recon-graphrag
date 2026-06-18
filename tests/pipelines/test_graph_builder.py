"""Tests for GraphBuilderPipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
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
    store.get_unembedded_entities = MagicMock(side_effect=[
        [  # first batch of unembedded entities
            {"id": "e1", "labels": ["Person"], "name": "Alice", "description": ""},
            {"id": "e2", "labels": ["Movie"], "name": "Inception", "description": ""},
        ],
        [],  # second batch (empty) -> break loop
    ])
    store.upsert_vectors = MagicMock()

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
    store.get_unembedded_entities.assert_called()
    store.upsert_vectors.assert_called_once()


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


def test_graph_store_name_uses_neo4j_class_name():
    pipeline = GraphBuilderPipeline(
        graph_store=Neo4jGraphStore(driver=MagicMock()),
        llm=MagicMock(),
        embedder=MagicMock(),
        schema=_make_test_schema(),
    )

    assert pipeline._graph_store_name() == "Neo4j"


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
    pipeline = GraphBuilderPipeline(
        graph_store=store,
        llm=llm,
        embedder=embedder,
        schema=movie_schema,
        entity_resolution_strategy="hybrid",
        entity_resolution_aliases=aliases,
        entity_resolution_llm_guidance=guidance,
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
    }
