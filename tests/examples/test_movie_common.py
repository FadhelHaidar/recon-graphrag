"""Tests for shared movie-example helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EXAMPLE_DIR))

from common import SEARCH_OPTIONS, configure_movie_rag, ingest_graph_document, run_movie_search_suite  # noqa: E402
from common import extract_graph_document_from_pages  # noqa: E402
import ingest  # noqa: E402
from ingest import ingest_artifact  # noqa: E402
from query_suite import MOVIE_QUERY_SUITE  # noqa: E402
from recon_graphrag.extraction.types import (  # noqa: E402
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    EvidenceLink,
    GraphDocument,
    RelationshipRecord,
)
from recon_graphrag.extraction.schema import (  # noqa: E402
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)


def test_search_options_are_shared_for_global_and_drift():
    assert SEARCH_OPTIONS["global"]["community_level"] == "coarsest"
    assert SEARCH_OPTIONS["drift"]["community_level"] == "finest"


def test_movie_query_suite_is_shared():
    assert MOVIE_QUERY_SUITE
    assert {"query", "modes", "test_objective"} <= set(MOVIE_QUERY_SUITE[0])


def test_configure_movie_rag_sets_prompts():
    graph_rag = MagicMock()

    configured = configure_movie_rag(graph_rag)

    assert configured is graph_rag
    assert graph_rag.local.answer_prompt
    assert graph_rag.drift.answer_prompt
    assert graph_rag.global_.map_prompt
    assert graph_rag.global_.reduce_prompt


@pytest.mark.asyncio
async def test_run_movie_search_suite_uses_shared_search_options(capsys):
    graph_rag = MagicMock()
    graph_rag.search = AsyncMock(return_value=MagicMock(answer="ok"))
    test_queries = [
        {
            "query": "What happened?",
            "modes": ["global"],
            "test_objective": "exercise shared runner",
        }
    ]

    await run_movie_search_suite(graph_rag, test_queries)

    graph_rag.search.assert_awaited_once_with(
        "What happened?",
        mode="global",
        **SEARCH_OPTIONS["global"],
    )
    assert "GLOBAL ANSWER" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_extract_graph_document_from_pages_does_not_need_graph_store():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="""
            {
              "nodes": [
                {"id": "alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "inception", "label": "Movie", "properties": {"name": "Inception"}}
              ],
              "relationships": [
                {"source_id": "alice", "target_id": "inception", "type": "DIRECTED"}
              ]
            }
            """
        )
    )
    schema = GraphSchema(
        node_types=[
            NodeType(label="Person", properties=[PropertyType(name="name", type="STRING")]),
            NodeType(label="Movie", properties=[PropertyType(name="name", type="STRING")]),
        ],
        relationship_types=[RelationshipType(label="DIRECTED")],
        patterns=[("Person", "DIRECTED", "Movie")],
    )

    graph_document = await extract_graph_document_from_pages(
        ["Alice directed Inception."],
        llm=llm,
        schema=schema,
        metadata={"source": "unit"},
        window_size=1,
        window_overlap=0,
    )

    assert graph_document.document.id == "doc:unit"
    assert len(graph_document.entities) == 2
    assert len(graph_document.relationships) == 1


@pytest.mark.asyncio
async def test_ingest_graph_document_resolves_after_write_before_embeddings(monkeypatch):
    events = []
    graph_document = GraphDocument(
        document=DocumentRecord(
            id="doc:unit",
            text_hash="hash",
            graph_name="movie-graph",
        ),
        chunks=[
            ChunkRecord(
                id="chunk:1",
                document_id="doc:unit",
                text="Alice directed Inception.",
                index=0,
                graph_name="movie-graph",
            )
        ],
        entities=[
            EntityRecord(
                id="alice",
                type="Person",
                properties={"name": "Alice"},
                graph_name="movie-graph",
            )
        ],
        relationships=[
            RelationshipRecord(
                id="rel:1",
                source_id="alice",
                target_id="alice",
                type="KNOWS",
                graph_name="movie-graph",
            )
        ],
        evidence_links=[
            EvidenceLink(
                chunk_id="chunk:1",
                entity_id="alice",
                graph_name="movie-graph",
            )
        ],
    )

    class FakeStore:
        def write_graph_document(self, document):
            events.append("write")
            assert document is graph_document
            return {"entities": 1}

        def backfill_descriptions(self):
            events.append("backfill")

        async def resolve_entities(self, **kwargs):
            events.append(("resolve", kwargs))
            return {"merged_nodes": 0}

        def validate_graph_build(self):
            events.append("validate")
            return {"entity_count": 1}

    class FakeIndexManager:
        def __init__(self, store, embedding_dim):
            events.append("index_init")

        def create_indexes(self):
            events.append("indexes")

    class FakeCommunityEmbedder:
        def __init__(self, store, embedder):
            events.append("embedder_init")

        async def embed_entities(self):
            events.append("embed")

    monkeypatch.setattr("common.CommunityEmbedder", FakeCommunityEmbedder)

    result = await ingest_graph_document(
        FakeStore(),
        graph_document,
        embedder=MagicMock(),
        index_manager_cls=FakeIndexManager,
        entity_resolution_strategy="normalized",
    )

    assert events == [
        "index_init",
        "indexes",
        "write",
        "backfill",
        ("resolve", {"graph_name": "movie-graph", "strategy": "normalized", "embedder": None}),
        "embedder_init",
        "embed",
        "validate",
    ]
    assert result["entity_resolution"] == {"merged_nodes": 0}


@pytest.mark.asyncio
async def test_ingest_artifact_finalizes_after_all_backend_writes(monkeypatch, tmp_path):
    events = []
    graph_document = GraphDocument(
        document=DocumentRecord(id="doc:unit", text_hash="hash"),
        chunks=[],
        entities=[],
        relationships=[],
        evidence_links=[],
    )

    class FakeStore:
        def __init__(self, name):
            self.name = name

        def write_graph_document(self, document):
            events.append((self.name, "write"))
            assert document is graph_document
            return {"entities": 0}

        def backfill_descriptions(self):
            events.append((self.name, "backfill"))

        async def resolve_entities(self, **kwargs):
            events.append((self.name, "resolve"))
            return {"merged_nodes": 0}

        def validate_graph_build(self):
            events.append((self.name, "validate"))
            return {"entity_count": 0}

    class FakeIndexManager:
        def __init__(self, store, embedding_dim):
            self.store = store

        def create_indexes(self):
            events.append((self.store.name, "indexes"))

    class FakeCommunityEmbedder:
        def __init__(self, store, embedder):
            self.store = store

        async def embed_entities(self):
            events.append((self.store.name, "embed"))

    monkeypatch.setattr(ingest, "load_graph_document_json", lambda path: graph_document)
    monkeypatch.setattr(ingest, "get_embedder", lambda provider: MagicMock())
    monkeypatch.setattr(ingest, "get_neo4j_store", lambda: FakeStore("neo4j"))
    monkeypatch.setattr(ingest, "get_memgraph_store", lambda: FakeStore("memgraph"))
    monkeypatch.setattr(ingest, "Neo4jIndexManager", FakeIndexManager)
    monkeypatch.setattr(ingest, "MemgraphIndexManager", FakeIndexManager)
    monkeypatch.setattr("common.CommunityEmbedder", FakeCommunityEmbedder)

    result = await ingest_artifact(
        tmp_path / "movie_graph.json",
        backend="both",
        embedder_provider="openai",
    )

    assert events == [
        ("neo4j", "indexes"),
        ("neo4j", "write"),
        ("memgraph", "indexes"),
        ("memgraph", "write"),
        ("neo4j", "backfill"),
        ("neo4j", "resolve"),
        ("neo4j", "embed"),
        ("neo4j", "validate"),
        ("memgraph", "backfill"),
        ("memgraph", "resolve"),
        ("memgraph", "embed"),
        ("memgraph", "validate"),
    ]
    assert set(result) == {"neo4j", "memgraph"}
