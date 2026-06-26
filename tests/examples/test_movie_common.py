"""Tests for shared movie-example helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


from examples import common, ingest
from examples.common import (
    SEARCH_OPTIONS,
    configure_movie_rag,
    extract_graph_document_from_pages,
    run_movie_search_suite,
)
from examples.ingest import ingest_artifact
from examples.prompts import GLOBAL_MAP_PROMPT, GLOBAL_REDUCE_PROMPT
from examples.query_suite import MOVIE_QUERY_SUITE
from recon_graphrag.extraction.types import (
    DocumentRecord,
    GraphDocument,
)
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)


def test_search_options_are_shared_for_global_and_drift():
    assert SEARCH_OPTIONS["global"]["community_level"] == "coarsest"
    assert SEARCH_OPTIONS["drift"]["community_level"] == "finest"


def test_all_backend_targets_expand_to_registered_backends(monkeypatch):
    assert "all" in common.BACKEND_CHOICES
    assert "both" not in common.BACKEND_CHOICES

    class FakeNeo4jIndexManager:
        pass

    class FakeMemgraphIndexManager:
        pass

    monkeypatch.setitem(
        common.BACKEND_REGISTRY,
        "neo4j",
        (lambda: "neo4j-store", FakeNeo4jIndexManager),
    )
    monkeypatch.setitem(
        common.BACKEND_REGISTRY,
        "memgraph",
        (lambda: "memgraph-store", FakeMemgraphIndexManager),
    )

    targets = common.get_backend_targets("all")

    assert targets == [
        ("neo4j", "neo4j-store", FakeNeo4jIndexManager),
        ("memgraph", "memgraph-store", FakeMemgraphIndexManager),
    ]
    with pytest.raises(ValueError):
        common.get_backend_targets("both")


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


def test_movie_global_prompts_match_retriever_contract():
    map_prompt = GLOBAL_MAP_PROMPT.format(query="q", batch_text="reports")
    reduce_prompt = GLOBAL_REDUCE_PROMPT.format(query="q", partial_text="partials")

    assert '"helpfulness"' in map_prompt
    assert '"answer"' in map_prompt
    assert "partials" in reduce_prompt


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
async def test_ingest_artifact_finalizes_after_all_backend_writes(monkeypatch):
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

    class FakeEntityEmbedder:
        def __init__(self, store, embedder):
            self.store = store

        async def embed_entities(self):
            events.append((self.store.name, "embed"))

    monkeypatch.setattr(ingest, "load_graph_document_json", lambda path: graph_document)
    monkeypatch.setattr(ingest, "get_embedder", lambda provider: MagicMock())
    monkeypatch.setattr(
        ingest,
        "get_backend_targets",
        lambda backend: [
            ("neo4j", FakeStore("neo4j"), FakeIndexManager),
            ("memgraph", FakeStore("memgraph"), FakeIndexManager),
        ],
    )
    monkeypatch.setattr(common, "EntityEmbedder", FakeEntityEmbedder)

    result = await ingest_artifact(
        Path("movie_graph.json"),
        backend="all",
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


@pytest.mark.asyncio
async def test_ingest_artifact_creates_llm_for_hybrid_resolution(monkeypatch):
    events = []
    graph_document = GraphDocument(
        document=DocumentRecord(id="doc:unit", text_hash="hash"),
        chunks=[],
        entities=[],
        relationships=[],
        evidence_links=[],
    )
    embedder = MagicMock(name="embedder")
    llm = MagicMock(name="llm")

    class FakeStore:
        def write_graph_document(self, document):
            return {"entities": 0}

    class FakeIndexManager:
        def __init__(self, store, embedding_dim):
            pass

        def create_indexes(self):
            pass

    async def fake_finalize(
        store,
        graph_document,
        embedder,
        llm=None,
        entity_resolution_strategy="normalized",
        resolve_entities=True,
        embed_entities=True,
        allow_ai_auto_merge=False,
    ):
        events.append(
            {
                "embedder": embedder,
                "llm": llm,
                "strategy": entity_resolution_strategy,
                "resolve_entities": resolve_entities,
                "embed_entities": embed_entities,
                "allow_ai_auto_merge": allow_ai_auto_merge,
            }
        )
        return {"entity_resolution": {"merged_nodes": 0}}

    monkeypatch.setattr(ingest, "load_graph_document_json", lambda path: graph_document)
    monkeypatch.setattr(ingest, "get_embedder", lambda provider: embedder)
    monkeypatch.setattr(ingest, "get_llm", lambda provider: llm)
    monkeypatch.setattr(
        ingest,
        "get_backend_targets",
        lambda backend: [("neo4j", FakeStore(), FakeIndexManager)],
    )
    monkeypatch.setattr(ingest, "finalize_graph_ingest", fake_finalize)

    await ingest_artifact(
        Path("movie_graph.json"),
        backend="neo4j",
        embedder_provider="openai",
        llm_provider="openrouter",
        entity_resolution_strategy="hybrid",
        allow_ai_auto_merge=True,
    )

    assert events == [
        {
            "embedder": embedder,
            "llm": llm,
            "strategy": "hybrid",
            "resolve_entities": True,
            "embed_entities": True,
            "allow_ai_auto_merge": True,
        }
    ]
