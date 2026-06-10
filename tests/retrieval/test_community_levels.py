"""Tests for semantic community-level selection."""

from types import SimpleNamespace

import pytest

from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval.community_levels import resolve_community_level
from recon_graphrag.retrieval.drift import DriftSearchRetriever
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever


class FakeGraphStore:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.calls.append((query.strip(), params))

        if "RETURN max(c.level) AS level" in query:
            return [{"level": 2}]

        if "db.index.vector.queryNodes" in query:
            return [
                {
                    "id": "c2",
                    "summary": "Coarse community summary",
                    "level": params.get("level"),
                    "score": 0.9,
                }
            ]

        if "RETURN c.id AS id, c.summary AS summary, c.level AS level" in query:
            return [{"id": "c2", "summary": "Coarse community summary", "level": 2}]

        if "RETURN DISTINCT e.name AS name" in query:
            return []

        return []

    @property
    def driver(self):
        return None


class FakeEmbedder:
    async def async_embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeLLM:
    async def ainvoke(self, prompt):
        return LLMResponse(content="answer")


class FakeHybridRetriever:
    async def search(self, query_text, top_k, **kwargs):
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    content={
                        "title": "Inception (Movie)",
                        "relationships": [],
                        "source_text": ["source"],
                        "communities": [
                            {
                                "id": "c0",
                                "level": 0,
                                "graph_name": "entity-graph",
                                "summary": "Fine summary",
                            },
                            {
                                "id": "c2",
                                "level": 2,
                                "graph_name": "entity-graph",
                                "summary": "Coarse summary",
                            },
                        ],
                    }
                )
            ]
        )


def test_resolve_community_level_aliases():
    store = FakeGraphStore()

    assert resolve_community_level(store, "entity-graph", None) is None
    assert resolve_community_level(store, "entity-graph", "all") is None
    assert resolve_community_level(store, "entity-graph", "finest") == 0
    assert resolve_community_level(store, "entity-graph", 1) == 1
    assert resolve_community_level(store, "entity-graph", "coarsest") == 2


def test_resolve_community_level_rejects_negative_level():
    with pytest.raises(ValueError):
        resolve_community_level(FakeGraphStore(), "entity-graph", -1)


@pytest.mark.asyncio
async def test_global_search_accepts_coarsest_alias():
    store = FakeGraphStore()
    retriever = GlobalSearchRetriever(store, FakeLLM(), FakeEmbedder())

    result = await retriever.search("themes", top_k=1, community_level="coarsest")

    assert result.answer == "answer"
    vector_params = [
        params for query, params in store.calls if "db.index.vector.queryNodes" in query
    ][0]
    assert vector_params["level"] == 2


@pytest.mark.asyncio
async def test_drift_search_accepts_coarsest_alias():
    store = FakeGraphStore()
    retriever = object.__new__(DriftSearchRetriever)
    retriever.graph_store = store
    retriever.llm = FakeLLM()
    retriever.graph_name = "entity-graph"
    retriever.community_level = "coarsest"
    retriever.answer_prompt = "{query}\n{entity_context}\n{community_context}\n{bridging_context}"
    retriever._retriever = FakeHybridRetriever()

    result = await retriever.search("themes", top_k=1)

    assert result.answer == "answer"
    summary_params = [
        params
        for query, params in store.calls
        if "RETURN c.id AS id, c.summary AS summary, c.level AS level" in query
    ][0]
    assert summary_params["keys"] == [{"id": "c2", "level": 2}]
