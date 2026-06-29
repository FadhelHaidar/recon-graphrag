"""Tests for semantic community-level selection."""

import pytest

from recon_graphrag.llm import LLMResponse
from recon_graphrag.retrieval.community_levels import resolve_community_level
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever


class FakeGraphStore:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, parameters=None):
        params = parameters or {}
        self.calls.append(("execute_query", {"query": query.strip(), "params": params}))

        if "RETURN max(c.level) AS level" in query:
            return [{"level": 2}]

        # Global search reads reports at a level
        if "Community" in query and "report_text" in query:
            level = params.get("level", 0)
            return [
                {"id": "c2", "level": level, "report_text": "Coarse community report"}
            ]

        return []

    def vector_search(self, index_name, query_vector, k, label=None, filters=None):
        return [{"id": "a", "score": 0.8}]

    def keyword_search(self, index_name, query_text, k, label=None, filters=None):
        return [{"id": "a", "score": 1.0}]

    def fetch_entity_context(self, matches, retrieval_query=None, query_params=None, mode="local", graph_name=None):
        return [
            {
                "title": "Test (Entity)",
                "relationships": [],
                "source_text": [],
                "source_chunk_ids": ["chunk:1"],
                "score": 0.8,
            }
        ]

    def resolve_chunk_citations(self, graph_name, chunk_ids):
        self.calls.append(
            (
                "resolve_chunk_citations",
                {"graph_name": graph_name, "chunk_ids": chunk_ids},
            )
        )
        return [
            {
                "document_id": "doc:1",
                "chunk_id": "chunk:1",
                "document_metadata": {"collection": "movies"},
                "chunk_metadata": {"record_id": "row-42", "source": "row-source"},
            }
        ]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return LLMResponse(content="answer")


class FakeEmbedder:
    async def async_embed_query(self, text):
        return [0.1, 0.2, 0.3]



def test_resolve_community_level_aliases():
    store = FakeGraphStore()

    assert resolve_community_level(store, "entity-graph", None) is None
    assert resolve_community_level(store, "entity-graph", "all") is None
    # After reversal: level 0 = coarsest, highest = finest
    assert resolve_community_level(store, "entity-graph", "coarsest") == 0
    assert resolve_community_level(store, "entity-graph", 0) == 0
    assert resolve_community_level(store, "entity-graph", 1) == 1
    assert resolve_community_level(store, "entity-graph", "finest") == 2


def test_resolve_community_level_rejects_negative_level():
    with pytest.raises(ValueError):
        resolve_community_level(FakeGraphStore(), "entity-graph", -1)


def test_resolve_community_level_rejects_invalid_string():
    with pytest.raises(ValueError):
        resolve_community_level(FakeGraphStore(), "entity-graph", "middle")


class EmptyGraphStore:
    def execute_query(self, query, parameters=None):
        if "RETURN max(c.level) AS level" in query:
            return [{"level": None}]
        return []


def test_resolve_finest_on_empty_graph_returns_none():
    store = EmptyGraphStore()
    assert resolve_community_level(store, "entity-graph", "finest") is None


def test_resolve_coarsest_on_empty_graph_returns_zero():
    store = EmptyGraphStore()
    assert resolve_community_level(store, "entity-graph", "coarsest") == 0


@pytest.mark.asyncio
async def test_global_search_accepts_coarsest_alias():
    store = FakeGraphStore()
    retriever = GlobalSearchRetriever(store, FakeLLM())

    result = await retriever.search("themes", community_level="coarsest")

    assert result.answer.strip()
    # After reversal: "coarsest" → level 0
    report_call = [
        c for c in store.calls
        if c[0] == "execute_query" and "report_text" in c[1]["query"]
    ]
    assert len(report_call) == 1
    assert report_call[0][1]["params"]["level"] == 0


@pytest.mark.asyncio
async def test_drift_search_accepts_coarsest_alias():
    """DriftSearchRetriever passes community_level to vector_search_community_reports."""
    from recon_graphrag.retrieval.drift import DriftSearchRetriever

    store = FakeGraphStore()
    store.vector_search_community_reports_calls = []

    def _mock_vscr(query_vector, graph_name, top_k=3, level=None):
        store.vector_search_community_reports_calls.append({"top_k": top_k, "level": level})
        return [{"id": "r1", "level": level, "report_text": "Test report"}]

    store.vector_search_community_reports = _mock_vscr
    llm = FakeLLM()
    embedder = FakeEmbedder()

    retriever = DriftSearchRetriever(store, llm, embedder)

    result = await retriever.search(
        "themes", top_k=1, community_level="coarsest"
    )

    # "coarsest" → level 0 after reversal
    assert store.vector_search_community_reports_calls[0]["level"] == 0



