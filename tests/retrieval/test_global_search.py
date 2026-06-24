"""Unit tests for GlobalSearchRetriever semantic behavior.

These tests lock the current map-reduce global search implementation so that
later paper-alignment work can add a separate mode without silently regressing
existing users.
"""

from __future__ import annotations

import pytest

from recon_graphrag.llm.base import LLMResponse
from recon_graphrag.retrieval.global_search import GlobalSearchRetriever


class FakeGraphStore:
    def __init__(self, communities=None):
        self.calls = []
        self._communities = communities or []

    def execute_query(self, query, parameters=None):
        self.calls.append(("execute_query", {"query": query.strip(), "params": parameters or {}}))
        return []

    def search_communities(self, index_name, query_vector, graph_name, top_k, level=None):
        self.calls.append(
            (
                "search_communities",
                {
                    "index_name": index_name,
                    "query_vector": query_vector,
                    "graph_name": graph_name,
                    "top_k": top_k,
                    "level": level,
                },
            )
        )
        return self._communities


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    async def async_embed_query(self, text):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class FakeLLM:
    def __init__(self, responses=None):
        self.calls = []
        self._responses = list(responses) if responses is not None else ["answer"]
        self._index = 0

    async def ainvoke(self, prompt):
        self.calls.append(prompt)
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        if isinstance(response, Exception):
            raise response
        return LLMResponse(content=response)


def _communities(n: int, level: int = 0) -> list[dict]:
    return [
        {
            "id": f"c{i}",
            "summary": f"summary {i}",
            "level": level,
            "score": 0.9 - (i * 0.05),
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_query_embedded_once():
    store = FakeGraphStore(_communities(1))
    embedder = FakeEmbedder()
    llm = FakeLLM()
    retriever = GlobalSearchRetriever(store, llm, embedder)

    await retriever.search("themes")

    assert len(embedder.calls) == 1
    assert embedder.calls[0] == "themes"


@pytest.mark.asyncio
async def test_search_communities_receives_correct_params():
    store = FakeGraphStore(_communities(1, level=2))
    embedder = FakeEmbedder()
    llm = FakeLLM()
    retriever = GlobalSearchRetriever(store, llm, embedder)

    await retriever.search("themes", top_k=5, level=2)

    search_calls = [c for c in store.calls if c[0] == "search_communities"]
    assert len(search_calls) == 1
    args = search_calls[0][1]
    assert args["top_k"] == 5
    assert args["level"] == 2
    assert args["index_name"] == "community-embeddings"
    assert args["graph_name"] == "entity-graph"


@pytest.mark.asyncio
async def test_map_calls_preserve_order():
    communities = _communities(3, level=1)
    store = FakeGraphStore(communities)
    embedder = FakeEmbedder()
    llm = FakeLLM(responses=["a", "b", "c", "final"])
    retriever = GlobalSearchRetriever(store, llm, embedder)

    await retriever.search("themes", top_k=3)

    map_prompts = [c for c in llm.calls if "Partial Answer:" in c]
    assert len(map_prompts) == 3
    for i, comm in enumerate(communities):
        assert comm["summary"] in map_prompts[i]


@pytest.mark.asyncio
async def test_map_calls_are_sequential():
    """Current implementation awaits each map call in a for loop."""
    communities = _communities(3, level=1)
    store = FakeGraphStore(communities)
    embedder = FakeEmbedder()

    order = []

    class TrackingLLM:
        async def ainvoke(self, prompt):
            order.append(len(order))
            return LLMResponse(content=f"ans{order[-1]}")

    retriever = GlobalSearchRetriever(store, TrackingLLM(), embedder)
    await retriever.search("themes", top_k=3)

    # Three map calls plus one reduce call happen in total.
    # Sequential execution means the first three calls are the map calls.
    assert order[:3] == [0, 1, 2]
    assert len(order) == 4


@pytest.mark.asyncio
async def test_every_map_answer_reaches_reduce():
    communities = _communities(3, level=1)
    store = FakeGraphStore(communities)
    embedder = FakeEmbedder()
    llm = FakeLLM(responses=["ans1", "", "ans3", "final"])
    retriever = GlobalSearchRetriever(store, llm, embedder)

    await retriever.search("themes", top_k=3)

    reduce_prompts = [c for c in llm.calls if "Final Answer:" in c]
    assert len(reduce_prompts) == 1
    reduce_prompt = reduce_prompts[0]
    assert "[Perspective 1]: ans1" in reduce_prompt
    assert "[Perspective 2]:" in reduce_prompt
    assert "[Perspective 3]: ans3" in reduce_prompt


@pytest.mark.asyncio
async def test_no_result_does_not_call_llm():
    store = FakeGraphStore([])
    embedder = FakeEmbedder()
    llm = FakeLLM()
    retriever = GlobalSearchRetriever(store, llm, embedder)

    result = await retriever.search("themes", top_k=5)

    assert result.answer == "No relevant communities found."
    assert result.context == ""
    assert llm.calls == []


@pytest.mark.asyncio
async def test_custom_map_reduce_prompts_work():
    communities = _communities(1, level=0)
    store = FakeGraphStore(communities)
    embedder = FakeEmbedder()
    llm = FakeLLM(responses=["partial", "final"])
    retriever = GlobalSearchRetriever(
        store,
        llm,
        embedder,
        map_prompt="CUSTOM MAP {query} {summary}",
        reduce_prompt="CUSTOM REDUCE {query} {partial_answers}",
    )

    await retriever.search("themes")

    assert any("CUSTOM MAP themes summary 0" in c for c in llm.calls)
    assert any("CUSTOM REDUCE themes [Perspective 1]: partial" in c for c in llm.calls)


@pytest.mark.asyncio
async def test_result_includes_formatted_context():
    communities = _communities(2, level=1)
    store = FakeGraphStore(communities)
    embedder = FakeEmbedder()
    llm = FakeLLM(responses=["a", "b", "final"])
    retriever = GlobalSearchRetriever(store, llm, embedder)

    result = await retriever.search("themes", top_k=2)

    assert "Report Segment c0 (level 1):" in result.context
    assert "Report Segment c1 (level 1):" in result.context
    assert "\n\n---\n\n" in result.context
